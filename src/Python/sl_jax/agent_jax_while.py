"""Fully-JAX trial loop with batch-level early termination via lax.while_loop.

Improvement over ``agent_jax_native``: the loop exits as soon as
*all seeds in the batch* have died, instead of running through the
fixed ``max_steps_per_trial`` regardless. Within the body, dead seeds
are masked out so their state doesn't update further.

This mirrors the early-termination behaviour of ``agent_batched``'s
Python ``for`` loop while keeping the whole loop GPU-resident.

Single jitted+vmapped kernel per trial; no Python↔GPU dispatch cost
per step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import sys
import time
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
from sl.env import GenerativeModel, initialise_environment  # noqa: E402

from .config import GridConfig, JaxOptions
from .env_jax import (
    calculate_posterior_jax,
    normalise_matrix_columns_jax,
    real_dirichlet_update_jax,
    sample_categorical,
    sample_observations_jax,
    update_environment_states_jax,
    update_needs_jax,
)
from .planning import PlannerInputsJax, plan_batched


# ---------------------------------------------------------------------------
# State container — easier to pass through while_loop.
# ---------------------------------------------------------------------------


def make_trial_runner(
    grid_food, grid_water, grid_sleep,
    A_pos, A_resource, A_hill, B_pos, B_ctx, bb_ctx,
    food_threshold, water_threshold, sleep_threshold, max_steps,
    horizon, w_novelty, w_learning, w_epistemic, pref_iprec,
    learning_proportion, learning_scale, learning_floor,
):
    """Returns a vmapped+jitted trial runner.

    Each call: takes (key, Q_pos, Q_ctx, true_pos, true_ctx, a_resource)
    per seed (axis-0 batched), runs one trial, returns
    (t_terminal, a_resource_final, key_final) per seed.
    """

    def trial_one_seed(key, Q_pos, Q_ctx, true_pos, true_ctx, a_resource):
        """Run one trial for one seed via lax.while_loop."""

        def cond_fn(state):
            (key_, Q_pos_, Q_ctx_, true_pos_, true_ctx_,
             t_food, t_water, t_sleep,
             a_, prev_chosen, alive, t_terminal, t) = state
            return alive & (t < max_steps)

        def body_fn(state):
            (key_, Q_pos_, Q_ctx_, true_pos_, true_ctx_,
             t_food, t_water, t_sleep,
             a_, prev_chosen, alive, t_terminal, t) = state

            # 1. env transition (only if t > 0)
            def transition_branch(args):
                k_, Qp, Qc, tp, tc = args
                return update_environment_states_jax(
                    Qp, Qc, tp, tc, prev_chosen,
                    B_pos, B_ctx, bb_ctx, k_,
                )

            def initial_branch(args):
                k_, Qp, Qc, tp, tc = args
                return Qp, Qc, tp.astype(jnp.int32), tc.astype(jnp.int32), k_

            Q_pos_, Q_ctx_, true_pos_, true_ctx_, key_ = jax.lax.cond(
                t > 0,
                transition_branch,
                initial_branch,
                (key_, Q_pos_, Q_ctx_, true_pos_, true_ctx_),
            )

            # 2. update needs
            t_food, t_water, t_sleep = update_needs_jax(
                grid_food, grid_water, grid_sleep,
                true_pos_, true_ctx_, t,
                t_food, t_water, t_sleep,
            )

            died_now = ((t_food >= food_threshold)
                        | (t_water >= water_threshold)
                        | (t_sleep >= sleep_threshold))

            # 3. observation
            O_pos, O_res, O_hill, key_ = sample_observations_jax(
                A_pos, A_resource, A_hill, true_pos_, true_ctx_, key_,
            )

            # 4. posterior
            y_resource = normalise_matrix_columns_jax(a_)
            P_pos, P_ctx = calculate_posterior_jax(
                Q_pos_, Q_ctx_, y_resource, A_hill, O_res, O_hill,
            )

            Q_pos_ = P_pos
            Q_ctx_ = P_ctx

            # 5. plan
            inputs = PlannerInputsJax(
                A_pos=A_pos, A_resource=A_resource, A_hill=A_hill,
                a_resource=a_, y_resource=y_resource,
                B_pos=B_pos, bb_ctx=bb_ctx,
                w_novelty=w_novelty, w_learning=w_learning,
                w_epistemic=w_epistemic,
                preference_inverse_precision=pref_iprec,
            )
            chosen, _G = plan_batched(
                P_pos, P_ctx, t_food, t_water, t_sleep,
                inputs, horizon, False,
            )

            # 6. Dirichlet update (t>0 AND not just died)
            do_update = (t > 0) & (~died_now)

            def do_update_branch(_):
                return real_dirichlet_update_jax(
                    a_, O_res, P_pos, P_ctx,
                    proportion=learning_proportion,
                    scale=learning_scale,
                    floor=learning_floor,
                )

            def skip_update_branch(_):
                return a_

            a_ = jax.lax.cond(do_update, do_update_branch, skip_update_branch, None)

            # 7. terminal step
            new_t_terminal = jnp.where(died_now, t + 1, t_terminal)
            new_alive = ~died_now

            return (
                key_, Q_pos_, Q_ctx_, true_pos_, true_ctx_,
                t_food, t_water, t_sleep,
                a_, jnp.int32(chosen), new_alive, new_t_terminal, t + 1,
            )

        init = (
            key, Q_pos, Q_ctx, true_pos, true_ctx,
            jnp.int32(0), jnp.int32(0), jnp.int32(0),
            a_resource,
            jnp.int32(0),     # prev_chosen
            jnp.bool_(True),  # alive
            jnp.int32(max_steps),  # t_terminal default
            jnp.int32(0),     # t (loop counter)
        )
        final = jax.lax.while_loop(cond_fn, body_fn, init)
        (_, _, _, _, _, _, _, _, a_final, _, _, t_terminal, _) = final
        return t_terminal, a_final

    # vmap across seeds; jit for fusion.
    in_axes = (0, None, None, None, 0, 0)  # key, Q_pos, Q_ctx, true_pos, true_ctx, a
    return jax.jit(jax.vmap(trial_one_seed, in_axes=in_axes))


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


@dataclass
class JaxWhileRun:
    seeds: List[int]
    survivals: List[List[int]]
    wall_seconds: float = 0.0


def run_while_batched(
    grid: GridConfig, opts: JaxOptions, seeds: List[int],
) -> JaxWhileRun:
    n_seeds = len(seeds)

    np_model: GenerativeModel = initialise_environment(grid)
    A_pos = jnp.asarray(np_model.A_pos)
    A_resource = jnp.asarray(np_model.A_resource)
    A_hill = jnp.asarray(np_model.A_hill)
    B_pos = jnp.asarray(np_model.B_pos)
    B_ctx = jnp.asarray(np_model.B_ctx)
    b_ctx = np_model.b_ctx
    bb_np = np.empty_like(b_ctx)
    for a in range(b_ctx.shape[2]):
        s = b_ctx[:, :, a].sum(axis=0, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        bb_np[:, :, a] = b_ctx[:, :, a] / s
    bb_ctx = jnp.asarray(bb_np)

    grid_food = jnp.asarray(grid.food_sources, dtype=jnp.int32)
    grid_water = jnp.asarray(grid.water_sources, dtype=jnp.int32)
    grid_sleep = jnp.asarray(grid.sleep_sources, dtype=jnp.int32)

    a0 = jnp.asarray(np_model.a_resource)
    a_per_seed = jnp.broadcast_to(a0, (n_seeds,) + a0.shape).copy()

    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])

    D_ctx = jnp.asarray(np_model.D_ctx)
    D_pos = jnp.asarray(np_model.D_pos)

    # Build the runner once; reuse across trials.
    trial_runner = make_trial_runner(
        grid_food, grid_water, grid_sleep,
        A_pos, A_resource, A_hill, B_pos, B_ctx, bb_ctx,
        opts.food_threshold, opts.water_threshold, opts.sleep_threshold,
        opts.max_steps_per_trial, opts.horizon,
        opts.w_novelty, opts.w_learning, opts.w_epistemic, opts.preference,
        opts.learning_proportion, opts.learning_scale, opts.learning_floor,
    )

    survivals: List[List[int]] = [[] for _ in range(n_seeds)]
    t_total0 = time.time()

    def _sample_initial_ctx(key):
        return sample_categorical(key, D_ctx)

    for trial in range(opts.num_trials):
        # Per-seed RNG splits.
        ks = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
        key_for_ctx = ks[:, 0]
        key_for_trial = ks[:, 1]
        keys = ks[:, 2]

        true_ctx0 = jax.vmap(_sample_initial_ctx)(key_for_ctx).astype(jnp.int32)
        true_pos0 = jnp.int32(grid.start_position)

        t_term, a_per_seed = trial_runner(
            key_for_trial, D_pos, D_ctx, true_pos0, true_ctx0, a_per_seed,
        )

        t_term_np = np.asarray(t_term)
        for i in range(n_seeds):
            survivals[i].append(int(t_term_np[i]))

    elapsed = time.time() - t_total0
    return JaxWhileRun(seeds=list(seeds), survivals=survivals, wall_seconds=elapsed)

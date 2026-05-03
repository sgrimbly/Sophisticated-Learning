"""Fully-JAX trial loop: a single ``lax.scan`` over trial steps.

Removes the per-seed Python overhead that plateaued ``agent_batched``
at ~115 ms/trial-seed. Each trial is one fused GPU kernel doing N
seeds in lockstep; trials remain a Python outer loop because each trial
re-initialises the per-trial state (Q_pos = D_pos, fresh true_ctx
sample, t_* timers reset) but keeps the learned ``a_resource`` from
prior trials.

Approximation parity: same as ``planning.py`` — full 5^H policy
enumeration, single-step novelty (SI) / approximated SL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
from sl.env import GenerativeModel, initialise_environment  # noqa: E402

from .config import GridConfig, JaxOptions
from .env_jax import (
    calculate_posterior_jax,
    normalise_flat,
    normalise_matrix_columns_jax,
    real_dirichlet_update_jax,
    sample_observations_jax,
    update_environment_states_jax,
    update_needs_jax,
)
from .planning import PlannerInputsJax, plan_batched


# ---------------------------------------------------------------------------
# Per-step lax-scan body for ONE seed.
# ---------------------------------------------------------------------------


def make_step_body(
    grid_food: jnp.ndarray,
    grid_water: jnp.ndarray,
    grid_sleep: jnp.ndarray,
    A_pos: jnp.ndarray,
    A_resource: jnp.ndarray,
    A_hill: jnp.ndarray,
    B_pos: jnp.ndarray,
    B_ctx: jnp.ndarray,
    bb_ctx: jnp.ndarray,
    food_threshold: int,
    water_threshold: int,
    sleep_threshold: int,
    max_steps: int,
    horizon: int,
    w_novelty: float,
    w_learning: float,
    w_epistemic: float,
    pref_iprec: float,
    learning_proportion: float,
    learning_scale: float,
    learning_floor: float,
):
    """Return a step-body function suitable for lax.scan over t in [0, max_steps)."""

    def step_body(carry, t):
        (key, Q_pos, Q_ctx, true_pos, true_ctx,
         t_food, t_water, t_sleep,
         a_resource, prev_chosen, alive, t_terminal) = carry

        # 1. env transition (only if t > 0)
        def transition_branch(args):
            key_in, Q_pos_in, Q_ctx_in, tp, tc, key_t = args
            return update_environment_states_jax(
                Q_pos_in, Q_ctx_in, tp, tc, prev_chosen,
                B_pos, B_ctx, bb_ctx, key_in,
            )

        def initial_branch(args):
            key_in, Q_pos_in, Q_ctx_in, tp, tc, key_t = args
            return Q_pos_in, Q_ctx_in, tp.astype(jnp.int32), tc.astype(jnp.int32), key_in

        Q_pos, Q_ctx, true_pos, true_ctx, key = jax.lax.cond(
            t > 0,
            transition_branch,
            initial_branch,
            (key, Q_pos, Q_ctx, true_pos, true_ctx, key),
        )

        # 2. update needs
        t_food, t_water, t_sleep = update_needs_jax(
            grid_food, grid_water, grid_sleep,
            true_pos, true_ctx, t,
            t_food, t_water, t_sleep,
        )

        # 3. alive check (after needs update — death detected at this step)
        died_now = ((t_food >= food_threshold)
                    | (t_water >= water_threshold)
                    | (t_sleep >= sleep_threshold))
        # Continue body even after death to keep static shape; mask at end.

        # 4. observation
        O_pos, O_res, O_hill, key = sample_observations_jax(
            A_pos, A_resource, A_hill, true_pos, true_ctx, key,
        )

        # 5. posterior at this step
        y_resource = normalise_matrix_columns_jax(a_resource)
        P_pos, P_ctx = calculate_posterior_jax(
            Q_pos, Q_ctx, y_resource, A_hill, O_res, O_hill,
        )

        # 6. Q-overwrite: Q ← P for next step's propagation
        Q_pos = P_pos
        Q_ctx = P_ctx

        # 7. plan
        inputs = PlannerInputsJax(
            A_pos=A_pos, A_resource=A_resource, A_hill=A_hill,
            a_resource=a_resource, y_resource=y_resource,
            B_pos=B_pos, bb_ctx=bb_ctx,
            w_novelty=w_novelty, w_learning=w_learning,
            w_epistemic=w_epistemic,
            preference_inverse_precision=pref_iprec,
        )
        chosen, _G = plan_batched(
            P_pos, P_ctx, t_food, t_water, t_sleep,
            inputs, horizon, False,
        )

        # 8. Dirichlet a-update (single-step SI, only when t > 0 AND alive)
        # In MATLAB, smoothing block fires when t > 0; here we only update
        # if also alive (the Python port's outer while-loop wouldn't have
        # entered another iteration anyway, so this is a no-op for dead).
        do_update = (t > 0) & alive & (~died_now)

        def do_update_branch(_):
            return real_dirichlet_update_jax(
                a_resource, O_res, P_pos, P_ctx,
                proportion=learning_proportion,
                scale=learning_scale,
                floor=learning_floor,
            )

        def skip_update_branch(_):
            return a_resource

        a_resource = jax.lax.cond(do_update, do_update_branch, skip_update_branch, None)

        # 9. record terminal step (first step of death)
        new_t_terminal = jnp.where(
            alive & died_now,
            t + 1,           # match unbatched agent.py: t_terminal = t+1
            t_terminal,
        )
        new_alive = alive & (~died_now)

        new_carry = (
            key, Q_pos, Q_ctx, true_pos, true_ctx,
            t_food, t_water, t_sleep,
            a_resource, jnp.int32(chosen), new_alive, new_t_terminal,
        )
        return new_carry, None

    return step_body


# ---------------------------------------------------------------------------
# Trial wrapper: single trial = one lax.scan over max_steps for one seed.
# ---------------------------------------------------------------------------


def run_trial_jax(
    key,
    Q_pos0,
    Q_ctx0,
    true_pos0,
    true_ctx0,
    a_resource0,
    grid_food, grid_water, grid_sleep,
    A_pos, A_resource, A_hill, B_pos, B_ctx, bb_ctx,
    food_threshold, water_threshold, sleep_threshold, max_steps,
    horizon, w_novelty, w_learning, w_epistemic, pref_iprec,
    learning_proportion, learning_scale, learning_floor,
):
    """Run one trial via lax.scan. Returns (t_terminal, a_resource_final, key)."""
    step_body = make_step_body(
        grid_food, grid_water, grid_sleep,
        A_pos, A_resource, A_hill, B_pos, B_ctx, bb_ctx,
        food_threshold, water_threshold, sleep_threshold, max_steps, horizon,
        w_novelty, w_learning, w_epistemic, pref_iprec,
        learning_proportion, learning_scale, learning_floor,
    )

    init = (
        key,                       # PRNG key
        Q_pos0, Q_ctx0,            # belief
        true_pos0, true_ctx0,      # true state (scalar int)
        jnp.int32(0), jnp.int32(0), jnp.int32(0),   # need timers
        a_resource0,               # learned (4, S, C)
        jnp.int32(0),              # prev chosen action
        jnp.bool_(True),           # alive
        jnp.int32(max_steps),      # t_terminal default to cap
    )
    ts = jnp.arange(max_steps, dtype=jnp.int32)
    final_carry, _ = jax.lax.scan(step_body, init, ts)
    (key_final, Q_pos_f, Q_ctx_f, _tp, _tc,
     _tf, _tw, _ts,
     a_final, _prev, _alive, t_terminal) = final_carry
    return t_terminal, a_final, key_final


# ---------------------------------------------------------------------------
# Top-level run: vmap across seeds, Python loop over trials.
# ---------------------------------------------------------------------------


@dataclass
class JaxNativeRun:
    seeds: List[int]
    survivals: List[List[int]]   # per-seed list of t_terminal across trials
    wall_seconds: float = 0.0


def run_native_batched(
    grid: GridConfig,
    opts: JaxOptions,
    seeds: List[int],
) -> JaxNativeRun:
    """All-JAX, all-GPU batched trial-loop runner."""
    n_seeds = len(seeds)

    # Build env once (shared across all seeds since grid is identical).
    np_model: GenerativeModel = initialise_environment(grid)
    A_pos = jnp.asarray(np_model.A_pos)
    A_resource = jnp.asarray(np_model.A_resource)
    A_hill = jnp.asarray(np_model.A_hill)
    B_pos = jnp.asarray(np_model.B_pos)
    B_ctx = jnp.asarray(np_model.B_ctx)
    # Normalise b_ctx columns (matches sl.env.normalise_b_ctx).
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

    # Per-seed initial a_resource (shared init, then divergent learning).
    a0 = jnp.asarray(np_model.a_resource)
    a_per_seed = jnp.broadcast_to(a0, (n_seeds,) + a0.shape).copy()

    # Per-seed PRNG keys.
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])

    # Initial true state per seed: pos = grid.start_position, ctx sampled from D_ctx.
    D_ctx = jnp.asarray(np_model.D_ctx)
    D_pos = jnp.asarray(np_model.D_pos)

    def sample_initial_ctx(key):
        return jax.random.categorical(key, jnp.log(jnp.maximum(D_ctx, 1e-16)))

    # in_axes for vmap: only key, true_ctx0, a_resource0 are batched.
    in_axes = [0, None, None, None, 0, 0,    # key, Q_pos0, Q_ctx0, true_pos0, true_ctx0, a0
               None, None, None,             # grid_food, grid_water, grid_sleep
               None, None, None, None, None, None]  # A_pos, A_resource, A_hill, B_pos, B_ctx, bb_ctx
    in_axes += [None] * 12   # all the scalar/option args (food_threshold..learning_floor = 12)
    from functools import partial
    run_trial_vmap = jax.jit(
        jax.vmap(run_trial_jax, in_axes=tuple(in_axes)),
        static_argnums=(15, 16, 17, 18, 19),  # food_threshold, water_threshold, sleep_threshold, max_steps, horizon
    )

    survivals: List[List[int]] = [[] for _ in range(n_seeds)]
    t_total0 = time.time()

    for trial in range(opts.num_trials):
        # Per-seed key splits: produce (k_ctx, k_trial, k_next) per seed.
        ks = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
        key_for_ctx = ks[:, 0]
        key_for_trial = ks[:, 1]
        keys = ks[:, 2]

        true_ctx0 = jax.vmap(sample_initial_ctx)(key_for_ctx).astype(jnp.int32)
        true_pos0 = jnp.int32(grid.start_position)

        t_term, a_per_seed, _ = run_trial_vmap(
            key_for_trial,
            D_pos, D_ctx,
            true_pos0, true_ctx0,
            a_per_seed,
            grid_food, grid_water, grid_sleep,
            A_pos, A_resource, A_hill,
            B_pos, B_ctx, bb_ctx,
            opts.food_threshold,
            opts.water_threshold,
            opts.sleep_threshold,
            opts.max_steps_per_trial,
            opts.horizon,
            opts.w_novelty, opts.w_learning, opts.w_epistemic,
            opts.preference,
            opts.learning_proportion, opts.learning_scale, opts.learning_floor,
        )

        # Pull terminals back to host.
        t_term_np = np.asarray(t_term)
        for i in range(n_seeds):
            survivals[i].append(int(t_term_np[i]))

    elapsed = time.time() - t_total0
    return JaxNativeRun(seeds=list(seeds), survivals=survivals, wall_seconds=elapsed)

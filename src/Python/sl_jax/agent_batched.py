"""Batched-across-seeds sl_jax agent.

Runs N seeds in parallel by vmapping the planner across N concurrent
states. The Python outer loop steps every seed forward in lockstep at
each timestep, with a per-seed alive mask that turns into ``True`` when
that seed dies (food/water/sleep timer breach or max-steps reached).
A dead seed contributes a no-op step until all seeds in the batch are
done; the actual recorded ``t_terminal`` is the step at which the seed
first died.

This trades some compute (running already-dead seeds for a few steps)
for a single GPU launch per timestep across all seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
from sl.env import sample_observations, update_environment_states, update_needs  # noqa: E402
from sl.inference import calculate_posterior, normalise_matrix_columns  # noqa: E402
from sl.learning import real_dirichlet_update  # noqa: E402
from sl.rng import make_rng, sample_categorical  # noqa: E402

from .config import GridConfig, JaxOptions
from .env import build_jax_model, update_a_resource
from .planning import PlannerInputsJax, plan_batched


def plan_batched_seeds(P_pos_b, P_ctx_b, tf_b, tw_b, ts_b, inputs_b, horizon, use_sl):
    """vmap plan_batched across the leading "seed" axis.

    ``inputs_b`` is a PlannerInputsJax whose tensor fields are batched
    along axis 0 (e.g., a_resource shape (B, 4, S, C)).
    """
    in_axes = PlannerInputsJax(
        A_pos=None, A_resource=None, A_hill=None,
        a_resource=0, y_resource=0,
        B_pos=None, bb_ctx=None,
        w_novelty=None, w_learning=None, w_epistemic=None,
        preference_inverse_precision=None,
    )
    fn = jax.vmap(
        plan_batched,
        in_axes=(0, 0, 0, 0, 0, in_axes, None, None),
    )
    return fn(P_pos_b, P_ctx_b, tf_b, tw_b, ts_b, inputs_b,
              horizon, use_sl)


@dataclass
class BatchedRunResult:
    seeds: List[int]
    survivals: List[List[int]]   # per-seed list of t_terminal across trials
    wall_seconds: float = 0.0


def run_batched(grid: GridConfig, opts: JaxOptions, seeds: List[int]) -> BatchedRunResult:
    """Run N seeds in parallel for opts.num_trials trials each.

    Each step batches the planner across all currently-alive seeds via
    jax.vmap. Dead seeds within a trial are kept in the batch but their
    chosen actions are ignored after death (recorded step is finalised).

    Returns per-seed survival lists.
    """
    n_seeds = len(seeds)
    rngs = [make_rng(s) for s in seeds]

    A_pos_np = None
    A_res_np = None
    A_hill_np = None
    B_pos_np = None
    B_ctx_np = None
    bb_ctx_np = None

    # Per-seed initial models.
    a_resources = []  # numpy arrays per seed
    for s in seeds:
        m = build_jax_model(grid)
        a_resources.append(np.asarray(m.a_resource).copy())
        if A_pos_np is None:
            A_pos_np = np.asarray(m.A_pos)
            A_res_np = np.asarray(m.A_resource)
            A_hill_np = np.asarray(m.A_hill)
            B_pos_np = np.asarray(m.B_pos)
            B_ctx_np = np.asarray(m.B_ctx)
            bb_ctx_np = np.asarray(m.bb_ctx)
            D_pos_np = np.asarray(m.D_pos)
            D_ctx_np = np.asarray(m.D_ctx)

    survivals: List[List[int]] = [[] for _ in range(n_seeds)]
    use_sl = opts.smoothing_on
    t_total0 = time.time()

    for trial in range(opts.num_trials):
        # Per-seed trial state.
        Q_pos = [D_pos_np.copy() for _ in range(n_seeds)]
        Q_ctx = [D_ctx_np.copy() for _ in range(n_seeds)]
        true_pos = [grid.start_position] * n_seeds
        true_ctx = [sample_categorical(D_ctx_np.copy(), rngs[i]) for i in range(n_seeds)]
        t_f = [0] * n_seeds
        t_w = [0] * n_seeds
        t_s = [0] * n_seeds
        chosen_prev = [0] * n_seeds
        true_states_prev = [(true_pos[i], true_ctx[i]) for i in range(n_seeds)]
        O_res_hist = [[] for _ in range(n_seeds)]
        Q_pos_hist = [[] for _ in range(n_seeds)]
        Q_ctx_hist = [[] for _ in range(n_seeds)]
        alive = [True] * n_seeds
        terminal = [None] * n_seeds

        for t in range(opts.max_steps_per_trial):
            for i in range(n_seeds):
                if not alive[i]:
                    continue
                # Env transition
                if t > 0:
                    Q_pos[i], Q_ctx[i], true_pos[i], true_ctx[i] = update_environment_states(
                        Q_pos[i], Q_ctx[i],
                        true_states_prev[i][0], true_states_prev[i][1],
                        chosen_prev[i],
                        B_pos_np, B_ctx_np, bb_ctx_np, rngs[i],
                    )
                true_states_prev[i] = (true_pos[i], true_ctx[i])
                t_f[i], t_w[i], t_s[i] = update_needs(
                    grid, true_pos[i], true_ctx[i], t, t_f[i], t_w[i], t_s[i],
                )
                if (t_f[i] >= opts.food_threshold or t_w[i] >= opts.water_threshold
                        or t_s[i] >= opts.sleep_threshold):
                    alive[i] = False
                    # Match unbatched agent.py: t_terminal counts bodies fully
                    # executed; the breach is detected after this body runs,
                    # so terminal = t + 1.
                    terminal[i] = t + 1
                    continue

                O_pos, O_res, O_hill = sample_observations(
                    A_pos_np, A_res_np, A_hill_np, true_pos[i], true_ctx[i], rngs[i],
                )
                O_res_hist[i].append(O_res)
                Q_pos_hist[i].append(Q_pos[i].copy())
                Q_ctx_hist[i].append(Q_ctx[i].copy())

                # Posterior at this step + Q overwrite.
                y_resource_now = normalise_matrix_columns(a_resources[i])
                P_pos_i, P_ctx_i = calculate_posterior(
                    Q_pos[i], Q_ctx[i], y_resource_now, A_hill_np, O_res, O_hill,
                )
                Q_pos[i] = P_pos_i
                Q_ctx[i] = P_ctx_i

            # End of any-alive check.
            if not any(alive):
                break

            # Build batched planner inputs across alive seeds.
            alive_idx = [i for i in range(n_seeds) if alive[i]]
            if not alive_idx:
                break
            P_pos_b = np.stack([Q_pos[i] for i in alive_idx])
            P_ctx_b = np.stack([Q_ctx[i] for i in alive_idx])
            tf_b = np.array([t_f[i] for i in alive_idx], dtype=np.int32)
            tw_b = np.array([t_w[i] for i in alive_idx], dtype=np.int32)
            ts_b = np.array([t_s[i] for i in alive_idx], dtype=np.int32)
            a_b = np.stack([a_resources[i] for i in alive_idx])
            y_b = np.stack([
                normalise_matrix_columns(a_resources[i]) for i in alive_idx
            ])

            inputs_b = PlannerInputsJax(
                A_pos=jnp.asarray(A_pos_np),
                A_resource=jnp.asarray(A_res_np),
                A_hill=jnp.asarray(A_hill_np),
                a_resource=jnp.asarray(a_b),
                y_resource=jnp.asarray(y_b),
                B_pos=jnp.asarray(B_pos_np),
                bb_ctx=jnp.asarray(bb_ctx_np),
                w_novelty=jnp.float64(opts.w_novelty),
                w_learning=jnp.float64(opts.w_learning),
                w_epistemic=jnp.float64(opts.w_epistemic),
                preference_inverse_precision=jnp.float64(opts.preference),
            )
            best_actions, _Gs = plan_batched_seeds(
                jnp.asarray(P_pos_b), jnp.asarray(P_ctx_b),
                jnp.asarray(tf_b), jnp.asarray(tw_b), jnp.asarray(ts_b),
                inputs_b, opts.horizon, use_sl,
            )
            best_actions = np.asarray(best_actions)
            for k, i in enumerate(alive_idx):
                chosen_prev[i] = int(best_actions[k])

            # Per-seed Dirichlet update (single-step SI; keeps Python loop)
            for i in alive_idx:
                if t > 0:
                    if not opts.smoothing_on:
                        a_resources[i] = real_dirichlet_update(
                            a_resources[i], O_res_hist[i][t],
                            Q_pos_hist[i][t], Q_ctx_hist[i][t],
                            proportion=opts.learning_proportion,
                            scale=opts.learning_scale,
                            floor=opts.learning_floor,
                        )
                    # SL smoothing path elided in batched proto for now

        # Trial done — record terminals.
        for i in range(n_seeds):
            survivals[i].append(terminal[i] if terminal[i] is not None
                                else opts.max_steps_per_trial)

    elapsed = time.time() - t_total0
    return BatchedRunResult(
        seeds=list(seeds), survivals=survivals, wall_seconds=elapsed,
    )

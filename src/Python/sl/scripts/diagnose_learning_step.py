"""Step-level diagnostic of a_resource evolution.

Runs one trial, intercepts every call to real_dirichlet_update, and prints:
  * step t, action chosen, true (pos, ctx), observation
  * a[2] sum, max, min at the canonical food_sources[true_ctx]
  * a[outcome=resource_obs, true_pos, true_ctx] before/after the update
  * count of update calls so far this trial
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from sl import learning as _learning
from sl.config import GridConfig, RunOptions, Weights
from sl.env import initialise_environment
from sl.rng import make_rng


_orig_update = _learning.real_dirichlet_update
_call_count = [0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--algorithm", default="SI")
    p.add_argument("--max-horizon", type=int, default=9)
    p.add_argument("--max-trial-steps", type=int, default=22)
    args = p.parse_args()

    grid = GridConfig.from_matlab_indices()
    weights = Weights()
    options = RunOptions(
        algorithm=args.algorithm, seed=args.seed,
        num_trials=1,
        max_horizon=args.max_horizon,
        use_jit_planner=False,  # NumPy path for clarity
    )

    # Verbose wrapper: log each call.
    def verbose_update(a_resource, O_res, P_pos, P_ctx, **kw):
        _call_count[0] += 1
        out = _orig_update(a_resource, O_res, P_pos, P_ctx, **kw)
        return out
    _learning.real_dirichlet_update = verbose_update
    # Also patch the import in agent.py
    from sl import agent as _agent
    _agent.real_dirichlet_update = verbose_update

    rng = make_rng(args.seed)
    model = initialise_environment(grid)

    # Run the trial step-by-step ourselves (mirroring run_trial logic) so we
    # can introspect per step.
    from sl.env import (
        normalise_b_ctx, sample_observations, update_environment_states,
        update_needs,
    )
    from sl.inference import (
        calculate_posterior, joint_pos_ctx, normalise, normalise_matrix_columns,
        spm_backwards,
    )

    print(f"=== {args.algorithm} seed={args.seed}, single trial diagnostic ===")
    print(f"food_sources (0-based) = {grid.food_sources}")
    print(f"water_sources (0-based) = {grid.water_sources}")
    print(f"sleep_sources (0-based) = {grid.sleep_sources}")
    print(f"start={grid.start_position}, hill={grid.hill_pos}\n")

    Q_pos = model.D_pos.copy()
    Q_ctx = model.D_ctx.copy()
    true_pos = grid.start_position
    true_ctx = int(np.argmax(np.cumsum(model.D_ctx) >= rng.random()))

    Q_pos_history = []
    Q_ctx_history = []
    O_resource_history = []
    O_hill_history = []
    chosen_actions = []
    t_food = t_water = t_sleep = 0

    for t in range(args.max_trial_steps):
        bb_ctx = normalise_b_ctx(model.b_ctx)

        if t > 0:
            ca = chosen_actions[-1]
            Q_pos, Q_ctx, true_pos, true_ctx = update_environment_states(
                Q_pos_history[-1], Q_ctx_history[-1],
                # use prev true state, not current
                int(np.argmax(model.D_pos)) if t == 1 else _last_true_pos,
                _last_true_ctx if t > 0 else true_ctx,
                ca, model.B_pos, model.B_ctx, bb_ctx, rng,
            )
        _last_true_pos = true_pos
        _last_true_ctx = true_ctx

        t_food, t_water, t_sleep = update_needs(grid, true_pos, true_ctx, t,
                                                  t_food, t_water, t_sleep)

        O_pos, O_res, O_hill = sample_observations(
            model.A_pos, model.A_resource, model.A_hill,
            true_pos, true_ctx, rng,
        )
        Q_pos_history.append(Q_pos.copy())
        Q_ctx_history.append(Q_ctx.copy())
        O_resource_history.append(O_res)
        O_hill_history.append(O_hill)

        # Sample obs index for printing
        res_obs = int(np.argmax(O_res))

        # a snapshot at canonical positions (before update this step)
        canonical = []
        for c in range(4):
            row = grid.food_sources[c]
            v = model.a_resource[:, row, c].copy()
            canonical.append((c, row, v))

        # smoothing learning
        n_updates_pre = _call_count[0]
        if t > 0:
            start = max(0, t - 6)
            for timey in range(start, t + 1):
                L_ctx = spm_backwards(
                    O_hill_history, Q_pos_history, Q_ctx_history[timey],
                    model.A_hill, bb_ctx, timey, t,
                )
                pre_round = np.round(L_ctx, 3)
                cmp_round = np.round(Q_ctx_history[timey], 3)
                if (timey > start and not np.array_equal(pre_round, cmp_round)) or timey == t:
                    model.a_resource = verbose_update(
                        model.a_resource, O_resource_history[timey],
                        Q_pos_history[timey], L_ctx,
                        proportion=0.3, scale=0.7, floor=0.05,
                    )
        n_updates_this_step = _call_count[0] - n_updates_pre

        # Now compute action — naive: pick argmax of (-water_threshold + tw, ...)
        # We don't actually plan here (just report). Pick action 0 (stay).
        chosen_actions.append(0)

        # Print this step
        on_food = grid.food_sources[true_ctx] == true_pos
        on_water = grid.water_sources[true_ctx] == true_pos
        on_sleep = grid.sleep_sources[true_ctx] == true_pos
        a_max_res = model.a_resource[1:, :, :].max()
        # Resource counts at canonical context-true food/water/sleep
        f_pos = grid.food_sources[true_ctx]
        w_pos = grid.water_sources[true_ctx]
        s_pos = grid.sleep_sources[true_ctx]
        a_food = model.a_resource[1, f_pos, true_ctx]
        a_water = model.a_resource[2, w_pos, true_ctx]
        a_sleep = model.a_resource[3, s_pos, true_ctx]
        a_empty_at_pos = model.a_resource[0, true_pos, true_ctx]
        a_resource_at_pos = model.a_resource[res_obs, true_pos, true_ctx] if res_obs > 0 else None

        marker = ""
        if on_food: marker = " ON_FOOD"
        elif on_water: marker = " ON_WATER"
        elif on_sleep: marker = " ON_SLEEP"

        print(f"t={t:2d}  pos={true_pos:3d} ctx={true_ctx} obs[res]={res_obs}{marker}  "
              f"updates={n_updates_this_step}  "
              f"a[1,f{true_ctx}]={a_food:.3f} a[2,w{true_ctx}]={a_water:.3f} a[3,s{true_ctx}]={a_sleep:.3f}  "
              f"a[0,pos]={a_empty_at_pos:.3f}", flush=True)

    print(f"\nTotal update calls in trial: {_call_count[0]}")
    print(f"a_resource: min={model.a_resource.min():.4f}, max={model.a_resource.max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

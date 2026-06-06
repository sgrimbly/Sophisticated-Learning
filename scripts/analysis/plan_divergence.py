"""Action-divergence diagnostic: does SL's roll-forward change decisions?

Runs one SL_noNovelty_adaptivePlan rollout with options.diagnose_plan_divergence
set, so at each real planning step the planner is also run with
adaptive_likelihood_in_plan toggled OFF on the identical state (fresh STM copy).
Records, per step, the action chosen with the roll-forward ON vs OFF — isolating
SL's mechanism (B) with the novelty reward (A) held off.

Writes a per-seed JSON: {seed, num_trials, trials:[{trial, steps, diffs, t_terminal}]}.
Aggregate across seeds + stratify by trial (early=learning vs late=saturated)
in a separate step.

Usage:
    python plan_divergence.py <seed> <num_trials> <output_json>
"""
import json
import sys

from sl.config import GridConfig, RunOptions, Weights
from sl.agent import run_experiment


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    out = sys.argv[3] if len(sys.argv) > 3 else f"plan_divergence_seed{seed}.json"

    grid = GridConfig.from_matlab_indices()
    opts = RunOptions(
        algorithm="SL_noNovelty_adaptivePlan", seed=seed, num_trials=ntr,
        max_horizon=9, max_steps_per_trial=100, use_jit_planner=True,
        diagnose_plan_divergence=True,
    )
    res = run_experiment(grid, opts, Weights())

    trials = []
    for i, tr in enumerate(res.trials):
        steps = len(tr.plan_divergence)
        diffs = sum(1 for (t, a_on, a_off) in tr.plan_divergence if a_on != a_off)
        trials.append({"trial": i + 1, "steps": steps, "diffs": diffs,
                       "t_terminal": tr.t_terminal})
    payload = {"seed": seed, "num_trials": ntr, "trials": trials}
    with open(out, "w") as fh:
        json.dump(payload, fh)

    tot_s = sum(t["steps"] for t in trials)
    tot_d = sum(t["diffs"] for t in trials)
    print(f"seed {seed}: {tot_d}/{tot_s} steps action differs "
          f"({100 * tot_d / max(1, tot_s):.2f}%) -> {out}")


if __name__ == "__main__":
    main()

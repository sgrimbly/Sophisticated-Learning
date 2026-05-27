"""Single-run CLI dispatch — analogue of MATLAB main.m.

Usage::

    python -m sl.runner --algorithm SL --seed 1 --num-trials 5 --max-horizon 9

Produces a JSON summary on stdout; pass ``--output PATH`` to also write a
per-step dump in the same shape as MATLAB's results files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import List

import numpy as np

from .config import GridConfig, RunOptions, Weights, ALGORITHM_VARIANTS
from .agent import run_experiment


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one Sophisticated-Learning experiment.")
    p.add_argument("--algorithm", choices=sorted(ALGORITHM_VARIANTS), default="SL")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-trials", type=int, default=120)
    p.add_argument("--max-horizon", type=int, default=9)
    p.add_argument("--max-steps", type=int, default=100,
                   help="Max steps per trial (MATLAB ``while t < 100``)")

    # Grid layout — defaults match MATLAB main.m
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--start-position", type=int, default=51,
                   help="MATLAB 1-based position of start cell")
    p.add_argument("--hill-pos", type=int, default=55)
    p.add_argument("--food-sources", type=int, nargs=4, default=[71, 43, 57, 78])
    p.add_argument("--water-sources", type=int, nargs=4, default=[73, 33, 48, 67])
    p.add_argument("--sleep-sources", type=int, nargs=4, default=[64, 44, 49, 59])
    p.add_argument("--grid-id", default="")

    # Weights
    p.add_argument("--w-novelty", type=float, default=10.0)
    p.add_argument("--w-learning", type=float, default=40.0)
    p.add_argument("--w-epistemic", type=float, default=1.0)
    p.add_argument("--w-preference", type=float, default=10.0)
    p.add_argument("--ucb-scale", type=float, default=5.0)

    # Variant flags
    p.add_argument("--no-real-smoothing", action="store_true")
    p.add_argument("--adaptive-likelihood-in-plan", action="store_true")
    p.add_argument("--learning-prune-threshold", type=float, default=0.2)
    p.add_argument("--state-selection", choices=["sample", "map"], default="sample")

    p.add_argument("--output", default="",
                   help="If set, write a JSON dump here. The summary is also "
                        "always printed on stdout.")
    p.add_argument("--enable-numba", action="store_true",
                   help="Try to swap in the Numba JIT G_epistemic_value at runtime.")
    return p.parse_args(argv)


def _build_config(ns: argparse.Namespace) -> tuple[GridConfig, RunOptions, Weights]:
    grid = GridConfig.from_matlab_indices(
        grid_size=ns.grid_size,
        start_position=ns.start_position,
        hill_pos=ns.hill_pos,
        food_sources=ns.food_sources,
        water_sources=ns.water_sources,
        sleep_sources=ns.sleep_sources,
        grid_id=ns.grid_id,
    )
    options = RunOptions(
        algorithm=ns.algorithm,
        seed=ns.seed,
        num_trials=ns.num_trials,
        max_horizon=ns.max_horizon,
        max_steps_per_trial=ns.max_steps,
        real_smoothing=not ns.no_real_smoothing,
        adaptive_likelihood_in_plan=ns.adaptive_likelihood_in_plan,
        learning_prune_threshold=ns.learning_prune_threshold,
        state_selection=ns.state_selection,
    )
    weights = Weights(
        novelty=ns.w_novelty,
        learning=ns.w_learning,
        epistemic=ns.w_epistemic,
        preference=ns.w_preference,
        ucb_scale=ns.ucb_scale,
    )
    return grid, options, weights


def main(argv: List[str] | None = None) -> int:
    ns = _parse_args(argv if argv is not None else sys.argv[1:])

    if ns.enable_numba:
        from .numba_kernels import maybe_swap_in_jit_efe, HAS_NUMBA
        if not HAS_NUMBA:
            print("[runner] WARNING: --enable-numba set but numba not importable.",
                  file=sys.stderr)
        else:
            maybe_swap_in_jit_efe()

    grid, options, weights = _build_config(ns)

    t0 = time.time()
    result = run_experiment(grid, options, weights)
    elapsed = time.time() - t0

    summary = {
        "algorithm": options.algorithm,
        "seed": options.seed,
        "num_trials": options.num_trials,
        "survived_count": result.survived_count,
        "survival_rate": result.survived_count / max(1, options.num_trials),
        "elapsed_seconds": elapsed,
        "trials": [
            {
                "trial": i + 1,
                "survived": tr.survived,
                "t_terminal": tr.t_terminal,
                "memory_resets": tr.memory_resets,
                "pe_resets": tr.pe_memory_resets,
                "hill_resets": tr.hill_memory_resets,
                "median_horizon": float(np.median(tr.horizons)) if tr.horizons else 0.0,
            }
            for i, tr in enumerate(result.trials)
        ],
    }

    print(json.dumps(summary, indent=2))

    if ns.output:
        os.makedirs(os.path.dirname(ns.output) or ".", exist_ok=True)
        # Full per-step dump for analysis. Heavy — only when explicitly requested.
        per_trial_full = []
        for tr in result.trials:
            per_trial_full.append({
                "survived": tr.survived,
                "t_terminal": tr.t_terminal,
                "chosen_actions": tr.chosen_actions,
                "true_states": tr.true_states,
                "observations": tr.observations,
                "horizons": tr.horizons,
                "memory_resets": tr.memory_resets,
                "pe_memory_resets": tr.pe_memory_resets,
                "hill_memory_resets": tr.hill_memory_resets,
            })
        with open(ns.output, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_trial": per_trial_full}, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())

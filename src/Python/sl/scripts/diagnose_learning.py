"""Diagnose whether Python's a_resource is learning at the same rate as MATLAB.

Computes the same per-trial summary as MATLAB's metrics.csv:
  * survival (= trial t_terminal)
  * p_food_c1, p_water_c1, p_sleep_c1, ..., p_sleep_c4
    where p_X_c{i} = (normalised a[2] at the resource X's canonical position
    in context i for outcome X).
  * param_update_kl: KL between this trial's final a[2] and the previous
    trial's final a[2].

If Python's p_food/p_water/p_sleep are growing toward 1 at a similar rate
to MATLAB's, then a-learning is fine. If they stay near 0.25 (uniform),
the learning update is broken.

Usage:
    python -m sl.scripts.diagnose_learning --seed 1 --num-trials 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from sl.config import GridConfig, RunOptions, Weights
from sl.env import initialise_environment
from sl.rng import make_rng
from sl.agent import run_trial


def normalise_a_at_position(a_resource: np.ndarray, pos: int, ctx: int) -> np.ndarray:
    """Returns probability over the 4 resource outcomes at (pos, ctx) — what
    MATLAB's metrics.csv reports as p_food/p_water/p_sleep at canonical
    locations.
    """
    col = a_resource[:, pos, ctx]
    s = col.sum()
    return col / s if s > 0 else col


def kl_dirichlet(a: np.ndarray, b: np.ndarray) -> float:
    """KL(normalised a flat || normalised b flat) — same formula as MATLAB."""
    af, bf = a.ravel(), b.ravel()
    sa, sb = af.sum(), bf.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    pa, pb = af / sa, bf / sb
    mask = (pa > 0) & (pb > 0)
    return float(np.sum(pa[mask] * np.log(pa[mask] / pb[mask])))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", default="SI", choices=["SI", "SL"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-trials", type=int, default=30)
    p.add_argument("--max-horizon", type=int, default=9)
    p.add_argument("--use-jit-planner", action="store_true", default=True)
    p.add_argument("--no-jit", action="store_true")
    p.add_argument("--no-smoothing", action="store_true",
                   help="Disable real_smoothing — single-step learning only.")
    args = p.parse_args()

    grid = GridConfig.from_matlab_indices()
    weights = Weights()
    options = RunOptions(
        algorithm=args.algorithm, seed=args.seed,
        num_trials=args.num_trials,
        max_horizon=args.max_horizon,
        use_jit_planner=args.use_jit_planner and not args.no_jit,
        real_smoothing=not args.no_smoothing,
    )

    print(f"=== {args.algorithm} seed={args.seed} num_trials={args.num_trials} "
          f"jit={options.use_jit_planner} ===\n")

    # Header — only show 6 of the 12 p_X_c{i} columns to keep it readable
    print(f"{'trial':>5} {'t_term':>6} {'kl_step':>9}  ",
          "p_food: ", "  ".join(f"c{i+1}" for i in range(4)),
          " | p_water: ", "  ".join(f"c{i+1}" for i in range(4)),
          " | p_sleep: ", "  ".join(f"c{i+1}" for i in range(4)))

    rng = make_rng(args.seed)
    model = initialise_environment(grid)
    a_prev = model.a_resource.copy()

    for trial in range(args.num_trials):
        t0 = time.perf_counter()
        result = run_trial(grid, options, weights, model, rng)
        elapsed = time.perf_counter() - t0
        kl = kl_dirichlet(model.a_resource, a_prev)
        a_prev = model.a_resource.copy()

        ps_food  = [normalise_a_at_position(model.a_resource, grid.food_sources[i],  i)[1]
                    for i in range(4)]
        ps_water = [normalise_a_at_position(model.a_resource, grid.water_sources[i], i)[2]
                    for i in range(4)]
        ps_sleep = [normalise_a_at_position(model.a_resource, grid.sleep_sources[i], i)[3]
                    for i in range(4)]
        line = f"{trial+1:>5} {result.t_terminal:>6} {kl:>9.4f}  "
        line += "p_food: " + "  ".join(f"{x:.3f}" for x in ps_food)
        line += " | p_water: " + "  ".join(f"{x:.3f}" for x in ps_water)
        line += " | p_sleep: " + "  ".join(f"{x:.3f}" for x in ps_sleep)
        line += f"  ({elapsed:.1f}s)"
        print(line, flush=True)

    # Final a stats
    print(f"\nFinal a_resource: min={model.a_resource.min():.4f} max={model.a_resource.max():.4f}")
    print(f"Mean a at canonical food locations: "
          f"{np.mean([model.a_resource[1, grid.food_sources[i], i] for i in range(4)]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

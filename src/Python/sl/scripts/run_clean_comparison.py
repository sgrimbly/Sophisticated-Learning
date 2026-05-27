"""Streaming MATLAB-vs-Python comparison.

Designed for the contended single-CPU node — prints progress as each seed
completes so we don't lose data on timeout. Defaults to small scope.

Usage:
    python -m sl.scripts.run_clean_comparison --algorithm SI --num-seeds 30 --num-trials 30
    python -m sl.scripts.run_clean_comparison --algorithm SL --num-seeds 10 --num-trials 10

Reads MATLAB ground truth from
    results/unknown_model/MATLAB/paper_si_sl_repro_default_env/{SI,SL}/*.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
MATLAB_RESULTS = REPO_ROOT / "results" / "unknown_model" / "MATLAB" / "paper_si_sl_repro_default_env"


def load_matlab_survival(algorithm: str, max_seeds: int) -> Dict[int, List[int]]:
    """Read MATLAB Seed{N}.txt files. Convert MATLAB t_terminal to Python's
    convention (steps_executed) by subtracting 1.
    """
    folder = MATLAB_RESULTS / algorithm
    out: Dict[int, List[int]] = {}
    if not folder.exists():
        print(f"[load] MATLAB folder not found: {folder}", file=sys.stderr)
        return out
    for path in sorted(folder.glob(f"{algorithm}_Seed*.txt")):
        stem = path.stem
        try:
            seed = int(stem.replace(f"{algorithm}_Seed", ""))
        except ValueError:
            continue
        if seed > max_seeds:
            continue
        try:
            vals = [int(float(line.strip())) - 1
                    for line in path.read_text().splitlines() if line.strip()]
        except Exception:
            continue
        if vals:
            out[seed] = vals
    return out


def summary_stats(per_seed: Dict[int, List[int]], max_steps: int) -> dict:
    if not per_seed:
        return {"n": 0}
    all_t = [t for vs in per_seed.values() for t in vs]
    per_seed_rate = [
        sum(1 for t in vs if t >= max_steps) / len(vs) for vs in per_seed.values()
    ]
    return {
        "n_trials": len(all_t),
        "n_seeds": len(per_seed),
        "trial_length_mean": float(mean(all_t)),
        "trial_length_median": float(median(all_t)),
        "trial_length_stdev": float(pstdev(all_t)),
        "per_seed_survival_mean": float(mean(per_seed_rate)),
        "per_seed_survival_stdev":
            float(pstdev(per_seed_rate)) if len(per_seed_rate) > 1 else 0.0,
    }


def ks_distance(a: List[int], b: List[int]) -> float:
    if not a or not b:
        return float("nan")
    aa = sorted(a); bb = sorted(b)
    vals = sorted(set(aa) | set(bb))
    sup = 0.0; i = j = 0
    for v in vals:
        while i < len(aa) and aa[i] <= v:
            i += 1
        while j < len(bb) and bb[j] <= v:
            j += 1
        sup = max(sup, abs(i / len(aa) - j / len(bb)))
    return sup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", choices=["SI", "SL"], default="SI")
    p.add_argument("--num-seeds", type=int, default=30)
    p.add_argument("--num-trials", type=int, default=30)
    p.add_argument("--max-horizon", type=int, default=9)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--use-jit-planner", action="store_true", default=True)
    p.add_argument("--no-jit", action="store_true",
                   help="Force the NumPy planner instead of JIT.")
    p.add_argument("--output", default="")
    args = p.parse_args()

    use_jit = not args.no_jit and args.use_jit_planner

    # Imports after argparse so usage prints fast.
    from sl.config import GridConfig, RunOptions, Weights
    from sl.agent import run_experiment

    print(f"=== {args.algorithm}  num_seeds={args.num_seeds}  num_trials={args.num_trials}  "
          f"horizon={args.max_horizon}  jit={use_jit} ===", flush=True)

    # 1. MATLAB ground truth
    print("[matlab] loading...", end=" ", flush=True)
    matlab_per_seed = load_matlab_survival(args.algorithm, max_seeds=args.num_seeds)
    matlab_per_seed = {s: t[:args.num_trials] for s, t in matlab_per_seed.items()}
    if not matlab_per_seed:
        print("NO FILES FOUND", flush=True)
    else:
        m_stats = summary_stats(matlab_per_seed, args.max_steps)
        print(f"{m_stats['n_seeds']} seeds × {args.num_trials} trials. "
              f"trial_len mean={m_stats['trial_length_mean']:.2f} "
              f"survival={m_stats['per_seed_survival_mean']:.3f}", flush=True)

    # 2. Python sweep, in-process, with progress per seed
    print("[python] running sweep with per-seed progress...", flush=True)
    grid = GridConfig.from_matlab_indices()
    weights = Weights()
    python_per_seed: Dict[int, List[int]] = {}

    sweep_start = time.perf_counter()
    for seed in range(1, args.num_seeds + 1):
        seed_start = time.perf_counter()
        opts = RunOptions(algorithm=args.algorithm, seed=seed,
                          num_trials=args.num_trials, max_horizon=args.max_horizon,
                          max_steps_per_trial=args.max_steps,
                          use_jit_planner=use_jit)
        result = run_experiment(grid, opts, weights)
        ts = [tr.t_terminal for tr in result.trials]
        python_per_seed[seed] = ts
        elapsed = time.perf_counter() - seed_start
        avg_t = mean(ts)
        survived = sum(1 for t in ts if t >= args.max_steps)
        print(f"  seed {seed:3d}: {elapsed:6.1f}s  avg_t={avg_t:5.1f}  "
              f"survived={survived}/{args.num_trials}", flush=True)

    sweep_elapsed = time.perf_counter() - sweep_start
    print(f"[python] sweep done in {sweep_elapsed:.1f}s", flush=True)

    # 3. Compare
    p_stats = summary_stats(python_per_seed, args.max_steps)

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    if matlab_per_seed:
        print(f"  MATLAB  trials={m_stats['n_trials']:5d}  "
              f"trial_len mean={m_stats['trial_length_mean']:6.2f} ± {m_stats['trial_length_stdev']:.2f}  "
              f"survival={m_stats['per_seed_survival_mean']:.3f} ± {m_stats['per_seed_survival_stdev']:.3f}")
    print(f"  Python  trials={p_stats['n_trials']:5d}  "
          f"trial_len mean={p_stats['trial_length_mean']:6.2f} ± {p_stats['trial_length_stdev']:.2f}  "
          f"survival={p_stats['per_seed_survival_mean']:.3f} ± {p_stats['per_seed_survival_stdev']:.3f}")

    if matlab_per_seed:
        m_all = [t for vs in matlab_per_seed.values() for t in vs]
        p_all = [t for vs in python_per_seed.values() for t in vs]
        ks = ks_distance(m_all, p_all)
        delta_mean = (p_stats['trial_length_mean'] - m_stats['trial_length_mean'])
        delta_pct = 100.0 * delta_mean / max(1e-12, m_stats['trial_length_mean'])
        delta_surv_pp = 100.0 * (p_stats['per_seed_survival_mean']
                                 - m_stats['per_seed_survival_mean'])
        print(f"  Δtrial_length  = {delta_mean:+.2f} steps  ({delta_pct:+.1f}%)")
        print(f"  Δsurvival_rate = {delta_surv_pp:+.2f} pp")
        print(f"  KS distance    = {ks:.3f}  (smaller = better)")

    if args.output:
        out = {
            "algorithm": args.algorithm,
            "num_seeds": args.num_seeds,
            "num_trials": args.num_trials,
            "use_jit_planner": use_jit,
            "matlab": m_stats if matlab_per_seed else None,
            "python": p_stats,
            "matlab_per_seed": {str(k): v for k, v in matlab_per_seed.items()},
            "python_per_seed": {str(k): v for k, v in python_per_seed.items()},
            "ks_distance": ks if matlab_per_seed else None,
            "sweep_seconds": sweep_elapsed,
        }
        Path(args.output).write_text(json.dumps(out, indent=2, default=str))
        print(f"  → {args.output}")


if __name__ == "__main__":
    main()

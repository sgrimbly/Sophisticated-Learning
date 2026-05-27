"""End-to-end statistical comparison + speed benchmark.

Workflow:
    1. Aggregate MATLAB ground-truth survival (per-trial t_terminal) from
       results/unknown_model/MATLAB/paper_si_sl_repro_default_env/{SI,SL}/.
    2. Run a matching Python sweep (multiprocessing) at horizon=9.
    3. Compare survival distributions: per-seed survival rate, mean
       trial length, KS statistic.
    4. Speed benchmark: NumPy single-trial timing, with vs without Numba JIT.

Run:
    python -m sl.scripts.benchmark_and_compare \
        --num-seeds 30 --num-trials 30 --workers 8

Defaults are conservative (small) so it finishes in minutes; bump
``--num-seeds 100 --num-trials 120`` for full paper-scale comparison.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
MATLAB_RESULTS = REPO_ROOT / "results" / "unknown_model" / "MATLAB" / "paper_si_sl_repro_default_env"


# -----------------------------------------------------------------------------
# 1. MATLAB ground truth aggregation
# -----------------------------------------------------------------------------


def load_matlab_survival(algorithm: str, max_seeds: int = 200) -> Dict[int, List[int]]:
    """Read MATLAB Seed{N}.txt per-trial t_terminal lists.

    Returns dict mapping seed -> list of trial terminal steps, *converted
    to Python's step-count convention*. MATLAB starts t=1 and increments
    after the body, so MATLAB t_terminal = (steps_executed + 1). We
    subtract 1 to align with Python's t_terminal = steps_executed.
    """
    folder = MATLAB_RESULTS / algorithm
    out: Dict[int, List[int]] = {}
    for path in sorted(folder.glob(f"{algorithm}_Seed*.txt")):
        stem = path.stem
        try:
            seed = int(stem.replace(f"{algorithm}_Seed", ""))
        except ValueError:
            continue
        if seed > max_seeds:
            continue
        try:
            vals = [int(float(line.strip())) - 1 for line in path.read_text().splitlines()
                    if line.strip()]
        except Exception:
            continue
        if vals:
            out[seed] = vals
    return out


def survival_summary(per_seed_trials: Dict[int, List[int]],
                     survive_at: int = 100) -> Dict[str, float]:
    """Compute aggregate survival statistics across seeds."""
    all_trials = [t for trials in per_seed_trials.values() for t in trials]
    if not all_trials:
        return {"n": 0}
    survival_count = sum(1 for t in all_trials if t >= survive_at)
    per_seed_rates = [
        sum(1 for t in trials if t >= survive_at) / len(trials)
        for trials in per_seed_trials.values()
    ]
    return {
        "n_trials": len(all_trials),
        "n_seeds": len(per_seed_trials),
        "survival_rate_overall": survival_count / len(all_trials),
        "trial_length_mean": float(mean(all_trials)),
        "trial_length_median": float(median(all_trials)),
        "trial_length_stdev": float(pstdev(all_trials)),
        "per_seed_survival_mean": float(mean(per_seed_rates)),
        "per_seed_survival_stdev": float(pstdev(per_seed_rates)) if len(per_seed_rates) > 1 else 0.0,
    }


# -----------------------------------------------------------------------------
# 2. Python sweep (re-uses the sweep worker via direct call)
# -----------------------------------------------------------------------------


def _python_run_one_seed(args) -> Tuple[int, List[int]]:
    """Worker: run one seed and return (seed, list of trial t_terminal)."""
    (algorithm, seed, num_trials, max_horizon,
     enable_numba, use_jit_planner) = args
    if enable_numba and not use_jit_planner:
        # Backward-compat moderate JIT (G_epistemic_value swap only).
        from sl.numba_kernels import maybe_swap_in_jit_efe, HAS_NUMBA
        if HAS_NUMBA:
            maybe_swap_in_jit_efe()

    from sl.config import GridConfig, RunOptions, Weights
    from sl.agent import run_experiment

    grid = GridConfig.from_matlab_indices()
    options = RunOptions(algorithm=algorithm, seed=seed,
                         num_trials=num_trials, max_horizon=max_horizon,
                         use_jit_planner=use_jit_planner)
    weights = Weights()
    result = run_experiment(grid, options, weights)
    return seed, [tr.t_terminal for tr in result.trials]


def run_python_sweep(algorithm: str, seeds: List[int], num_trials: int,
                     max_horizon: int, workers: int,
                     enable_numba: bool = False,
                     use_jit_planner: bool = False) -> Dict[int, List[int]]:
    tasks = [(algorithm, s, num_trials, max_horizon, enable_numba, use_jit_planner)
             for s in seeds]
    if workers <= 1:
        results = [_python_run_one_seed(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = pool.map(_python_run_one_seed, tasks)
    return dict(results)


# -----------------------------------------------------------------------------
# 3. Statistical comparison
# -----------------------------------------------------------------------------


def ks_statistic(a: List[int], b: List[int]) -> float:
    """Two-sample Kolmogorov-Smirnov (no scipy needed for the simple case).

    Returns the supremum-norm distance between empirical CDFs.
    """
    if not a or not b:
        return float("nan")
    aa = sorted(a); bb = sorted(b)
    all_vals = sorted(set(aa) | set(bb))
    cdf_a, cdf_b = 0, 0
    i = j = 0
    sup = 0.0
    for v in all_vals:
        while i < len(aa) and aa[i] <= v:
            i += 1
        while j < len(bb) and bb[j] <= v:
            j += 1
        sup = max(sup, abs(i / len(aa) - j / len(bb)))
    return sup


def compare(matlab_per_seed: Dict[int, List[int]],
            python_per_seed: Dict[int, List[int]]) -> dict:
    matlab_summary = survival_summary(matlab_per_seed)
    python_summary = survival_summary(python_per_seed)
    # KS over the *all-trials* distributions (not seed-paired).
    matlab_all = [t for v in matlab_per_seed.values() for t in v]
    python_all = [t for v in python_per_seed.values() for t in v]
    ks = ks_statistic(matlab_all, python_all)
    return {
        "matlab": matlab_summary,
        "python": python_summary,
        "ks_distance": ks,
        "trial_length_mean_delta_pct": (
            100.0 * (python_summary["trial_length_mean"] - matlab_summary["trial_length_mean"])
            / max(1e-12, matlab_summary["trial_length_mean"])
            if matlab_summary.get("trial_length_mean") else float("nan")
        ),
        "survival_rate_delta_pp": (
            100.0 * (python_summary["per_seed_survival_mean"] - matlab_summary["per_seed_survival_mean"])
            if matlab_summary.get("per_seed_survival_mean") is not None else float("nan")
        ),
    }


# -----------------------------------------------------------------------------
# 4. Speed benchmark — single-trial timing
# -----------------------------------------------------------------------------


def benchmark_single_trial(algorithm: str, max_horizon: int = 9, seed: int = 1,
                            mode: str = "numpy", warmup: bool = True) -> dict:
    """Run one trial; returns per-step time and total.

    mode :
        'numpy'      — plain NumPy planner
        'numba_efe'  — moderate JIT (only G_epistemic_value swapped)
        'jit'        — full JIT tree-search recursion
    """
    use_jit_planner = (mode == "jit")
    if mode == "numba_efe":
        from sl.numba_kernels import maybe_swap_in_jit_efe, HAS_NUMBA
        if HAS_NUMBA:
            maybe_swap_in_jit_efe(extended=False)

    from sl.config import GridConfig, RunOptions, Weights
    from sl.agent import run_trial
    from sl.env import initialise_environment
    from sl.rng import make_rng

    grid = GridConfig.from_matlab_indices()
    options = RunOptions(algorithm=algorithm, seed=seed, num_trials=1,
                         max_horizon=max_horizon, use_jit_planner=use_jit_planner)
    weights = Weights()

    if warmup and (mode != "numpy"):
        # Trigger JIT compilation; the timed run reuses the cached kernels.
        model = initialise_environment(grid)
        run_trial(grid, RunOptions(algorithm=algorithm, seed=seed, num_trials=1,
                                    max_horizon=2, max_steps_per_trial=3,
                                    use_jit_planner=use_jit_planner),
                  weights, model, make_rng(seed))

    model = initialise_environment(grid)
    t0 = time.perf_counter()
    result = run_trial(grid, options, weights, model, make_rng(seed))
    elapsed = time.perf_counter() - t0
    n_steps = result.t_terminal
    return {
        "algorithm": algorithm,
        "max_horizon": max_horizon,
        "mode": mode,
        "t_terminal": n_steps,
        "wall_seconds": elapsed,
        "ms_per_step": 1000.0 * elapsed / max(1, n_steps),
        "survived": result.survived,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--algorithms", nargs="+", default=["SI", "SL"])
    p.add_argument("--num-seeds", type=int, default=30,
                   help="Number of MATLAB seeds to load AND Python seeds to run")
    p.add_argument("--num-trials", type=int, default=30,
                   help="Trials per seed for Python (MATLAB has 120, we slice to this)")
    p.add_argument("--max-horizon", type=int, default=9)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--enable-numba", action="store_true",
                   help="Enable moderate Numba JIT (G_epistemic_value swap).")
    p.add_argument("--use-jit-planner", action="store_true",
                   help="Enable the full JIT tree-search recursion.")
    p.add_argument("--skip-sweep", action="store_true",
                   help="Skip the multi-seed sweep; only run the single-trial benchmark.")
    p.add_argument("--bench-modes", nargs="+",
                   default=["numpy", "numba_efe", "jit"],
                   help="Modes to time in the single-trial benchmark.")
    p.add_argument("--output", default="",
                   help="Optional JSON path for the full report.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    ns = _parse_args(argv if argv is not None else sys.argv[1:])
    seeds = list(range(1, ns.num_seeds + 1))

    print("=" * 78)
    print(f"Benchmark + statistical comparison")
    print(f"  algorithms      : {ns.algorithms}")
    print(f"  num_seeds       : {ns.num_seeds}  (Python; MATLAB load up to same)")
    print(f"  num_trials/seed : {ns.num_trials}  (Python; MATLAB has 120, sliced)")
    print(f"  max_horizon     : {ns.max_horizon}")
    print(f"  workers         : {ns.workers}")
    print(f"  numba           : {ns.enable_numba}")
    print("=" * 78)

    full_report = {"comparisons": {}, "benchmarks": {}}

    # ---------- Speed benchmark ---------------------------------------------
    print("\n--- Single-trial speed benchmark (max_horizon=9, seed=1) ---")
    for algo in ns.algorithms:
        for mode in ns.bench_modes:
            r = benchmark_single_trial(algo, max_horizon=ns.max_horizon,
                                        seed=1, mode=mode)
            tag = f"{algo}_{mode}"
            full_report["benchmarks"][tag] = r
            print(f"  {algo:5s} {mode:10s}: "
                  f"{r['wall_seconds']:7.2f}s for {r['t_terminal']:3d} steps "
                  f"({r['ms_per_step']:7.1f} ms/step) survived={r['survived']}")

    if ns.skip_sweep:
        if ns.output:
            Path(ns.output).write_text(json.dumps(full_report, indent=2))
        return 0

    # ---------- Statistical comparison --------------------------------------
    for algo in ns.algorithms:
        print(f"\n--- {algo}: MATLAB vs Python ---")

        print("  loading MATLAB ground truth...", end=" ", flush=True)
        matlab_per_seed = load_matlab_survival(algo, max_seeds=ns.num_seeds)
        # Slice to first num_trials per seed to match Python sweep.
        matlab_per_seed = {s: t[:ns.num_trials] for s, t in matlab_per_seed.items()}
        print(f"loaded {len(matlab_per_seed)} seeds × {ns.num_trials} trials")

        print(f"  running Python sweep ({len(seeds)} seeds × {ns.num_trials} trials)...",
              end=" ", flush=True)
        t0 = time.perf_counter()
        python_per_seed = run_python_sweep(
            algo, seeds, ns.num_trials, ns.max_horizon, ns.workers,
            enable_numba=ns.enable_numba,
            use_jit_planner=ns.use_jit_planner,
        )
        elapsed = time.perf_counter() - t0
        print(f"done in {elapsed:.1f}s")

        cmp = compare(matlab_per_seed, python_per_seed)
        cmp["python_sweep_seconds"] = elapsed
        full_report["comparisons"][algo] = cmp

        m = cmp["matlab"]; p = cmp["python"]
        print(f"  MATLAB  trials={m['n_trials']:5d} mean_t={m['trial_length_mean']:6.2f} "
              f"survival={m['per_seed_survival_mean']:.3f}±{m['per_seed_survival_stdev']:.3f}")
        print(f"  Python  trials={p['n_trials']:5d} mean_t={p['trial_length_mean']:6.2f} "
              f"survival={p['per_seed_survival_mean']:.3f}±{p['per_seed_survival_stdev']:.3f}")
        print(f"  Δmean_trial_length = {cmp['trial_length_mean_delta_pct']:+.1f}%")
        print(f"  Δsurvival_rate     = {cmp['survival_rate_delta_pp']:+.1f} pp")
        print(f"  KS distance        = {cmp['ks_distance']:.3f}  (smaller = better)")

    if ns.output:
        Path(ns.output).write_text(json.dumps(full_report, indent=2, default=str))
        print(f"\nFull report → {ns.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Multiprocessing sweep driver.

Embarrassingly-parallel over (algorithm, seed). Mirrors what
``parallel_runs_PC.m`` does in MATLAB. Use one worker per Slurm CPU.

Each worker runs a full ``run_experiment`` (one seed × num_trials trials).
Results are written incrementally to ``--output-dir`` as JSON files —
matches the per-seed flat-file convention the analysis scripts expect.

Usage::

    python -m sl.sweep \\
        --algorithms SL SI BA BAUCB \\
        --seeds 1-30 \\
        --num-trials 120 \\
        --output-dir results/python_sweep \\
        --workers 8
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
from typing import List, Tuple

import numpy as np

from .agent import run_experiment
from .config import GridConfig, RunOptions, Weights, ALGORITHM_VARIANTS


def _parse_seed_spec(spec: str) -> List[int]:
    """Accept formats: '1', '1-5', '1,3,5,7' or combinations '1-3,7,10-12'."""
    seeds: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(chunk))
    return sorted(set(seeds))


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed sweep driver.")
    p.add_argument("--algorithms", nargs="+", default=["SI", "SL"],
                   help="One or more algorithm names from ALGORITHM_VARIANTS.")
    p.add_argument("--seeds", default="1-30",
                   help="Seed specification, e.g. '1-30' or '1,3,5'.")
    p.add_argument("--num-trials", type=int, default=120)
    p.add_argument("--max-horizon", type=int, default=9)
    p.add_argument("--max-steps", type=int, default=100)

    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--start-position", type=int, default=51)
    p.add_argument("--hill-pos", type=int, default=55)
    p.add_argument("--food-sources", type=int, nargs=4, default=[71, 43, 57, 78])
    p.add_argument("--water-sources", type=int, nargs=4, default=[73, 33, 48, 67])
    p.add_argument("--sleep-sources", type=int, nargs=4, default=[64, 44, 49, 59])
    p.add_argument("--grid-id", default="default10")

    p.add_argument("--w-novelty", type=float, default=10.0)
    p.add_argument("--w-learning", type=float, default=40.0)
    p.add_argument("--w-epistemic", type=float, default=1.0)
    p.add_argument("--w-preference", type=float, default=10.0)
    p.add_argument("--ucb-scale", type=float, default=5.0)

    p.add_argument("--learning-prune-threshold", type=float, default=0.2)
    p.add_argument("--no-real-smoothing", action="store_true")
    p.add_argument("--adaptive-likelihood-in-plan", action="store_true")
    p.add_argument("--state-selection", choices=["sample", "map"], default="sample")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--enable-numba", action="store_true",
                   help="Moderate JIT (G_epistemic_value swap only).")
    p.add_argument("--use-jit-planner", action="store_true",
                   help="Full JIT tree-search recursion (recommended).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (algo, seed) tasks whose JSON output already exists. "
                        "Useful when resuming a partial sweep after a Slurm time-limit kill.")
    return p.parse_args(argv)


def _worker(task: Tuple[str, int, dict]) -> dict:
    algorithm, seed, kw = task
    use_jit_planner = kw.pop("use_jit_planner", False)
    if kw.pop("enable_numba", False) and not use_jit_planner:
        from .numba_kernels import maybe_swap_in_jit_efe
        maybe_swap_in_jit_efe()

    grid = GridConfig.from_matlab_indices(
        grid_size=kw["grid_size"],
        start_position=kw["start_position"],
        hill_pos=kw["hill_pos"],
        food_sources=kw["food_sources"],
        water_sources=kw["water_sources"],
        sleep_sources=kw["sleep_sources"],
        grid_id=kw["grid_id"],
    )
    options = RunOptions(
        algorithm=algorithm,
        seed=seed,
        num_trials=kw["num_trials"],
        max_horizon=kw["max_horizon"],
        max_steps_per_trial=kw["max_steps"],
        real_smoothing=not kw["no_real_smoothing"],
        adaptive_likelihood_in_plan=kw["adaptive_likelihood_in_plan"],
        learning_prune_threshold=kw["learning_prune_threshold"],
        state_selection=kw["state_selection"],
        use_jit_planner=use_jit_planner,
    )
    weights = Weights(
        novelty=kw["w_novelty"],
        learning=kw["w_learning"],
        epistemic=kw["w_epistemic"],
        preference=kw["w_preference"],
        ucb_scale=kw["ucb_scale"],
    )

    t0 = time.time()
    result = run_experiment(grid, options, weights)
    elapsed = time.time() - t0

    summary = {
        "algorithm": algorithm,
        "seed": seed,
        "grid_id": kw["grid_id"],
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

    out_path = Path(kw["output_dir"]) / f"{algorithm}_seed{seed}_{kw['grid_id']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return {"algorithm": algorithm, "seed": seed,
            "survival_rate": summary["survival_rate"],
            "elapsed_seconds": elapsed, "path": str(out_path)}


def main(argv: List[str] | None = None) -> int:
    ns = _parse_args(argv if argv is not None else sys.argv[1:])
    seeds = _parse_seed_spec(ns.seeds)
    if not seeds:
        print("[sweep] no seeds parsed; exiting.", file=sys.stderr)
        return 2

    Path(ns.output_dir).mkdir(parents=True, exist_ok=True)

    kw = {k: v for k, v in vars(ns).items()
          if k not in {"algorithms", "seeds", "workers", "skip_existing"}}
    # Interleave (seed-major) so a partial run still has balanced samples
    # across algorithms if it gets killed at a time limit.
    tasks: List[Tuple[str, int, dict]] = []
    skipped = 0
    for seed in seeds:
        for algo in ns.algorithms:
            if ns.skip_existing:
                expected = Path(ns.output_dir) / f"{algo}_seed{seed}_{kw['grid_id']}.json"
                if expected.exists():
                    skipped += 1
                    continue
            tasks.append((algo, seed, dict(kw)))
    if ns.skip_existing and skipped:
        print(f"[sweep] skipping {skipped} already-completed tasks")

    print(f"[sweep] {len(tasks)} tasks, {ns.workers} workers, "
          f"output={ns.output_dir}")

    t0 = time.time()
    if ns.workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        # ``spawn`` is safer with NumPy/MKL than the Linux ``fork`` default.
        ctx = mp.get_context("spawn")
        with ctx.Pool(ns.workers) as pool:
            results = pool.map(_worker, tasks)
    total = time.time() - t0

    # Print per-task summary line; analysis scripts can rebuild from JSONs.
    for r in results:
        print(f"[sweep] {r['algorithm']:8s} seed={r['seed']:3d} "
              f"survival={r['survival_rate']:.3f} "
              f"time={r['elapsed_seconds']:.1f}s -> {r['path']}")
    print(f"[sweep] total wall time: {total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

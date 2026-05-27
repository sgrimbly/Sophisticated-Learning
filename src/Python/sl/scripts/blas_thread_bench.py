"""Quick benchmark to determine optimal BLAS-thread × worker combo.

Measures wall-time for a fixed workload across a few candidate
configurations:

    4 workers × 1 thread  (current default)
    2 workers × 2 threads
    1 worker  × 4 threads

Workload: SL with 1 seed × 5 trials at horizon=9.

Run on a node with 4 isolated cores and a JIT cache pre-warmed (we run
a 1-seed warmup first to compile, otherwise the timed run is dominated
by Numba compilation time).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def time_run(workers: int, blas_threads: int, n_seeds: int, n_trials: int,
              algorithm: str = "SL") -> dict:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(blas_threads)
    env["MKL_NUM_THREADS"] = str(blas_threads)
    env["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    env["NUMEXPR_NUM_THREADS"] = str(blas_threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)

    out_dir = Path(f"/tmp/sl_blas_bench_{algorithm}_w{workers}_t{blas_threads}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any prior output so skip-existing doesn't kick in.
    for p in out_dir.glob("*.json"):
        p.unlink()

    cmd = [
        sys.executable, "-m", "sl.sweep",
        "--algorithms", algorithm,
        "--seeds", f"1-{n_seeds}",
        "--num-trials", str(n_trials),
        "--max-horizon", "9",
        "--max-steps", "100",
        "--output-dir", str(out_dir),
        "--workers", str(workers),
        "--use-jit-planner",
        "--grid-id", "default10",
    ]
    cwd = "/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python"
    t0 = time.perf_counter()
    res = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"  ERR ({workers}×{blas_threads}): {res.stderr[-500:]}", flush=True)
        return {"workers": workers, "threads": blas_threads, "elapsed": float("nan")}
    return {"workers": workers, "threads": blas_threads, "elapsed": elapsed}


def main() -> int:
    # Warm-up the JIT cache outside the loop so all subsequent runs have it.
    print("Warming JIT cache...", flush=True)
    time_run(workers=1, blas_threads=1, n_seeds=1, n_trials=2, algorithm="SL")
    time_run(workers=1, blas_threads=1, n_seeds=1, n_trials=2, algorithm="SI")

    print("\n--- Benchmark: SL 1 seed × 5 trials, varying (workers, BLAS threads) ---", flush=True)
    results = []
    configs = [(4, 1), (2, 2), (1, 4), (1, 1)]
    for w, t in configs:
        print(f"  running workers={w} threads={t}...", end=" ", flush=True)
        r = time_run(workers=w, blas_threads=t, n_seeds=1, n_trials=5, algorithm="SL")
        results.append(r)
        print(f"{r['elapsed']:.1f}s", flush=True)

    print("\n--- Summary (lower is better) ---")
    print(f"  {'config':<20} {'wall_seconds':>12}")
    for r in sorted(results, key=lambda x: x["elapsed"]):
        cfg = f"{r['workers']}w × {r['threads']}t"
        print(f"  {cfg:<20} {r['elapsed']:>12.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

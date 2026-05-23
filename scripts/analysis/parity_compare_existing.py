"""Compare existing Python sweep results against MATLAB reference outputs.

For each algorithm present in --python-dir, loads the per-seed JSON files,
extracts per-trial t_terminal values, and compares against the matching
MATLAB reference run in --matlab-dir. MATLAB values are decremented by 1
to align with Python's step-count convention (MATLAB starts t=1 and
increments after the body, so MATLAB t_terminal = steps_executed + 1).

Reports per-algorithm:
  - n seeds and trials on both sides
  - mean trial length and SD
  - mean-difference (Python − MATLAB), absolute and percent
  - KS distance and α=0.05 critical value
  - pass/fail at α=0.05

Use as a quick parity check when a full sweep cannot complete; with even
small n the KS test against a 200-seed MATLAB reference is informative.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


PYTHON_JSON_RE = re.compile(r"^(.+?)_seed(\d+)_(?:.+)?\.json$")


def load_python_results(out_dir: Path) -> dict[str, dict[int, list[int]]]:
    """Return {algorithm: {seed: [t_terminal, ...]}} from a sl.sweep output dir."""
    out: dict[str, dict[int, list[int]]] = defaultdict(dict)
    for f in sorted(out_dir.glob("*.json")):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        algo = d.get("algorithm")
        seed = d.get("seed")
        trials = d.get("trials")
        if not (algo and seed is not None and trials):
            continue
        out[algo][int(seed)] = [int(t["t_terminal"]) for t in trials]
    return out


def load_matlab_results(ref_dir: Path, algorithm: str) -> dict[int, list[int]]:
    """Return {seed: [t_terminal, ...]} for one algorithm. Decrements by 1."""
    out: dict[int, list[int]] = {}
    alg_dir = ref_dir / algorithm
    if not alg_dir.is_dir():
        return out
    pat = re.compile(rf"^{re.escape(algorithm)}_Seed_?(\d+)(?:_.*)?\.txt$")
    for f in sorted(alg_dir.glob("*.txt")):
        m = pat.match(f.name)
        if not m:
            continue
        seed = int(m.group(1))
        try:
            vals = [int(float(line.strip())) - 1 for line in open(f) if line.strip()]
        except (ValueError, OSError):
            continue
        if vals:
            out[seed] = vals
    return out


def ks_2samp_manual(a: list[float], b: list[float]) -> tuple[float, float]:
    """Manual KS 2-sample test, returns (statistic, pvalue_approx)."""
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    combined = sorted(set(a_sorted + b_sorted))
    na, nb = len(a_sorted), len(b_sorted)

    def ecdf(sorted_data, x, n):
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_data[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo / n

    d = max(abs(ecdf(a_sorted, x, na) - ecdf(b_sorted, x, nb)) for x in combined)
    en = np.sqrt(na * nb / (na + nb))
    pvalue = np.exp(-2.0 * (en * d) ** 2)
    return d, pvalue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python-dir", type=Path, required=True,
                    help="sl.sweep output directory with *.json result files")
    ap.add_argument("--matlab-dir", type=Path, required=True,
                    help="Directory containing {algorithm}/{algorithm}_Seed{N}.txt")
    ap.add_argument("--output", type=Path, required=True,
                    help="CSV output path for the comparison summary")
    args = ap.parse_args()

    py = load_python_results(args.python_dir)
    if not py:
        print(f"[parity] no Python results found in {args.python_dir}")
        return

    rows = []
    missing = []
    for algo, seeds in sorted(py.items()):
        ml = load_matlab_results(args.matlab_dir, algo)
        if not ml:
            missing.append(algo)
            continue
        py_trials = [t for trials in seeds.values() for t in trials]
        ml_trials = [t for trials in ml.values() for t in trials]
        py_arr = np.asarray(py_trials, dtype=float)
        ml_arr = np.asarray(ml_trials, dtype=float)

        if HAS_SCIPY:
            ks = _stats.ks_2samp(py_arr, ml_arr)
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        else:
            ks_stat, ks_p = ks_2samp_manual(py_arr.tolist(), ml_arr.tolist())

        na, nb = len(py_arr), len(ml_arr)
        ks_crit_05 = 1.358 * np.sqrt((na + nb) / (na * nb))
        py_mean = float(py_arr.mean())
        ml_mean = float(ml_arr.mean())
        py_std = float(py_arr.std(ddof=1)) if na > 1 else 0.0
        ml_std = float(ml_arr.std(ddof=1)) if nb > 1 else 0.0
        delta_pct = (py_mean - ml_mean) / ml_mean * 100 if ml_mean else float("nan")

        rows.append({
            "algorithm": algo,
            "py_seeds": len(seeds),
            "py_trials": na,
            "ml_seeds": len(ml),
            "ml_trials": nb,
            "py_mean": py_mean,
            "py_std": py_std,
            "ml_mean": ml_mean,
            "ml_std": ml_std,
            "delta_mean_pct": delta_pct,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "ks_crit_05": float(ks_crit_05),
            "pass_alpha_05": ks_stat < ks_crit_05,
        })

    print(f"\n{'algorithm':30s} {'py_seeds':>9s} {'ml_seeds':>9s} {'py_mean':>9s} {'ml_mean':>9s} {'Δ%':>7s} {'KS':>6s} {'KS₀.₀₅':>7s} {'pass':>5s}")
    print("-" * 105)
    for r in rows:
        flag = "PASS" if r["pass_alpha_05"] else "FAIL"
        print(f"{r['algorithm']:30s} {r['py_seeds']:>9d} {r['ml_seeds']:>9d} "
              f"{r['py_mean']:>9.2f} {r['ml_mean']:>9.2f} "
              f"{r['delta_mean_pct']:>+6.1f}% {r['ks_stat']:>6.3f} "
              f"{r['ks_crit_05']:>7.3f} {flag:>5s}")
    if missing:
        print(f"\nMissing MATLAB reference for: {', '.join(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()

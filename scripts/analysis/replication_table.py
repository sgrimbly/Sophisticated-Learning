"""Merge the canonical-8 replication sweep into one parity + runtime table.

For each algorithm variant, compares the Python (Numba) sweep against the
MATLAB diagnosis batch and reports:

  - survival parity, scored PER SEED (each seed's AUC = mean trial-length
    over its 120 trials), so the KS test treats one independent
    trajectory per seed -- NOT 120 correlated trials as independent
    (the pseudo-replication bug in parity_compare_existing.py).
  - runtime: MATLAB ms/step (from the perf_time_step log) and Python
    ms/step (elapsed_seconds / total executed steps, averaged over seeds),
    plus the speed ratio.

Inputs
  --python-dir         sl.sweep output dir of {algo}_seed{N}_{grid}.json
  --matlab-survival-dir  diagnosis batch dir with {algo}/{algo}_Seed{N}_metrics.csv
  --matlab-timing-log  perf_time_step .out log (ms_per_step per variant)
  --output             CSV path for the merged table

MATLAB survival is decremented by 1 to match Python's step convention.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

CANONICAL_8 = [
    "SI", "SI_noNovelty", "SI_novelty_smooth", "SI_smooth_noNovelty",
    "SL_adaptivePlan", "SL_noNovelty_adaptivePlan",
    "SL_noSmooth_adaptivePlan", "SL_noNovelty_noSmooth_adaptivePlan",
]


def load_python(python_dir: Path) -> dict[str, dict[int, dict]]:
    """{algo: {seed: {"t": [t_terminal...], "elapsed": s}}}."""
    out: dict[str, dict[int, dict]] = {}
    for f in sorted(python_dir.glob("*.json")):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        algo, seed, trials = d.get("algorithm"), d.get("seed"), d.get("trials")
        if not (algo and seed is not None and trials):
            continue
        out.setdefault(algo, {})[int(seed)] = {
            "t": [int(t["t_terminal"]) for t in trials],
            "elapsed": float(d.get("elapsed_seconds", float("nan"))),
        }
    return out


def load_matlab_survival(ref_dir: Path, algo: str) -> dict[int, list[int]]:
    """{seed: [t_terminal...]} from {algo}/{algo}_Seed{N}_metrics.csv, minus 1."""
    out: dict[int, list[int]] = {}
    alg_dir = ref_dir / algo
    if not alg_dir.is_dir():
        return out
    pat = re.compile(rf"^{re.escape(algo)}_Seed_?(\d+)_metrics\.csv$")
    for f in sorted(alg_dir.glob("*_metrics.csv")):
        m = pat.match(f.name)
        if not m:
            continue
        seed = int(m.group(1))
        try:
            with open(f) as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or "survival" not in reader.fieldnames:
                    continue
                vals = [int(float(r["survival"])) - 1 for r in reader if r.get("survival")]
        except (ValueError, OSError):
            continue
        if vals:
            out[seed] = vals
    return out


def load_matlab_timing(log: Path) -> dict[str, float]:
    """{algo: ms_per_step} parsed from a perf_time_step log."""
    out: dict[str, float] = {}
    pat = re.compile(r"\[([A-Za-z0-9_]+)\].*ms_per_step=([\d.]+)")
    if not log or not log.exists():
        return out
    for line in open(log):
        m = pat.search(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def _parse_hms(s: str) -> float:
    """'HH:MM:SS' (hours may exceed 24) -> seconds."""
    parts = [int(x) for x in s.strip().split(":")]
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + sec


def load_matlab_walltime(slurm_logs_root: Path, algo: str) -> dict[int, float]:
    """{seed: full-120-trial wall seconds} from diagnosis slurm logs.

    Layout: {root}/{algo}/<...>_Seed{N}_<...>/slurm-*.out, each containing a
    'TOTAL RUNTIME (hours/minutes/seconds): HH:MM:SS' line. This is the matched
    full-run cost -- the perf_time_step ms/step is 5 *early* trials and badly
    understates the late-trial-dominated full run.
    """
    out: dict[int, float] = {}
    base = slurm_logs_root / algo
    if not base.is_dir():
        return out
    seed_re = re.compile(r"_Seed(\d+)_")
    rt_re = re.compile(r"TOTAL RUNTIME.*?:\s*([\d:]+)")
    for d in sorted(base.glob("*Seed*")):
        m = seed_re.search(d.name)
        if not m:
            continue
        seed = int(m.group(1))
        for log in sorted(d.glob("slurm-*.out")):
            try:
                txt = log.read_text(errors="ignore")
            except OSError:
                continue
            rm = rt_re.search(txt)
            if rm:
                out[seed] = _parse_hms(rm.group(1))
                break
    return out


def per_seed_auc(by_seed: dict[int, list[int]]) -> np.ndarray:
    """One AUC (mean trial-length) per seed -> independent-sample array."""
    return np.array([float(np.mean(v)) for v in by_seed.values() if v], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python-dir", type=Path, required=True)
    ap.add_argument("--matlab-survival-dir", type=Path, required=True)
    ap.add_argument("--matlab-timing-log", type=Path, default=None,
                    help="(optional) perf_time_step log; early-trial ms/step only.")
    ap.add_argument("--matlab-slurm-logs", type=Path, default=None,
                    help="diagnosis slurm_logs/<batch> dir for full-run wall time.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    py = load_python(args.python_dir)
    ml_timing = load_matlab_timing(args.matlab_timing_log)

    rows = []
    for algo in CANONICAL_8:
        py_seeds = py.get(algo, {})
        ml_seeds = load_matlab_survival(args.matlab_survival_dir, algo)
        if not py_seeds or not ml_seeds:
            rows.append({"algorithm": algo, "n_py": len(py_seeds), "n_ml": len(ml_seeds),
                         "note": "missing data"})
            continue

        py_auc = per_seed_auc({s: d["t"] for s, d in py_seeds.items()})
        ml_auc = per_seed_auc(ml_seeds)
        n_py, n_ml = len(py_auc), len(ml_auc)

        ks = stats.ks_2samp(py_auc, ml_auc)
        ks_crit = 1.358 * np.sqrt((n_py + n_ml) / (n_py * n_ml))
        delta_pct = (py_auc.mean() - ml_auc.mean()) / ml_auc.mean() * 100

        # Full-run wall time per seed (the matched, honest runtime metric).
        py_wall = [d["elapsed"] for d in py_seeds.values()
                   if np.isfinite(d["elapsed"])]
        py_h = float(np.median(py_wall)) / 3600 if py_wall else float("nan")
        ml_wall = (load_matlab_walltime(args.matlab_slurm_logs, algo)
                   if args.matlab_slurm_logs else {})
        ml_h = (float(np.median(list(ml_wall.values()))) / 3600
                if ml_wall else float("nan"))
        speedup = (ml_h / py_h if (py_h and np.isfinite(ml_h)
                   and np.isfinite(py_h) and py_h > 0) else float("nan"))

        rows.append({
            "algorithm": algo, "n_py": n_py, "n_ml": n_ml,
            "py_auc": py_auc.mean(), "ml_auc": ml_auc.mean(),
            "delta_pct": delta_pct,
            "ks_stat": float(ks.statistic), "ks_crit_05": float(ks_crit),
            "ks_pvalue": float(ks.pvalue),
            "parity_pass": bool(ks.statistic < ks_crit),
            "py_wall_h": py_h, "ml_wall_h": ml_h, "speedup": speedup,
            "ml_ms_step_early": ml_timing.get(algo, float("nan")),
        })

    hdr = (f"\n{'variant':36s} {'n_py':>4s} {'n_ml':>4s} {'py_AUC':>7s} {'ml_AUC':>7s} "
           f"{'Δ%':>6s} {'KS':>6s} {'KScrit':>6s} {'par':>4s} "
           f"{'py_h':>6s} {'ml_h':>6s} {'×':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "note" in r:
            print(f"{r['algorithm']:36s} {r['n_py']:>4d} {r['n_ml']:>4d}   {r['note']}")
            continue
        print(f"{r['algorithm']:36s} {r['n_py']:>4d} {r['n_ml']:>4d} "
              f"{r['py_auc']:>7.2f} {r['ml_auc']:>7.2f} {r['delta_pct']:>+5.1f}% "
              f"{r['ks_stat']:>6.3f} {r['ks_crit_05']:>6.3f} "
              f"{'OK' if r['parity_pass'] else 'X':>4s} "
              f"{r['py_wall_h']:>6.2f} {r['ml_wall_h']:>6.2f} {r['speedup']:>5.1f}x")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["algorithm", "n_py", "n_ml", "py_auc", "ml_auc", "delta_pct",
              "ks_stat", "ks_crit_05", "ks_pvalue", "parity_pass",
              "py_wall_h", "ml_wall_h", "speedup", "ml_ms_step_early", "note"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()

"""Plot Python sweep results, optionally overlaid on MATLAB ground truth.

Reads JSON files produced by :mod:`sl.sweep` (or :mod:`sl.scripts.run_clean_comparison`)
and produces the canonical algorithm-comparison plots:

  1. Learning curve: mean trial length per trial number, with shaded
     ±1 stderr-of-mean across seeds. One curve per algorithm. Optional
     MATLAB overlay (dashed).
  2. Cumulative survival: fraction of trials that ran to ``max_steps``,
     cumulative across the trial axis.
  3. Trial-length distribution: boxplot per algorithm.
  4. Per-seed survival rate distribution: dot-plot per algorithm.

All four plots can be produced from the same data; pass ``--plots all``
or any subset of ``learning_curve survival_cdf boxplot scatter``.

Usage:
    python -m sl.scripts.plot_results \\
        --input-dir results/python_sweep_20260501 \\
        --algorithms SI SL BA BAUCB \\
        --include-matlab \\
        --output-dir results/python_sweep_20260501/plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit(f"matplotlib required: {e}")


# Canonical colours mirroring scripts/analysis/plot_MATLAB_results.py
ALGORITHM_COLOURS = {
    "SL": "#1f77b4",      # blue
    "SI": "#2ca02c",      # green
    "BA": "#ff7f0e",      # orange
    "BAUCB": "#d62728",   # red
}

REPO_ROOT = Path(__file__).resolve().parents[4]
MATLAB_RESULTS = REPO_ROOT / "results" / "unknown_model" / "MATLAB" / "paper_si_sl_repro_default_env"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_python_sweep(input_dir: Path, algorithm: str) -> Dict[int, List[int]]:
    """Load per-seed t_terminal lists from sweep JSON outputs.

    Accepts either:
      * sweep.py output: one JSON per (algo, seed) with key 'trials'[i]['t_terminal']
      * run_clean_comparison.py output: combined JSON with 'python_per_seed' map
    """
    out: Dict[int, List[int]] = {}
    if not input_dir.exists():
        raise SystemExit(f"input dir not found: {input_dir}")

    # First try the per-seed flat-file layout (sweep.py).
    for path in sorted(input_dir.glob(f"{algorithm}_seed*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        if "trials" in d:
            seed = int(d.get("seed", 0))
            out[seed] = [int(t["t_terminal"]) for t in d["trials"]]

    # Fallback: combined file from run_clean_comparison.py
    if not out:
        for path in input_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
            except Exception:
                continue
            if d.get("algorithm") == algorithm and "python_per_seed" in d:
                for seed_str, trials in d["python_per_seed"].items():
                    out[int(seed_str)] = list(map(int, trials))

    return out


def load_matlab_ground_truth(algorithm: str, max_seeds: int = 200) -> Dict[int, List[int]]:
    """Load MATLAB Seed{N}.txt files (paper_si_sl_repro_default_env) and
    convert to Python step-count convention (subtract 1).
    """
    folder = MATLAB_RESULTS / algorithm
    out: Dict[int, List[int]] = {}
    if not folder.exists():
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


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _stack_per_trial(per_seed: Dict[int, List[int]], n_trials: int) -> np.ndarray:
    """Return (n_seeds, n_trials) array, padding short rows with the last value."""
    rows = []
    for seed, vs in sorted(per_seed.items()):
        if len(vs) >= n_trials:
            rows.append(vs[:n_trials])
        else:
            # Pad with NaN — incomplete seeds shouldn't bias the curve.
            row = list(vs) + [float("nan")] * (n_trials - len(vs))
            rows.append(row)
    return np.array(rows, dtype=np.float64)


def _learning_curve(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-trial mean and ±1 SEM across seeds."""
    mean = np.nanmean(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)
    n = np.where(n > 0, n, 1)
    std = np.nanstd(arr, axis=0)
    sem = std / np.sqrt(n)
    return mean, mean - sem, mean + sem


def _cumulative_survival(arr: np.ndarray, survive_at: int) -> np.ndarray:
    """Per-trial cumulative survival rate (fraction of seeds with ≥1 surviving
    trial up to and including this trial)."""
    survived = (arr >= survive_at).astype(np.float64)
    cum = np.maximum.accumulate(survived, axis=1)  # cumulative max per seed
    return np.nanmean(cum, axis=0)


def _moving_average(x: np.ndarray, w: int = 10) -> np.ndarray:
    """Simple moving average for smoothing."""
    if w <= 1 or len(x) < w:
        return x.copy()
    pad = np.full(w - 1, x[0])
    padded = np.concatenate([pad, x])
    cumsum = np.cumsum(np.insert(padded, 0, 0))
    return (cumsum[w:] - cumsum[:-w]) / w


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_learning_curve(
    python_data: Dict[str, Dict[int, List[int]]],
    matlab_data: Optional[Dict[str, Dict[int, List[int]]]],
    n_trials: int,
    output_path: Path,
    smooth_window: int = 10,
):
    fig, ax = plt.subplots(figsize=(11, 6))

    for algo in python_data:
        colour = ALGORITHM_COLOURS.get(algo, "black")
        py_arr = _stack_per_trial(python_data[algo], n_trials)
        py_mean, py_lo, py_hi = _learning_curve(py_arr)
        py_smooth = _moving_average(py_mean, smooth_window)
        x = np.arange(1, n_trials + 1)
        ax.plot(x, py_smooth, color=colour, label=f"{algo} (Python, n={py_arr.shape[0]} seeds)")
        ax.fill_between(x, py_lo, py_hi, color=colour, alpha=0.18, linewidth=0)

        if matlab_data and algo in matlab_data and matlab_data[algo]:
            ml_arr = _stack_per_trial(matlab_data[algo], n_trials)
            ml_mean, _, _ = _learning_curve(ml_arr)
            ml_smooth = _moving_average(ml_mean, smooth_window)
            ax.plot(x, ml_smooth, color=colour, linestyle="--", alpha=0.7,
                    label=f"{algo} (MATLAB, n={ml_arr.shape[0]} seeds)")

    ax.set_xlabel("Trial number")
    ax.set_ylabel(f"Mean trial length (steps)  —  {smooth_window}-trial moving average")
    ax.set_title("Learning curve: mean trial length per trial, ±1 SEM across seeds")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {output_path}")


def plot_cumulative_survival(
    python_data: Dict[str, Dict[int, List[int]]],
    matlab_data: Optional[Dict[str, Dict[int, List[int]]]],
    n_trials: int,
    survive_at: int,
    output_path: Path,
):
    fig, ax = plt.subplots(figsize=(11, 6))

    for algo in python_data:
        colour = ALGORITHM_COLOURS.get(algo, "black")
        py_arr = _stack_per_trial(python_data[algo], n_trials)
        py_cdf = _cumulative_survival(py_arr, survive_at)
        x = np.arange(1, n_trials + 1)
        ax.plot(x, py_cdf, color=colour, label=f"{algo} (Python, n={py_arr.shape[0]} seeds)")
        if matlab_data and algo in matlab_data and matlab_data[algo]:
            ml_arr = _stack_per_trial(matlab_data[algo], n_trials)
            ml_cdf = _cumulative_survival(ml_arr, survive_at)
            ax.plot(x, ml_cdf, color=colour, linestyle="--", alpha=0.7,
                    label=f"{algo} (MATLAB, n={ml_arr.shape[0]} seeds)")

    ax.set_xlabel("Trial number")
    ax.set_ylabel(f"Cumulative survival rate (≥{survive_at} steps reached)")
    ax.set_title("Cumulative survival: probability of having survived ≥1 trial by trial t")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {output_path}")


def plot_distribution_boxplot(
    python_data: Dict[str, Dict[int, List[int]]],
    matlab_data: Optional[Dict[str, Dict[int, List[int]]]],
    output_path: Path,
):
    """Side-by-side boxplots of trial-length distributions per algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = []
    labels = []
    data: List[List[int]] = []
    colours: List[str] = []

    pos = 1
    for algo in python_data:
        colour = ALGORITHM_COLOURS.get(algo, "black")
        py_all = [t for vs in python_data[algo].values() for t in vs]
        if py_all:
            data.append(py_all)
            positions.append(pos)
            labels.append(f"{algo}\n(Py, n={len(py_all)})")
            colours.append(colour)
            pos += 1
        if matlab_data and algo in matlab_data and matlab_data[algo]:
            ml_all = [t for vs in matlab_data[algo].values() for t in vs]
            data.append(ml_all)
            positions.append(pos)
            labels.append(f"{algo}\n(ML, n={len(ml_all)})")
            colours.append(colour)
            pos += 1
        pos += 0.5  # gap between algorithms

    bplot = ax.boxplot(data, positions=positions, widths=0.7,
                        patch_artist=True, showfliers=True)
    for patch, c in zip(bplot["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel("Trial length (steps)")
    ax.set_title("Trial-length distribution: Python (full JIT) vs MATLAB ground truth")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {output_path}")


def plot_per_seed_survival_scatter(
    python_data: Dict[str, Dict[int, List[int]]],
    matlab_data: Optional[Dict[str, Dict[int, List[int]]]],
    survive_at: int,
    output_path: Path,
):
    """Strip-plot: per-seed survival rates per algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pos = 1
    for algo in python_data:
        colour = ALGORITHM_COLOURS.get(algo, "black")
        py_rates = [
            sum(1 for t in vs if t >= survive_at) / max(1, len(vs))
            for vs in python_data[algo].values()
        ]
        if py_rates:
            xs = np.full(len(py_rates), pos) + np.random.uniform(-0.1, 0.1, len(py_rates))
            ax.scatter(xs, py_rates, color=colour, alpha=0.6, s=24, label=f"{algo} Python")
            ax.hlines(np.mean(py_rates), pos - 0.2, pos + 0.2,
                       colors="black", linewidth=2, alpha=0.8)
            pos += 1

        if matlab_data and algo in matlab_data and matlab_data[algo]:
            ml_rates = [
                sum(1 for t in vs if t >= survive_at) / max(1, len(vs))
                for vs in matlab_data[algo].values()
            ]
            if ml_rates:
                xs = np.full(len(ml_rates), pos) + np.random.uniform(-0.1, 0.1, len(ml_rates))
                ax.scatter(xs, ml_rates, color=colour, alpha=0.4, s=24, marker="x",
                           label=f"{algo} MATLAB")
                ax.hlines(np.mean(ml_rates), pos - 0.2, pos + 0.2,
                           colors="black", linewidth=2, alpha=0.6)
                pos += 1

        pos += 0.5

    ax.set_ylabel(f"Per-seed survival rate (fraction of trials reaching ≥{survive_at} steps)")
    ax.set_title("Per-seed survival rates")
    ax.set_xlim(0.5, pos - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Directory with JSON outputs from sl.sweep or run_clean_comparison.")
    p.add_argument("--algorithms", nargs="+", default=["SI", "SL", "BA", "BAUCB"])
    p.add_argument("--n-trials", type=int, default=120,
                   help="Number of trials per seed to align on (truncate longer; pad shorter).")
    p.add_argument("--survive-at", type=int, default=99,
                   help="Step count counted as 'survived' (default 99 — captures "
                        "both MATLAB convention max=99 and Python convention max=100). "
                        "MATLAB's loop runs t=1..99 (99 iterations) while Python's "
                        "runs t=0..99 (100 iterations); both correspond to '100 "
                        "conceptual steps without dying'.")
    p.add_argument("--include-matlab", action="store_true",
                   help="Overlay MATLAB ground-truth curves where available.")
    p.add_argument("--matlab-max-seeds", type=int, default=200)
    p.add_argument("--smooth-window", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write PNGs (default: <input-dir>/plots).")
    p.add_argument("--plots", nargs="+",
                   default=["learning_curve", "survival_cdf", "boxplot", "scatter"],
                   choices=["learning_curve", "survival_cdf", "boxplot", "scatter", "all"])
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if "all" in args.plots:
        args.plots = ["learning_curve", "survival_cdf", "boxplot", "scatter"]

    output_dir = args.output_dir or (args.input_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot] input dir: {args.input_dir}")
    print(f"[plot] output dir: {output_dir}")

    python_data: Dict[str, Dict[int, List[int]]] = {}
    matlab_data: Dict[str, Dict[int, List[int]]] = {}

    for algo in args.algorithms:
        py = load_python_sweep(args.input_dir, algo)
        if py:
            python_data[algo] = py
            n_seeds = len(py)
            n_trials_avg = int(np.mean([len(v) for v in py.values()])) if py else 0
            print(f"  loaded Python {algo}: {n_seeds} seeds, ~{n_trials_avg} trials/seed")
        else:
            print(f"  no Python data found for {algo}")

        if args.include_matlab:
            ml = load_matlab_ground_truth(algo, max_seeds=args.matlab_max_seeds)
            if ml:
                matlab_data[algo] = ml
                print(f"  loaded MATLAB {algo}: {len(ml)} seeds")

    if not python_data:
        print("[plot] ERROR: no Python data found; did the sweep run?", file=sys.stderr)
        return 2

    print()
    if "learning_curve" in args.plots:
        plot_learning_curve(python_data, matlab_data if args.include_matlab else None,
                             n_trials=args.n_trials,
                             output_path=output_dir / "learning_curve.png",
                             smooth_window=args.smooth_window)
    if "survival_cdf" in args.plots:
        plot_cumulative_survival(python_data,
                                  matlab_data if args.include_matlab else None,
                                  n_trials=args.n_trials,
                                  survive_at=args.survive_at,
                                  output_path=output_dir / "cumulative_survival.png")
    if "boxplot" in args.plots:
        plot_distribution_boxplot(python_data,
                                   matlab_data if args.include_matlab else None,
                                   output_path=output_dir / "trial_length_boxplot.png")
    if "scatter" in args.plots:
        plot_per_seed_survival_scatter(python_data,
                                        matlab_data if args.include_matlab else None,
                                        survive_at=args.survive_at,
                                        output_path=output_dir / "per_seed_survival.png")

    print(f"\n[plot] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

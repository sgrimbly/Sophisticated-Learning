from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FILE_PATTERN = re.compile(r"^(BAUCB|BA|SI|SL)_Seed_?(\d+)(?:_.*)?\.txt$")
DEFAULT_ALGORITHMS = ["BA", "BAUCB", "SI", "SL"]
DISPLAY_LABELS = {
    "BA": "BARL",
    "BAUCB": "BARL-UCB",
    "SI": "SI",
    "SL": "SL",
}
COLORS = {
    "BA": "#457b9d",
    "BAUCB": "#2a9d8f",
    "SI": "#6c757d",
    "SL": "#d62828",
}


@dataclass
class LoadedRun:
    algorithm: str
    seed: int
    path: Path
    mtime: float
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether manuscript-regime learning curves plateau.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing survival .txt files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for figures and summary tables.")
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS, help="Algorithms to include.")
    parser.add_argument("--num-trials", type=int, default=120, help="Expected number of trials per file.")
    parser.add_argument("--window", type=int, default=20, help="Plateau summary window size.")
    parser.add_argument(
        "--learning-curve-output",
        default="learning_curves.png",
        help="Filename for the learning-curve figure inside --output-dir.",
    )
    parser.add_argument(
        "--final-window-figure-output",
        default="final_window_summary.png",
        help="Filename for the final-window figure inside --output-dir.",
    )
    parser.add_argument(
        "--curve-summary-output",
        default="learning_curve_summary.csv",
        help="Filename for the learning-curve CSV inside --output-dir.",
    )
    parser.add_argument(
        "--plateau-summary-output",
        default="plateau_summary.csv",
        help="Filename for the plateau/final-window CSV inside --output-dir.",
    )
    parser.add_argument(
        "--duplicate-summary-output",
        default="duplicate_seed_files.csv",
        help="Filename for the duplicate-seed CSV inside --output-dir.",
    )
    parser.add_argument(
        "--late-slope-threshold",
        type=float,
        default=0.05,
        help="Absolute slope threshold for the last window.",
    )
    return parser.parse_args()


def load_runs(input_dir: Path, algorithms: list[str], num_trials: int) -> tuple[list[LoadedRun], list[dict[str, object]]]:
    candidates: dict[tuple[str, int], LoadedRun] = {}
    duplicate_rows: list[dict[str, object]] = []

    for path in sorted(input_dir.rglob("*.txt")):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue

        algorithm, seed_text = match.groups()
        if algorithm not in algorithms:
            continue

        values = np.loadtxt(path, dtype=float)
        if values.ndim == 0:
            values = np.array([float(values)])
        if len(values) != num_trials:
            continue

        seed = int(seed_text)
        loaded = LoadedRun(
            algorithm=algorithm,
            seed=seed,
            path=path,
            mtime=path.stat().st_mtime,
            values=values,
        )
        key = (algorithm, seed)
        if key in candidates:
            prior = candidates[key]
            if loaded.mtime >= prior.mtime:
                kept = loaded.path
                discarded = prior.path
                candidates[key] = loaded
            else:
                kept = prior.path
                discarded = loaded.path
            duplicate_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "kept_file": str(kept),
                    "discarded_file": str(discarded),
                }
            )
        else:
            candidates[key] = loaded

    runs = sorted(candidates.values(), key=lambda item: (item.algorithm, item.seed))
    return runs, duplicate_rows


def compute_curve_summary(runs: list[LoadedRun], algorithms: list[str], num_trials: int) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for algorithm in algorithms:
        algo_runs = [run for run in runs if run.algorithm == algorithm]
        if not algo_runs:
            continue
        matrix = np.vstack([run.values for run in algo_runs])
        means = np.mean(matrix, axis=0)
        stds = np.std(matrix, axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(num_trials)
        sems = stds / np.sqrt(matrix.shape[0])
        ci95 = 1.96 * sems
        for trial_idx in range(num_trials):
            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "trial": trial_idx + 1,
                    "mean": float(means[trial_idx]),
                    "std": float(stds[trial_idx]),
                    "count": int(matrix.shape[0]),
                    "sem": float(sems[trial_idx]),
                    "ci95": float(ci95[trial_idx]),
                }
            )
    return summary_rows


def compute_plateau_table(
    runs: list[LoadedRun],
    summary_rows: list[dict[str, object]],
    algorithms: list[str],
    num_trials: int,
    window: int,
    late_slope_threshold: float,
) -> list[dict[str, object]]:
    previous_start = num_trials - (2 * window) + 1
    previous_end = num_trials - window
    recent_start = num_trials - window + 1
    recent_end = num_trials

    plateau_rows: list[dict[str, object]] = []
    for algorithm in algorithms:
        algo_runs = [run for run in runs if run.algorithm == algorithm]
        if not algo_runs:
            continue

        matrix = np.vstack([run.values for run in algo_runs])
        previous_mean_by_seed = np.mean(matrix[:, previous_start - 1 : previous_end], axis=1)
        recent_mean_by_seed = np.mean(matrix[:, recent_start - 1 : recent_end], axis=1)
        mean_previous_window = float(np.mean(previous_mean_by_seed))
        mean_recent_window = float(np.mean(recent_mean_by_seed))
        window_diff = mean_recent_window - mean_previous_window

        final_window_mean = float(np.mean(recent_mean_by_seed))
        final_window_std = float(np.std(recent_mean_by_seed, ddof=1)) if len(recent_mean_by_seed) > 1 else 0.0
        final_window_sem = final_window_std / math.sqrt(len(recent_mean_by_seed)) if len(recent_mean_by_seed) > 0 else 0.0

        late_curve = [row for row in summary_rows if row["algorithm"] == algorithm and recent_start <= int(row["trial"]) <= recent_end]
        late_curve = sorted(late_curve, key=lambda row: int(row["trial"]))
        x = np.array([int(row["trial"]) for row in late_curve], dtype=float)
        y = np.array([float(row["mean"]) for row in late_curve], dtype=float)
        late_slope = float(np.polyfit(x, y, 1)[0])
        plateau_pass = abs(window_diff) < 1.0 and abs(late_slope) <= late_slope_threshold

        plateau_rows.append(
            {
                "algorithm": algorithm,
                "display_label": DISPLAY_LABELS.get(algorithm, algorithm),
                "n_seeds": len(algo_runs),
                "mean_previous_window": mean_previous_window,
                "mean_recent_window": mean_recent_window,
                "window_diff_recent_minus_previous": window_diff,
                "abs_window_diff": abs(window_diff),
                "late_slope_last_window": late_slope,
                "late_slope_threshold": late_slope_threshold,
                "plateau_pass": plateau_pass,
                "final_window_mean": final_window_mean,
                "final_window_sem": final_window_sem,
                "recommended_extension_trials": 0 if plateau_pass else 180,
                "extension_note": "Passes plateau criterion" if plateau_pass else "Fails local plateau check; extend to 300 trials",
            }
        )

    return plateau_rows


def plot_learning_curves(summary_rows: list[dict[str, object]], algorithms: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for algorithm in algorithms:
        algo_rows = sorted(
            (row for row in summary_rows if row["algorithm"] == algorithm),
            key=lambda row: int(row["trial"]),
        )
        if not algo_rows:
            continue
        x = np.array([int(row["trial"]) for row in algo_rows], dtype=int)
        y = np.array([float(row["mean"]) for row in algo_rows], dtype=float)
        ci = np.array([float(row["ci95"]) for row in algo_rows], dtype=float)
        color = COLORS.get(algorithm, None)
        ax.plot(x, y, label=DISPLAY_LABELS.get(algorithm, algorithm), linewidth=2, color=color)
        ax.fill_between(x, y - ci, y + ci, alpha=0.18, color=color)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Average Survival Steps")
    ax.set_title("Locked manuscript-regime learning curves")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_window_summary(plateau_rows: list[dict[str, object]], output_path: Path) -> None:
    sorted_rows = sorted(plateau_rows, key=lambda row: float(row["final_window_mean"]), reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(sorted_rows))
    heights = np.array([float(row["final_window_mean"]) for row in sorted_rows], dtype=float)
    yerr = np.array([1.96 * float(row["final_window_sem"]) for row in sorted_rows], dtype=float)
    colors = [COLORS.get(str(row["algorithm"]), "#4c566a") for row in sorted_rows]
    labels = [str(row["display_label"]) for row in sorted_rows]

    ax.bar(x, heights, yerr=yerr, color=colors, alpha=0.9, capsize=4)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean Survival, Last 20 Trials")
    ax.set_title("Final-window summary")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    learning_curve_output = output_dir / args.learning_curve_output
    final_window_figure_output = output_dir / args.final_window_figure_output
    curve_summary_output = output_dir / args.curve_summary_output
    plateau_summary_output = output_dir / args.plateau_summary_output
    duplicate_summary_output = output_dir / args.duplicate_summary_output

    runs, duplicate_rows = load_runs(input_dir, args.algorithms, args.num_trials)
    if not runs:
        raise FileNotFoundError(f"No valid survival files found in {input_dir}")

    summary_rows = compute_curve_summary(runs, args.algorithms, args.num_trials)
    plateau_rows = compute_plateau_table(
        runs=runs,
        summary_rows=summary_rows,
        algorithms=args.algorithms,
        num_trials=args.num_trials,
        window=args.window,
        late_slope_threshold=args.late_slope_threshold,
    )

    plot_learning_curves(summary_rows, args.algorithms, learning_curve_output)
    plot_final_window_summary(plateau_rows, final_window_figure_output)

    write_csv(summary_rows, curve_summary_output)
    write_csv(plateau_rows, plateau_summary_output)
    write_csv(duplicate_rows, duplicate_summary_output)

    print(f"Loaded {len(runs)} deduplicated runs from {input_dir}")
    print(f"Saved learning curves to {learning_curve_output}")
    print(f"Saved final-window summary to {final_window_figure_output}")
    print(f"Saved plateau table to {plateau_summary_output}")
    if duplicate_rows:
        unique_duplicate_pairs = {(row['algorithm'], row['seed']) for row in duplicate_rows}
        print(f"Detected duplicates for {len(unique_duplicate_pairs)} algorithm/seed pairs")


if __name__ == "__main__":
    main()

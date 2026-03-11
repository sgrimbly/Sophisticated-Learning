from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_LABELS = {
    "SI": "SI (no novelty)",
    "SI_novelty": "SI + novelty",
    "SI_novelty_smooth": "SI + novelty + smoothed novelty estimate",
    "SL_noSmooth": "SL without planning-time backward smoothing",
    "SL": "SL",
}

DEFAULT_ORDER = [
    "SI",
    "SI_novelty",
    "SI_novelty_smooth",
    "SL_noSmooth",
    "SL",
]

COLORS = {
    "SI": "#4c566a",
    "SI_novelty": "#2a9d8f",
    "SI_novelty_smooth": "#1d3557",
    "SL_noSmooth": "#e76f51",
    "SL": "#d62828",
}

REQUIRED_COLUMNS = {
    "config_id",
    "algorithm",
    "seed",
    "trial",
    "real_step",
    "planning_novelty_term_mean",
    "param_update_kl_step",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reviewer-facing novelty traces over real time.")
    parser.add_argument("inputs", nargs="+", help="Step-metrics CSV files, directories, or glob patterns.")
    parser.add_argument("--output", type=Path, required=True, help="Output figure path.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional CSV summary path. Defaults next to the figure.",
    )
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ORDER, help="Algorithm plotting order.")
    parser.add_argument("--band", choices=("sem", "95ci"), default="95ci", help="Error band type.")
    return parser.parse_args()


def resolve_inputs(raw_inputs: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in raw_inputs:
        path = Path(raw)
        if any(token in raw for token in ("*", "?", "[")):
            resolved.extend(sorted(Path().glob(raw)))
        elif path.is_dir():
            resolved.extend(sorted(path.rglob("*_step_metrics.csv")))
        elif path.is_file():
            resolved.append(path)
        else:
            resolved.extend(sorted(Path().glob(raw)))

    unique_paths = sorted({p.resolve() for p in resolved if p.suffix.lower() == ".csv"})
    if not unique_paths:
        raise FileNotFoundError("No step-metrics CSV files matched the provided inputs.")
    return unique_paths


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_step_metrics(paths: list[Path]) -> list[dict[str, object]]:
    deduped: dict[tuple[str, str, int, int, int], dict[str, object]] = {}
    for path in paths:
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

            for row in reader:
                key = (
                    row["config_id"],
                    row["algorithm"],
                    int(row["seed"]),
                    int(row["trial"]),
                    int(row["real_step"]),
                )
                deduped[key] = {
                    "config_id": row["config_id"],
                    "algorithm": row["algorithm"],
                    "seed": int(row["seed"]),
                    "trial": int(row["trial"]),
                    "real_step": int(row["real_step"]),
                    "planning_novelty_term_mean": parse_float(row["planning_novelty_term_mean"]),
                    "param_update_kl_step": parse_float(row["param_update_kl_step"]),
                }

    return list(deduped.values())


def summarise_metric(rows: list[dict[str, object]], algorithms: list[str], value_key: str, band: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        algorithm = str(row["algorithm"])
        if algorithm not in algorithms:
            continue
        value = float(row[value_key])
        if np.isnan(value):
            continue
        grouped[(algorithm, int(row["real_step"]))].append(value)

    multiplier = 1.96 if band == "95ci" else 1.0
    summary_rows: list[dict[str, object]] = []
    for algorithm in algorithms:
        steps = sorted(step for algo, step in grouped if algo == algorithm)
        for step in steps:
            values = np.array(grouped[(algorithm, step)], dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            sem = std / np.sqrt(len(values)) if len(values) > 0 else float("nan")
            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "real_step": step,
                    "metric": value_key,
                    "mean": mean,
                    "std": std,
                    "count": len(values),
                    "sem": sem,
                    "band_halfwidth": sem * multiplier,
                }
            )
    return summary_rows


def plot_metric(ax: plt.Axes, summary_rows: list[dict[str, object]], algorithms: list[str], ylabel: str) -> None:
    for algorithm in algorithms:
        algo_rows = sorted(
            (row for row in summary_rows if row["algorithm"] == algorithm),
            key=lambda row: int(row["real_step"]),
        )
        if not algo_rows:
            continue

        x = np.array([int(row["real_step"]) for row in algo_rows], dtype=int)
        y = np.array([float(row["mean"]) for row in algo_rows], dtype=float)
        band = np.array([float(row["band_halfwidth"]) for row in algo_rows], dtype=float)
        color = COLORS.get(algorithm, None)
        ax.plot(x, y, label=DISPLAY_LABELS.get(algorithm, algorithm), linewidth=2, color=color)
        ax.fill_between(x, y - band, y + band, alpha=0.18, color=color)

    ax.set_xlabel("Real Step")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_summary(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = ["algorithm", "real_step", "metric", "mean", "std", "count", "sem", "band_halfwidth"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    input_paths = resolve_inputs(args.inputs)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else output_path.with_name(f"{output_path.stem}_summary.csv")
    )

    rows = load_step_metrics(input_paths)
    present_algorithms = {str(row["algorithm"]) for row in rows}
    algorithms = [algorithm for algorithm in args.algorithms if algorithm in present_algorithms]
    if not algorithms:
        raise ValueError("None of the requested algorithms were found in the supplied step metrics.")

    novelty_summary = summarise_metric(rows, algorithms, "planning_novelty_term_mean", args.band)
    kl_summary = summarise_metric(rows, algorithms, "param_update_kl_step", args.band)
    has_kl_panel = any(row["count"] > 0 for row in kl_summary)

    nrows = 2 if has_kl_panel else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(9, 4 + 2.8 * (nrows - 1)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    plot_metric(axes[0], novelty_summary, algorithms, "Mean Planning Novelty per Node")
    axes[0].set_title("Reviewer novelty traces over real steps")
    axes[0].legend(frameon=False)

    if has_kl_panel:
        plot_metric(axes[1], kl_summary, algorithms, "Mean Parameter-Update KL per Step")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_summary(novelty_summary + kl_summary, summary_output)

    print(f"Loaded {len(input_paths)} step-metrics files")
    print(f"Rows after deduplication: {len(rows)}")
    print(f"Saved figure to {output_path}")
    print(f"Saved summary to {summary_output}")


if __name__ == "__main__":
    main()

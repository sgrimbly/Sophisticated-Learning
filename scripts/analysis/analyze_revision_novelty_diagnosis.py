from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ALGORITHMS = [
    "SI",
    "SI_noNovelty",
    "SI_novelty_smooth",
    "SI_smooth_noNovelty",
    "SL_adaptivePlan",
    "SL_noNovelty_adaptivePlan",
    "SL_noSmooth_adaptivePlan",
    "SL_noNovelty_noSmooth_adaptivePlan",
]

DISPLAY_LABELS = {
    "SI": "SI",
    "SI_noNovelty": "SI no novelty",
    "SI_novelty_smooth": "SI novelty smooth",
    "SI_smooth_noNovelty": "SI smooth no novelty",
    "SL_adaptivePlan": "SL adaptive",
    "SL_noNovelty_adaptivePlan": "SL no novelty adaptive",
    "SL_noSmooth_adaptivePlan": "SL no smooth adaptive",
    "SL_noNovelty_noSmooth_adaptivePlan": "SL no novelty no smooth adaptive",
}

COLORS = {
    "SI": "#005f73",
    "SI_noNovelty": "#0a9396",
    "SI_novelty_smooth": "#94d2bd",
    "SI_smooth_noNovelty": "#e9d8a6",
    "SL_adaptivePlan": "#bb3e03",
    "SL_noNovelty_adaptivePlan": "#ca6702",
    "SL_noSmooth_adaptivePlan": "#ee9b00",
    "SL_noNovelty_noSmooth_adaptivePlan": "#ae2012",
}

NOVELTY_PAIRS = [
    ("SI", "SI_noNovelty"),
    ("SI_novelty_smooth", "SI_smooth_noNovelty"),
    ("SL_adaptivePlan", "SL_noNovelty_adaptivePlan"),
    ("SL_noSmooth_adaptivePlan", "SL_noNovelty_noSmooth_adaptivePlan"),
]


@dataclass
class RunSummary:
    algorithm: str
    seed: int
    is_complete: bool
    num_rows: int
    final_survival: float | None
    auc_mean: float | None
    frame: pd.DataFrame | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the clean novelty diagnosis batch.")
    parser.add_argument("results_root", type=Path, help="Results root containing per-algorithm folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and summary files.")
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS, help="Algorithms to summarize.")
    parser.add_argument("--seed-start", type=int, default=1, help="First expected seed.")
    parser.add_argument("--seed-end", type=int, default=200, help="Last expected seed.")
    parser.add_argument("--num-trials", type=int, default=120, help="Expected complete trial count.")
    return parser.parse_args()


def discover_runs(results_root: Path, algorithm: str) -> dict[int, Path]:
    pattern = re.compile(rf"^{re.escape(algorithm)}_Seed(\d+)_metrics\.csv$")
    discovered: dict[int, Path] = {}
    algorithm_dir = results_root / algorithm
    if not algorithm_dir.exists():
        return discovered
    for path in sorted(algorithm_dir.glob(f"{algorithm}_Seed*_metrics.csv")):
        match = pattern.match(path.name)
        if match:
            discovered[int(match.group(1))] = path
    return discovered


def load_run(algorithm: str, seed: int, path: Path, num_trials: int) -> RunSummary:
    frame = pd.read_csv(path)
    if "trial" not in frame.columns or "survival" not in frame.columns:
        raise ValueError(f"{path} is missing required columns.")

    frame = frame.sort_values("trial").drop_duplicates(subset=["trial"], keep="last")
    is_complete = len(frame.index) >= num_trials and int(frame["trial"].max()) >= num_trials
    final_survival = None
    auc_mean = None
    if is_complete:
        complete_frame = frame[frame["trial"] <= num_trials].copy()
        complete_frame = complete_frame.sort_values("trial")
        final_row = complete_frame[complete_frame["trial"] == num_trials]
        if final_row.empty:
            final_row = complete_frame.tail(1)
        final_survival = float(final_row["survival"].iloc[-1])
        auc_mean = float(complete_frame["survival"].mean())
        frame = complete_frame
    return RunSummary(
        algorithm=algorithm,
        seed=seed,
        is_complete=is_complete,
        num_rows=int(len(frame.index)),
        final_survival=final_survival,
        auc_mean=auc_mean,
        frame=frame if is_complete else None,
    )


def build_algorithm_summary(algorithm: str, run_map: dict[int, RunSummary], seed_start: int, seed_end: int) -> dict[str, float | int | str]:
    complete_runs = [run for run in run_map.values() if run.is_complete]
    partial_runs = [run for run in run_map.values() if not run.is_complete]
    expected_seed_count = seed_end - seed_start + 1
    missing_runs = expected_seed_count - len(run_map)

    final_values = [run.final_survival for run in complete_runs if run.final_survival is not None]
    auc_values = [run.auc_mean for run in complete_runs if run.auc_mean is not None]

    return {
        "algorithm": algorithm,
        "complete_runs": len(complete_runs),
        "partial_runs": len(partial_runs),
        "missing_runs": missing_runs,
        "final_mean": float(pd.Series(final_values).mean()) if final_values else math.nan,
        "final_sd": float(pd.Series(final_values).std(ddof=1)) if len(final_values) > 1 else math.nan,
        "auc_mean": float(pd.Series(auc_values).mean()) if auc_values else math.nan,
        "auc_sd": float(pd.Series(auc_values).std(ddof=1)) if len(auc_values) > 1 else math.nan,
        "max_complete_seed": max((run.seed for run in complete_runs), default=0),
    }


def build_trial_means(run_summaries: dict[str, dict[int, RunSummary]], algorithms: list[str], common_seeds: set[int] | None = None) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for algorithm in algorithms:
        frames = []
        for seed, run in run_summaries[algorithm].items():
            if not run.is_complete or run.frame is None:
                continue
            if common_seeds is not None and seed not in common_seeds:
                continue
            frame = run.frame[["trial", "survival"]].copy()
            frame["seed"] = seed
            frames.append(frame)
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        grouped = merged.groupby("trial")["survival"]
        summary = grouped.agg(["mean", "std", "count"]).reset_index()
        summary["sem"] = summary["std"].fillna(0.0) / summary["count"].pow(0.5)
        summary["algorithm"] = algorithm
        rows.extend(summary.to_dict("records"))
    return pd.DataFrame(rows)


def build_novelty_delta_table(run_summaries: dict[str, dict[int, RunSummary]]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for novelty_on, novelty_off in NOVELTY_PAIRS:
        if novelty_on not in run_summaries or novelty_off not in run_summaries:
            continue
        on_complete = {seed for seed, run in run_summaries[novelty_on].items() if run.is_complete}
        off_complete = {seed for seed, run in run_summaries[novelty_off].items() if run.is_complete}
        paired_seeds = sorted(on_complete & off_complete)
        if not paired_seeds:
            continue

        on_final = []
        off_final = []
        on_auc = []
        off_auc = []
        for seed in paired_seeds:
            on_run = run_summaries[novelty_on][seed]
            off_run = run_summaries[novelty_off][seed]
            on_final.append(on_run.final_survival)
            off_final.append(off_run.final_survival)
            on_auc.append(on_run.auc_mean)
            off_auc.append(off_run.auc_mean)

        rows.append(
            {
                "novelty_on": novelty_on,
                "novelty_off": novelty_off,
                "n_common": len(paired_seeds),
                "final_mean_delta": float(pd.Series(on_final).mean() - pd.Series(off_final).mean()),
                "auc_mean_delta": float(pd.Series(on_auc).mean() - pd.Series(off_auc).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_exact_equivalence_table(run_summaries: dict[str, dict[int, RunSummary]], algorithms: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for left, right in combinations(algorithms, 2):
        left_complete = {seed for seed, run in run_summaries[left].items() if run.is_complete}
        right_complete = {seed for seed, run in run_summaries[right].items() if run.is_complete}
        paired_seeds = sorted(left_complete & right_complete)
        if not paired_seeds:
            continue

        all_identical = True
        for seed in paired_seeds:
            left_frame = run_summaries[left][seed].frame
            right_frame = run_summaries[right][seed].frame
            if left_frame is None or right_frame is None:
                all_identical = False
                break
            merged = left_frame.merge(right_frame, on="trial", suffixes=("_left", "_right"))
            if merged.empty or not (merged["survival_left"] == merged["survival_right"]).all():
                all_identical = False
                break

        if all_identical:
            rows.append(
                {
                    "left_algorithm": left,
                    "right_algorithm": right,
                    "n_common": len(paired_seeds),
                    "status": "identical_survival_trajectories",
                }
            )
    return pd.DataFrame(rows)


def plot_trial_means(frame: pd.DataFrame, algorithms: list[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for algorithm in algorithms:
        alg_frame = frame[frame["algorithm"] == algorithm].sort_values("trial")
        if alg_frame.empty:
            continue
        x = alg_frame["trial"]
        y = alg_frame["mean"]
        band = 1.96 * alg_frame["sem"].fillna(0.0)
        color = COLORS.get(algorithm)
        ax.plot(x, y, linewidth=2, label=DISPLAY_LABELS.get(algorithm, algorithm), color=color)
        ax.fill_between(x, y - band, y + band, alpha=0.18, color=color)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean survival")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_novelty_deltas(run_summaries: dict[str, dict[int, RunSummary]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for novelty_on, novelty_off in NOVELTY_PAIRS:
        if novelty_on not in run_summaries or novelty_off not in run_summaries:
            continue
        on_complete = {seed for seed, run in run_summaries[novelty_on].items() if run.is_complete}
        off_complete = {seed for seed, run in run_summaries[novelty_off].items() if run.is_complete}
        paired_seeds = sorted(on_complete & off_complete)
        if not paired_seeds:
            continue

        delta_rows = []
        for seed in paired_seeds:
            on_frame = run_summaries[novelty_on][seed].frame
            off_frame = run_summaries[novelty_off][seed].frame
            if on_frame is None or off_frame is None:
                continue
            merged = on_frame.merge(off_frame, on="trial", suffixes=("_on", "_off"))
            merged["delta"] = merged["survival_on"] - merged["survival_off"]
            delta_rows.append(merged[["trial", "delta"]])

        if not delta_rows:
            continue

        delta_frame = pd.concat(delta_rows, ignore_index=True)
        summary = delta_frame.groupby("trial")["delta"].mean().reset_index()
        ax.plot(summary["trial"], summary["delta"], linewidth=2, label=f"{DISPLAY_LABELS[novelty_on]} - {DISPLAY_LABELS[novelty_off]}")
        plotted = True

    ax.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean survival delta")
    ax.set_title("Novelty-on minus novelty-off across paired seeds")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if plotted:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_text_summary(
    output_path: Path,
    algorithm_summary: pd.DataFrame,
    common_seed_count: int,
    novelty_delta_frame: pd.DataFrame,
    exact_equivalence_frame: pd.DataFrame,
    results_root: Path,
) -> None:
    lines = [
        f"Results root: {results_root}",
        "",
        "Algorithm summary",
    ]
    for row in algorithm_summary.to_dict("records"):
        lines.append(
            (
                f"{row['algorithm']}: complete={row['complete_runs']}, partial={row['partial_runs']}, "
                f"missing={row['missing_runs']}, final_mean={row['final_mean']:.2f}, auc_mean={row['auc_mean']:.2f}, "
                f"max_complete_seed={row['max_complete_seed']}"
            )
        )

    lines.extend(
        [
            "",
            f"Common complete seeds across all algorithms: {common_seed_count}",
            "",
            "Within-family novelty deltas",
        ]
    )

    if novelty_delta_frame.empty:
        lines.append("No paired novelty comparisons were available.")
    else:
        for row in novelty_delta_frame.to_dict("records"):
            lines.append(
                (
                    f"{row['novelty_on']} vs {row['novelty_off']}: n_common={row['n_common']}, "
                    f"final_mean_delta={row['final_mean_delta']:.2f}, auc_mean_delta={row['auc_mean_delta']:.2f}"
                )
            )

    lines.extend(
        [
            "",
            "Exact equivalence audit",
        ]
    )
    if exact_equivalence_frame.empty:
        lines.append("No exact trajectory-equivalence pairs were detected among the compared algorithms.")
    else:
        for row in exact_equivalence_frame.to_dict("records"):
            lines.append(
                f"{row['left_algorithm']} == {row['right_algorithm']}: n_common={row['n_common']}, status={row['status']}"
            )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    stats_dir = output_dir / "stats"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: dict[str, dict[int, RunSummary]] = {}
    algorithm_summary_rows: list[dict[str, float | int | str]] = []

    for algorithm in args.algorithms:
        discovered = discover_runs(results_root, algorithm)
        run_map = {
            seed: load_run(algorithm, seed, path, args.num_trials)
            for seed, path in discovered.items()
            if args.seed_start <= seed <= args.seed_end
        }
        run_summaries[algorithm] = run_map
        algorithm_summary_rows.append(build_algorithm_summary(algorithm, run_map, args.seed_start, args.seed_end))

    algorithm_summary = pd.DataFrame(algorithm_summary_rows)
    common_complete_seeds = None
    for algorithm in args.algorithms:
        complete_seeds = {seed for seed, run in run_summaries[algorithm].items() if run.is_complete}
        if common_complete_seeds is None:
            common_complete_seeds = complete_seeds
        else:
            common_complete_seeds &= complete_seeds
    common_complete_seeds = common_complete_seeds or set()

    all_complete_means = build_trial_means(run_summaries, args.algorithms)
    paired_means = build_trial_means(run_summaries, args.algorithms, common_complete_seeds)
    novelty_delta_frame = build_novelty_delta_table(run_summaries)
    exact_equivalence_frame = build_exact_equivalence_table(run_summaries, args.algorithms)

    algorithm_summary.to_csv(data_dir / "algorithm_summary.csv", index=False)
    all_complete_means.to_csv(data_dir / "trial_means_all_complete.csv", index=False)
    paired_means.to_csv(data_dir / "trial_means_paired.csv", index=False)
    novelty_delta_frame.to_csv(data_dir / "novelty_deltas.csv", index=False)
    exact_equivalence_frame.to_csv(data_dir / "exact_equivalences.csv", index=False)

    if not all_complete_means.empty:
        plot_trial_means(all_complete_means, args.algorithms, plots_dir / "trial_means_all_complete.png", "All complete runs")
    if not paired_means.empty:
        plot_trial_means(paired_means, args.algorithms, plots_dir / "trial_means_paired.png", f"Paired complete seeds (n={len(common_complete_seeds)})")
    plot_novelty_deltas(run_summaries, plots_dir / "novelty_deltas.png")

    summary_txt = stats_dir / "summary.txt"
    write_text_summary(summary_txt, algorithm_summary, len(common_complete_seeds), novelty_delta_frame, exact_equivalence_frame, results_root)

    summary_json = {
        "results_root": str(results_root),
        "algorithms": args.algorithms,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "num_trials": args.num_trials,
        "common_complete_seed_count": len(common_complete_seeds),
        "algorithm_summary": algorithm_summary.to_dict("records"),
        "novelty_deltas": novelty_delta_frame.to_dict("records"),
        "exact_equivalences": exact_equivalence_frame.to_dict("records"),
    }
    (stats_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()

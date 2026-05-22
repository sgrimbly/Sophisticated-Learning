"""Per-trial-bin diagnostic for novelty term behaviour during planning.

Aggregates step_metrics.csv outputs across seeds and produces per-trial
summaries answering three questions:

  H3 (does novelty decay?):       planning_novelty_term_mean vs trial index
  H1 indirect (mis-scaling):      novelty/extrinsic magnitude ratio vs trial
  Sanity (imagined vs real):      planning_novelty_term_mean vs
                                  param_update_kl_step (actual KL after
                                  Dirichlet update from real observation)

Reads files matching ``{algorithm}_Seed{N}_step_metrics.csv`` under
``--input-dir`` (recursively).
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FILE_PATTERN = re.compile(r"^([A-Za-z][A-Za-z_]*?)_Seed(\d+)_step_metrics\.csv$")


def _safe_float(x: str) -> float:
    if x is None or x == "" or x.lower() == "nan":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def load_per_seed_per_trial(
    input_dir: Path, algorithm: str, num_trials: int
) -> dict[int, dict[int, dict[str, float]]]:
    """Return per_seed_per_trial[seed][trial] -> {metric: per-trial mean}."""
    out: dict[int, dict[int, dict[str, float]]] = {}
    for path in sorted(input_dir.rglob("*.csv")):
        m = FILE_PATTERN.match(path.name)
        if not m:
            continue
        alg, seed_text = m.groups()
        if alg != algorithm:
            continue
        seed = int(seed_text)
        # Group rows by trial; compute per-trial means
        trial_buckets: dict[int, list[dict[str, str]]] = defaultdict(list)
        with open(path) as fh:
            for row in csv.DictReader(fh):
                t = int(row["trial"])
                if 1 <= t <= num_trials:
                    trial_buckets[t].append(row)

        per_trial: dict[int, dict[str, float]] = {}
        for t, rows in trial_buckets.items():
            novs = [_safe_float(r["planning_novelty_term_mean"]) for r in rows]
            exts = [_safe_float(r["planning_extrinsic_term_mean"]) for r in rows]
            kls = [_safe_float(r["param_update_kl_step"]) for r in rows]

            novs = [x for x in novs if not np.isnan(x)]
            exts = [x for x in exts if not np.isnan(x)]
            kls = [x for x in kls if not np.isnan(x)]

            per_trial[t] = {
                "novelty_mean": float(np.mean(novs)) if novs else float("nan"),
                "extrinsic_mean": float(np.mean(exts)) if exts else float("nan"),
                "kl_actual_mean": float(np.mean(kls)) if kls else float("nan"),
            }
        out[seed] = per_trial
    return out


def aggregate_across_seeds(
    per_seed_per_trial: dict[int, dict[int, dict[str, float]]],
    num_trials: int,
) -> list[dict[str, float]]:
    """Return rows: {trial, n_seeds, novelty_mean, novelty_sem, ...}."""
    rows: list[dict[str, float]] = []
    for trial in range(1, num_trials + 1):
        vals_by_metric: dict[str, list[float]] = defaultdict(list)
        for seed_dict in per_seed_per_trial.values():
            if trial not in seed_dict:
                continue
            for metric, v in seed_dict[trial].items():
                if not np.isnan(v):
                    vals_by_metric[metric].append(v)
        if not vals_by_metric.get("novelty_mean"):
            continue

        def mean_sem(xs: list[float]) -> tuple[float, float]:
            arr = np.asarray(xs)
            return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else (float(arr.mean()), 0.0)

        novelty_mean, novelty_sem = mean_sem(vals_by_metric["novelty_mean"])
        extrinsic_mean = float(np.mean(vals_by_metric["extrinsic_mean"])) if vals_by_metric.get("extrinsic_mean") else float("nan")
        kl_actual_mean = float(np.mean(vals_by_metric["kl_actual_mean"])) if vals_by_metric.get("kl_actual_mean") else float("nan")

        rows.append({
            "trial": trial,
            "n_seeds": len(vals_by_metric["novelty_mean"]),
            "novelty_mean": novelty_mean,
            "novelty_sem": novelty_sem,
            "extrinsic_mean": extrinsic_mean,
            "kl_actual_mean": kl_actual_mean,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--algorithms", nargs="+", required=True)
    parser.add_argument("--num-trials", type=int, default=120)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summary: list[dict[str, object]] = []
    for algo in args.algorithms:
        print(f"[novelty_decay] loading {algo}...")
        per_seed_per_trial = load_per_seed_per_trial(args.input_dir, algo, args.num_trials)
        if not per_seed_per_trial:
            print(f"  no step_metrics CSVs for {algo}")
            continue
        rows = aggregate_across_seeds(per_seed_per_trial, args.num_trials)
        print(f"  {len(per_seed_per_trial)} seeds, {len(rows)} per-trial rows")
        for r in rows:
            r["algorithm"] = algo
            all_summary.append(r)

    csv_path = args.output_dir / "novelty_decay_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "algorithm", "trial", "n_seeds",
            "novelty_mean", "novelty_sem",
            "extrinsic_mean", "kl_actual_mean",
        ])
        for r in all_summary:
            writer.writerow([
                r["algorithm"], r["trial"], r["n_seeds"],
                r["novelty_mean"], r["novelty_sem"],
                r["extrinsic_mean"], r["kl_actual_mean"],
            ])
    print(f"[novelty_decay] wrote {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for algo in args.algorithms:
        rows = [r for r in all_summary if r["algorithm"] == algo]
        if not rows:
            continue
        ts = np.asarray([r["trial"] for r in rows])
        novs = np.asarray([r["novelty_mean"] for r in rows])
        exts = np.asarray([r["extrinsic_mean"] for r in rows])
        kls = np.asarray([r["kl_actual_mean"] for r in rows])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(exts) > 1e-9, novs / np.abs(exts), np.nan)

        axes[0, 0].plot(ts, novs, label=algo, linewidth=1.3)
        axes[0, 1].plot(ts, exts, label=algo, linewidth=1.3)
        axes[1, 0].plot(ts, kls, label=algo, linewidth=1.3)
        axes[1, 1].plot(ts, ratio, label=algo, linewidth=1.3)

    axes[0, 0].set_title("Mean novelty term during planning (H3: does it decay?)")
    axes[0, 0].set_ylabel("planning_novelty_term_mean")
    axes[0, 1].set_title("Mean extrinsic term during planning")
    axes[0, 1].set_ylabel("planning_extrinsic_term_mean")
    axes[1, 0].set_title("Mean actual KL update per step (real learning)")
    axes[1, 0].set_ylabel("param_update_kl_step")
    axes[1, 0].set_xlabel("Trial")
    axes[1, 1].set_title("|Novelty / Extrinsic| during planning (H1 indirect)")
    axes[1, 1].set_ylabel("ratio (log scale)")
    axes[1, 1].set_xlabel("Trial")
    axes[1, 1].set_yscale("log")
    for ax in axes.flat:
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = args.output_dir / "novelty_decay_diagnostic.png"
    plt.savefig(png_path, dpi=150)
    print(f"[novelty_decay] wrote {png_path}")


if __name__ == "__main__":
    main()

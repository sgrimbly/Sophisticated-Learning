"""Early-learning + model-saturation figure (SL vs SI x novelty).

Reads the revision diagnosis batch ({algo}/{algo}_Seed{N}_metrics.csv, with
`survival` and `param_update_kl` columns, ~200 seeds/variant) and renders a
three-panel figure aimed at the early-learning question (which converged AUC
hides):

  A. Learning curves (mean survival per trial, +/-95% CI) for the SL/SI x
     novelty 2x2, with the survival>=threshold crossing marked -- shows SL
     learns faster *with* novelty and ties *without*.
  B. Trials-to-threshold (from the across-seed mean curve) per variant -- a
     learning-speed summary.
  C. param_update_kl per trial (log-y) -- shows the model saturates ~25x
     early->late and is ~novelty-invariant, i.e. novelty buys no extra learning.

Usage::

    python plot_early_learning.py \
        --input-dir .../revision_novelty_diagnosis_default_env \
        --output-dir .../early_learning_figure
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# SL/SI x novelty 2x2: colour by family, linestyle by novelty.
KEY = [
    ("SL_adaptivePlan",            "SL (novelty+smooth)", "#d62828", "-"),
    ("SL_noNovelty_adaptivePlan",  "SL (no novelty)",     "#d62828", "--"),
    ("SI_novelty_smooth",          "SI (novelty+smooth)", "#1f5fa8", "-"),
    ("SI_smooth_noNovelty",        "SI (no novelty)",     "#1f5fa8", "--"),
]
THRESHOLD = 40.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output", default="early_learning.png")
    p.add_argument("--num-trials", type=int, default=120)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--smooth-window", type=int, default=1,
                   help="Rolling-mean window for the learning-curve panel only "
                        "(cosmetic; threshold crossings use the raw mean). 1 = off.")
    return p.parse_args()


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average; edges shrink the window. window<=1 is a no-op."""
    if window <= 1:
        return y
    half = window // 2
    out = np.empty_like(y, dtype=np.float64)
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out[i] = np.nanmean(y[lo:hi])
    return out


def load_variant(input_dir: Path, algo: str, num_trials: int):
    """Return (survival_matrix, paramkl_matrix) of shape (n_seeds, num_trials)."""
    surv, pkl = [], []
    for f in glob.glob(str(input_dir / algo / f"{algo}_Seed*_metrics.csv")):
        if "step_metrics" in f:
            continue
        rows = list(csv.DictReader(open(f)))
        s = [float(r["survival"]) for r in rows if r.get("survival") not in (None, "")]
        k = [float(r["param_update_kl"]) for r in rows if r.get("param_update_kl") not in (None, "")]
        if len(s) == num_trials:
            surv.append(s)
            pkl.append(k if len(k) == num_trials else [np.nan] * num_trials)
    return np.array(surv), np.array(pkl)


def mean_ci(matrix: np.ndarray):
    mean = np.nanmean(matrix, axis=0)
    n = np.sum(~np.isnan(matrix), axis=0)
    sd = np.nanstd(matrix, axis=0, ddof=1)
    ci = 1.96 * sd / np.sqrt(np.maximum(n, 1))
    return mean, ci


def trials_to_threshold(mean_curve: np.ndarray, thr: float):
    idx = np.where(mean_curve >= thr)[0]
    return int(idx[0] + 1) if len(idx) else None


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for algo, label, color, ls in KEY:
        surv, pkl = load_variant(args.input_dir.resolve(), algo, args.num_trials)
        if surv.size:
            data[algo] = (surv, pkl, label, color, ls)

    trials = np.arange(1, args.num_trials + 1)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 5.2))

    # --- Panel A: learning curves + threshold crossings ---
    t2t = {}
    sw = args.smooth_window
    for algo, (surv, pkl, label, color, ls) in data.items():
        m, ci = mean_ci(surv)
        tt = trials_to_threshold(m, args.threshold)   # crossing on the RAW mean
        t2t[algo] = tt
        ms, cis = _smooth(m, sw), _smooth(ci, sw)      # smooth only for display
        axA.plot(trials, ms, color=color, linestyle=ls, linewidth=2.2,
                 label=label, zorder=3)
        axA.fill_between(trials, ms - cis, ms + cis, color=color, alpha=0.12, zorder=1)
        if tt:
            axA.plot([tt], [args.threshold], marker="o", color=color, ms=6, zorder=4)
    axA.axhline(args.threshold, color="grey", lw=0.8, ls=":", zorder=0)
    axA.set_xlabel("Trial")
    axA.set_ylabel("Mean survival steps (+/-95% CI)")
    axA.set_title("Learning curves: SL learns faster *with* novelty, ties without")
    axA.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # --- Panel B: trials-to-threshold bar ---
    labels = [data[a][2] for a in data]
    colors = [data[a][3] for a in data]
    vals = [t2t[a] if t2t[a] else 0 for a in data]
    x = np.arange(len(labels))
    axB.bar(x, vals, color=colors, alpha=0.9)
    for xi, v in zip(x, vals):
        axB.text(xi, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axB.set_ylabel(f"Trials to reach mean survival >= {args.threshold:.0f}")
    axB.set_title("Learning speed (lower = faster)")
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    # --- Panel C: param_update_kl per trial (model learning rate) ---
    for algo, (surv, pkl, label, color, ls) in data.items():
        m, _ = mean_ci(pkl)
        axC.plot(trials, m, color=color, linestyle=ls, linewidth=2.0, label=label)
    axC.set_yscale("log")
    axC.set_xlabel("Trial")
    axC.set_ylabel("param_update_kl (model-update magnitude, log)")
    axC.set_title("Model saturates ~25x early->late, ~novelty-invariant")
    axC.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        axC.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out / args.output}")
    print("trials-to-threshold:", {data[a][2]: t2t[a] for a in data})


if __name__ == "__main__":
    main()

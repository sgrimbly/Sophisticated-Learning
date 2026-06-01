"""Novelty x learning weight grid analysis (Rowan's ask #1).

Reads the per-(cell, seed) sweep JSONs produced by ``sl.sweep`` for a grid of
``novelty_weight x learning_weight`` cells (one output dir, grid_id encoded as
``g_n{N}_l{L}``), computes each cell's mean per-seed AUC (= mean trial length
over the run), and produces:

  * a heatmap of mean AUC over the novelty x learning grid, and
  * a novelty-response plot (AUC vs novelty_weight, one line per learning
    weight plus the across-learning mean), to show whether an *intermediate*
    novelty weight beats novelty-off.

Usage::

    python plot_novelty_learning_grid.py \
        --input-dir /scratch/.../rowan_grid_20260601/python_sweep \
        --output-dir /scratch/.../rowan_grid_20260601/figure
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

GRID_ID_RE = re.compile(r"g_n(\d+)_l(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Dir of {algo}_seed{S}_g_n{N}_l{L}.json sweep outputs.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--figure-output", default="novelty_learning_grid.png")
    p.add_argument("--csv-output", default="novelty_learning_grid_summary.csv")
    return p.parse_args()


def load_cells(input_dir: Path) -> dict[tuple[int, int], list[float]]:
    """{(novelty, learning): [per-seed AUC, ...]} where AUC = mean t_terminal."""
    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for f in sorted(glob.glob(str(input_dir / "*.json"))):
        d = json.load(open(f))
        m = GRID_ID_RE.search(d.get("grid_id", ""))
        if not m:
            continue
        novelty, learning = int(m.group(1)), int(m.group(2))
        trials = d.get("trials", [])
        if not trials:
            continue
        auc = float(np.mean([t["t_terminal"] for t in trials]))
        cells[(novelty, learning)].append(auc)
    return cells


def axes_values(cells: dict[tuple[int, int], list[float]]) -> tuple[list[int], list[int]]:
    novelties = sorted({n for n, _ in cells})
    learnings = sorted({l for _, l in cells})
    return novelties, learnings


def mean_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sem


def plot_grid(cells, novelties, learnings, output_path: Path) -> None:
    mean_grid = np.full((len(novelties), len(learnings)), np.nan)
    for i, n in enumerate(novelties):
        for j, l in enumerate(learnings):
            vals = cells.get((n, l), [])
            if vals:
                mean_grid[i, j] = float(np.mean(vals))

    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Heatmap ---
    im = ax_h.imshow(mean_grid, origin="lower", aspect="auto", cmap="viridis")
    ax_h.set_xticks(range(len(learnings)), learnings)
    ax_h.set_yticks(range(len(novelties)), novelties)
    ax_h.set_xlabel("learning weight")
    ax_h.set_ylabel("novelty weight")
    ax_h.set_title("Mean AUC (mean trial length) over the grid")
    for i in range(len(novelties)):
        for j in range(len(learnings)):
            if not np.isnan(mean_grid[i, j]):
                ax_h.text(j, i, f"{mean_grid[i, j]:.1f}", ha="center", va="center",
                          color="white", fontsize=9)
    fig.colorbar(im, ax=ax_h, label="mean AUC")

    # --- Novelty-response lines ---
    cmap = plt.get_cmap("plasma")
    for k, l in enumerate(learnings):
        ys = [np.mean(cells.get((n, l), [np.nan])) for n in novelties]
        ax_l.plot(novelties, ys, marker="o", linewidth=1.4,
                  color=cmap(k / max(1, len(learnings) - 1)),
                  label=f"learning={l}")
    # Across-learning mean +/- SEM (bold).
    agg_mean, agg_sem = [], []
    for n in novelties:
        allv = [a for l in learnings for a in cells.get((n, l), [])]
        m, s = mean_sem(allv)
        agg_mean.append(m)
        agg_sem.append(s)
    agg_mean = np.array(agg_mean)
    agg_sem = np.array(agg_sem)
    ax_l.plot(novelties, agg_mean, color="black", linewidth=3.0, marker="s",
              zorder=5, label="mean across learning")
    ax_l.fill_between(novelties, agg_mean - agg_sem, agg_mean + agg_sem,
                      color="black", alpha=0.15, zorder=4)
    ax_l.set_xlabel("novelty weight")
    ax_l.set_ylabel("mean AUC (mean trial length)")
    ax_l.set_title("Novelty response (SL adaptive) — does some novelty beat none?")
    ax_l.legend(frameon=False, fontsize=8)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(cells, novelties, learnings, output_path: Path) -> None:
    with output_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["novelty_weight", "learning_weight", "n_seeds", "mean_auc", "sem_auc"])
        for n in novelties:
            for l in learnings:
                vals = cells.get((n, l), [])
                mean, sem = mean_sem(vals)
                w.writerow([n, l, len(vals), f"{mean:.4f}", f"{sem:.4f}"])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells(args.input_dir.resolve())
    if not cells:
        raise FileNotFoundError(f"No grid JSONs with g_n*_l* grid_id in {args.input_dir}")
    novelties, learnings = axes_values(cells)
    plot_grid(cells, novelties, learnings, output_dir / args.figure_output)
    write_csv(cells, novelties, learnings, output_dir / args.csv_output)
    n_cells = len(cells)
    n_seeds = sum(len(v) for v in cells.values())
    print(f"Loaded {n_seeds} runs across {n_cells} cells "
          f"({len(novelties)}x{len(learnings)} novelty x learning grid)")
    print(f"Saved figure to {output_dir / args.figure_output}")
    print(f"Saved summary to {output_dir / args.csv_output}")


if __name__ == "__main__":
    main()

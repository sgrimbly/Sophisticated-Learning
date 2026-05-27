"""Load a MATLAB reference trace and run a side-by-side comparison.

Pairs with :file:`scripts/dump_matlab_reference.m`. The MATLAB script saves
a ``.mat`` file with one trial's worth of canonical state. This module:

  * Loads it.
  * Runs the Python port with the same config.
  * Reports per-step deltas in the quantities that *should* match
    statistically (action distribution, posterior shape, learning curve).

Bit-exactness across MATLAB and Python RNG streams is not a goal; the
report is meant for human inspection to confirm shape/scale parity.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from scipy.io import loadmat
except Exception as e:  # pragma: no cover
    raise SystemExit(f"reference_trace requires scipy: {e}")

from .agent import run_experiment
from .config import GridConfig, RunOptions, Weights


def load_reference(path: str) -> Dict[str, Any]:
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    out = {k: v for k, v in raw.items() if not k.startswith("__")}
    return out


def _compare_action_distribution(matlab_actions: np.ndarray, python_actions: np.ndarray) -> dict:
    """Count action frequencies; return histograms and KL divergence between them."""
    matlab_actions = np.asarray(matlab_actions, dtype=int).ravel()
    python_actions = np.asarray(python_actions, dtype=int).ravel()
    # MATLAB actions are 1-based; convert to 0-based.
    if matlab_actions.size and matlab_actions.min() >= 1 and matlab_actions.max() <= 5:
        matlab_actions = matlab_actions - 1
    bins = np.arange(6)  # 0..5 → 5 bins
    h_m, _ = np.histogram(matlab_actions, bins=bins)
    h_p, _ = np.histogram(python_actions, bins=bins)
    p_m = h_m / max(1, h_m.sum())
    p_p = h_p / max(1, h_p.sum())
    eps = 1e-12
    kl = float(np.sum(p_m * np.log((p_m + eps) / (p_p + eps))))
    return {"matlab": h_m.tolist(), "python": h_p.tolist(), "kl_m_to_p": kl}


def _compare_terminal_steps(matlab_t, python_t) -> dict:
    return {"matlab": int(matlab_t), "python": int(python_t),
            "delta": int(python_t) - int(matlab_t)}


def _compare_a_resource(a_matlab: np.ndarray, a_python: np.ndarray) -> dict:
    a_m = np.asarray(a_matlab, dtype=np.float64)
    a_p = np.asarray(a_python, dtype=np.float64)
    if a_m.shape != a_p.shape:
        return {"shape_match": False, "matlab_shape": a_m.shape, "python_shape": a_p.shape}
    diff = a_m - a_p
    return {
        "shape_match": True,
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "matlab_l2": float(np.linalg.norm(a_m.ravel())),
        "python_l2": float(np.linalg.norm(a_p.ravel())),
        "ratio_l2": float(np.linalg.norm(a_p.ravel()) /
                          max(1e-12, np.linalg.norm(a_m.ravel()))),
    }


def compare(reference_path: str) -> dict:
    ref = load_reference(reference_path)
    meta = ref["meta"]
    # squeeze_me + struct_as_record=False yields scalar fields directly.
    seed = int(meta.seed)
    algorithm = str(meta.algorithm)
    num_trials = int(meta.num_trials)
    max_horizon = int(meta.max_horizon)
    grid_size = int(meta.grid_size)

    grid = GridConfig.from_matlab_indices(
        grid_size=grid_size,
        start_position=int(meta.start_position),
        hill_pos=int(meta.hill_pos),
        food_sources=tuple(int(x) for x in np.atleast_1d(meta.food_sources)),
        water_sources=tuple(int(x) for x in np.atleast_1d(meta.water_sources)),
        sleep_sources=tuple(int(x) for x in np.atleast_1d(meta.sleep_sources)),
    )
    weights_arr = np.atleast_1d(meta.weights)
    weights = Weights(
        novelty=float(weights_arr[0]),
        learning=float(weights_arr[1]),
        epistemic=float(weights_arr[2]),
        preference=float(weights_arr[3]),
    )
    options = RunOptions(
        algorithm=algorithm,
        seed=seed,
        num_trials=num_trials,
        max_horizon=max_horizon,
    )

    py_result = run_experiment(grid, options, weights)

    # Per-trial comparison
    per_trial_ref = ref["per_trial"]
    if not isinstance(per_trial_ref, np.ndarray):
        per_trial_ref = np.array([per_trial_ref])

    per_trial_reports = []
    for i in range(min(len(per_trial_ref), len(py_result.trials))):
        ref_tr = per_trial_ref[i]
        py_tr = py_result.trials[i]
        report = {
            "trial": i + 1,
            "actions": _compare_action_distribution(
                ref_tr.chosen_action, np.asarray(py_tr.chosen_actions)
            ),
            "t_terminal": _compare_terminal_steps(ref_tr.t_terminal, py_tr.t_terminal),
            "a_resource": _compare_a_resource(
                np.asarray(ref_tr.a2_snapshots)[-1], py_tr.a_resource_final,
            ),
        }
        per_trial_reports.append(report)

    return {
        "algorithm": algorithm,
        "seed": seed,
        "matlab_survived_count": int(sum(int(t.survived) for t in per_trial_ref)),
        "python_survived_count": int(py_result.survived_count),
        "per_trial": per_trial_reports,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("reference_mat", help="Path to .mat from dump_matlab_reference.m")
    args = p.parse_args(argv)
    out = compare(args.reference_mat)
    import json
    print(json.dumps(out, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

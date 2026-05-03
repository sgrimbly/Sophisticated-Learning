"""Run-time configuration for sl_jax.

Re-uses GridConfig from the canonical sl package so the env layout is
identical. The sl_jax-specific options bundle is separate from sl's
RunOptions to keep the two paths decoupled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

# Reuse the canonical GridConfig.
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
from sl.config import GridConfig  # noqa: E402  re-export


@dataclass(frozen=True)
class JaxOptions:
    """SI/SL planner options for the JAX prototype."""

    algorithm: str = "SI"  # 'SI' or 'SL'
    seed: int = 1
    num_trials: int = 60
    max_steps_per_trial: int = 100

    # Planning horizon. With H=4 we enumerate 5**4 = 625 policies; with
    # H=5 → 3125. The canonical MATLAB port uses 9 (1.95M policies),
    # which is feasible on GPU but is overkill for the proof of concept.
    horizon: int = 4

    # Survival thresholds — MATCH MATLAB / sl package.
    food_threshold: int = 22
    water_threshold: int = 20
    sleep_threshold: int = 25

    # Algorithm flags
    novelty_on: bool = True
    epistemic_on: bool = True
    real_smoothing: bool = False  # SI: False, SL: True (overridden by family)
    smoothing_window: int = 6     # used only when real_smoothing=True

    # EFE term weights (match canonical sl.config defaults).
    w_novelty: float = 10.0
    w_learning: float = 40.0
    w_epistemic: float = 1.0
    preference: float = 10.0  # treated as preference_inverse_precision (MATLAB inverse_precision mode)

    # Dirichlet a-update params (match sl/learning.py).
    learning_proportion: float = 0.3
    learning_scale: float = 0.7
    learning_floor: float = 0.05
    learning_prune_threshold: float = 0.2  # used inside in-tree imagined updates only

    # Initial Dirichlet concentration.
    initial_a_concentration: float = 0.1

    @property
    def family(self) -> str:
        a = self.algorithm.upper()
        if a.startswith("SI"):
            return "SI"
        if a.startswith("SL"):
            return "SL"
        raise ValueError(f"Unsupported algorithm '{self.algorithm}' for sl_jax")

    @property
    def smoothing_on(self) -> bool:
        # Mirror dashboard_run_one's algorithm_spec override:
        # SI family => no smoothing; SL family => smoothing on.
        return self.family == "SL"

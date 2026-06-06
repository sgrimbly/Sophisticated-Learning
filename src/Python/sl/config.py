"""Run-configuration dataclasses.

Single source of truth for thresholds, weights and grid layout. All array
indices stored here are 0-based (Python convention); helpers convert from
the MATLAB 1-based grid_configs.txt format.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence


@dataclass(frozen=True)
class GridConfig:
    """Grid-world layout. All position fields are 0-based.

    The four resource lists are indexed by *context* (latent season). Each
    list has one position per context.
    """

    grid_size: int = 10
    start_position: int = 50  # MATLAB 51 - 1
    hill_pos: int = 54        # MATLAB 55 - 1
    food_sources: tuple = (70, 42, 56, 77)   # MATLAB [71,43,57,78] - 1
    water_sources: tuple = (72, 32, 47, 66)
    sleep_sources: tuple = (63, 43, 48, 58)
    grid_id: str = ""

    @property
    def num_states(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def num_contexts(self) -> int:
        return 4

    @property
    def num_joint_states(self) -> int:
        return self.num_states * self.num_contexts

    @classmethod
    def from_matlab_indices(
        cls,
        grid_size: int = 10,
        start_position: int = 51,
        hill_pos: int = 55,
        food_sources: Sequence[int] = (71, 43, 57, 78),
        water_sources: Sequence[int] = (73, 33, 48, 67),
        sleep_sources: Sequence[int] = (64, 44, 49, 59),
        grid_id: str = "",
    ) -> "GridConfig":
        return cls(
            grid_size=grid_size,
            start_position=start_position - 1,
            hill_pos=hill_pos - 1,
            food_sources=tuple(p - 1 for p in food_sources),
            water_sources=tuple(p - 1 for p in water_sources),
            sleep_sources=tuple(p - 1 for p in sleep_sources),
            grid_id=grid_id,
        )


@dataclass(frozen=True)
class Weights:
    """EFE term scales.

    Mirrors MATLAB ``weights = [novelty, learning, epistemic, preference]``.

    ``preference_param`` selects how ``preference`` is interpreted:
      * ``'inverse_precision'`` (MATLAB runner default): ``preference`` IS the
        preference inverse-precision used to divide ``C`` in the planner —
        larger ``preference`` ⇒ weaker extrinsic signal.
      * ``'weight'`` (legacy): ``preference`` is a weight, so the inverse
        precision is ``1/preference`` — larger ``preference`` ⇒ stronger
        extrinsic signal.
    """

    novelty: float = 10.0
    learning: float = 40.0
    epistemic: float = 1.0
    preference: float = 10.0
    preference_param: str = "inverse_precision"
    ucb_scale: float = 5.0  # only used by BAUCB

    @property
    def preference_inverse_precision(self) -> float:
        if self.preference_param == "inverse_precision":
            # ``preference`` IS the inverse precision (matches MATLAB
            # ``preference_inverse_precision = preference_value``). 0 ⇒ ∞.
            if self.preference == 0:
                return float("inf")
            return float(self.preference)
        if self.preference_param == "weight":
            if self.preference == 0:
                return float("inf")
            return 1.0 / self.preference
        raise ValueError(
            f"Unknown preference_param={self.preference_param!r}. "
            "Expected 'inverse_precision' or 'weight'."
        )


@dataclass(frozen=True)
class RunOptions:
    """Algorithm flags and survival/horizon thresholds.

    Defaults match the MATLAB canonical loop. Variants flip individual
    flags (real_smoothing, adaptive_likelihood_in_plan, etc.).
    """

    algorithm: str = "SL"
    seed: int = 1
    num_trials: int = 120
    max_horizon: int = 9
    max_steps_per_trial: int = 100  # MATLAB ``while t < 100``

    # Survival thresholds. Agent dies when time_since >= threshold.
    food_threshold: int = 22
    water_threshold: int = 20
    sleep_threshold: int = 25

    real_smoothing: bool = True
    adaptive_likelihood_in_plan: bool = False
    learning_prune_threshold: float = 0.2
    state_selection: str = "sample"  # 'sample' or 'map'
    rng_algorithm: str = "twister"  # informational only; we use np.random.default_rng

    # BAUCB-specific
    baucb_variant: str = "legacy"

    # Performance: opt-in fully-JIT'd tree-search planner (requires numba).
    # When True, the agent dispatches to planning.si_jit / planning.sl_jit
    # instead of the NumPy planners. Falls back silently to NumPy if numba
    # is unavailable.
    use_jit_planner: bool = False

    # Diagnostic (off by default): for the SL family, at each real planning step
    # also run the planner with adaptive_likelihood_in_plan toggled (on the same
    # state, fresh STM copy) and record both chosen actions. Used to test whether
    # SL's roll-forward actually changes decisions vs SI-style frozen planning.
    diagnose_plan_divergence: bool = False

    def is_alive(self, t: int, t_food: int, t_water: int, t_sleep: int) -> bool:
        return (
            t < self.max_steps_per_trial
            and t_food < self.food_threshold
            and t_water < self.water_threshold
            and t_sleep < self.sleep_threshold
        )

    def horizon(self, t_food: int, t_water: int, t_sleep: int) -> int:
        h = min(
            self.max_horizon,
            self.food_threshold - t_food,
            self.water_threshold - t_water,
            self.sleep_threshold - t_sleep,
        )
        if h <= 0:
            h = 1
        return h


# Algorithm name -> variant flag map. Labels and (novelty, smoothing,
# adaptive_plan) semantics mirror MATLAB ``resolve_algorithm_spec.m`` 1:1 so
# the two implementations can be paired by name. ``SI_smooth`` is
# novelty-OFF (identical to ``SI_smooth_noNovelty``), matching the MATLAB
# definition; use ``SI_novelty_smooth`` for the novelty-ON smoothing variant.
ALGORITHM_VARIANTS = {
    # SI family
    "SI":               dict(family="SI", novelty=True,  smoothing=False, adaptive_plan=False),
    "SI_noNovelty":     dict(family="SI", novelty=False, smoothing=False, adaptive_plan=False),
    "SI_smooth":        dict(family="SI", novelty=False, smoothing=True,  adaptive_plan=False),
    "SI_novelty_smooth": dict(family="SI", novelty=True, smoothing=True,  adaptive_plan=False),
    "SI_smooth_noNovelty": dict(family="SI", novelty=False, smoothing=True, adaptive_plan=False),
    # SL family
    "SL":               dict(family="SL", novelty=True,  smoothing=True,  adaptive_plan=False),
    "SL_noNovelty":     dict(family="SL", novelty=False, smoothing=True,  adaptive_plan=False),
    "SL_noSmooth":      dict(family="SL", novelty=True,  smoothing=False, adaptive_plan=False),
    "SL_noNovelty_noSmooth": dict(family="SL", novelty=False, smoothing=False, adaptive_plan=False),
    "SL_adaptivePlan":  dict(family="SL", novelty=True,  smoothing=True,  adaptive_plan=True),
    "SL_noNovelty_adaptivePlan":  dict(family="SL", novelty=False, smoothing=True,  adaptive_plan=True),
    "SL_noSmooth_adaptivePlan":   dict(family="SL", novelty=True,  smoothing=False, adaptive_plan=True),
    "SL_noNovelty_noSmooth_adaptivePlan": dict(family="SL", novelty=False, smoothing=False, adaptive_plan=True),
    # Baselines
    "BA":               dict(family="BA"),
    "BAUCB":            dict(family="BAUCB"),
}


def resolve_algorithm(name: str) -> dict:
    if name not in ALGORITHM_VARIANTS:
        raise ValueError(f"Unknown algorithm '{name}'. Known: {sorted(ALGORITHM_VARIANTS)}")
    return dict(ALGORITHM_VARIANTS[name])

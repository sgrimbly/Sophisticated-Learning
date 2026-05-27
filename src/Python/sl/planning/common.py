"""Shared planner data structures and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PlannerInputs:
    """All read-mostly state passed into a tree-search call.

    Mutable items (short_term_memory, Q histories) are explicitly passed
    *separately* so the planner can update them in place where needed,
    matching MATLAB's pass-by-reference semantics for output cell arrays.
    """
    A_pos: np.ndarray
    A_resource: np.ndarray
    A_hill: np.ndarray
    a_resource: np.ndarray  # MATLAB a{2}; the only learned likelihood
    y_pos: np.ndarray       # = A_pos (known)
    y_resource: np.ndarray  # = normalise_matrix_columns(a_resource)
    y_hill: np.ndarray      # = A_hill (known)
    B_pos: np.ndarray       # (S, S, 5) deterministic-ish
    bb_ctx: np.ndarray      # (C, C, 5) column-normalised; only [:,:,0] is used
    weights: "Weights"  # noqa: F821 — forward ref to sl.config.Weights


@dataclass
class PlanResult:
    G: float
    best_actions: List[int]
    memory_hits: int = 0
    memory_misses: int = 0
    node_count: int = 0


def index_clip(value: int, lo: int = 0, hi: int = 34) -> int:
    """Clamp need-timer to [lo, hi]. Memory tensor is 35 deep (0..34)."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def likely_state_indices(qs: np.ndarray, threshold: float = 0.125) -> np.ndarray:
    """Return joint-state indices whose probability mass exceeds ``threshold``.

    Falls back to MATLAB's tiny-jitter rule when the strict threshold yields
    an empty set (lines 132-136 of tree_search_frwd_SI.m):
        threshold = 1/N**2; likely = find(qs > 1/N - threshold)
    """
    likely = np.where(qs > threshold)[0]
    if likely.size == 0:
        n = qs.size
        eps = 1.0 / (n * n)
        likely = np.where(qs > (1.0 / n - eps))[0]
    return likely


def imagined_observations(
    y_pos: np.ndarray,
    y_resource: np.ndarray,
    y_hill: np.ndarray,
    joint_state: int,
    num_states: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute imagined ``O`` distributions at a fictive next state.

    Mirrors ``O{modal, t+1} = normalise(y{modal}(:, state)')`` for each
    modality, with ``state`` as a column-major linear index into
    (num_states, num_contexts).
    """
    pos = joint_state % num_states
    ctx = joint_state // num_states
    from ..inference import normalise

    return (
        normalise(y_pos[:, pos, ctx]),
        normalise(y_resource[:, pos, ctx]),
        normalise(y_hill[:, pos, ctx]),
    )

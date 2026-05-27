"""Dirichlet learning update for ``a[2]`` (resource-cue likelihood).

Port of the inline smoothing+learning block in SI_modular.m (lines 350-426)
and the per-step learning update in tree_search_frwd_SL.m (lines 65-89).

The two contexts where this is invoked:

1. Real-trial learning (in :mod:`sl.agent`): given the smoothed *real*
   posterior over context at each ``timey`` in [t-6, t], update ``a[2]``
   with the row-0 penalty and the 0.7-scale + 0.05-floor.

2. SL planning-time learning (in :mod:`sl.planning.sl`): given the
   *imagined* smoothed posterior, build ``a_learning``, optionally apply
   the prune threshold, then add to imagined ``a[2]`` (no scale, no floor).

Both share the same cross-product builder; the wrappers below select the
right post-processing for each call site.
"""
from __future__ import annotations

import numpy as np

from .inference import spm_cross


def build_a_learning(
    O_resource_at_timey: np.ndarray,
    P_pos_at_timey: np.ndarray,
    P_ctx_smoothed: np.ndarray,
    a_resource_mask: np.ndarray,
) -> np.ndarray:
    """Construct the unnormalised Dirichlet update tensor.

    Parameters
    ----------
    O_resource_at_timey : (4,)   resource observation at timey (one-hot in real trials)
    P_pos_at_timey      : (S,)   position posterior at timey
    P_ctx_smoothed      : (C,)   smoothed context posterior at timey
    a_resource_mask     : (4, S, C) boolean / float — gates positions where
                                    a_resource > 0.

    Returns
    -------
    a_learning : (4, S, C)   the cross-product times the mask.
    """
    L = spm_cross(np.asarray(O_resource_at_timey, dtype=np.float64).ravel(),
                  np.asarray(P_pos_at_timey,      dtype=np.float64).ravel(),
                  np.asarray(P_ctx_smoothed,      dtype=np.float64).ravel())
    return L * (np.asarray(a_resource_mask, dtype=np.float64) > 0)


def apply_row0_penalty(a_learning: np.ndarray, proportion: float = 0.3) -> np.ndarray:
    """Apply the MATLAB row-0 (empty) penalty (lines 380-388 of SI_modular).

    For each (state, context) where the *empty* row of ``a_learning`` is
    zero, subtract ``proportion * max(a_learning[1:, j, i])`` from row 0
    *only* — matching MATLAB exactly. The Python predecessor mistakenly
    subtracted from all rows.
    """
    a_learning = a_learning.copy()
    # max over outcomes 1..end, broadcast (S, C)
    max_above_empty = np.max(a_learning[1:, :, :], axis=0)
    amount = proportion * max_above_empty
    # Mask: where empty row is zero, subtract amount from row 0 only.
    empty_zero = a_learning[0, :, :] == 0
    a_learning[0, :, :] = np.where(
        empty_zero,
        a_learning[0, :, :] - amount,
        a_learning[0, :, :],
    )
    return a_learning


def real_dirichlet_update(
    a_resource: np.ndarray,
    O_resource_at_timey: np.ndarray,
    P_pos_at_timey: np.ndarray,
    P_ctx_smoothed: np.ndarray,
    proportion: float = 0.3,
    scale: float = 0.7,
    floor: float = 0.05,
) -> np.ndarray:
    """Real-trial update: row-0 penalty, 0.7 scaling, 0.05 floor.

    Returns the *new* ``a_resource`` tensor (does not mutate the input).
    """
    a_learning = build_a_learning(O_resource_at_timey, P_pos_at_timey,
                                  P_ctx_smoothed, a_resource)
    a_learning = apply_row0_penalty(a_learning, proportion=proportion)
    out = a_resource + scale * a_learning
    out = np.where(out <= floor, floor, out)
    return out


def planning_dirichlet_update(
    a_resource_imag: np.ndarray,
    O_resource_at_timey: np.ndarray,
    P_pos_at_timey: np.ndarray,
    P_ctx_smoothed: np.ndarray,
    learning_weight: float,
    prune_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SL planning-time update.

    Returns ``(a_imag_new, a_learning_unweighted, a_learning_weighted)``:
      - ``a_imag_new`` = a_resource_imag + a_learning_unweighted
      - ``a_learning_weighted`` is used to compute ``a_temp`` for the KL
        novelty term: ``a_temp = a_prior + a_learning_weighted``.

    The MATLAB pattern (lines 75-88 of tree_search_frwd_SL):
        a_learning(a_learning <= 0.2) = 0          % prune
        a_learning_weighted = a_learning           % copy
        a_learning_weighted(2:end, :) = w_l * ...  % scale rows >0
        a_learning_weighted(1, :)     = a_learning(1, :)   % keep row 0
        a{modality} = a{modality} + a_learning     % unscaled imagined update
        a_temp      = a_prior + a_learning_weighted

    Note the imagined ``a`` is updated by the *unweighted* tensor, but
    novelty is computed against the *weighted* version. This is faithful
    to MATLAB; a refactor of either is out of scope.
    """
    a_learning = build_a_learning(O_resource_at_timey, P_pos_at_timey,
                                  P_ctx_smoothed, a_resource_imag)
    if prune_threshold > 0:
        a_learning = np.where(a_learning <= prune_threshold, 0.0, a_learning)

    a_weighted = a_learning.copy()
    a_weighted[1:, :, :] = learning_weight * a_learning[1:, :, :]
    # row 0 left unscaled (already correct via the copy)

    a_new = a_resource_imag + a_learning
    return a_new, a_learning, a_weighted

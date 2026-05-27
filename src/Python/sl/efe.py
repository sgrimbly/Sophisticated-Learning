"""Expected Free Energy term helpers.

Port of:
    src/MATLAB/utils/G_epistemic_value.m
    src/MATLAB/utils/kldir.m
    src/MATLAB/utils/determineObservationPreference.m
    src/MATLAB/utils/nat_log.m
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .inference import spm_cross


_LOG_FLOOR = np.exp(-16.0)
_REALMAX = float(np.finfo(np.float64).max)


def nat_log(x: np.ndarray) -> np.ndarray:
    """MATLAB nat_log.m (per spm_norm convention)."""
    x = np.asarray(x, dtype=np.float64)
    return np.log(x + _LOG_FLOOR)


def kldir(a: np.ndarray, b: np.ndarray) -> float:
    """Port of kldir.m: sum(a .* log(a./b), 'all')."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Input matrices must have the same dimensions, got {a.shape} vs {b.shape}")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = a / b
        kl = float(np.sum(a * np.log(ratio)))
    if not np.isfinite(kl):
        return _REALMAX
    return kl


def G_epistemic_value(A_modalities: Sequence[np.ndarray], P_factors: Sequence[np.ndarray]) -> float:
    """Bayesian surprise / mutual information.

    Port of G_epistemic_value.m. ``A_modalities`` is a list of likelihood
    arrays (one per modality), each with leading axis = outcomes and
    subsequent axes = state factors. ``P_factors`` is a list of posterior
    distributions over each state factor (e.g., [P_pos, P_ctx]).
    """
    P_factors = [np.asarray(p, dtype=np.float64).ravel() for p in P_factors]
    qx = spm_cross(*P_factors)  # joint distribution over factors
    qx_flat = qx.ravel(order="F")
    nz = np.nonzero(qx_flat > _LOG_FLOOR)[0]

    # Reshape each A_modal so the trailing factor axes can be linearly indexed
    # in column-major order to match MATLAB's `A{g}(:, i)` linearisation.
    flat_modalities = []
    for A in A_modalities:
        A = np.asarray(A, dtype=np.float64)
        flat = A.reshape(A.shape[0], -1, order="F")
        flat_modalities.append(flat)

    G = 0.0
    qo = None  # accumulator over outcome combinations

    for i in nz:
        # po = outer product of A_modal[:, i] across modalities
        po = np.array([1.0])
        for flat in flat_modalities:
            po = spm_cross(po, flat[:, i])
        po = po.ravel(order="F")

        weight = qx_flat[i]
        if qo is None:
            qo = weight * po
        else:
            qo = qo + weight * po
        G = G + float(weight * (po @ nat_log(po)))

    if qo is None:
        return 0.0
    G = G - float(qo @ nat_log(qo))
    return G


def determine_observation_preference(
    t_food: int,
    t_water: int,
    t_sleep: int,
    preference_inverse_precision: float,
) -> np.ndarray:
    """Port of determineObservationPreference.m, returning the resource-
    modality preference vector C{2}, scaled by 1/preference_inverse_precision.

    Returns array of shape (4,) ordered [empty, food, water, sleep].

    Threshold semantics: matches MATLAB exactly — flip-to-(-500) triggers
    when ``t_water > 19``, ``t_food > 21``, ``t_sleep > 24`` (i.e., the
    *next* step would breach the survival constraint).
    """
    empty = -1.0
    f = float(t_food)
    w = float(t_water)
    s = float(t_sleep)

    if w > 19:
        f, s, empty = -500.0, -500.0, -500.0
    if f > 21:
        w, s, empty = -500.0, -500.0, -500.0
    if s > 24:
        f, w, empty = -500.0, -500.0, -500.0

    C = np.array([empty, f, w, s], dtype=np.float64)
    if np.isfinite(preference_inverse_precision):
        return C / preference_inverse_precision
    # preference weight == 0 → infinite inverse-precision → C ≈ 0
    return np.zeros_like(C)

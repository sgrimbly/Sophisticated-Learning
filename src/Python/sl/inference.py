"""SPM-style inference primitives.

Port of:
    src/MATLAB/utils/spm/spm_cross.m
    src/MATLAB/utils/spm/spm_norm.m
    src/MATLAB/utils/spm/spm_backwards.m
    src/MATLAB/utils/calculate_posterior.m
    src/MATLAB/utils/normalise.m
    src/MATLAB/utils/normalise_matrix.m

Conventions
-----------
Two latent factors, ordered (position, context) — matches MATLAB ``Q{t,1}``
and ``Q{t,2}``.

Likelihood tensors:
    A_pos    : shape (num_states, num_states, num_contexts)   modality 0 (= MATLAB modality 1)
    A_resource: shape (4, num_states, num_contexts)             modality 1 (= MATLAB modality 2)
    A_hill   : shape (5, num_states, num_contexts)             modality 2 (= MATLAB modality 3)

Posterior arrays (per timestep):
    P_pos : shape (num_states,)
    P_ctx : shape (num_contexts,)

Joint state ordering (joint = pos + num_states * ctx) matches MATLAB
column-major ravel of ``spm_cross(P_pos, P_ctx)``.

NO sampling occurs anywhere in this module. Observation distributions are
treated as probability vectors (one-hot or otherwise) and marginalised. This
is the single most important fix versus the prior Python port — see
docs MATLAB_PYTHON_EQUIVALENCE.md §3.4.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def normalise(x: np.ndarray) -> np.ndarray:
    """Port of normalise.m. Returns uniform vector if input sums to 0/NaN."""
    x = np.asarray(x, dtype=np.float64)
    s = x.sum()
    if not np.isfinite(s) or s <= 0.0:
        out = np.full(x.shape, 1.0 / x.size, dtype=np.float64)
        return out
    return x / s


def normalise_matrix_columns(m: np.ndarray) -> np.ndarray:
    """Port of normalise_matrix.m: column-stochastic.

    Operates on the *first* axis (columns), preserving any trailing axes —
    matches MATLAB's behaviour on 3-D arrays where it normalises along axis 1
    of size N×K (or N×K×L, normalising columns slice-by-slice).
    """
    m = np.asarray(m, dtype=np.float64)
    sums = m.sum(axis=0, keepdims=True)
    # Avoid divide-by-zero. Where the sum is zero, leave as zeros (matches
    # MATLAB behaviour of producing NaN there; we substitute zero so downstream
    # code does not propagate NaN through `0 * NaN`).
    safe = np.where(sums > 0, sums, 1.0)
    out = m / safe
    return out


def spm_norm(A: np.ndarray) -> np.ndarray:
    """Port of spm_norm.m: column-normalise a transition matrix.

    Substitutes 1/N for any column that summed to NaN/inf/0.
    """
    A = np.asarray(A, dtype=np.float64)
    s = A.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = A / s
    bad = ~np.isfinite(out)
    if bad.any():
        out = np.where(bad, 1.0 / A.shape[0], out)
    return out


def spm_cross(*factors: np.ndarray) -> np.ndarray:
    """Multidimensional outer product of any number of arrays.

    Equivalent to MATLAB ``spm_cross(X, x, varargin{:})``: each new factor
    extends the last axis. With two 1-D inputs of shapes (M,), (N,) the
    result is shape (M, N). With three (M,), (N,), (K,) → (M, N, K).

    The function also accepts Python lists/tuples (MATLAB cell-array
    behaviour: recursively cross all members) for compatibility with code
    that calls ``spm_cross(Q[t])``.
    """
    if len(factors) == 0:
        raise ValueError("spm_cross requires at least one input")

    # Single argument: if list/tuple, recurse over its members.
    if len(factors) == 1:
        x = factors[0]
        if isinstance(x, (list, tuple)):
            return spm_cross(*x)
        return np.asarray(x, dtype=np.float64)

    out = np.asarray(factors[0], dtype=np.float64)
    for nxt in factors[1:]:
        nxt = np.asarray(nxt, dtype=np.float64)
        # Append nxt's dims to the trailing axes of out.
        # out.shape == (a1, ..., aD); nxt.shape == (b1, ..., bE)
        # Result shape == (a1, ..., aD, b1, ..., bE)
        out = out.reshape(out.shape + (1,) * nxt.ndim) * nxt.reshape(
            (1,) * out.ndim + nxt.shape
        )
    return out


def joint_pos_ctx(P_pos: np.ndarray, P_ctx: np.ndarray) -> np.ndarray:
    """Joint distribution over (position × context), ravelled in MATLAB order.

    Matches MATLAB ``qs = spm_cross(Q(t,:)); qs(:)`` — column-major flattening
    so that index ``j = pos + num_states * ctx``.
    """
    pos = np.asarray(P_pos, dtype=np.float64).ravel()
    ctx = np.asarray(P_ctx, dtype=np.float64).ravel()
    # Outer in axis order (pos, ctx) then F-order ravel == pos varies fastest.
    return np.outer(pos, ctx).ravel(order="F")


def calculate_posterior(
    P_pos: np.ndarray,
    P_ctx: np.ndarray,
    A_resource: np.ndarray,
    A_hill: np.ndarray,
    O_resource: np.ndarray,
    O_hill: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of calculate_posterior.m.

    Updates the *context* posterior (factor 2 in MATLAB / index 1 in Python)
    by treating each non-position modality's observation as a probability
    vector and marginalising.

    Position posterior is returned unchanged (matches MATLAB, which also
    only updates ``fact = 2``).

    Parameters
    ----------
    P_pos : (num_states,)         predictive position posterior
    P_ctx : (num_contexts,)       predictive context posterior
    A_resource : (4, num_states, num_contexts)
    A_hill : (5, num_states, num_contexts)
    O_resource : (4,)             observation distribution over resource outcomes
    O_hill : (5,)                 observation distribution over hill outcomes

    Returns
    -------
    P_pos, P_ctx_updated
    """
    P_pos = np.asarray(P_pos, dtype=np.float64).ravel()
    P_ctx = np.asarray(P_ctx, dtype=np.float64).ravel()

    # Likelihood per (state, context):
    #   L_modal[s,c] = sum_o O_modal[o] * A_modal[o,s,c]
    L_resource = np.einsum("o,osc->sc", np.asarray(O_resource, dtype=np.float64).ravel(),
                           np.asarray(A_resource, dtype=np.float64))
    L_hill = np.einsum("o,osc->sc", np.asarray(O_hill, dtype=np.float64).ravel(),
                       np.asarray(A_hill, dtype=np.float64))
    L = L_resource * L_hill  # (num_states, num_contexts)

    # MATLAB marginalises out the *position* factor (the "other" factor):
    #   LL = L * P{t,1}'    -> for fact==2, the column from P_pos
    # gives a (num_contexts,) vector: integrand over s.
    LL = P_pos @ L  # (num_contexts,)

    y = LL * P_ctx
    return P_pos, normalise(y)


def spm_backwards(
    O_history_hill: Sequence[np.ndarray],
    P_history_pos: Sequence[np.ndarray],
    P_history_ctx_t: np.ndarray,
    A_hill: np.ndarray,
    B_ctx: np.ndarray,
    timey: int,
    t: int,
) -> np.ndarray:
    """Port of spm_backwards.m for the hill modality only.

    The MATLAB function is called with the full ``O`` and ``Q`` cell arrays
    plus indices ``timey`` and ``t``. It computes the smoothed posterior
    over the *context* at ``timey`` given hill observations from
    ``timey+1..t``.

    Parameters
    ----------
    O_history_hill : sequence of length >= t+1
        ``O_history_hill[k]`` is the hill-modality observation distribution
        at imagined timestep ``k`` (0-based: index ``k`` corresponds to
        MATLAB ``O{3, k+1}``). Must extend up to and including index ``t``.
    P_history_pos : sequence of length >= t+1
        Position posterior at each imagined timestep.
    P_history_ctx_t : (num_contexts,)
        Context posterior at the start time ``timey`` — this is the prior
        we update.
    A_hill : (5, num_states, num_contexts)
    B_ctx : (num_contexts, num_contexts, ...)
        Context transition tensor. Action-axis 0 is used (context evolves
        independently of action in this task).
    timey : int (0-based)
    t     : int (0-based, inclusive end)

    Returns
    -------
    L : (num_contexts,)
        Smoothed (and normalised) posterior over context at ``timey``.
    """
    L = np.asarray(P_history_ctx_t, dtype=np.float64).ravel().copy()
    n_ctx = L.size
    p = np.eye(n_ctx, dtype=np.float64)  # propagator from `timey` to current step

    # B_ctx shape is (num_contexts, num_contexts, num_actions); we use action 0.
    B_ctx_step = np.asarray(B_ctx, dtype=np.float64)
    if B_ctx_step.ndim == 3:
        B_ctx_step = B_ctx_step[:, :, 0]

    A_hill = np.asarray(A_hill, dtype=np.float64)

    for timestep in range(timey + 1, t + 1):
        p = B_ctx_step @ p  # (n_ctx, n_ctx)

        O_hill_step = np.asarray(O_history_hill[timestep], dtype=np.float64).ravel()
        Q_pos_step = np.asarray(P_history_pos[timestep], dtype=np.float64).ravel()

        # likelihood over (state, context) marginalised over outcomes:
        #   L_modal[s, c] = sum_o O[o] * A[o, s, c]
        L_modal = np.einsum("o,osc->sc", O_hill_step, A_hill)
        # marginalise out state:
        #   temp(c) = L_modal[:, c] · Q_pos_step
        temp_c = Q_pos_step @ L_modal  # (n_ctx,)

        # For each candidate state at `timey` (i.e., each entry of L), compute
        #   aaa(state) = temp_c · p[:, state]
        # which is a vector of length n_ctx (one per "starting" context).
        aaa = temp_c @ p  # (n_ctx,)
        L = L * aaa

    return spm_norm(L.reshape(-1, 1)).ravel()


def imagined_observation_dist(y_modal: np.ndarray, joint_state: int, num_states: int) -> np.ndarray:
    """Mirror MATLAB ``normalise(y{modal}(:, state))`` where ``state`` is a
    1-based linear index into the joint (state × context) space.

    Parameters
    ----------
    y_modal : (num_outcomes, num_states, num_contexts)
    joint_state : 0-based linear index, with column-major (pos, ctx)
        ordering — i.e., joint_state = pos + num_states * ctx.
    """
    pos = joint_state % num_states
    ctx = joint_state // num_states
    col = y_modal[:, pos, ctx]
    return normalise(col)

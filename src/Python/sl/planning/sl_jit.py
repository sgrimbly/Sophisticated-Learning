"""Fully-JIT'd Sophisticated Learning tree search.

Same approach as :mod:`sl.planning.si_jit`, with the additional complication
that SL has *history buffers* the novelty loop walks over (``timey``) and
imagined-learning state that mutates inside the recursion.

Design:
  * History is fixed-shape arrays: ``hist_O_res`` (T_max+1, 4),
    ``hist_O_hill`` (T_max+1, 5), ``hist_P_pos`` (T_max+1, S),
    ``hist_P_ctx`` (T_max+1, C). T_max = max_horizon + max_real_steps + 1.
  * The recursion threads ``a_resource_imag`` and (optionally) ``y_resource``
    through; both are fresh allocations per recursion-level call so back-
    tracking doesn't corrupt sibling expansions. We allocate them once on
    the stack with ``.copy()``.
  * ``adaptive_likelihood_in_plan=True`` is supported via a flag.

This is the second leg of the big refactor. Numba-recursive performance
on SL is bounded by the ``spm_backwards`` step inside the smoothing window;
the inner loop here is fully unrolled so it should hit C-level throughput.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import numba as _nb
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

from .common import PlanResult


_LOG_FLOOR = float(np.exp(-16.0))
_REALMAX = float(np.finfo(np.float64).max)


if _HAS_NUMBA:

    # Re-import the small JIT primitives from si_jit to avoid duplicating them.
    # (Numba caches compilation per-function so this doesn't cost extra.)
    from .si_jit import (
        _normalise as _normalise,                 # noqa: F401
        _kldir as _kldir,                          # noqa: F401
        _calculate_posterior as _calculate_posterior,
        _G_epistemic as _G_epistemic,
        _determine_observation_preference as _determine_observation_preference,
        _index_clip as _index_clip,
        _imagined_obs_modality as _imagined_obs_modality,
    )

    @_nb.njit(cache=True, fastmath=True)
    def _spm_backwards(L_init, hist_O_hill, hist_P_pos, A_hill, B_ctx_step,
                        timey, t):
        """Hill-only backward smoothing for the JIT SL recursion."""
        n_o, n_s, n_c = A_hill.shape
        L = L_init.copy()
        p = np.eye(n_c, dtype=np.float64)
        new_p = np.empty((n_c, n_c), dtype=np.float64)

        for timestep in range(timey + 1, t + 1):
            # p = B_ctx_step @ p
            for i in range(n_c):
                for j in range(n_c):
                    v = 0.0
                    for k in range(n_c):
                        v += B_ctx_step[i, k] * p[k, j]
                    new_p[i, j] = v
            for i in range(n_c):
                for j in range(n_c):
                    p[i, j] = new_p[i, j]

            # temp_c[c] = sum_s P_pos[s] * sum_o O_hill[o] * A_hill[o, s, c]
            temp_c = np.zeros(n_c, dtype=np.float64)
            for c in range(n_c):
                v = 0.0
                for s in range(n_s):
                    lm = 0.0
                    for o in range(n_o):
                        lm += hist_O_hill[timestep, o] * A_hill[o, s, c]
                    v += hist_P_pos[timestep, s] * lm
                temp_c[c] = v

            # aaa = temp_c @ p
            for c in range(n_c):
                v = 0.0
                for cc in range(n_c):
                    v += temp_c[cc] * p[cc, c]
                L[c] = L[c] * v

        # spm_norm
        s = 0.0
        for i in range(n_c):
            v = L[i]
            if v != v or v == np.inf or v == -np.inf:
                v = 0.0
            L[i] = v
            s += v
        if s <= 0.0:
            for i in range(n_c):
                L[i] = 1.0 / n_c
        else:
            for i in range(n_c):
                L[i] = L[i] / s
        return L

    @_nb.njit(cache=True, fastmath=True)
    def _planning_dirichlet_update(a_imag, O_res_t, P_pos_t, P_ctx_smoothed,
                                    learning_weight, prune_threshold):
        """In-line port of learning.planning_dirichlet_update for the SL JIT loop.

        Returns (a_new, a_weighted).  a_learning is folded in.
        """
        n_o, n_s, n_c = a_imag.shape
        a_new = a_imag.copy()
        a_weighted = np.zeros((n_o, n_s, n_c), dtype=np.float64)

        for o in range(n_o):
            ov = O_res_t[o]
            for s in range(n_s):
                pv = ov * P_pos_t[s]
                for c in range(n_c):
                    if a_imag[o, s, c] <= 0.0:
                        continue
                    al = pv * P_ctx_smoothed[c]
                    if prune_threshold > 0.0 and al <= prune_threshold:
                        al = 0.0
                    a_new[o, s, c] = a_imag[o, s, c] + al
                    if o == 0:
                        a_weighted[o, s, c] = al
                    else:
                        a_weighted[o, s, c] = learning_weight * al
        return a_new, a_weighted

    @_nb.njit(cache=True, fastmath=True)
    def _kldir_normalised_flat(a_temp, a_prior):
        """KL(normalise(a_temp.flat), normalise(a_prior.flat)) — fused."""
        n_o, n_s, n_c = a_temp.shape
        sum_t = 0.0; sum_p = 0.0
        for o in range(n_o):
            for s in range(n_s):
                for c in range(n_c):
                    if a_temp[o, s, c] > 0.0:
                        sum_t += a_temp[o, s, c]
                    if a_prior[o, s, c] > 0.0:
                        sum_p += a_prior[o, s, c]
        if sum_t <= 0.0 or sum_p <= 0.0:
            return 0.0
        kl = 0.0
        for o in range(n_o):
            for s in range(n_s):
                for c in range(n_c):
                    p_t = a_temp[o, s, c] / sum_t
                    p_p = a_prior[o, s, c] / sum_p
                    if p_t <= 0.0:
                        continue
                    if p_p <= 0.0:
                        return _REALMAX
                    kl += p_t * np.log(p_t / p_p)
        if kl != kl or kl == np.inf or kl == -np.inf:
            return _REALMAX
        return kl

    @_nb.njit(cache=True)
    def _sl_recurse(
        stm,
        hist_O_res, hist_O_hill, hist_P_pos, hist_P_ctx,
        max_t_in_hist,                # current populated extent (exclusive upper)
        a_resource_imag,
        A_pos_flat, A_res_flat, A_hill_flat,
        A_pos, A_resource, A_hill,
        y_pos, y_resource, y_hill,
        B_pos, bb_ctx,
        w_novelty, w_learning, w_epistemic, pref_iprec,
        prune_threshold, adaptive_likelihood,
        t, N, t_food, t_water, t_sleep, true_t,
        novelty_on, epistemic_on, smoothing_on,
        node_count, memory_hits, memory_misses,
    ):
        node_count[0] += 1
        G = 0.02

        P_pos_prior = hist_P_pos[t].copy()
        P_ctx_prior = hist_P_ctx[t].copy()

        P_pos, P_ctx = _calculate_posterior(
            P_pos_prior, P_ctx_prior,
            y_resource, y_hill,
            hist_O_res[t], hist_O_hill[t],
        )

        # MATLAB tree_search_frwd_SL line 29 writes the posterior back into
        # P{t, ...} in-place. Mirror that here so the smoothing window's
        # spm_backwards (called by deeper recursions for timey < current t)
        # reads posterior context, not the parent-stored predictive Q.
        # Same fix as planning/sl.py and the agent.py outer loop.
        for s_ in range(P_pos.shape[0]):
            hist_P_pos[t, s_] = P_pos[s_]
        for c_ in range(P_ctx.shape[0]):
            hist_P_ctx[t, c_] = P_ctx[c_]

        t_food_idx = _index_clip(int(round(float(t_food))) + 1)
        t_water_idx = _index_clip(int(round(float(t_water))) + 1)
        t_sleep_idx = _index_clip(int(round(float(t_sleep))) + 1)

        # Save a copy so when we backtrack the caller's a_imag is unchanged.
        a_imag_local = a_resource_imag.copy()
        y_resource_local = y_resource

        if t > true_t:
            if novelty_on and w_novelty != 0.0:
                if smoothing_on:
                    start = max(0, t - 6)
                    novelty_total = 0.0
                    for timey in range(start, t + 1):
                        if timey != t:
                            L_ctx = _spm_backwards(
                                hist_P_ctx[timey].copy(),
                                hist_O_hill, hist_P_pos,
                                A_hill, bb_ctx[:, :, 0], timey, t,
                            )
                        else:
                            L_ctx = P_ctx.copy()
                        P_pos_t = hist_P_pos[timey]
                        O_res_t = hist_O_res[timey]
                        a_prior = a_imag_local.copy()
                        a_imag_local, a_weighted = _planning_dirichlet_update(
                            a_imag_local, O_res_t, P_pos_t, L_ctx,
                            w_learning, prune_threshold,
                        )
                        a_temp = a_prior + a_weighted
                        novelty_total += _kldir_normalised_flat(a_temp, a_prior)
                    if adaptive_likelihood:
                        # y_resource = normalise_matrix_columns(a_imag_local)
                        n_o = a_imag_local.shape[0]
                        n_s = a_imag_local.shape[1]
                        n_c = a_imag_local.shape[2]
                        y_resource_local = np.empty_like(a_imag_local)
                        for s in range(n_s):
                            for c in range(n_c):
                                colsum = 0.0
                                for o in range(n_o):
                                    colsum += a_imag_local[o, s, c]
                                if colsum > 0.0:
                                    for o in range(n_o):
                                        y_resource_local[o, s, c] = a_imag_local[o, s, c] / colsum
                                else:
                                    for o in range(n_o):
                                        y_resource_local[o, s, c] = 0.0
                else:
                    # SL_noSmooth: single-step novelty using current node only
                    a_prior = a_imag_local.copy()
                    a_imag_local, a_weighted = _planning_dirichlet_update(
                        a_imag_local, hist_O_res[t], P_pos, P_ctx,
                        w_learning, prune_threshold,
                    )
                    a_temp = a_prior + a_weighted
                    novelty_total = _kldir_normalised_flat(a_temp, a_prior)

                G += w_novelty * novelty_total

            if epistemic_on and w_epistemic != 0.0:
                # Use the (possibly updated) y_resource_local — flatten on the fly
                if adaptive_likelihood and smoothing_on:
                    A_res_flat_local = np.ascontiguousarray(
                        y_resource_local.reshape(y_resource_local.shape[0], -1).copy()
                    )
                    epi = _G_epistemic(A_pos_flat, A_res_flat_local, A_hill_flat,
                                        P_pos_prior, P_ctx_prior)
                else:
                    epi = _G_epistemic(A_pos_flat, A_res_flat, A_hill_flat,
                                        P_pos_prior, P_ctx_prior)
                G += w_epistemic * epi

            C = _determine_observation_preference(t_food, t_water, t_sleep, pref_iprec)
            extrinsic = (hist_O_res[t, 0] * C[0] + hist_O_res[t, 1] * C[1]
                         + hist_O_res[t, 2] * C[2] + hist_O_res[t, 3] * C[3])
            G += extrinsic

            t_food = int(round((t_food + 1) * (1.0 - hist_O_res[t, 1])))
            t_water = int(round((t_water + 1) * (1.0 - hist_O_res[t, 2])))
            t_sleep = int(round((t_sleep + 1) * (1.0 - hist_O_res[t, 3])))
            t_food_idx = _index_clip(t_food + 1)
            t_water_idx = _index_clip(t_water + 1)
            t_sleep_idx = _index_clip(t_sleep + 1)

        best_action = -1
        if t < N:
            n_states = A_pos_flat.shape[0]
            n_ctx = bb_ctx.shape[0]
            n_joint = n_states * n_ctx
            n_actions = B_pos.shape[2]
            efe = np.zeros(n_actions, dtype=np.float64)

            for action in range(n_actions):
                Q_pos_a = np.dot(B_pos[:, :, action], P_pos)
                Q_ctx_a = np.dot(bb_ctx[:, :, 0], P_ctx)

                qs = np.empty(n_joint, dtype=np.float64)
                idx = 0
                for c in range(n_ctx):
                    pcv = Q_ctx_a[c]
                    for p_ in range(n_states):
                        qs[idx] = Q_pos_a[p_] * pcv
                        idx += 1

                threshold = 1.0 / 8.0
                count_likely = 0
                for k in range(n_joint):
                    if qs[k] > threshold:
                        count_likely += 1
                if count_likely == 0:
                    eps = 1.0 / (n_joint * n_joint)
                    threshold = 1.0 / n_joint - eps

                K = np.zeros(n_joint, dtype=np.float64)
                action_fe = 0.0
                for state in range(n_joint):
                    if qs[state] <= threshold:
                        continue
                    cache = stm[t_food_idx, t_water_idx, t_sleep_idx, state]
                    if cache != 0.0:
                        K[state] = cache
                        memory_hits[0] += 1
                    else:
                        # Save current history slot t+1 (if previously written, restore later).
                        # We overwrite slots [t+1] for the imagined child step.
                        for o in range(hist_O_res.shape[1]):
                            hist_O_res[t + 1, o] = _imagined_obs_modality(
                                y_resource_local, state, n_states
                            )[o]
                        for o in range(hist_O_hill.shape[1]):
                            hist_O_hill[t + 1, o] = _imagined_obs_modality(
                                y_hill, state, n_states
                            )[o]
                        for s_ in range(n_states):
                            hist_P_pos[t + 1, s_] = Q_pos_a[s_]
                        for c_ in range(n_ctx):
                            hist_P_ctx[t + 1, c_] = Q_ctx_a[c_]

                        G_child, _ = _sl_recurse(
                            stm,
                            hist_O_res, hist_O_hill, hist_P_pos, hist_P_ctx,
                            max(max_t_in_hist, t + 2),
                            a_imag_local,
                            A_pos_flat, A_res_flat, A_hill_flat,
                            A_pos, A_resource, A_hill,
                            y_pos, y_resource_local, y_hill,
                            B_pos, bb_ctx,
                            w_novelty, w_learning, w_epistemic, pref_iprec,
                            prune_threshold, adaptive_likelihood,
                            t + 1, N, t_food, t_water, t_sleep, true_t,
                            novelty_on, epistemic_on, smoothing_on,
                            node_count, memory_hits, memory_misses,
                        )
                        K[state] = G_child
                        stm[t_food_idx, t_water_idx, t_sleep_idx, state] = G_child
                        memory_misses[0] += 1
                    action_fe += K[state] * qs[state]

                efe[action] = 0.7 * action_fe

            best_action = 0
            best_val = efe[0]
            for k in range(1, n_actions):
                if efe[k] > best_val:
                    best_val = efe[k]
                    best_action = k
            G += best_val

        return G, best_action


def tree_search_sl_jit_run(
    short_term_memory: np.ndarray,
    O_pos_root: np.ndarray, O_res_root: np.ndarray, O_hill_root: np.ndarray,
    P_pos_root: np.ndarray, P_ctx_root: np.ndarray,
    inputs,                # PlannerInputs
    t: int, N: int,
    t_food: int, t_water: int, t_sleep: int,
    true_t: int,
    *,
    novelty_on: bool = True, epistemic_on: bool = True,
    smoothing_on: bool = True,
    adaptive_likelihood_in_plan: bool = False,
    learning_prune_threshold: float = 0.2,
    history_O_resource: List[np.ndarray] = None,
    history_O_hill: List[np.ndarray] = None,
    history_P_pos: List[np.ndarray] = None,
    history_P_ctx: List[np.ndarray] = None,
) -> PlanResult:
    """JIT entry for SL with the same return type as :func:`tree_search_sl`."""
    if not _HAS_NUMBA:
        from .sl import tree_search_sl as _np_sl
        return _np_sl(
            short_term_memory, O_pos_root, O_res_root, O_hill_root,
            P_pos_root, P_ctx_root, inputs,
            t=t, N=N, t_food=t_food, t_water=t_water, t_sleep=t_sleep,
            true_t=true_t,
            novelty_on=novelty_on, epistemic_on=epistemic_on,
            smoothing_on=smoothing_on,
            adaptive_likelihood_in_plan=adaptive_likelihood_in_plan,
            learning_prune_threshold=learning_prune_threshold,
            history_O_resource=history_O_resource,
            history_O_hill=history_O_hill,
            history_P_pos=history_P_pos,
            history_P_ctx=history_P_ctx,
        )

    n_states = inputs.A_pos.shape[0]
    n_ctx = inputs.A_resource.shape[2]

    T_max = N + 2  # safe upper bound for imagined trajectory writes
    hist_O_res = np.zeros((T_max, 4), dtype=np.float64)
    hist_O_hill = np.zeros((T_max, 5), dtype=np.float64)
    hist_P_pos = np.zeros((T_max, n_states), dtype=np.float64)
    hist_P_ctx = np.zeros((T_max, n_ctx), dtype=np.float64)

    # Populate the real-trial history portion.
    if history_O_resource is not None:
        for k in range(min(t + 1, len(history_O_resource))):
            v = history_O_resource[k]
            if v is not None:
                hist_O_res[k, :len(v)] = v
    if history_O_hill is not None:
        for k in range(min(t + 1, len(history_O_hill))):
            v = history_O_hill[k]
            if v is not None:
                hist_O_hill[k, :len(v)] = v
    if history_P_pos is not None:
        for k in range(min(t + 1, len(history_P_pos))):
            v = history_P_pos[k]
            if v is not None:
                hist_P_pos[k, :len(v)] = v
    if history_P_ctx is not None:
        for k in range(min(t + 1, len(history_P_ctx))):
            v = history_P_ctx[k]
            if v is not None:
                hist_P_ctx[k, :len(v)] = v

    # Root values.
    hist_O_res[t] = np.asarray(O_res_root, dtype=np.float64).ravel()
    hist_O_hill[t] = np.asarray(O_hill_root, dtype=np.float64).ravel()
    hist_P_pos[t] = np.asarray(P_pos_root, dtype=np.float64).ravel()
    hist_P_ctx[t] = np.asarray(P_ctx_root, dtype=np.float64).ravel()

    A_pos_flat = np.ascontiguousarray(
        inputs.y_pos.reshape(inputs.y_pos.shape[0], -1, order="F"), dtype=np.float64,
    )
    A_res_flat = np.ascontiguousarray(
        inputs.y_resource.reshape(inputs.y_resource.shape[0], -1, order="F"), dtype=np.float64,
    )
    A_hill_flat = np.ascontiguousarray(
        inputs.y_hill.reshape(inputs.y_hill.shape[0], -1, order="F"), dtype=np.float64,
    )

    node_count = np.zeros(1, dtype=np.int64)
    memory_hits = np.zeros(1, dtype=np.int64)
    memory_misses = np.zeros(1, dtype=np.int64)

    G, best_action = _sl_recurse(
        short_term_memory,
        hist_O_res, hist_O_hill, hist_P_pos, hist_P_ctx,
        t + 1,                                         # max_t_in_hist
        np.ascontiguousarray(inputs.a_resource, dtype=np.float64),
        A_pos_flat, A_res_flat, A_hill_flat,
        np.ascontiguousarray(inputs.A_pos, dtype=np.float64),
        np.ascontiguousarray(inputs.A_resource, dtype=np.float64),
        np.ascontiguousarray(inputs.A_hill, dtype=np.float64),
        np.ascontiguousarray(inputs.y_pos, dtype=np.float64),
        np.ascontiguousarray(inputs.y_resource, dtype=np.float64),
        np.ascontiguousarray(inputs.y_hill, dtype=np.float64),
        np.ascontiguousarray(inputs.B_pos, dtype=np.float64),
        np.ascontiguousarray(inputs.bb_ctx, dtype=np.float64),
        float(inputs.weights.novelty),
        float(inputs.weights.learning),
        float(inputs.weights.epistemic),
        float(inputs.weights.preference_inverse_precision),
        float(learning_prune_threshold),
        bool(adaptive_likelihood_in_plan),
        int(t), int(N),
        int(t_food), int(t_water), int(t_sleep), int(true_t),
        bool(novelty_on), bool(epistemic_on), bool(smoothing_on),
        node_count, memory_hits, memory_misses,
    )

    return PlanResult(
        G=float(G),
        best_actions=[int(best_action)] if best_action >= 0 else [],
        memory_hits=int(memory_hits[0]),
        memory_misses=int(memory_misses[0]),
        node_count=int(node_count[0]),
    )

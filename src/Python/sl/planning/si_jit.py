"""Fully-JIT'd SI tree search (the big refactor for 5-10× target speedup).

Design constraints:
  * No PlannerInputs dataclass, no _RecState class. Everything passes as
    bare numpy arrays + scalars.
  * No best_actions list — recursion returns only the scalar G and the
    best action at the current level; the agent only ever needs the root's
    best action.
  * G_epistemic_value uses the F-order flattened A modality arrays
    (pre-flatten once at entry; the y_* arrays are constant through the
    recursion in SI).
  * No history buffers (SI has none — single-timey novelty).

Compatibility with the NumPy SI planner:
  * Same recursion shape and per-node EFE breakdown.
  * Same need-timer update timing (after extrinsic).
  * Same likely-state pruning: ``qs > 1/8`` with the tiny-jitter fallback.
  * Same memoisation cache semantics.

Entry point: :func:`tree_search_si_jit_run` returns a
:class:`sl.planning.common.PlanResult` so callers don't have to know about
the JIT specialisation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import numba as _nb
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

from .common import PlanResult


_LOG_FLOOR = float(np.exp(-16.0))
_REALMAX = float(np.finfo(np.float64).max)


# =============================================================================
# JIT primitives — kept local so the whole recursion can call them inline
# =============================================================================


if _HAS_NUMBA:

    @_nb.njit(cache=True, fastmath=True, inline="always")
    def _normalise(x):
        n = x.shape[0]
        out = np.empty(n, dtype=np.float64)
        s = 0.0
        for i in range(n):
            v = x[i]
            if v != v or v < 0.0 or v == np.inf or v == -np.inf:
                v = 0.0
            s += v
        if s <= 0.0:
            for i in range(n):
                out[i] = 1.0 / n
            return out
        for i in range(n):
            out[i] = x[i] / s
        return out

    @_nb.njit(cache=True, fastmath=True, inline="always")
    def _kldir(a, b):
        n = a.size
        af = a.ravel(); bf = b.ravel()
        kl = 0.0
        for i in range(n):
            av = af[i]; bv = bf[i]
            if av <= 0.0:
                continue
            if bv <= 0.0:
                return _REALMAX
            kl += av * np.log(av / bv)
        if kl != kl or kl == np.inf or kl == -np.inf:
            return _REALMAX
        return kl

    @_nb.njit(cache=True, fastmath=True)
    def _calculate_posterior(P_pos, P_ctx, A_resource, A_hill, O_resource, O_hill):
        n_o_r, n_s, n_c = A_resource.shape
        n_o_h = A_hill.shape[0]
        # L[s, c] = sum_o O_r[o] * A_r[o,s,c] * sum_o O_h[o] * A_h[o,s,c]
        # collapsed into a single inner loop.
        LL_c = np.zeros(n_c, dtype=np.float64)
        for c in range(n_c):
            v = 0.0
            for s in range(n_s):
                lr = 0.0
                for o in range(n_o_r):
                    lr += O_resource[o] * A_resource[o, s, c]
                lh = 0.0
                for o in range(n_o_h):
                    lh += O_hill[o] * A_hill[o, s, c]
                v += P_pos[s] * lr * lh
            LL_c[c] = v

        y = np.empty(n_c, dtype=np.float64)
        for c in range(n_c):
            y[c] = LL_c[c] * P_ctx[c]
        return P_pos, _normalise(y)

    @_nb.njit(cache=True, fastmath=True)
    def _G_epistemic(A_pos_flat, A_res_flat, A_hill_flat, P_pos, P_ctx):
        """3-modality epistemic value computed in F-order linear index space."""
        n_pos = P_pos.shape[0]
        n_ctx = P_ctx.shape[0]
        n_joint = n_pos * n_ctx
        n_o_pos = A_pos_flat.shape[0]
        n_o_res = A_res_flat.shape[0]
        n_o_hill = A_hill_flat.shape[0]
        n_outcomes_total = n_o_pos * n_o_res * n_o_hill

        qx = np.empty(n_joint, dtype=np.float64)
        idx = 0
        for j in range(n_ctx):
            v = P_ctx[j]
            for i in range(n_pos):
                qx[idx] = P_pos[i] * v
                idx += 1

        qo = np.zeros(n_outcomes_total, dtype=np.float64)
        G = 0.0

        for k in range(n_joint):
            w = qx[k]
            if w <= _LOG_FLOOR:
                continue

            po = np.empty(n_outcomes_total, dtype=np.float64)
            idx = 0
            for hh in range(n_o_hill):
                vh = A_hill_flat[hh, k]
                for rr in range(n_o_res):
                    vrh = vh * A_res_flat[rr, k]
                    for pp in range(n_o_pos):
                        po[idx] = A_pos_flat[pp, k] * vrh
                        idx += 1

            ent = 0.0
            for ii in range(n_outcomes_total):
                p_o = po[ii]
                qo[ii] += w * p_o
                ent += p_o * np.log(p_o + _LOG_FLOOR)
            G += w * ent

        ent_qo = 0.0
        for ii in range(n_outcomes_total):
            ent_qo += qo[ii] * np.log(qo[ii] + _LOG_FLOOR)
        return G - ent_qo

    @_nb.njit(cache=True, fastmath=True, inline="always")
    def _novelty_si(a_resource, O_res, P_pos, P_ctx, learning_weight):
        """Single-step SI novelty term: KL(normalise(a_temp) || normalise(a_prior))."""
        n_o, n_s, n_c = a_resource.shape
        # Build a_learning (with mask a_resource > 0) and a_weighted in one pass.
        # For SI we only need (a_temp = a_prior + a_weighted) flattened and normalised.
        sum_temp = 0.0
        sum_prior = 0.0
        for o in range(n_o):
            ov = O_res[o]
            for s in range(n_s):
                pv = P_pos[s]
                for c in range(n_c):
                    cv = P_ctx[c]
                    a_prior_v = a_resource[o, s, c]
                    sum_prior += a_prior_v
                    if a_prior_v <= 0.0:
                        sum_temp += a_prior_v
                        continue
                    al = ov * pv * cv
                    if o == 0:
                        weighted = al
                    else:
                        weighted = learning_weight * al
                    sum_temp += a_prior_v + weighted

        # KL(a_temp_norm, a_prior_norm) — second pass (Numba can't easily store
        # the intermediate flattened tensors without big alloc).
        if sum_prior <= 0.0 or sum_temp <= 0.0:
            return 0.0
        kl = 0.0
        for o in range(n_o):
            ov = O_res[o]
            for s in range(n_s):
                pv = P_pos[s]
                for c in range(n_c):
                    cv = P_ctx[c]
                    a_prior_v = a_resource[o, s, c]
                    if a_prior_v <= 0.0:
                        continue
                    al = ov * pv * cv
                    if o == 0:
                        weighted = al
                    else:
                        weighted = learning_weight * al
                    a_temp_v = a_prior_v + weighted
                    if a_temp_v <= 0.0:
                        continue
                    p_t = a_temp_v / sum_temp
                    p_p = a_prior_v / sum_prior
                    if p_t > 0.0 and p_p > 0.0:
                        kl += p_t * np.log(p_t / p_p)
        if kl != kl or kl == np.inf or kl == -np.inf:
            return _REALMAX
        return kl

    @_nb.njit(cache=True, fastmath=True, inline="always")
    def _determine_observation_preference(t_food, t_water, t_sleep, pref_iprec):
        """Returns 4-vector [empty, food, water, sleep] / pref_iprec."""
        empty = -1.0
        f = float(t_food); w = float(t_water); s = float(t_sleep)
        if w > 19.0:
            f = -500.0; s = -500.0; empty = -500.0
        if f > 21.0:
            w = -500.0; s = -500.0; empty = -500.0
        if s > 24.0:
            f = -500.0; w = -500.0; empty = -500.0
        out = np.empty(4, dtype=np.float64)
        if pref_iprec == np.inf or pref_iprec != pref_iprec:
            out[0] = 0.0; out[1] = 0.0; out[2] = 0.0; out[3] = 0.0
        else:
            out[0] = empty / pref_iprec
            out[1] = f / pref_iprec
            out[2] = w / pref_iprec
            out[3] = s / pref_iprec
        return out

    @_nb.njit(cache=True, fastmath=True, inline="always")
    def _index_clip(value):
        if value < 0:
            return 0
        if value > 34:
            return 34
        return value

    @_nb.njit(cache=True, fastmath=True)
    def _imagined_obs_modality(y_modal, joint_state, num_states):
        """Equivalent to inference.imagined_observation_dist."""
        pos = joint_state % num_states
        ctx = joint_state // num_states
        n_o = y_modal.shape[0]
        col = np.empty(n_o, dtype=np.float64)
        s = 0.0
        for o in range(n_o):
            v = y_modal[o, pos, ctx]
            col[o] = v
            s += v
        out = np.empty(n_o, dtype=np.float64)
        if s <= 0.0:
            for o in range(n_o):
                out[o] = 1.0 / n_o
        else:
            for o in range(n_o):
                out[o] = col[o] / s
        return out

    # =========================================================================
    # Smoothing-window primitives.
    #
    # These are shared with the SL JIT planner (sl_jit imports them from here).
    # They live in si_jit — the "lower" module in the import graph — so the
    # SI-smooth recursion below and sl_jit's recursion both reuse one compiled
    # copy without an import cycle (sl_jit -> si_jit only).
    # =========================================================================

    @_nb.njit(cache=True, fastmath=True)
    def _spm_backwards(L_init, hist_O_hill, hist_P_pos, A_hill, B_ctx_step,
                        timey, t):
        """Hill-only backward smoothing for the JIT recursions."""
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
        """In-line port of learning.planning_dirichlet_update for the JIT loops.

        Returns (a_new, a_weighted).  a_learning is folded in.  Pass
        ``prune_threshold = 0.0`` to disable pruning (SI smoothing uses the
        ``a > 0`` mask only, no magnitude prune).
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

    # =========================================================================
    # Tree search recursion (SI)
    # =========================================================================

    @_nb.njit(cache=True)
    def _si_recurse(
        stm,
        O_pos, O_res, O_hill,
        P_pos_prior, P_ctx_prior,
        A_pos_flat, A_res_flat, A_hill_flat,
        a_resource,
        y_pos, y_resource, y_hill,
        B_pos, bb_ctx,
        w_novelty, w_learning, w_epistemic, pref_iprec,
        t, N, t_food, t_water, t_sleep, true_t,
        novelty_on, epistemic_on,
        node_count,           # 1-element array (mutable counter)
        memory_hits,          # 1-element array
        memory_misses,        # 1-element array
        A_resource_full,      # full A_resource (for likely_state expansion)
    ):
        node_count[0] += 1

        G = 0.02

        P_pos, P_ctx = _calculate_posterior(
            P_pos_prior, P_ctx_prior,
            y_resource, y_hill, O_res, O_hill,
        )

        t_food_idx = _index_clip(int(round(float(t_food))) + 1)
        t_water_idx = _index_clip(int(round(float(t_water))) + 1)
        t_sleep_idx = _index_clip(int(round(float(t_sleep))) + 1)

        if t > true_t:
            if novelty_on and w_novelty != 0.0:
                novelty = _novelty_si(a_resource, O_res, P_pos, P_ctx, w_learning)
                G += w_novelty * novelty
            if epistemic_on and w_epistemic != 0.0:
                epi = _G_epistemic(A_pos_flat, A_res_flat, A_hill_flat,
                                    P_pos_prior, P_ctx_prior)
                G += w_epistemic * epi

            C = _determine_observation_preference(t_food, t_water, t_sleep, pref_iprec)
            extrinsic = O_res[0] * C[0] + O_res[1] * C[1] + O_res[2] * C[2] + O_res[3] * C[3]
            G += extrinsic

            t_food = int(round((t_food + 1) * (1.0 - O_res[1])))
            t_water = int(round((t_water + 1) * (1.0 - O_res[2])))
            t_sleep = int(round((t_sleep + 1) * (1.0 - O_res[3])))
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
                # Use BLAS-backed @ via np.dot (Numba lowers this to gemv).
                Q_pos_a = np.dot(B_pos[:, :, action], P_pos)
                Q_ctx_a = np.dot(bb_ctx[:, :, 0], P_ctx)

                # qs = column-major outer-product flatten — equivalent to
                # joint_state index = pos + n_states * ctx.
                qs = np.empty(n_joint, dtype=np.float64)
                idx = 0
                for c in range(n_ctx):
                    pcv = Q_ctx_a[c]
                    for p in range(n_states):
                        qs[idx] = Q_pos_a[p] * pcv
                        idx += 1

                threshold = 1.0 / 8.0
                count_likely = 0
                for k in range(n_joint):
                    if qs[k] > threshold:
                        count_likely += 1
                if count_likely == 0:
                    eps = 1.0 / (n_joint * n_joint)
                    threshold = 1.0 / n_joint - eps

                action_fe = 0.0
                for state in range(n_joint):
                    if qs[state] <= threshold:
                        continue
                    cache = stm[t_food_idx, t_water_idx, t_sleep_idx, state]
                    if cache != 0.0:
                        K_state = cache
                        memory_hits[0] += 1
                    else:
                        O_pos_n = _imagined_obs_modality(y_pos, state, n_states)
                        O_res_n = _imagined_obs_modality(y_resource, state, n_states)
                        O_hill_n = _imagined_obs_modality(y_hill, state, n_states)
                        G_child, _ = _si_recurse(
                            stm,
                            O_pos_n, O_res_n, O_hill_n,
                            Q_pos_a, Q_ctx_a,
                            A_pos_flat, A_res_flat, A_hill_flat,
                            a_resource,
                            y_pos, y_resource, y_hill,
                            B_pos, bb_ctx,
                            w_novelty, w_learning, w_epistemic, pref_iprec,
                            t + 1, N, t_food, t_water, t_sleep, true_t,
                            novelty_on, epistemic_on,
                            node_count, memory_hits, memory_misses,
                            A_resource_full,
                        )
                        K_state = G_child
                        stm[t_food_idx, t_water_idx, t_sleep_idx, state] = G_child
                        memory_misses[0] += 1
                    action_fe += K_state * qs[state]

                efe[action] = 0.7 * action_fe

            # argmax (deterministic tie-break: returns smallest index)
            best_action = 0
            best_val = efe[0]
            for k in range(1, n_actions):
                if efe[k] > best_val:
                    best_val = efe[k]
                    best_action = k
            G += best_val

        return G, best_action

    # =========================================================================
    # Tree search recursion (SI smooth — windowed novelty)
    # =========================================================================

    @_nb.njit(cache=True)
    def _si_smooth_recurse(
        stm,
        hist_O_res, hist_O_hill, hist_P_pos, hist_P_ctx,
        a_resource,                       # CONSTANT real a (never mutated)
        A_pos_flat, A_res_flat, A_hill_flat, A_hill,
        y_pos, y_resource, y_hill,
        B_pos, bb_ctx,
        w_novelty, w_learning, w_epistemic, pref_iprec,
        t, N, t_food, t_water, t_sleep, true_t,
        novelty_on, epistemic_on,
        node_count, memory_hits, memory_misses,
    ):
        """JIT port of tree_search_frwd_SI_smooth.m.

        Same shape as :func:`_si_recurse`, but the novelty term is summed over
        a backward-smoothed t-6..t window. ``a_resource`` is held constant
        (SI does not imaginarily learn), so it is passed unchanged to every
        child — there is no a/y threading as in the SL JIT recursion.
        """
        node_count[0] += 1
        G = 0.02

        P_pos_prior = hist_P_pos[t].copy()
        P_ctx_prior = hist_P_ctx[t].copy()

        P_pos, P_ctx = _calculate_posterior(
            P_pos_prior, P_ctx_prior,
            y_resource, y_hill,
            hist_O_res[t], hist_O_hill[t],
        )

        # Posterior writeback (MATLAB line 29): deeper recursions' spm_backwards
        # reads P{timey, ...} for timey < t and must see the posterior.
        for s_ in range(P_pos.shape[0]):
            hist_P_pos[t, s_] = P_pos[s_]
        for c_ in range(P_ctx.shape[0]):
            hist_P_ctx[t, c_] = P_ctx[c_]

        t_food_idx = _index_clip(int(round(float(t_food))) + 1)
        t_water_idx = _index_clip(int(round(float(t_water))) + 1)
        t_sleep_idx = _index_clip(int(round(float(t_sleep))) + 1)

        if t > true_t:
            if novelty_on and w_novelty != 0.0:
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
                    # Reuse the single-step novelty term: a_prior is the
                    # constant real a_resource each window step (SI does not
                    # imaginarily learn), and _novelty_si masks by a>0 with no
                    # prune. Same algebra as the NumPy _novelty_si_smooth, so
                    # the two paths agree to FP precision.
                    novelty_total += _novelty_si(
                        a_resource, O_res_t, P_pos_t, L_ctx, w_learning,
                    )
                G += w_novelty * novelty_total

            if epistemic_on and w_epistemic != 0.0:
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
                        O_res_n = _imagined_obs_modality(y_resource, state, n_states)
                        O_hill_n = _imagined_obs_modality(y_hill, state, n_states)
                        for o in range(hist_O_res.shape[1]):
                            hist_O_res[t + 1, o] = O_res_n[o]
                        for o in range(hist_O_hill.shape[1]):
                            hist_O_hill[t + 1, o] = O_hill_n[o]
                        for s_ in range(n_states):
                            hist_P_pos[t + 1, s_] = Q_pos_a[s_]
                        for c_ in range(n_ctx):
                            hist_P_ctx[t + 1, c_] = Q_ctx_a[c_]

                        G_child, _ = _si_smooth_recurse(
                            stm,
                            hist_O_res, hist_O_hill, hist_P_pos, hist_P_ctx,
                            a_resource,
                            A_pos_flat, A_res_flat, A_hill_flat, A_hill,
                            y_pos, y_resource, y_hill,
                            B_pos, bb_ctx,
                            w_novelty, w_learning, w_epistemic, pref_iprec,
                            t + 1, N, t_food, t_water, t_sleep, true_t,
                            novelty_on, epistemic_on,
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


def tree_search_si_jit_run(
    short_term_memory: np.ndarray,
    O_pos_root: np.ndarray, O_res_root: np.ndarray, O_hill_root: np.ndarray,
    P_pos_root: np.ndarray, P_ctx_root: np.ndarray,
    inputs,                # PlannerInputs (dataclass)
    t: int, N: int,
    t_food: int, t_water: int, t_sleep: int,
    true_t: int,
    novelty_on: bool = True, epistemic_on: bool = True,
    smoothing_on: bool = False,
    history_O_resource: Optional[list] = None,
    history_O_hill: Optional[list] = None,
    history_P_pos: Optional[list] = None,
    history_P_ctx: Optional[list] = None,
) -> PlanResult:
    """JIT entry point with the same return type as :func:`tree_search_si`.

    Falls back to the NumPy planner if numba is unavailable. When
    ``smoothing_on`` and novelty are both active, dispatches to the windowed
    -novelty recursion (``tree_search_frwd_SI_smooth.m``); otherwise the
    single-step recursion (the smooth/non-smooth trees are identical when
    novelty is off).
    """
    if not _HAS_NUMBA:
        from .si import tree_search_si as _np_si
        return _np_si(short_term_memory, O_pos_root, O_res_root, O_hill_root,
                      P_pos_root, P_ctx_root, inputs,
                      t, N, t_food, t_water, t_sleep, true_t,
                      novelty_on=novelty_on, epistemic_on=epistemic_on,
                      smoothing_on=smoothing_on,
                      history_O_resource=history_O_resource,
                      history_O_hill=history_O_hill,
                      history_P_pos=history_P_pos,
                      history_P_ctx=history_P_ctx)

    # Pre-flatten A modalities to F-order linear index for the JIT epistemic.
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

    if smoothing_on and novelty_on and inputs.weights.novelty != 0:
        n_states = inputs.A_pos.shape[0]
        n_ctx = inputs.A_resource.shape[2]
        T_max = N + 2  # safe upper bound for imagined trajectory writes
        hist_O_res = np.zeros((T_max, 4), dtype=np.float64)
        hist_O_hill = np.zeros((T_max, 5), dtype=np.float64)
        hist_P_pos = np.zeros((T_max, n_states), dtype=np.float64)
        hist_P_ctx = np.zeros((T_max, n_ctx), dtype=np.float64)

        # Populate the real-trial history portion (indices 0..t-1; the root
        # slot t is overwritten just below).
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

        hist_O_res[t] = np.asarray(O_res_root, dtype=np.float64).ravel()
        hist_O_hill[t] = np.asarray(O_hill_root, dtype=np.float64).ravel()
        hist_P_pos[t] = np.asarray(P_pos_root, dtype=np.float64).ravel()
        hist_P_ctx[t] = np.asarray(P_ctx_root, dtype=np.float64).ravel()

        G, best_action = _si_smooth_recurse(
            short_term_memory,
            hist_O_res, hist_O_hill, hist_P_pos, hist_P_ctx,
            np.ascontiguousarray(inputs.a_resource, dtype=np.float64),
            A_pos_flat, A_res_flat, A_hill_flat,
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
            int(t), int(N),
            int(t_food), int(t_water), int(t_sleep), int(true_t),
            bool(novelty_on), bool(epistemic_on),
            node_count, memory_hits, memory_misses,
        )
        return PlanResult(
            G=float(G),
            best_actions=[int(best_action)] if best_action >= 0 else [],
            memory_hits=int(memory_hits[0]),
            memory_misses=int(memory_misses[0]),
            node_count=int(node_count[0]),
        )

    G, best_action = _si_recurse(
        short_term_memory,
        np.ascontiguousarray(O_pos_root, dtype=np.float64).ravel(),
        np.ascontiguousarray(O_res_root, dtype=np.float64).ravel(),
        np.ascontiguousarray(O_hill_root, dtype=np.float64).ravel(),
        np.ascontiguousarray(P_pos_root, dtype=np.float64).ravel(),
        np.ascontiguousarray(P_ctx_root, dtype=np.float64).ravel(),
        A_pos_flat, A_res_flat, A_hill_flat,
        np.ascontiguousarray(inputs.a_resource, dtype=np.float64),
        np.ascontiguousarray(inputs.y_pos, dtype=np.float64),
        np.ascontiguousarray(inputs.y_resource, dtype=np.float64),
        np.ascontiguousarray(inputs.y_hill, dtype=np.float64),
        np.ascontiguousarray(inputs.B_pos, dtype=np.float64),
        np.ascontiguousarray(inputs.bb_ctx, dtype=np.float64),
        float(inputs.weights.novelty),
        float(inputs.weights.learning),
        float(inputs.weights.epistemic),
        float(inputs.weights.preference_inverse_precision),
        int(t), int(N),
        int(t_food), int(t_water), int(t_sleep), int(true_t),
        bool(novelty_on), bool(epistemic_on),
        node_count, memory_hits, memory_misses,
        np.ascontiguousarray(inputs.A_resource, dtype=np.float64),
    )

    return PlanResult(
        G=float(G),
        best_actions=[int(best_action)] if best_action >= 0 else [],
        memory_hits=int(memory_hits[0]),
        memory_misses=int(memory_misses[0]),
        node_count=int(node_count[0]),
    )

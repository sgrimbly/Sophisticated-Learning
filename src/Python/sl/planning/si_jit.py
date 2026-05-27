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


def tree_search_si_jit_run(
    short_term_memory: np.ndarray,
    O_pos_root: np.ndarray, O_res_root: np.ndarray, O_hill_root: np.ndarray,
    P_pos_root: np.ndarray, P_ctx_root: np.ndarray,
    inputs,                # PlannerInputs (dataclass)
    t: int, N: int,
    t_food: int, t_water: int, t_sleep: int,
    true_t: int,
    novelty_on: bool = True, epistemic_on: bool = True,
) -> PlanResult:
    """JIT entry point with the same return type as :func:`tree_search_si`.

    Falls back to the NumPy planner if numba is unavailable.
    """
    if not _HAS_NUMBA:
        from .si import tree_search_si as _np_si
        return _np_si(short_term_memory, O_pos_root, O_res_root, O_hill_root,
                      P_pos_root, P_ctx_root, inputs,
                      t, N, t_food, t_water, t_sleep, true_t,
                      novelty_on=novelty_on, epistemic_on=epistemic_on)

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

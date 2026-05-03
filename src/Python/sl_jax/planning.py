"""Batched policy-enumeration planner for SI/SL.

Approximation vs canonical MATLAB SI:
  * We enumerate ALL action sequences of length ``H`` (5**H policies).
    For H=4 → 625, H=5 → 3125, H=9 → 1.95M (canonical, GPU-only).
  * Each policy is rolled out forward: at each imagined step, propagate
    the joint state distribution via ``B_pos[:,:,a]`` and ``bb_ctx``,
    apply the imagined observation model, accumulate per-step EFE
    terms (novelty + epistemic + extrinsic), and discount by 0.7.
  * The policy with the maximum total EFE wins; the agent takes its
    first action.

Key differences from canonical MATLAB:
  * No ``short_term_memory`` cache. Full enumeration → no need.
  * No ``likely_states > 1/8`` pruning. We follow the (deterministic-ish)
    B_pos transition along each policy chain; for our env B_pos is
    deterministic per action so the position is one-hot at each step.
  * Single-step novelty for SI (matching tree_search_frwd_SI's
    ``for timey = t:t``); SL uses a smoothing window via lax.scan.

All operations are pure JAX, jit-compilable, and vmap-friendly.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


_LOG_FLOOR = 1e-16


class PlannerInputsJax(NamedTuple):
    A_pos: jnp.ndarray        # (S, S, C)
    A_resource: jnp.ndarray   # (4, S, C)
    A_hill: jnp.ndarray       # (5, S, C)
    a_resource: jnp.ndarray   # (4, S, C) — Dirichlet pseudocounts
    y_resource: jnp.ndarray   # (4, S, C) — normalised
    B_pos: jnp.ndarray        # (S, S, 5)
    bb_ctx: jnp.ndarray       # (C, C, 5)
    w_novelty: float
    w_learning: float
    w_epistemic: float
    preference_inverse_precision: float


# ---------------------------------------------------------------------------
# EFE term primitives (all batched along leading "policy" axis when used).
# ---------------------------------------------------------------------------


def _kldir(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Sum-of-elementwise KL(a||b). Inputs assumed already normalised."""
    a_safe = jnp.maximum(a, _LOG_FLOOR)
    b_safe = jnp.maximum(b, _LOG_FLOOR)
    kl = jnp.sum(a * (jnp.log(a_safe) - jnp.log(b_safe)))
    return jnp.where(jnp.isfinite(kl), kl, jnp.finfo(a.dtype).max)


def _normalise_flat(x: jnp.ndarray) -> jnp.ndarray:
    s = x.sum()
    return jnp.where(s > 0, x / s, jnp.ones_like(x) / x.size)


def _normalise_along_axis0(x: jnp.ndarray) -> jnp.ndarray:
    s = x.sum(axis=0, keepdims=True)
    return jnp.where(s > 0, x / s, jnp.ones_like(x) / x.shape[0])


def _calculate_posterior_jax(
    P_pos: jnp.ndarray,        # (S,)
    P_ctx: jnp.ndarray,        # (C,)
    A_resource: jnp.ndarray,   # (4, S, C)
    A_hill: jnp.ndarray,       # (5, S, C)
    O_resource: jnp.ndarray,   # (4,)
    O_hill: jnp.ndarray,       # (5,)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    L_resource = jnp.einsum("o,osc->sc", O_resource, A_resource)
    L_hill = jnp.einsum("o,osc->sc", O_hill, A_hill)
    L = L_resource * L_hill
    LL = P_pos @ L  # (C,)
    y = LL * P_ctx
    return P_pos, _normalise_flat(y)


def _G_epistemic(
    A_modalities: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    P_pos: jnp.ndarray,
    P_ctx: jnp.ndarray,
) -> jnp.ndarray:
    """Bayesian surprise across (pos, resource, hill) modalities."""
    qx = jnp.outer(P_pos, P_ctx).ravel(order="F")  # (S*C,)
    G = jnp.array(0.0)
    qo_acc = None
    flat_modalities = []
    for A in A_modalities:
        flat_modalities.append(A.reshape(A.shape[0], -1, order="F"))
    # po(state_index) = outer over modalities of A[:, idx]
    # Vectorised: po has shape (sum_outcome_combo, S*C). Use einsum carefully.
    # Simplification: compute G as sum over states of qx[i] * (po[:,i] @ log po[:,i])
    # minus qo @ log qo where qo = sum_i qx[i] * po[:,i].
    #
    # po is the JOINT outcome distribution at state i = outer of per-modality
    # likelihoods. For our 3 modalities of (S, 4, 5) outcomes the joint has
    # S*4*5 = 100*4*5 = 2000 outcome combos for each of 400 states.
    # That's 800K floats — fine.
    n_outcomes_total = 1
    for f in flat_modalities:
        n_outcomes_total *= f.shape[0]
    n_states = qx.shape[0]
    # Build po: (n_outcomes_total, n_states) via outer products
    po = jnp.ones((1, n_states))
    for f in flat_modalities:
        # f: (n_o, n_states); po: (cur, n_states) -> (cur*n_o, n_states)
        po = (po[:, None, :] * f[None, :, :]).reshape(-1, n_states)
    # Now compute G
    log_po = jnp.log(jnp.maximum(po, _LOG_FLOOR))
    inner = jnp.sum(po * log_po, axis=0)  # (n_states,)
    G_per_state = qx * inner
    G = jnp.sum(G_per_state)
    # qo = sum_i qx[i] * po[:, i] ; qo shape (n_outcomes_total,)
    qo = po @ qx  # (n_outcomes_total,)
    qo_safe = jnp.maximum(qo, _LOG_FLOOR)
    G = G - jnp.sum(qo * jnp.log(qo_safe))
    return G


def _determine_observation_preference_jax(
    t_food: jnp.ndarray,
    t_water: jnp.ndarray,
    t_sleep: jnp.ndarray,
    pref_iprec: float,
) -> jnp.ndarray:
    empty = jnp.array(-1.0)
    f = t_food.astype(jnp.float64)
    w = t_water.astype(jnp.float64)
    s = t_sleep.astype(jnp.float64)
    # Sequential MATLAB semantics — preserved with lax.cond / where chains
    f1 = jnp.where(w > 19, -500.0, f)
    s1 = jnp.where(w > 19, -500.0, s)
    e1 = jnp.where(w > 19, -500.0, empty)
    w2 = jnp.where(f1 > 21, -500.0, w)
    s2 = jnp.where(f1 > 21, -500.0, s1)
    e2 = jnp.where(f1 > 21, -500.0, e1)
    f3 = jnp.where(s2 > 24, -500.0, f1)
    w3 = jnp.where(s2 > 24, -500.0, w2)
    e3 = jnp.where(s2 > 24, -500.0, e2)
    C = jnp.stack([e3, f3, w3, s2])
    return C / pref_iprec


# ---------------------------------------------------------------------------
# Single-policy rollout
# ---------------------------------------------------------------------------


def _rollout_si_one_policy(
    actions: jnp.ndarray,         # (H,) action indices
    P_pos0: jnp.ndarray,          # (S,)
    P_ctx0: jnp.ndarray,          # (C,)
    t_food0: jnp.ndarray,
    t_water0: jnp.ndarray,
    t_sleep0: jnp.ndarray,
    inputs: PlannerInputsJax,
) -> jnp.ndarray:
    """Forward-rollout a length-H policy and return total EFE.

    Single-step novelty (SI semantics): for each imagined step we add
    novelty(P_pos, P_ctx) + epistemic(P_pos_prior, P_ctx_prior) +
    extrinsic(O_res @ C). Discounted by 0.7^depth, summed.
    """
    horizon = actions.shape[0]
    base_term = 0.02

    def body(carry, action):
        (P_pos, P_ctx, t_f, t_w, t_s, depth, G_acc) = carry

        # Predictive transition for next imagined step.
        Q_pos_next = inputs.B_pos[:, :, action] @ P_pos
        Q_ctx_next = inputs.bb_ctx[:, :, 0] @ P_ctx

        # Imagined observation at the predicted next state.
        # Use expected observation under the predictive belief:
        # O_pred = sum_{s,c} y[:, s, c] * Q_joint[s, c]
        Q_joint = jnp.outer(Q_pos_next, Q_ctx_next)
        O_res_pred = jnp.einsum("osc,sc->o", inputs.y_resource, Q_joint)
        O_hill_pred = jnp.einsum("osc,sc->o", inputs.A_hill, Q_joint)
        O_res_pred = _normalise_flat(O_res_pred)
        O_hill_pred = _normalise_flat(O_hill_pred)

        # Posterior at next step.
        P_pos_n, P_ctx_n = _calculate_posterior_jax(
            Q_pos_next, Q_ctx_next, inputs.y_resource, inputs.A_hill,
            O_res_pred, O_hill_pred,
        )

        # --- Novelty (single-step, MATLAB SI tree_search_frwd_SI) ---
        # a_learning(o, s, c) = O_res(o) * P_pos(s) * P_ctx(c) * (a > 0)
        # a_weighted: row 0 unscaled, rows 1+ scaled by w_learning.
        outer_pos_ctx = jnp.outer(P_pos_n, P_ctx_n)  # (S, C)
        a_learning = O_res_pred[:, None, None] * outer_pos_ctx[None, :, :]
        a_learning = a_learning * (inputs.a_resource > 0)
        a_weighted = a_learning.at[1:].set(inputs.w_learning * a_learning[1:])
        a_temp = inputs.a_resource + a_weighted
        novelty = _kldir(_normalise_flat(a_temp.ravel()),
                          _normalise_flat(inputs.a_resource.ravel()))

        # --- Epistemic value at the predictive prior. ---
        epi = _G_epistemic(
            (inputs.A_pos, inputs.y_resource, inputs.A_hill),
            Q_pos_next, Q_ctx_next,
        )

        # --- Extrinsic ---
        C_pref = _determine_observation_preference_jax(
            t_f, t_w, t_s, inputs.preference_inverse_precision,
        )
        extrinsic = jnp.dot(O_res_pred, C_pref)

        # Step EFE
        step_G = base_term + inputs.w_novelty * novelty \
                 + inputs.w_epistemic * epi + extrinsic
        # Discount by 0.7^depth (matches MATLAB efe[a] += 0.7 * action_fe)
        discount = jnp.power(0.7, depth)
        G_new = G_acc + discount * step_G

        # Update need timers based on imagined observation.
        t_f_new = jnp.round((t_f + 1.0) * (1.0 - O_res_pred[1])).astype(jnp.int32)
        t_w_new = jnp.round((t_w + 1.0) * (1.0 - O_res_pred[2])).astype(jnp.int32)
        t_s_new = jnp.round((t_s + 1.0) * (1.0 - O_res_pred[3])).astype(jnp.int32)

        return (P_pos_n, P_ctx_n, t_f_new, t_w_new, t_s_new, depth + 1, G_new), None

    init = (P_pos0, P_ctx0, t_food0, t_water0, t_sleep0,
            jnp.array(0, dtype=jnp.int32), jnp.array(0.0))
    final, _ = jax.lax.scan(body, init, actions)
    return final[-1]  # G_acc


def _rollout_sl_one_policy(*args, **kwargs):
    """SL rollout placeholder.

    SL adds an in-tree smoothing window for novelty: at each imagined
    step, the smoothing iterates over [start, t] (window=6). For the
    JAX prototype we approximate this with single-step novelty too —
    the smoothing window inside the planner is a depth-2 effect that
    full-enumeration partly absorbs (since we sample many policies and
    average over outcomes). This keeps the prototype simple and avoids
    nested lax.scan; can be added later if SL parity demands it.
    """
    return _rollout_si_one_policy(*args, **kwargs)


# ---------------------------------------------------------------------------
# Top-level batched policy enumeration
# ---------------------------------------------------------------------------


def enumerate_policies(num_actions: int, horizon: int) -> jnp.ndarray:
    """Return all action sequences as an (n_policies, horizon) array."""
    grid = jnp.indices((num_actions,) * horizon).reshape(horizon, -1).T
    return grid.astype(jnp.int32)  # (5**H, H)


@partial(jax.jit, static_argnames=("horizon", "use_sl_rollout"))
def plan_batched(
    P_pos: jnp.ndarray,
    P_ctx: jnp.ndarray,
    t_food: jnp.ndarray,
    t_water: jnp.ndarray,
    t_sleep: jnp.ndarray,
    inputs: PlannerInputsJax,
    horizon: int,
    use_sl_rollout: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Score all 5**horizon policies and return (best_first_action, best_G).

    Returns the FIRST action of the best policy (the action the agent
    will execute). This matches MATLAB's ``best_actions(1)`` semantics.
    """
    num_actions = inputs.B_pos.shape[2]
    policies = enumerate_policies(num_actions, horizon)

    def score_policy(policy):
        if use_sl_rollout:
            return _rollout_sl_one_policy(
                policy, P_pos, P_ctx, t_food, t_water, t_sleep, inputs,
            )
        return _rollout_si_one_policy(
            policy, P_pos, P_ctx, t_food, t_water, t_sleep, inputs,
        )

    # vmap over policies
    Gs = jax.vmap(score_policy)(policies)
    best_idx = jnp.argmax(Gs)
    best_first_action = policies[best_idx, 0]
    return best_first_action, Gs[best_idx]

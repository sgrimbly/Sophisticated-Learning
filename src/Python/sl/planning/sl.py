"""Sophisticated Learning tree search.

Faithful port of ``src/MATLAB/tree-search/tree_search_frwd_SL.m``.

The two key differences from SI:

1. Novelty loop walks ``timey = max(0, t-6) .. t`` and uses the *historical*
   observation and position posterior at each ``timey`` (this was the
   single biggest bug in the prior Python attempt — it reused the current
   step's O and P_pos for every loop iteration, see equivalence doc §3.8.2).

2. The agent's ``a_resource`` is mutated *imaginarily* during the recursion
   — i.e., the planner explores futures in which it has already learned
   from the imagined trajectory. The mutation uses the *unweighted*
   pruned ``a_learning``; the KL is computed against the *weighted*
   ``a_temp``.

3. Optional ``adaptive_likelihood_in_plan``: when ``True``, refresh
   ``y_resource = normalise(a_resource_imag)`` after each update so future
   imagined observations reflect the imagined learning. Default ``False``
   (matches MATLAB default).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ..efe import G_epistemic_value, kldir, determine_observation_preference
from ..inference import (
    calculate_posterior, joint_pos_ctx, normalise, normalise_matrix_columns,
    spm_backwards,
)
from ..learning import planning_dirichlet_update
from .common import (
    PlannerInputs, PlanResult, imagined_observations, index_clip,
    likely_state_indices,
)


@dataclass
class _History:
    """Imagined trajectory buffers (one entry per imagined timestep).

    Each list grows as recursion descends; on backtrack the recursion
    truncates.  ``O_resource`` and ``O_hill`` and ``P_pos``/``P_ctx`` are
    indexed by absolute time index ``t`` (not depth).  We size them to
    ``max_t = N + 1`` and overwrite slots as the search descends.
    """
    O_resource: List[np.ndarray]
    O_hill: List[np.ndarray]
    P_pos: List[np.ndarray]
    P_ctx: List[np.ndarray]


def _ensure_size(hist: _History, t: int) -> None:
    """Pad each history list with None placeholders up to length ``t+1``."""
    n = max(t + 1, max(len(hist.O_resource), len(hist.O_hill),
                       len(hist.P_pos), len(hist.P_ctx)))
    for arr in (hist.O_resource, hist.O_hill, hist.P_pos, hist.P_ctx):
        while len(arr) < n:
            arr.append(None)  # type: ignore[arg-type]


def _novelty_sl(
    a_resource_imag: np.ndarray,
    hist: _History,
    A_hill: np.ndarray,
    bb_ctx: np.ndarray,
    P_pos_now: np.ndarray,
    P_ctx_now: np.ndarray,
    t: int,
    learning_weight: float,
    prune_threshold: float,
    adaptive_likelihood_in_plan: bool,
    y_resource: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """SL novelty loop (lines 44-89 of tree_search_frwd_SL.m).

    Returns ``(novelty_total, a_resource_imag_updated, y_resource_maybe_updated)``.
    """
    start = max(0, t - 6)
    novelty = 0.0
    a_imag = a_resource_imag.copy()

    for timey in range(start, t + 1):
        if timey != t:
            # Use historical posterior at timey as starting context belief,
            # smoothed forward to current timestep.
            L_ctx = spm_backwards(
                hist.O_hill, hist.P_pos, hist.P_ctx[timey],
                A_hill, bb_ctx, timey, t,
            )
        else:
            L_ctx = P_ctx_now

        P_pos_t = hist.P_pos[timey] if hist.P_pos[timey] is not None else P_pos_now
        O_res_t = hist.O_resource[timey]
        if O_res_t is None:
            # No history at this depth — skip (shouldn't happen if root pre-fills)
            continue

        a_prior = a_imag.copy()
        a_imag, _, a_weighted = planning_dirichlet_update(
            a_imag, O_res_t, P_pos_t, L_ctx,
            learning_weight=learning_weight,
            prune_threshold=prune_threshold,
        )

        a_temp = a_prior + a_weighted
        novelty += kldir(normalise(a_temp.ravel()), normalise(a_prior.ravel()))

    if adaptive_likelihood_in_plan:
        y_resource = normalise_matrix_columns(a_imag)

    return novelty, a_imag, y_resource


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def tree_search_sl(
    short_term_memory: np.ndarray,
    O_pos_root: np.ndarray,
    O_resource_root: np.ndarray,
    O_hill_root: np.ndarray,
    P_pos_root: np.ndarray,
    P_ctx_root: np.ndarray,
    inputs: PlannerInputs,
    t: int,
    N: int,
    t_food: int,
    t_water: int,
    t_sleep: int,
    true_t: int,
    *,
    novelty_on: bool = True,
    epistemic_on: bool = True,
    smoothing_on: bool = True,
    adaptive_likelihood_in_plan: bool = False,
    learning_prune_threshold: float = 0.2,
    history_O_resource: List[np.ndarray] = None,
    history_O_hill: List[np.ndarray] = None,
    history_P_pos: List[np.ndarray] = None,
    history_P_ctx: List[np.ndarray] = None,
) -> PlanResult:
    """SL planner.

    The history buffers are the *real-trial* histories up to (and including)
    ``true_t``, supplied by :mod:`sl.agent`. Inside recursion we extend
    them with the imagined trajectory.
    """
    # Build a private history (deep-copied) so recursion can extend without
    # corrupting the caller's record.
    hist = _History(
        O_resource=list(history_O_resource or []),
        O_hill=list(history_O_hill or []),
        P_pos=list(history_P_pos or []),
        P_ctx=list(history_P_ctx or []),
    )

    # Make sure the slot for the root timestep is populated. This mirrors
    # MATLAB SI/SL where O{:,t} and Q{t,:} exist before the planner runs.
    _ensure_size(hist, t)
    hist.O_resource[t] = O_resource_root
    hist.O_hill[t] = O_hill_root
    hist.P_pos[t] = P_pos_root
    hist.P_ctx[t] = P_ctx_root

    rec = _RecState(
        inputs=inputs, true_t=true_t,
        novelty_on=novelty_on, epistemic_on=epistemic_on,
        smoothing_on=smoothing_on,
        adaptive=adaptive_likelihood_in_plan,
        prune=learning_prune_threshold,
        memory_hits=0, memory_misses=0, node_count=0,
        best_actions=[],
        a_resource_imag=inputs.a_resource.copy(),
        y_resource=inputs.y_resource.copy(),
    )
    G = _node(short_term_memory, hist, rec, t, N, t_food, t_water, t_sleep)
    return PlanResult(
        G=float(np.max(G)),
        best_actions=rec.best_actions,
        memory_hits=rec.memory_hits,
        memory_misses=rec.memory_misses,
        node_count=rec.node_count,
    )


# ---------------------------------------------------------------------------
# Recursion body
# ---------------------------------------------------------------------------


class _RecState:
    __slots__ = ("inputs", "true_t", "novelty_on", "epistemic_on",
                 "smoothing_on", "adaptive", "prune",
                 "memory_hits", "memory_misses", "node_count",
                 "best_actions", "a_resource_imag", "y_resource")

    def __init__(self, inputs, true_t, novelty_on, epistemic_on, smoothing_on,
                 adaptive, prune, memory_hits, memory_misses, node_count,
                 best_actions, a_resource_imag, y_resource):
        self.inputs = inputs
        self.true_t = true_t
        self.novelty_on = novelty_on
        self.epistemic_on = epistemic_on
        self.smoothing_on = smoothing_on
        self.adaptive = adaptive
        self.prune = prune
        self.memory_hits = memory_hits
        self.memory_misses = memory_misses
        self.node_count = node_count
        self.best_actions = best_actions
        self.a_resource_imag = a_resource_imag
        self.y_resource = y_resource


def _node(stm, hist, rec, t, N, t_food, t_water, t_sleep):
    inp = rec.inputs
    rec.node_count += 1
    G = 0.02

    P_pos_prior = hist.P_pos[t]
    P_ctx_prior = hist.P_ctx[t]

    P_pos, P_ctx = calculate_posterior(
        P_pos_prior, P_ctx_prior,
        rec.y_resource, inp.y_hill,
        hist.O_resource[t], hist.O_hill[t],
    )

    # MATLAB tree_search_frwd_SL line 29 writes the posterior back into the
    # P cell in-place. Deeper recursions' smoothing block reads back
    # P{timey, 2} for timey < current_t and gets the just-updated posteriors,
    # not the parent-stored predictive Q. Without this writeback the smoothing
    # window's spm_backwards is seeded from stale predictive context, which
    # under-credits learning at past timesteps. Mirrors the agent.py outer
    # loop's Q_history posterior overwrite.
    hist.P_pos[t] = P_pos
    hist.P_ctx[t] = P_ctx

    t_food_idx = index_clip(int(round(t_food)) + 1)
    t_water_idx = index_clip(int(round(t_water)) + 1)
    t_sleep_idx = index_clip(int(round(t_sleep)) + 1)

    if t > rec.true_t:
        # Novelty (with imagined learning across smoothing window)
        novelty_val = 0.0
        a_imag_pre = rec.a_resource_imag
        if rec.novelty_on and inp.weights.novelty != 0:
            if rec.smoothing_on:
                novelty_val, rec.a_resource_imag, rec.y_resource = _novelty_sl(
                    rec.a_resource_imag, hist,
                    inp.A_hill, inp.bb_ctx,
                    P_pos, P_ctx, t,
                    learning_weight=inp.weights.learning,
                    prune_threshold=rec.prune,
                    adaptive_likelihood_in_plan=rec.adaptive,
                    y_resource=rec.y_resource,
                )
            else:
                # SL_noSmooth path: single-step novelty using current node only
                a_prior = rec.a_resource_imag.copy()
                rec.a_resource_imag, _, a_weighted = planning_dirichlet_update(
                    rec.a_resource_imag, hist.O_resource[t],
                    P_pos, P_ctx,
                    learning_weight=inp.weights.learning,
                    prune_threshold=rec.prune,
                )
                a_temp = a_prior + a_weighted
                novelty_val = kldir(
                    normalise(a_temp.ravel()),
                    normalise(a_prior.ravel()),
                )
                if rec.adaptive:
                    rec.y_resource = normalise_matrix_columns(rec.a_resource_imag)

            G += inp.weights.novelty * novelty_val

        if rec.epistemic_on and inp.weights.epistemic != 0:
            epi = G_epistemic_value(
                [inp.y_pos, rec.y_resource, inp.y_hill],
                [P_pos_prior, P_ctx_prior],
            )
            G += inp.weights.epistemic * epi

        C = determine_observation_preference(
            t_food, t_water, t_sleep,
            inp.weights.preference_inverse_precision,
        )
        G += float(hist.O_resource[t] @ C)

        t_food = int(round((t_food + 1) * (1 - float(hist.O_resource[t][1]))))
        t_water = int(round((t_water + 1) * (1 - float(hist.O_resource[t][2]))))
        t_sleep = int(round((t_sleep + 1) * (1 - float(hist.O_resource[t][3]))))
        t_food_idx = index_clip(t_food + 1)
        t_water_idx = index_clip(t_water + 1)
        t_sleep_idx = index_clip(t_sleep + 1)

    if t < N:
        S = inp.A_pos.shape[0]
        C_ = inp.A_resource.shape[2]
        K = np.zeros(S * C_, dtype=np.float64)
        efe = np.zeros(inp.B_pos.shape[2], dtype=np.float64)

        for action in range(inp.B_pos.shape[2]):
            Q_pos_a = inp.B_pos[:, :, action] @ P_pos
            Q_ctx_a = inp.bb_ctx[:, :, 0] @ P_ctx
            qs = joint_pos_ctx(Q_pos_a, Q_ctx_a)
            likely = likely_state_indices(qs, threshold=1.0 / 8.0)

            for state in likely:
                cache = stm[t_food_idx, t_water_idx, t_sleep_idx, state]
                if cache != 0.0:
                    K[state] = cache
                    rec.memory_hits += 1
                else:
                    Op, Or, Oh = imagined_observations(
                        inp.y_pos, rec.y_resource, inp.y_hill,
                        int(state), S,
                    )
                    _ensure_size(hist, t + 1)
                    hist.O_resource[t + 1] = Or
                    hist.O_hill[t + 1] = Oh
                    hist.P_pos[t + 1] = Q_pos_a
                    hist.P_ctx[t + 1] = Q_ctx_a

                    G_child = _node(stm, hist, rec, t + 1, N, t_food, t_water, t_sleep)
                    K[state] = G_child
                    stm[t_food_idx, t_water_idx, t_sleep_idx, state] = G_child
                    rec.memory_misses += 1

            efe[action] += 0.7 * float(K[likely] @ qs[likely])

        best_action = int(np.argmax(efe))
        G += float(efe[best_action])
        rec.best_actions.insert(0, best_action)

    return G

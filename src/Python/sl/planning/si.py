"""Sophisticated Inference tree search.

Faithful port of ``src/MATLAB/tree-search/tree_search_frwd_SI.m``.

Differences from the prior Python attempt this fixes:
  * No sampling inside ``calculate_posterior``/``spm_backwards`` — they take
    full distributions.
  * Need-timers are updated *after* the extrinsic term is added (lines
    93-114 of MATLAB), not before.
  * Action loop is deterministic ``range(5)``, not ``np.random.permutation``.
  * Memory counter increments on cache *hits*, matching MATLAB.
  * Learning-update KL uses the row-0-penalty-aware ``a_temp`` per
    learning.real_dirichlet_update style, but the imagined ``a`` is *not*
    mutated in SI — only in SL (see :mod:`sl.planning.sl`).
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..efe import G_epistemic_value, kldir, determine_observation_preference
from ..inference import (
    calculate_posterior, joint_pos_ctx, normalise, spm_cross,
)
from .common import (
    PlannerInputs, PlanResult, imagined_observations, index_clip,
    likely_state_indices,
)


def _novelty_si(
    a_resource: np.ndarray,
    O_resource_t: np.ndarray,
    P_pos_t: np.ndarray,
    P_ctx_t: np.ndarray,
    learning_weight: float,
) -> float:
    """SI novelty term: single-step (no smoothing window).

    Builds ``a_learning`` at the current node, applies the row-1+-scaling
    rule, then KL between normalised ``a_temp`` and ``a_prior``.
    """
    a_prior = a_resource
    L = spm_cross(O_resource_t, P_pos_t, P_ctx_t)
    a_learning = L * (a_prior > 0)

    a_weighted = a_learning.copy()
    a_weighted[1:, :, :] = learning_weight * a_learning[1:, :, :]
    # row 0 unscaled, already correct via copy

    a_temp = a_prior + a_weighted
    return kldir(normalise(a_temp.ravel()), normalise(a_prior.ravel()))


def tree_search_si(
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
    novelty_on: bool = True,
    epistemic_on: bool = True,
) -> PlanResult:
    """Top-level entry point.

    Parameters
    ----------
    short_term_memory : (35, 35, 35, num_joint_states) memoisation cache
                        (mutated in place; pass a fresh zeros for each call
                        from agent.run_trial since the agent resets between
                        steps when triggered).
    O_*_root : observation distributions at the root (one-hot during real
               trials; never used as posterior input directly — calculate
               posterior internally).
    P_pos_root, P_ctx_root : *predictive* posteriors at the root (the
               "prior" that calculate_posterior will fold the observation
               into).
    inputs : PlannerInputs bundle.
    t, N    : current and final imagined timesteps. The MATLAB call passes
              ``t = real_t`` and ``N = real_t + horizon``.
    true_t  : the actual real-trial timestep — used to gate "is this an
              imagined node or the root?" via ``t > true_t``.
    novelty_on, epistemic_on : flags from the algorithm-variant resolution.
    """
    inputs_recursion = _RecursionState(
        inputs=inputs,
        true_t=true_t,
        novelty_on=novelty_on,
        epistemic_on=epistemic_on,
        memory_hits=0,
        memory_misses=0,
        node_count=0,
        best_actions=[],
    )
    G = _tree_search_si_node(
        short_term_memory,
        O_pos_root, O_resource_root, O_hill_root,
        P_pos_root, P_ctx_root,
        inputs_recursion,
        t=t, N=N,
        t_food=t_food, t_water=t_water, t_sleep=t_sleep,
    )
    return PlanResult(
        G=float(np.max(G)),
        best_actions=inputs_recursion.best_actions,
        memory_hits=inputs_recursion.memory_hits,
        memory_misses=inputs_recursion.memory_misses,
        node_count=inputs_recursion.node_count,
    )


# ---------------------------------------------------------------------------
# Internal recursion state + body
# ---------------------------------------------------------------------------


class _RecursionState:
    __slots__ = ("inputs", "true_t", "novelty_on", "epistemic_on",
                 "memory_hits", "memory_misses", "node_count", "best_actions")

    def __init__(self, inputs: PlannerInputs, true_t: int,
                 novelty_on: bool, epistemic_on: bool,
                 memory_hits: int, memory_misses: int, node_count: int,
                 best_actions: List[int]):
        self.inputs = inputs
        self.true_t = true_t
        self.novelty_on = novelty_on
        self.epistemic_on = epistemic_on
        self.memory_hits = memory_hits
        self.memory_misses = memory_misses
        self.node_count = node_count
        self.best_actions = best_actions


def _tree_search_si_node(
    stm: np.ndarray,
    O_pos: np.ndarray, O_res: np.ndarray, O_hill: np.ndarray,
    P_pos_prior: np.ndarray, P_ctx_prior: np.ndarray,
    rec: _RecursionState,
    t: int, N: int,
    t_food: int, t_water: int, t_sleep: int,
) -> float:
    inp = rec.inputs
    rec.node_count += 1

    G = 0.02  # base term (MATLAB constant)

    # --- Bayesian posterior given (imagined or real) observation ---------
    P_pos, P_ctx = calculate_posterior(
        P_pos_prior, P_ctx_prior,
        inp.y_resource, inp.y_hill,
        O_res, O_hill,
    )

    # Indices into the short-term-memory tensor.
    t_food_idx = index_clip(int(round(t_food)) + 1)
    t_water_idx = index_clip(int(round(t_water)) + 1)
    t_sleep_idx = index_clip(int(round(t_sleep)) + 1)

    # --- Per-node EFE terms (only at imagined nodes, not the root) -------
    if t > rec.true_t:
        if rec.novelty_on and inp.weights.novelty != 0:
            novelty = _novelty_si(
                inp.a_resource, O_res, P_pos, P_ctx, inp.weights.learning,
            )
            G += inp.weights.novelty * novelty

        if rec.epistemic_on and inp.weights.epistemic != 0:
            epi = G_epistemic_value(
                [inp.y_pos, inp.y_resource, inp.y_hill],
                [P_pos_prior, P_ctx_prior],
            )
            G += inp.weights.epistemic * epi

        # Extrinsic / preference (uses CURRENT need-timers, before increment)
        C = determine_observation_preference(
            t_food, t_water, t_sleep,
            inp.weights.preference_inverse_precision,
        )
        extrinsic = float(O_res @ C)
        G += extrinsic

        # NOW update the need-timers per MATLAB's formula.
        # round((t + 1) * (1 - O_res[k]))  with k = 1 (food), 2 (water), 3 (sleep)
        t_food = int(round((t_food + 1) * (1 - float(O_res[1]))))
        t_water = int(round((t_water + 1) * (1 - float(O_res[2]))))
        t_sleep = int(round((t_sleep + 1) * (1 - float(O_res[3]))))
        t_food_idx = index_clip(t_food + 1)
        t_water_idx = index_clip(t_water + 1)
        t_sleep_idx = index_clip(t_sleep + 1)

    # --- Recurse over actions if there's depth left ---------------------
    if t < N:
        num_states = inp.A_pos.shape[0]
        num_actions = inp.B_pos.shape[2]
        K = np.zeros(num_states * inp.A_resource.shape[2], dtype=np.float64)
        efe = np.zeros(num_actions, dtype=np.float64)

        for action in range(num_actions):
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
                    O_pos_n, O_res_n, O_hill_n = imagined_observations(
                        inp.y_pos, inp.y_resource, inp.y_hill,
                        int(state), num_states,
                    )
                    G_child = _tree_search_si_node(
                        stm, O_pos_n, O_res_n, O_hill_n,
                        Q_pos_a, Q_ctx_a, rec,
                        t + 1, N, t_food, t_water, t_sleep,
                    )
                    K[state] = G_child
                    stm[t_food_idx, t_water_idx, t_sleep_idx, state] = G_child
                    rec.memory_misses += 1

            efe[action] += 0.7 * float(K[likely] @ qs[likely])

        best_action = int(np.argmax(efe))  # deterministic tie-break
        G += float(efe[best_action])
        rec.best_actions.insert(0, best_action)

    return G

"""Bayes-Adaptive RL planner (extrinsic-only EFE).

Faithful port of ``src/MATLAB/tree-search/tree_search_frwd.m``.

Differences from SI:
  * No novelty, no epistemic — only ``extrinsic = O_resource @ C``.
  * Memory cache is indexed by (t_food, t_water, t_sleep, joint_state, action)
    — one extra trailing axis. The cached value at action level stores
    ``0.7 * action_fe`` directly, not the recursive G.
  * State at the *current* timestep is selected from the joint posterior
    via ``select_from_posterior`` (the function tracks a single state, not
    a distribution over states like SI).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..efe import determine_observation_preference
from ..inference import calculate_posterior, joint_pos_ctx, normalise
from ..rng import select_from_posterior
from .common import (
    PlannerInputs, PlanResult, imagined_observations, index_clip,
    likely_state_indices,
)


def tree_search_ba(
    short_term_memory: np.ndarray,  # shape (35,35,35, joint_states, num_actions)
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
    state_selection: str = "sample",
    rng: Optional[np.random.Generator] = None,
) -> PlanResult:
    rec = _RecBA(
        inputs=inputs, true_t=true_t, state_selection=state_selection, rng=rng,
        memory_hits=0, memory_misses=0, node_count=0, best_actions=[],
    )
    G = _ba_node(short_term_memory,
                 O_pos_root, O_resource_root, O_hill_root,
                 P_pos_root, P_ctx_root,
                 rec, t, N, t_food, t_water, t_sleep)
    return PlanResult(
        G=float(np.max(G)),
        best_actions=rec.best_actions,
        memory_hits=rec.memory_hits,
        memory_misses=rec.memory_misses,
        node_count=rec.node_count,
    )


class _RecBA:
    __slots__ = ("inputs", "true_t", "state_selection", "rng",
                 "memory_hits", "memory_misses", "node_count", "best_actions")

    def __init__(self, inputs, true_t, state_selection, rng,
                 memory_hits, memory_misses, node_count, best_actions):
        self.inputs = inputs
        self.true_t = true_t
        self.state_selection = state_selection
        self.rng = rng
        self.memory_hits = memory_hits
        self.memory_misses = memory_misses
        self.node_count = node_count
        self.best_actions = best_actions


def _ba_node(stm, O_pos, O_res, O_hill, P_pos_prior, P_ctx_prior,
             rec, t, N, t_food, t_water, t_sleep):
    inp = rec.inputs
    rec.node_count += 1
    G = 0.02

    P_pos, P_ctx = calculate_posterior(
        P_pos_prior, P_ctx_prior,
        inp.y_resource, inp.y_hill,
        O_res, O_hill,
    )

    # Clamp need timers to memory cache extent.
    t_food = min(t_food, 35)
    t_water = min(t_water, 35)
    t_sleep = min(t_sleep, 35)

    # Always-add extrinsic (BA computes preference at every node, including root).
    C = determine_observation_preference(
        t_food, t_water, t_sleep,
        inp.weights.preference_inverse_precision,
    )
    G += float(O_res @ C)

    # Update need timers (after extrinsic) — mirrors MATLAB lines 43-49.
    t_food = int(round((t_food + 1) * (1 - float(O_res[1]))))
    t_water = int(round((t_water + 1) * (1 - float(O_res[2]))))
    t_sleep = int(round((t_sleep + 1) * (1 - float(O_res[3]))))
    t_food_idx = index_clip(t_food + 1)
    t_water_idx = index_clip(t_water + 1)
    t_sleep_idx = index_clip(t_sleep + 1)

    if t < N:
        # BA selects a single "current" joint state from the posterior, then
        # caches per (state, action). Matches MATLAB lines 53-55.
        joint_dist = joint_pos_ctx(P_pos, P_ctx)
        cur_state = select_from_posterior(joint_dist, rec.state_selection, rec.rng)

        S = inp.A_pos.shape[0]
        efe = np.zeros(inp.B_pos.shape[2], dtype=np.float64)

        for action in range(inp.B_pos.shape[2]):
            cache = stm[t_food_idx, t_water_idx, t_sleep_idx, cur_state, action]
            if cache != 0.0:
                efe[action] = cache
                rec.memory_hits += 1
                continue

            Q_pos_a = inp.B_pos[:, :, action] @ P_pos
            Q_ctx_a = inp.bb_ctx[:, :, 0] @ P_ctx
            qs = joint_pos_ctx(Q_pos_a, Q_ctx_a)
            likely = likely_state_indices(qs, threshold=1.0 / 8.0)

            K = np.zeros(S * inp.A_resource.shape[2], dtype=np.float64)
            for state in likely:
                Op, Or, Oh = imagined_observations(
                    inp.y_pos, inp.y_resource, inp.y_hill,
                    int(state), S,
                )
                G_child = _ba_node(stm, Op, Or, Oh, Q_pos_a, Q_ctx_a, rec,
                                   t + 1, N, t_food, t_water, t_sleep)
                K[state] = G_child

            action_fe = 0.7 * float(K[likely] @ qs[likely])
            efe[action] = action_fe
            stm[t_food_idx, t_water_idx, t_sleep_idx, cur_state, action] = action_fe
            rec.memory_misses += 1

        best_action = int(np.argmax(efe))
        G += float(efe[best_action])
        rec.best_actions.insert(0, best_action)

    return G

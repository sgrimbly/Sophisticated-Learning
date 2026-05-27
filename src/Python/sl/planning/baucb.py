"""Bayes-Adaptive RL with UCB exploration bonus.

Faithful port of ``src/MATLAB/tree-search/tree_search_frwd_UCB.m``.

Differences from BA:
  * Maintains visit counts ``Nt[joint_state]`` updated *only* at the root
    (``t == true_t``).
  * Adds ``ucb_scale * sqrt(log(total_visits + 1) / Nt[cur_state])`` to the
    extrinsic at every node — the count comes from the agent's persistent
    visit log, not the recursion's local state.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..efe import determine_observation_preference
from ..inference import calculate_posterior, joint_pos_ctx
from ..rng import select_from_posterior
from .common import (
    PlannerInputs, PlanResult, imagined_observations, index_clip,
    likely_state_indices,
)


def tree_search_baucb(
    short_term_memory: np.ndarray,
    Nt: np.ndarray,                  # (joint_states,) visit counts (mutated)
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
    current_joint_state: int = 0,
    ucb_scale: float = 5.0,
) -> PlanResult:
    rec = _RecUCB(
        inputs=inputs, true_t=true_t, state_selection=state_selection, rng=rng,
        ucb_scale=ucb_scale, Nt=Nt,
        memory_hits=0, memory_misses=0, node_count=0, best_actions=[],
    )
    G = _baucb_node(
        short_term_memory,
        O_pos_root, O_resource_root, O_hill_root,
        P_pos_root, P_ctx_root,
        rec, t, N, t_food, t_water, t_sleep,
        current_joint_state=current_joint_state,
    )
    return PlanResult(
        G=float(np.max(G)),
        best_actions=rec.best_actions,
        memory_hits=rec.memory_hits,
        memory_misses=rec.memory_misses,
        node_count=rec.node_count,
    )


class _RecUCB:
    __slots__ = ("inputs", "true_t", "state_selection", "rng", "ucb_scale", "Nt",
                 "memory_hits", "memory_misses", "node_count", "best_actions")

    def __init__(self, inputs, true_t, state_selection, rng, ucb_scale, Nt,
                 memory_hits, memory_misses, node_count, best_actions):
        self.inputs = inputs
        self.true_t = true_t
        self.state_selection = state_selection
        self.rng = rng
        self.ucb_scale = ucb_scale
        self.Nt = Nt
        self.memory_hits = memory_hits
        self.memory_misses = memory_misses
        self.node_count = node_count
        self.best_actions = best_actions


def _baucb_node(stm, O_pos, O_res, O_hill, P_pos_prior, P_ctx_prior, rec,
                t, N, t_food, t_water, t_sleep, current_joint_state):
    inp = rec.inputs
    rec.node_count += 1
    G = 0.02

    P_pos, P_ctx = calculate_posterior(
        P_pos_prior, P_ctx_prior,
        inp.y_resource, inp.y_hill,
        O_res, O_hill,
    )

    # Bump visit count only at the root.
    if t == rec.true_t:
        rec.Nt[current_joint_state] += 1

    t_food = min(t_food, 35); t_water = min(t_water, 35); t_sleep = min(t_sleep, 35)

    # Extrinsic + UCB exploration bonus.
    C = determine_observation_preference(
        t_food, t_water, t_sleep,
        inp.weights.preference_inverse_precision,
    )
    extrinsic = float(O_res @ C)
    total_visits = float(rec.Nt.sum())
    visits = float(rec.Nt[current_joint_state])
    if visits > 0:
        exploration = rec.ucb_scale * np.sqrt(np.log(total_visits + 1.0) / visits)
    else:
        exploration = rec.ucb_scale * np.sqrt(np.log(total_visits + 1.0))
    G += extrinsic + exploration

    t_food = int(round((t_food + 1) * (1 - float(O_res[1]))))
    t_water = int(round((t_water + 1) * (1 - float(O_res[2]))))
    t_sleep = int(round((t_sleep + 1) * (1 - float(O_res[3]))))
    t_food_idx = index_clip(t_food + 1)
    t_water_idx = index_clip(t_water + 1)
    t_sleep_idx = index_clip(t_sleep + 1)

    if t < N:
        S = inp.A_pos.shape[0]
        efe = np.zeros(inp.B_pos.shape[2], dtype=np.float64)

        for action in range(inp.B_pos.shape[2]):
            cache = stm[t_food_idx, t_water_idx, t_sleep_idx,
                         current_joint_state, action]
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
                G_child = _baucb_node(stm, Op, Or, Oh, Q_pos_a, Q_ctx_a, rec,
                                      t + 1, N, t_food, t_water, t_sleep,
                                      current_joint_state=int(state))
                K[state] = G_child

            action_fe = 0.7 * float(K[likely] @ qs[likely])
            efe[action] = action_fe
            stm[t_food_idx, t_water_idx, t_sleep_idx, current_joint_state, action] = action_fe
            rec.memory_misses += 1

        best_action = int(np.argmax(efe))
        G += float(efe[best_action])
        rec.best_actions.insert(0, best_action)

    return G

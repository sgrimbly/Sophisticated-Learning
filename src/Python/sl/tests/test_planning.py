"""Smoke + invariant tests for tree-search planners.

These don't try to bit-match MATLAB — they assert structural invariants
that any correct port of the algorithm must satisfy.
"""
from __future__ import annotations

import unittest

import numpy as np

from sl.config import GridConfig, RunOptions, Weights
from sl.env import initialise_environment, normalise_b_ctx
from sl.inference import normalise_matrix_columns
from sl.planning.common import PlannerInputs
from sl.planning.si import tree_search_si
from sl.planning.sl import tree_search_sl
from sl.planning.ba import tree_search_ba
from sl.planning.baucb import tree_search_baucb
from sl.rng import make_rng


def _setup(weights=None, max_horizon=2):
    grid = GridConfig.from_matlab_indices()
    options = RunOptions(algorithm="SI", seed=1, num_trials=1, max_horizon=max_horizon)
    weights = weights or Weights()
    model = initialise_environment(grid)
    bb_ctx = normalise_b_ctx(model.b_ctx)
    inputs = PlannerInputs(
        A_pos=model.A_pos, A_resource=model.A_resource, A_hill=model.A_hill,
        a_resource=model.a_resource,
        y_pos=model.A_pos,
        y_resource=normalise_matrix_columns(model.a_resource),
        y_hill=model.A_hill,
        B_pos=model.B_pos, bb_ctx=bb_ctx,
        weights=weights,
    )
    # Synthetic root observation/posterior (match what the agent would pass).
    O_pos = np.zeros(grid.num_states); O_pos[grid.start_position] = 1.0
    O_res = np.zeros(4); O_res[0] = 1.0  # 'empty'
    O_hill = np.zeros(5); O_hill[4] = 1.0  # 'none'
    P_pos = model.D_pos
    P_ctx = model.D_ctx
    return grid, model, inputs, (O_pos, O_res, O_hill), (P_pos, P_ctx)


class TestSI(unittest.TestCase):
    def test_returns_at_least_one_action(self):
        """The shared best_actions list grows beyond ``horizon`` because each
        recursive call's prepend is on the same Python list (this matches
        MATLAB pass-by-value semantics in spirit: the agent only ever reads
        ``best_actions[0]`` which corresponds to the root's chosen action).
        """
        grid, model, inputs, (Op, Or, Oh), (Pp, Pc) = _setup(max_horizon=3)
        stm = np.zeros((35, 35, 35, grid.num_joint_states))
        result = tree_search_si(stm, Op, Or, Oh, Pp, Pc, inputs,
                                t=0, N=3, t_food=5, t_water=3, t_sleep=4, true_t=0)
        self.assertGreaterEqual(len(result.best_actions), 1)
        # All actions in valid range.
        for a in result.best_actions:
            self.assertGreaterEqual(a, 0)
            self.assertLess(a, 5)

    def test_no_rng_consumed(self):
        """SI tree search must never call np.random — verify the global RNG
        state is identical before and after.
        """
        grid, model, inputs, (Op, Or, Oh), (Pp, Pc) = _setup(max_horizon=3)
        stm = np.zeros((35, 35, 35, grid.num_joint_states))
        before = np.random.get_state()
        tree_search_si(stm, Op, Or, Oh, Pp, Pc, inputs, 0, 3, 5, 3, 4, 0)
        after = np.random.get_state()
        # Compare the bytes-identical state tuples.
        self.assertEqual(before[1].tobytes(), after[1].tobytes())

    def test_short_term_memory_is_filled(self):
        grid, model, inputs, (Op, Or, Oh), (Pp, Pc) = _setup(max_horizon=4)
        stm = np.zeros((35, 35, 35, grid.num_joint_states))
        tree_search_si(stm, Op, Or, Oh, Pp, Pc, inputs, 0, 4, 5, 3, 4, 0)
        self.assertGreater((stm != 0.0).sum(), 0,
                           "tree search should populate the short-term-memory cache")


class TestSL(unittest.TestCase):
    def test_runs_with_history(self):
        grid, model, inputs, (Op, Or, Oh), (Pp, Pc) = _setup(max_horizon=2)
        stm = np.zeros((35, 35, 35, grid.num_joint_states))
        result = tree_search_sl(
            stm, Op, Or, Oh, Pp, Pc, inputs,
            t=0, N=2, t_food=5, t_water=3, t_sleep=4, true_t=0,
            history_O_resource=[Or],
            history_O_hill=[Oh],
            history_P_pos=[Pp],
            history_P_ctx=[Pc],
        )
        self.assertGreaterEqual(len(result.best_actions), 1)


class TestBA(unittest.TestCase):
    def test_extrinsic_only_planner_runs(self):
        grid, model, inputs, (Op, Or, Oh), (Pp, Pc) = _setup(max_horizon=2)
        stm = np.zeros((35, 35, 35, grid.num_joint_states, 5))
        rng = make_rng(0)
        result = tree_search_ba(stm, Op, Or, Oh, Pp, Pc, inputs,
                                t=0, N=2, t_food=5, t_water=3, t_sleep=4,
                                true_t=0, state_selection="map", rng=rng)
        self.assertGreaterEqual(len(result.best_actions), 1)


class TestBAUCB(unittest.TestCase):
    def test_visit_counts_increment_only_at_root(self):
        grid, model, inputs, (Op, Or, Oh), (Pp, Pc) = _setup(max_horizon=3)
        stm = np.zeros((35, 35, 35, grid.num_joint_states, 5))
        Nt = np.zeros(grid.num_joint_states)
        rng = make_rng(0)
        cur_joint = grid.start_position + grid.num_states * 0
        tree_search_baucb(stm, Nt, Op, Or, Oh, Pp, Pc, inputs,
                          t=0, N=3, t_food=5, t_water=3, t_sleep=4,
                          true_t=0, state_selection="map", rng=rng,
                          current_joint_state=cur_joint, ucb_scale=5.0)
        self.assertEqual(int(Nt[cur_joint]), 1)


if __name__ == "__main__":
    unittest.main()

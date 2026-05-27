"""Tests for the Dirichlet learning update.

Guards against the prior Python attempt's bug of subtracting the row-0
penalty from *all* rows instead of row 0 only.
"""
from __future__ import annotations

import unittest

import numpy as np

from sl.learning import (
    apply_row0_penalty, build_a_learning, planning_dirichlet_update,
    real_dirichlet_update,
)


class TestRow0Penalty(unittest.TestCase):
    def test_penalty_only_subtracts_from_row_0(self):
        # 4 outcomes × 3 states × 2 contexts.
        a = np.zeros((4, 3, 2))
        # Row 0 is zero. Rows 1..3 have varying mass.
        a[1, 0, 0] = 1.0
        a[2, 0, 0] = 0.5
        a[3, 0, 0] = 0.2
        a_after = apply_row0_penalty(a, proportion=0.3)
        # Max above empty at (0,0) = 1.0. Penalty = 0.3.
        # Row 0 should drop from 0 to -0.3. Others unchanged.
        self.assertAlmostEqual(a_after[0, 0, 0], -0.3)
        self.assertAlmostEqual(a_after[1, 0, 0], 1.0)
        self.assertAlmostEqual(a_after[2, 0, 0], 0.5)
        self.assertAlmostEqual(a_after[3, 0, 0], 0.2)

    def test_penalty_skips_when_empty_row_is_nonzero(self):
        """No subtraction if empty row is already > 0."""
        a = np.zeros((4, 1, 1))
        a[0, 0, 0] = 0.1
        a[1, 0, 0] = 1.0
        a_after = apply_row0_penalty(a, proportion=0.3)
        np.testing.assert_array_equal(a_after, a)


class TestRealDirichletUpdate(unittest.TestCase):
    def test_update_increases_resource_count_at_observed_position(self):
        S, C = 5, 2
        a = np.full((4, S, C), 0.1)
        O_res = np.array([0.0, 1.0, 0.0, 0.0])  # observed food
        P_pos = np.zeros(S); P_pos[2] = 1.0  # certain about being at pos 2
        P_ctx = np.zeros(C); P_ctx[0] = 1.0  # certain about context 0

        new_a = real_dirichlet_update(a, O_res, P_pos, P_ctx,
                                       proportion=0.3, scale=0.7, floor=0.05)
        # The (food, pos=2, ctx=0) entry increased: 0.1 + 0.7*1.0 = 0.8.
        self.assertAlmostEqual(new_a[1, 2, 0], 0.8)
        # The empty-row entry at (pos=2, ctx=0) was zero in `a_learning` (only
        # row 1 was hit by the cross product), so the penalty *is* applied:
        # a_learning[0,2,0] becomes -0.3, then a + 0.7 * a_learning gives
        # 0.1 - 0.21 = -0.11, then the 0.05 floor clamps it.
        self.assertAlmostEqual(new_a[0, 2, 0], 0.05)
        # Floor: nothing below 0.05.
        self.assertGreaterEqual(new_a.min(), 0.05)


class TestPlanningDirichletUpdate(unittest.TestCase):
    def test_prune_threshold_zeros_small_values(self):
        S, C = 4, 2
        a_imag = np.full((4, S, C), 0.1)
        # Construct a low-mass observation distribution to keep a_learning small.
        O_res = np.array([0.0, 0.05, 0.0, 0.0])
        P_pos = np.zeros(S); P_pos[0] = 1.0
        P_ctx = np.zeros(C); P_ctx[0] = 1.0
        new_a, a_learn, a_weighted = planning_dirichlet_update(
            a_imag, O_res, P_pos, P_ctx, learning_weight=40.0, prune_threshold=0.2,
        )
        # 0.05 < 0.2 → all values pruned to zero.
        np.testing.assert_array_equal(a_learn, 0.0)
        np.testing.assert_array_equal(new_a, a_imag)

    def test_no_prune_keeps_small_updates(self):
        S, C = 4, 2
        a_imag = np.full((4, S, C), 0.1)
        O_res = np.array([0.0, 0.05, 0.0, 0.0])
        P_pos = np.zeros(S); P_pos[0] = 1.0
        P_ctx = np.zeros(C); P_ctx[0] = 1.0
        new_a, a_learn, _ = planning_dirichlet_update(
            a_imag, O_res, P_pos, P_ctx, learning_weight=40.0, prune_threshold=0.0,
        )
        self.assertGreater(a_learn.max(), 0.0)


if __name__ == "__main__":
    unittest.main()

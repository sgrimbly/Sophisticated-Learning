"""Tests for the generative environment.

Covers the issues that broke the prior Python attempt:
  * B[0] (position transition) reachability per action
  * Resource-cue likelihood A_resource: deterministic at source positions,
    'empty' elsewhere
  * Hill-cue likelihood A_hill: deterministic per-context at hill_pos,
    'none' elsewhere
  * Initial state priors normalised
"""
from __future__ import annotations

import unittest

import numpy as np

from sl.config import GridConfig
from sl.env import (
    GenerativeModel, initialise_environment, normalise_b_ctx,
    sample_observations, update_environment_states, update_needs,
)
from sl.rng import make_rng


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.grid = GridConfig.from_matlab_indices()
        self.model = initialise_environment(self.grid)

    def test_shapes(self):
        m = self.model
        S, C = self.grid.num_states, self.grid.num_contexts
        self.assertEqual(m.A_pos.shape, (S, S, C))
        self.assertEqual(m.A_resource.shape, (4, S, C))
        self.assertEqual(m.A_hill.shape, (5, S, C))
        self.assertEqual(m.B_pos.shape, (S, S, 5))
        self.assertEqual(m.B_ctx.shape, (C, C, 5))
        self.assertEqual(m.D_pos.shape, (S,))
        self.assertEqual(m.D_ctx.shape, (C,))

    def test_position_transition_column_stochastic(self):
        """Every column of B_pos[:,:,action] must sum to 1 — probability mass preserved."""
        for action in range(5):
            sums = self.model.B_pos[:, :, action].sum(axis=0)
            np.testing.assert_allclose(sums, 1.0, atol=1e-12)

    def test_position_transition_stay_is_identity(self):
        """Action 0 = stay = identity."""
        np.testing.assert_array_equal(
            self.model.B_pos[:, :, 0],
            np.eye(self.grid.num_states),
        )

    def test_position_transition_left_right_inverse_in_interior(self):
        """For an interior cell (not on column boundary), left then right
        returns to the same column. Validates that the action labels
        consistently move along the same axis with opposite signs.
        """
        # Pick a cell guaranteed to be in the interior of a row (e.g., 0-based 55):
        # row=5, col=5 in a 10×10 grid.
        i = 55
        # Apply 'left' (action 1): result is the post-state distribution.
        post_left = self.model.B_pos[:, i, 1]
        # Reverse: from the state that result point-mass is on, go right.
        new_state = int(np.argmax(post_left))
        post_right = self.model.B_pos[:, new_state, 2]
        recovered = int(np.argmax(post_right))
        self.assertEqual(recovered, i,
                         f"left then right should return to {i}, got {recovered}")

    def test_position_transition_up_down_inverse_in_interior(self):
        """Same invariant for up/down on a cell not on a row boundary."""
        i = 55  # interior row 5, column 5 (0-based)
        post_up = self.model.B_pos[:, i, 3]
        new_state = int(np.argmax(post_up))
        post_down = self.model.B_pos[:, new_state, 4]
        recovered = int(np.argmax(post_down))
        self.assertEqual(recovered, i,
                         f"up then down should return to {i}, got {recovered}")

    def test_position_transition_boundaries_are_self_loops(self):
        """Cells on the boundary of the appropriate edge map to themselves
        for the move that would leave the grid. Matches MATLAB's `mod` and
        `i > grid_size` boundary excludes which leave that column as the
        identity (stay)."""
        gs = self.grid.grid_size
        # leftmost column (action 1 = left) — column 0 in 0-based
        for row in range(gs):
            i = row * gs  # leftmost column
            self.assertEqual(int(np.argmax(self.model.B_pos[:, i, 1])), i,
                             f"leftmost cell {i} should not move on action 'left'")
        # rightmost column (action 2 = right)
        for row in range(gs):
            i = row * gs + (gs - 1)
            self.assertEqual(int(np.argmax(self.model.B_pos[:, i, 2])), i)

    def test_A_resource_at_source_positions(self):
        """At each (food_source, context) the resource A puts mass on outcome 1 (food)."""
        for ctx, pos in enumerate(self.grid.food_sources):
            self.assertEqual(self.model.A_resource[1, pos, ctx], 1.0)
            self.assertEqual(self.model.A_resource[0, pos, ctx], 0.0)
        for ctx, pos in enumerate(self.grid.water_sources):
            self.assertEqual(self.model.A_resource[2, pos, ctx], 1.0)
        for ctx, pos in enumerate(self.grid.sleep_sources):
            self.assertEqual(self.model.A_resource[3, pos, ctx], 1.0)

    def test_A_resource_default_outcome_is_empty(self):
        """A position not assigned to any resource for a context has outcome 0 (empty)."""
        # Pick a cell unlikely to be any resource: 0-based position 0 in context 0.
        self.assertEqual(self.model.A_resource[0, 0, 0], 1.0)

    def test_A_hill_per_context(self):
        """At hill_pos, A_hill points to the context's index; elsewhere 'none' (4)."""
        h = self.grid.hill_pos
        for ctx in range(self.grid.num_contexts):
            self.assertEqual(self.model.A_hill[ctx, h, ctx], 1.0)
            self.assertEqual(self.model.A_hill[4, h, ctx], 0.0)
        # Non-hill cell: outcome 4 ('none').
        self.assertEqual(self.model.A_hill[4, 0, 0], 1.0)

    def test_a_resource_initial_is_uniform(self):
        """Agent's prior on a_resource starts at 0.1 everywhere (Dirichlet pseudocounts)."""
        np.testing.assert_allclose(self.model.a_resource, 0.1)

    def test_D_normalised(self):
        np.testing.assert_allclose(self.model.D_pos.sum(), 1.0)
        np.testing.assert_allclose(self.model.D_ctx.sum(), 1.0)
        # Initial position is the start cell.
        self.assertEqual(int(np.argmax(self.model.D_pos)), self.grid.start_position)


class TestPerStepUpdates(unittest.TestCase):
    def setUp(self):
        self.grid = GridConfig.from_matlab_indices()
        self.model = initialise_environment(self.grid)

    def test_update_needs_at_food_resets_food_only(self):
        ctx = 0
        food = self.grid.food_sources[ctx]
        f, w, s = update_needs(self.grid, food, ctx, t=5,
                               t_food=10, t_water=5, t_sleep=3)
        self.assertEqual(f, 0)
        self.assertEqual(w, 6)
        self.assertEqual(s, 4)

    def test_update_needs_t0_no_increment_off_resource(self):
        # t=0 with a non-resource cell should not increment.
        f, w, s = update_needs(self.grid, true_pos=5, true_ctx=0, t=0,
                               t_food=0, t_water=0, t_sleep=0)
        self.assertEqual((f, w, s), (0, 0, 0))

    def test_normalise_b_ctx_columns_sum_to_one(self):
        bb = normalise_b_ctx(self.model.b_ctx)
        for action in range(bb.shape[2]):
            sums = bb[:, :, action].sum(axis=0)
            np.testing.assert_allclose(sums, 1.0, atol=1e-12)

    def test_sample_observations_returns_one_hot(self):
        rng = make_rng(0)
        O_pos, O_res, O_hill = sample_observations(
            self.model.A_pos, self.model.A_resource, self.model.A_hill,
            true_pos=self.grid.start_position, true_ctx=0, rng=rng,
        )
        for O in (O_pos, O_res, O_hill):
            self.assertAlmostEqual(O.sum(), 1.0)
            self.assertEqual(int((O == 1.0).sum()), 1)

    def test_update_environment_states_preserves_distributional_normalisation(self):
        rng = make_rng(0)
        bb = normalise_b_ctx(self.model.b_ctx)
        Q_pos = self.model.D_pos.copy()
        Q_ctx = self.model.D_ctx.copy()
        Q_pos_n, Q_ctx_n, tp, tc = update_environment_states(
            Q_pos, Q_ctx,
            true_pos_prev=self.grid.start_position, true_ctx_prev=0,
            chosen_action=0,  # stay
            B_pos=self.model.B_pos, B_ctx=self.model.B_ctx, bb_ctx=bb,
            rng=rng,
        )
        np.testing.assert_allclose(Q_pos_n.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(Q_ctx_n.sum(), 1.0, atol=1e-12)
        self.assertEqual(tp, self.grid.start_position)  # 'stay' keeps position


if __name__ == "__main__":
    unittest.main()

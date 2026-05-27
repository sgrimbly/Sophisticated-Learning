"""Tests for inference primitives.

The biggest single regression these guard against is the prior Python
attempt's mistake of *sampling* from O inside calculate_posterior /
spm_backwards. These tests exercise the distribution path directly.
"""
from __future__ import annotations

import unittest

import numpy as np

from sl.inference import (
    calculate_posterior, joint_pos_ctx, normalise, normalise_matrix_columns,
    spm_backwards, spm_cross, spm_norm,
)


class TestPrimitives(unittest.TestCase):
    def test_normalise_uniform_when_sum_zero(self):
        x = np.zeros(4)
        out = normalise(x)
        np.testing.assert_allclose(out, np.full(4, 0.25))

    def test_normalise_simple(self):
        x = np.array([1.0, 1.0, 2.0, 4.0])
        out = normalise(x)
        np.testing.assert_allclose(out, x / x.sum())

    def test_normalise_matrix_columns(self):
        m = np.array([[1.0, 2.0], [1.0, 2.0]])
        out = normalise_matrix_columns(m)
        np.testing.assert_allclose(out, np.array([[0.5, 0.5], [0.5, 0.5]]))

    def test_spm_norm_handles_nan(self):
        m = np.array([[0.0, 1.0], [0.0, 1.0]])
        out = spm_norm(m)
        # First column was all zero → spm_norm fills with 1/N.
        np.testing.assert_allclose(out[:, 0], 0.5)
        np.testing.assert_allclose(out[:, 1], 0.5)

    def test_spm_cross_two_factors(self):
        a = np.array([0.5, 0.5])
        b = np.array([0.25, 0.75])
        out = spm_cross(a, b)
        self.assertEqual(out.shape, (2, 2))
        np.testing.assert_allclose(out, np.outer(a, b))

    def test_spm_cross_three_factors(self):
        a = np.array([0.5, 0.5])
        b = np.array([1.0, 0.0])
        c = np.array([0.5, 0.5])
        out = spm_cross(a, b, c)
        self.assertEqual(out.shape, (2, 2, 2))
        # Hand-check a few entries.
        self.assertAlmostEqual(out[0, 0, 0], 0.5 * 1.0 * 0.5)
        self.assertAlmostEqual(out[0, 1, 0], 0.5 * 0.0 * 0.5)

    def test_spm_cross_recurses_lists(self):
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        out_list = spm_cross([a, b])
        out_args = spm_cross(a, b)
        np.testing.assert_allclose(out_list, out_args)

    def test_joint_pos_ctx_matches_matlab_order(self):
        """qs(:) in MATLAB column-major order — pos varies fastest, then ctx."""
        p = np.array([0.7, 0.3])
        c = np.array([0.4, 0.6])
        joint = joint_pos_ctx(p, c)
        # Expect [p0c0, p1c0, p0c1, p1c1]
        np.testing.assert_allclose(joint, [0.28, 0.12, 0.42, 0.18])


class TestCalculatePosterior(unittest.TestCase):
    """The defining test: verify posterior is computed by *marginalising*
    over the observation distribution, not by sampling.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        # Tiny model: 3 positions, 2 contexts, 2 resource outcomes, 2 hill outcomes.
        self.S = 3; self.C = 2
        self.A_resource = np.array([
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],   # outcome 0 likelihoods
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],   # outcome 1
        ])
        # Shape: (n_o, S, C) = (2, 3, 2)
        self.A_hill = np.zeros((2, 3, 2))
        self.A_hill[0, :, :] = 1.0  # outcome 0 ('none') everywhere

        self.P_pos = normalise(np.ones(3))
        self.P_ctx = normalise(np.ones(2))

    def test_one_hot_observation_concentrates_posterior(self):
        # Set hill to known: position 1 in context 0 yields outcome 1.
        self.A_hill[1, 1, 0] = 1.0; self.A_hill[0, 1, 0] = 0.0
        # Resource observation: at (pos=1, ctx=0), the resource A says outcome 1.
        # So observe outcome 1 — consistent with (pos=1, ctx=0).
        O_res = np.array([0.0, 1.0])
        O_hill = np.array([0.0, 1.0])
        _, P_ctx_post = calculate_posterior(
            self.P_pos, self.P_ctx, self.A_resource, self.A_hill, O_res, O_hill,
        )
        # Joint observation only possible at (pos=1, ctx=0). So ctx 0 should
        # dominate the posterior.
        self.assertGreater(P_ctx_post[0], P_ctx_post[1])
        np.testing.assert_allclose(P_ctx_post.sum(), 1.0)

    def test_distributional_observation_does_not_sample(self):
        """Same call twice with same input must return identical results.

        If the function were sampling internally, repeated calls would diverge
        with a fresh global RNG state. We don't pass an rng — and the function
        must not need one.
        """
        O_res = np.array([0.7, 0.3])
        O_hill = np.array([0.5, 0.5])
        _, p1 = calculate_posterior(self.P_pos, self.P_ctx,
                                    self.A_resource, self.A_hill, O_res, O_hill)
        _, p2 = calculate_posterior(self.P_pos, self.P_ctx,
                                    self.A_resource, self.A_hill, O_res, O_hill)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_allclose(p1.sum(), 1.0)

    def test_uniform_observation_matches_uniform_prior(self):
        """When O is uniform across outcomes, posterior over context == prior."""
        n_o_res = self.A_resource.shape[0]
        n_o_hill = self.A_hill.shape[0]
        O_res = np.full(n_o_res, 1.0 / n_o_res)
        O_hill = np.full(n_o_hill, 1.0 / n_o_hill)
        _, P_ctx_post = calculate_posterior(
            self.P_pos, self.P_ctx, self.A_resource, self.A_hill, O_res, O_hill,
        )
        # Likelihood becomes a constant → posterior == prior.
        np.testing.assert_allclose(P_ctx_post, self.P_ctx, atol=1e-12)


class TestSpmBackwards(unittest.TestCase):
    """spm_backwards must (a) not sample and (b) produce a normalised posterior."""

    def test_no_evidence_is_identity(self):
        # If timey == t (no integration steps), should return spm_norm of input.
        rng = np.random.default_rng(0)
        A_hill = np.ones((2, 3, 2)) * 0.5
        B_ctx = np.eye(2)[:, :, None] * np.ones((1, 1, 5))
        P_pos_hist = [normalise(np.ones(3))]
        O_hill_hist = [np.array([0.5, 0.5])]
        L = spm_backwards(O_hill_hist, P_pos_hist, normalise(np.ones(2)),
                          A_hill, B_ctx, timey=0, t=0)
        np.testing.assert_allclose(L, [0.5, 0.5])

    def test_deterministic_repeated_calls(self):
        A_hill = np.array([
            [[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]],
            [[0.0, 1.0], [0.5, 0.5], [0.0, 1.0]],
        ])
        B_ctx = np.broadcast_to(np.eye(2)[:, :, None], (2, 2, 5)).copy()
        P_pos = [normalise(np.ones(3)) for _ in range(2)]
        O_h = [np.array([0.6, 0.4]), np.array([0.3, 0.7])]
        L1 = spm_backwards(O_h, P_pos, normalise(np.ones(2)), A_hill, B_ctx, 0, 1)
        L2 = spm_backwards(O_h, P_pos, normalise(np.ones(2)), A_hill, B_ctx, 0, 1)
        np.testing.assert_array_equal(L1, L2)
        np.testing.assert_allclose(L1.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()

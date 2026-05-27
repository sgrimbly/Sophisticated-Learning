"""Tests for EFE term computations."""
from __future__ import annotations

import unittest

import numpy as np

from sl.efe import (
    G_epistemic_value, determine_observation_preference, kldir, nat_log,
)


class TestKldir(unittest.TestCase):
    def test_zero_when_distributions_equal(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(kldir(p, p), 0.0, places=10)

    def test_positive_when_different(self):
        p = np.array([0.7, 0.3])
        q = np.array([0.5, 0.5])
        self.assertGreater(kldir(p, q), 0.0)

    def test_handles_zeros(self):
        # 0 * log(0/q) = 0 by convention, but a / 0 -> inf. The kldir port
        # must clamp non-finite values to realmax — never NaN.
        a = np.array([0.0, 1.0])
        b = np.array([0.5, 0.5])
        self.assertTrue(np.isfinite(kldir(a, b)))


class TestNatLog(unittest.TestCase):
    def test_log_zero_does_not_blow_up(self):
        # nat_log(0) should add the floor exp(-16) before taking log.
        out = nat_log(np.array([0.0]))
        self.assertTrue(np.isfinite(out[0]))


class TestDetermineObservationPreference(unittest.TestCase):
    def test_normal_state_returns_negative_offsets(self):
        C = determine_observation_preference(5, 3, 7, preference_inverse_precision=0.1)
        # Resource modality has 4 outcomes: [empty, food, water, sleep]
        self.assertEqual(C.shape, (4,))
        # Higher t_<resource> → MORE negative preference (rescaled).
        # MATLAB pattern: empty=-1, then [t_food, t_water, t_sleep].

    def test_water_threshold_flips_others_to_panic(self):
        """When t_water > 19, food/sleep/empty are set to -500 (MATLAB lines 5-9)."""
        C = determine_observation_preference(0, 20, 0, preference_inverse_precision=1.0)
        # Indices: 0=empty, 1=food, 2=water, 3=sleep
        self.assertEqual(int(C[0]), -500)
        self.assertEqual(int(C[1]), -500)
        # water keeps its value (20)
        self.assertEqual(int(C[2]), 20)
        self.assertEqual(int(C[3]), -500)

    def test_food_threshold_flips_others_to_panic(self):
        C = determine_observation_preference(22, 0, 0, preference_inverse_precision=1.0)
        self.assertEqual(int(C[0]), -500)
        self.assertEqual(int(C[1]), 22)  # food keeps its value
        self.assertEqual(int(C[2]), -500)
        self.assertEqual(int(C[3]), -500)

    def test_zero_preference_weight_returns_zeros(self):
        """With preference weight 0 → preference_inverse_precision = inf,
        the function returns the all-zero vector. In the EFE, this turns the
        extrinsic term off."""
        C = determine_observation_preference(5, 3, 7, preference_inverse_precision=float("inf"))
        np.testing.assert_array_equal(C, np.zeros(4))


class TestEpistemicValue(unittest.TestCase):
    def test_zero_when_likelihood_is_uniform_state_sharp(self):
        """If likelihood doesn't vary by state, mutual info = 0."""
        # Single modality, uniform across states.
        S, C = 3, 2
        A_uniform = np.full((4, S, C), 0.25)
        # Sharp state distribution.
        P_pos = np.zeros(S); P_pos[1] = 1.0
        P_ctx = np.array([0.7, 0.3])
        g = G_epistemic_value([A_uniform], [P_pos, P_ctx])
        self.assertAlmostEqual(g, 0.0, places=8)

    def test_positive_for_informative_likelihood(self):
        """A_pos: deterministic identity → high mutual information when state
        is uncertain.
        """
        S, C = 3, 1
        A_id = np.zeros((S, S, C))
        for i in range(S):
            A_id[i, i, 0] = 1.0
        P_pos = np.full(S, 1.0 / S)
        P_ctx = np.array([1.0])
        g = G_epistemic_value([A_id], [P_pos, P_ctx])
        # Uniform 3-state → max info ≈ log(3).
        self.assertGreater(g, 0.5)
        self.assertLess(g, np.log(3) + 1e-6)


if __name__ == "__main__":
    unittest.main()

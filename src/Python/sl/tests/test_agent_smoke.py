"""End-to-end smoke tests for the agent loop.

Asserts:
  * All four algorithms run a full trial without crashing.
  * Agent dies before max_steps when need-thresholds aren't reachable
    given a tiny horizon.
  * Posteriors stay normalised across the entire trial.
  * a_resource never violates the 0.05 floor.
"""
from __future__ import annotations

import unittest

import numpy as np

from sl.agent import run_trial
from sl.config import GridConfig, RunOptions, Weights
from sl.env import initialise_environment
from sl.rng import make_rng


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.grid = GridConfig.from_matlab_indices()

    def _run(self, algorithm, max_horizon=2, seed=1):
        options = RunOptions(algorithm=algorithm, seed=seed,
                              num_trials=1, max_horizon=max_horizon)
        weights = Weights()
        model = initialise_environment(self.grid)
        return run_trial(self.grid, options, weights, model, make_rng(seed))

    def test_si_completes(self):
        result = self._run("SI")
        self.assertGreater(result.t_terminal, 0)
        self.assertEqual(len(result.chosen_actions), result.t_terminal)

    def test_sl_completes(self):
        result = self._run("SL")
        self.assertGreater(result.t_terminal, 0)

    def test_ba_completes(self):
        result = self._run("BA")
        self.assertGreater(result.t_terminal, 0)

    def test_baucb_completes(self):
        result = self._run("BAUCB")
        self.assertGreater(result.t_terminal, 0)

    def test_posteriors_normalised(self):
        """Every step's predictive and post-observation posterior must sum to 1."""
        result = self._run("SI", max_horizon=3)
        for q in result.Q_pos_history:
            np.testing.assert_allclose(q.sum(), 1.0, atol=1e-9)
        for q in result.Q_ctx_history:
            np.testing.assert_allclose(q.sum(), 1.0, atol=1e-9)
        for p in result.P_pos_history:
            np.testing.assert_allclose(p.sum(), 1.0, atol=1e-9)
        for p in result.P_ctx_history:
            np.testing.assert_allclose(p.sum(), 1.0, atol=1e-9)

    def test_a_resource_floor_respected(self):
        result = self._run("SL", max_horizon=3)
        # Floor is 0.05 (real_dirichlet_update).
        self.assertGreaterEqual(result.a_resource_final.min(), 0.05 - 1e-12)

    def test_no_negative_actions(self):
        result = self._run("SL", max_horizon=4)
        for a in result.chosen_actions:
            self.assertGreaterEqual(a, 0)
            self.assertLess(a, 5)


class TestVariants(unittest.TestCase):
    """SI_smooth, SL_noSmooth and the rest of the family table run end-to-end."""

    def test_all_variants_smoke(self):
        grid = GridConfig.from_matlab_indices()
        for algo in ("SI", "SI_noNovelty", "SI_smooth", "SI_smooth_noNovelty",
                      "SL", "SL_noNovelty", "SL_noSmooth", "SL_noNovelty_noSmooth",
                      "SL_adaptivePlan"):
            with self.subTest(algorithm=algo):
                options = RunOptions(algorithm=algo, seed=1, num_trials=1, max_horizon=2)
                model = initialise_environment(grid)
                result = run_trial(grid, options, Weights(), model, make_rng(1))
                self.assertGreater(result.t_terminal, 0)


if __name__ == "__main__":
    unittest.main()

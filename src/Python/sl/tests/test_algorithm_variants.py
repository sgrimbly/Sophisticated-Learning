"""Guard the algorithm-variant registry against drift from MATLAB.

The Python ``ALGORITHM_VARIANTS`` table must mirror MATLAB
``resolve_algorithm_spec.m`` 1:1 so the two implementations can be paired by
name in replication/parity runs. These tests pin the (novelty, smoothing,
adaptive_plan) semantics of the canonical ablation set and the previously
conflicting ``SI_smooth`` label.
"""
from __future__ import annotations

import unittest

from sl.config import ALGORITHM_VARIANTS, resolve_algorithm


# (family, novelty, smoothing, adaptive_plan) — the contract from MATLAB
# resolve_algorithm_spec.m. The first eight rows are the canonical
# diagnosis-batch ablation set.
EXPECTED = {
    "SI":                                 ("SI", True,  False, False),
    "SI_noNovelty":                       ("SI", False, False, False),
    "SI_novelty_smooth":                  ("SI", True,  True,  False),
    "SI_smooth_noNovelty":                ("SI", False, True,  False),
    "SL_adaptivePlan":                    ("SL", True,  True,  True),
    "SL_noNovelty_adaptivePlan":          ("SL", False, True,  True),
    "SL_noSmooth_adaptivePlan":           ("SL", True,  False, True),
    "SL_noNovelty_noSmooth_adaptivePlan": ("SL", False, False, True),
}

CANONICAL_8 = list(EXPECTED)


class TestAlgorithmVariantRegistry(unittest.TestCase):
    def test_canonical_eight_present_with_expected_flags(self):
        for name, (family, novelty, smoothing, adaptive) in EXPECTED.items():
            with self.subTest(variant=name):
                spec = resolve_algorithm(name)
                self.assertEqual(spec["family"], family)
                self.assertEqual(spec["novelty"], novelty)
                self.assertEqual(spec["smoothing"], smoothing)
                self.assertEqual(spec["adaptive_plan"], adaptive)

    def test_canonical_eight_are_distinct_specs(self):
        # No two canonical variants should resolve to the same behaviour.
        seen = {}
        for name in CANONICAL_8:
            s = resolve_algorithm(name)
            key = (s["family"], s["novelty"], s["smoothing"], s["adaptive_plan"])
            self.assertNotIn(key, seen, f"{name} duplicates {seen.get(key)}")
            seen[key] = name

    def test_si_smooth_is_novelty_off(self):
        # MATLAB resolve_algorithm_spec.m defines SI_smooth as novelty-OFF
        # (identical to SI_smooth_noNovelty); novelty-ON smoothing is
        # SI_novelty_smooth. Guards the prior Python/MATLAB conflict.
        self.assertFalse(resolve_algorithm("SI_smooth")["novelty"])
        self.assertEqual(
            resolve_algorithm("SI_smooth"),
            resolve_algorithm("SI_smooth_noNovelty"),
        )

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            resolve_algorithm("not_a_real_variant")


if __name__ == "__main__":
    unittest.main()

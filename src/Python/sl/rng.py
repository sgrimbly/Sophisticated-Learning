"""Centralized RNG plumbing.

The MATLAB code mixes ``rng(seed,'twister')`` and global ``rand`` calls. We
replace this with an explicit ``numpy.random.Generator`` (PCG64) that is
threaded through every function that *legitimately* needs randomness:

  - environment state transitions (env.update_environment_states)
  - environment observation sampling (env.sample_observations)
  - posterior-based action sampling (state_selection='sample')

Anything else (calculate_posterior, spm_backwards, tree-search action loops)
must NOT touch this generator. The Python port must drain the RNG only at
the same conceptual events as MATLAB does, otherwise statistical results
diverge from the reference.

We intentionally do NOT try to match MATLAB's Mersenne-Twister sequence —
seed-exact parity isn't a project goal. Different streams, same statistics.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Build a per-run Generator. Use one per worker / per seed."""
    return np.random.default_rng(seed)


def sample_categorical(p: np.ndarray, rng: np.random.Generator) -> int:
    """Inverse-CDF categorical sample, matching MATLAB ``find(cumsum(p)>=rand,1)``.

    Tolerates unnormalised input (renormalises). Handles all-zero by sampling
    uniformly. Returns a 0-based integer index.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    p[~np.isfinite(p)] = 0.0
    p[p < 0.0] = 0.0
    total = p.sum()
    if total <= 0.0:
        return int(rng.integers(0, p.size))
    p = p / total
    r = rng.random()
    # np.searchsorted(cumsum, r) is the same as MATLAB find(cumsum>=r, 1)
    # provided we pass side='left' (default). Clamp for r→1.0 boundary.
    idx = int(np.searchsorted(np.cumsum(p), r, side="left"))
    if idx >= p.size:
        idx = p.size - 1
    return idx


def select_from_posterior(
    p: np.ndarray, mode: str, rng: Optional[np.random.Generator] = None
) -> int:
    """Port of select_from_posterior.m. ``mode`` is 'sample' or 'map'."""
    p = np.asarray(p, dtype=np.float64).ravel()
    p[~np.isfinite(p)] = 0.0
    p[p < 0.0] = 0.0
    if p.sum() <= 0.0:
        p = np.ones_like(p) / p.size

    if mode.lower() == "map":
        return int(np.argmax(p))
    if mode.lower() == "sample":
        if rng is None:
            raise ValueError("select_from_posterior(mode='sample') requires rng")
        return sample_categorical(p, rng)
    raise ValueError(f"Unknown mode {mode!r}. Expected 'sample' or 'map'.")

"""Faithful Python port of the MATLAB Sophisticated-Learning algorithm suite.

Subpackages:
    planning : SI / SL / BA / BAUCB tree-search planners
Modules:
    config   : dataclasses for run configuration
    env      : generative model and per-step state/observation updates
    inference: spm_cross / spm_backwards / calculate_posterior (no sampling)
    efe      : G_epistemic_value, kldir, observation-preference helper
    learning : Dirichlet update for a[2] with row-0 penalty
    agent    : per-trial main loop
    rng      : centralized numpy.random.Generator usage
    runner   : single-run CLI dispatch
    sweep    : multiprocessing sweep driver

Indexing convention (entire package): 0-based throughout. Resource locations,
start_position and hill_pos are converted from MATLAB 1-based at config
ingestion time only (see :func:`sl.config.GridConfig.from_matlab_indices`).

Action numbering: 0 = stay, 1 = left, 2 = right, 3 = up, 4 = down — matching
MATLAB ``B{1}(:,:,1..5)`` after a 1-based -> 0-based shift.
"""
from .config import GridConfig, RunOptions, Weights  # noqa: F401

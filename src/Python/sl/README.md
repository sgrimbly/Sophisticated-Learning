# `sl` — Faithful Python port of the Sophisticated-Learning MATLAB suite

This package replaces the older `src/Python/` attempt, which had several
algorithmic divergences from the canonical MATLAB code (see
`MATLAB_PYTHON_EQUIVALENCE.md` for the full list). It is designed to
replicate MATLAB's *statistical performance* (not seed-exact behaviour),
to be runnable on the Ada Slurm cluster, and to be straightforward to
parallelise across seeds via `multiprocessing`.

**See `BENCHMARK_RESULTS.md` for empirical speed and statistical-parity
numbers** comparing this package against the canonical MATLAB
`paper_si_sl_repro_default_env` reference run. TL;DR: the full JIT path
is **6-9× faster** than NumPy and statistically indistinguishable from
MATLAB on a 10×5 SI sweep (KS=0.10) and a 5×5 SL sweep (KS=0.16).

## Layout

```
sl/
  config.py             dataclasses: GridConfig, RunOptions, Weights
  rng.py                centralised np.random.Generator helpers
  env.py                generative model + per-step state/observation updates
  inference.py          spm_cross / spm_backwards / calculate_posterior (no sampling)
  efe.py                G_epistemic_value, kldir, observation-preference
  learning.py           Dirichlet update with row-0 penalty
  planning/
    common.py           PlannerInputs, PlanResult, helpers
    si.py               SI tree search
    sl.py               SL tree search (historical novelty loop)
    ba.py               BA (extrinsic-only)
    baucb.py            BAUCB (with visit counts and exploration bonus)
  agent.py              per-trial main loop (replaces sophisticated_agent.py)
  runner.py             single-run CLI dispatch (analogue of MATLAB main.m)
  sweep.py              multiprocessing sweep driver
  numba_kernels.py      optional JIT acceleration of G_epistemic_value
  reference_trace.py    side-by-side comparison with MATLAB .mat dumps
  tests/                unit-test suite (algebraic invariants)
  scripts/
    dump_matlab_reference.m   run in MATLAB to produce a reference trace
```

## Quick start

```bash
# 1. Run one experiment (one seed, default canonical config):
python -m sl.runner --algorithm SL --seed 1 --num-trials 5 --max-horizon 9

# 2. Multi-seed sweep across algorithms (uses multiprocessing.Pool):
python -m sl.sweep \
    --algorithms SL SI BA BAUCB \
    --seeds 1-30 \
    --num-trials 120 \
    --max-horizon 9 \
    --output-dir results/python_sweep_$(date +%Y%m%d) \
    --workers 8

# 3. Run the unit-test suite:
python -m unittest discover -s sl/tests -t .
```

## Algorithm names

```
SI                     SL                     BA           BAUCB
SI_noNovelty           SL_noNovelty
SI_smooth              SL_noSmooth
SI_smooth_noNovelty    SL_noNovelty_noSmooth
                       SL_adaptivePlan
```

Variants share the SI/SL planner code via flags resolved through
`config.resolve_algorithm`. See `ALGORITHM_VARIANTS` in `config.py` for
the exact mapping.

## Validating against MATLAB

The intended workflow (one-time per major change):

```matlab
% In MATLAB, from src/Python/sl/scripts/:
dump_matlab_reference('si_seed1.mat', 'SI', 1, 2, 4)
dump_matlab_reference('sl_seed1.mat', 'SL', 1, 2, 4)
```

Then in Python:

```bash
python -m sl.reference_trace si_seed1.mat
python -m sl.reference_trace sl_seed1.mat
```

The harness reports per-trial deltas in chosen-action histograms,
terminal step, and final `a_resource`. Expectation: distributions are
close, individual seeds are not bit-identical. KL between MATLAB and
Python action histograms < 0.1 over 30 seeds is a reasonable bar for
"statistical parity".

## Numba JIT (optional, two tiers)

There are two levels of Numba acceleration. Both require `pip install numba`.

### Tier 1 — moderate JIT (`--enable-numba`)
Swaps in a JIT'd `G_epistemic_value` only. Modest speedup on SI (~1.6×),
negligible on SL (the SL bottleneck is elsewhere).

```bash
python -m sl.runner --enable-numba --algorithm SI --seed 1
```

### Tier 2 — full JIT tree-search recursion (`use_jit_planner=True`)
The entire recursive tree search is compiled. Empirical speedup on a
single CPU core (canonical horizon=9 setup):

| Algorithm | NumPy ms/step | Full JIT ms/step | Speedup |
|---|---|---|---|
| SI | ~46 | ~7 | **~6.7×** |
| SL | ~273 | ~43 | **~6.4×** |

Enable via either:
  * `RunOptions(use_jit_planner=True)` directly, or
  * `python -m sl.scripts.benchmark_and_compare --use-jit-planner ...`
    for a sweep, or
  * adding `use_jit_planner=True` to the worker arg in `sl.sweep`
    (currently you can subclass / patch — a CLI flag is on the to-do list).

The JIT path is numerically identical to the NumPy path to FP precision
(unit tests assert ``allclose(G_jit, G_numpy, rtol=1e-12)``). Action
choice at the root may differ by one slot when two actions have EFE
values tied within rounding error — that's expected statistical noise,
not a parity failure.

Compilation cost: ~1-2 seconds for SI, ~2-5 seconds for SL on the first
call per Python process. Numba's ``cache=True`` writes the compiled code
to ``__pycache__`` so subsequent processes can re-use it. With
``multiprocessing.Pool`` over seeds, each worker pays the cache-load
cost once then runs all assigned trials JIT-compiled.

## Known limitations

* No seed-exact MATLAB parity. RNG sequencing diverges (MATLAB uses
  Mersenne-Twister `rng(seed,'twister')`, Python uses NumPy PCG64).
  Statistical-level parity is the design goal.

* Single-trial runtime for SL at horizon=9 is around 10-20 s on Ada CPU
  hardware; SI is closer to 2-5 s. With multiprocessing across seeds
  the throughput is roughly equivalent to MATLAB's parfor approach.

* `--enable-numba` is opt-in; without numba the package depends only on
  NumPy + SciPy.

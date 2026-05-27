# MATLAB experiments
This folder contains the MATLAB scripts for the experiments in the paper. The scripts are organized as follows:

- `main.m`: The main entrypoint (function) to run experiments.
- `algorithms`: Contains the implementations of the four algorithms presented in the paper.
- `utils`: Contains utility functions used in the experiments.
- `tree-search`: Contains the implementation of the tree search algorithm used in the experiments.
- [`PERFORMANCE.md`](PERFORMANCE.md): Bottleneck audit and ranked CPU-only speedup recommendations (memoisation, vectorisation, MEX/Coder), plus a calibration vs. the Python+Numba port.

## Quick start
From the repo root in MATLAB:

```matlab
addpath(genpath('src/MATLAB'));
main('SL', 1);            % algorithm, seed (defaults match paper settings)
main('SI', 1);            % legacy-compatible SI (novelty enabled)
main('SI_noNovelty', 1);  % SI ablation (novelty disabled)
main('SL_noSmooth', 1);   % SL ablation (no backward smoothing)
main('SI_smooth_noNovelty', 1); % SI smoothed ablation without novelty
main('SI_novelty_smooth', 1);   % SI smoothed variant with novelty
main('SL_noNovelty', 1);  % SL ablation (novelty term disabled)
main('SL_noNovelty_adaptivePlan', 1); % novelty-off SL with adaptive planning
main('SL_noNovelty_noSmooth_adaptivePlan', 1); % no-smooth novelty-off SL with adaptive planning
```

## Legacy mapping
- `rowan-version/SI_rowan.m` is the older Rowan SI baseline: novelty on, uniform `a{2}` prior, 120 trials, default MATLAB RNG.
- `rowan-version/SL_rowan.m` is the older Rowan SL baseline: novelty on, informative `a{2}` prior, 120 trials, default MATLAB RNG.
- `algorithms/unknown-models/SI.m` is the later direct SI descendant with the same uniform-prior semantics as Rowan SI, but using `rng(seed, "threefry")` and 200 trials.
- `algorithms/unknown-models/SL.m` is the later direct SL descendant with the same informative-prior semantics as Rowan SL, but using `rng(seed, "threefry")` and 200 trials.
- The modular labels in `main.m` are explicit ablations of the modular codepath. `SI` is the closest current modular match to legacy SI semantics. Current modular `SL`-family labels do not reproduce legacy/Rowan `SL` exactly, because the modular initializer uses a uniform `a{2}` prior rather than the informative prior in `SL_rowan.m` and `SL.m`.

## Reproducibility flags
These are passed via the `weights` struct and recorded into the per-run `config_id` and `run_meta`:

- `weights.state_selection`: `'sample'` (legacy) or `'map'` (deterministic)
- `weights.preference_param`:
  - `'weight'` (default): `weights.preference` strengthens extrinsic terms as it increases
  - `'inverse_precision'`: `weights.preference_inverse_precision` (or `weights.preference`) weakens extrinsic terms as it increases
- `weights.baucb_variant`: `'legacy'` (default) or `'fixed_joint_counts'` (bugfixed BAUCB)

## Outputs
- Results (survival per trial) are written to `results/` by default. Override with env var `SL_RESULTS_ROOT`.
- Optional per-trial diagnostic metrics are written alongside results when `SL_LOG_METRICS=1` (CSV with `_metrics.csv` suffix).

## Horizon
The `horizon` argument in `main(...)` controls the maximum planning horizon for modular unknown-model algorithms (default: 9), and is also used as the horizon for known-model experiments.

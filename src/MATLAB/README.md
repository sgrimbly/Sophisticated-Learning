# MATLAB experiments
This folder contains the MATLAB scripts for the experiments in the paper. The scripts are organized as follows:

- `main.m`: The main entrypoint (function) to run experiments.
- `algorithms`: Contains the implementations of the four algorithms presented in the paper.
- `utils`: Contains utility functions used in the experiments.
- `tree-search`: Contains the implementation of the tree search algorithm used in the experiments.

## Quick start
From the repo root in MATLAB:

```matlab
addpath(genpath('src/MATLAB'));
main('SL', 1);            % algorithm, seed (defaults match paper settings)
main('SI', 1);            % SI defaults to no novelty term
main('SI_novelty', 1);    % SI ablation (novelty term enabled)
main('SL_noSmooth', 1);   % SL ablation (no backward smoothing)
main('SI_smooth', 1);     % SI ablation (with backward smoothing)
main('SL_noNovelty', 1);  % SL ablation (novelty term disabled)
```

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

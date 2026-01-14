# Changelog

## 2026-01-14
- Added `real_smoothing` and `adaptive_likelihood_in_plan` run flags (hashed into `config_id` and encoded in filenames) to support reviewer-facing ablations without silently changing defaults.
- Implemented the long-standing TODO in SL tree search as an opt-in: when `adaptive_likelihood_in_plan=true`, SL updates `y{2}=normalise_matrix(a{2})` after simulated learning updates so future simulated observations reflect the updated model.
- Added a `real_smoothing=false` mode for modular runners (SI/SL/BA/BAUCB) that performs filtered-only parameter updates (no `spm_backwards` window) for clean “real smoothing” ablations.
- Updated local dashboard + UCT-HPC-HEX MATLAB runners to pass the new flags.

## 2025-12-10
- Adopted per-(state, context) visit counts for BAUCB/BA_UCB, fixed the `cur_state` typo, moved count updates into the UCB tree search root, and threaded `Nt` through recursion alongside a tunable `ucb_scale` (default 5). Scripts/figures: not re-run yet.
- Exposed preference precision to BA/BAUCB (and legacy BA/BA_UCB) by passing weights from `main`, scaling preferences inside tree search, and adding `ucb_scale` to the run metadata. Scripts/figures: not re-run yet.
- Added ablation variants `SL_noSmooth` and `SI_smooth` (new tree searches, modular wrappers, and main/parallel hooks) with distinct result/state file prefixes. Scripts/figures: not re-run yet.

# Changelog

## 2025-12-10
- Adopted per-(state, context) visit counts for BAUCB/BA_UCB, fixed the `cur_state` typo, moved count updates into the UCB tree search root, and threaded `Nt` through recursion alongside a tunable `ucb_scale` (default 5). Scripts/figures: not re-run yet.
- Exposed preference precision to BA/BAUCB (and legacy BA/BA_UCB) by passing weights from `main`, scaling preferences inside tree search, and adding `ucb_scale` to the run metadata. Scripts/figures: not re-run yet.
- Added ablation variants `SL_noSmooth` and `SI_smooth` (new tree searches, modular wrappers, and main/parallel hooks) with distinct result/state file prefixes. Scripts/figures: not re-run yet.

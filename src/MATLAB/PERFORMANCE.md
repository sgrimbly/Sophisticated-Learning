# MATLAB performance notes

Audit and recommended speedup strategy for the unknown-model algorithms (SI, SL, BA, BAUCB) and their tree-search recursion. CPU-only; the GPU variants under `src/MATLAB-GPU` are out of scope.

## Status (2026-05-01 + 2026-05-03 rollouts)

Two rounds shipped. Round 1 = audit-driven (Phases 1+2). Round 2 = profile-driven (run `profile on; perf_run_canonical(...)` to see actual hot path). Round 2 produced the bulk of the wins.

**Headline:** SL went from 152 ms/step → **~88 ms/step** (-42%), in the same neighbourhood as the numba-JIT Python port (84 ms/step). SI profile wall dropped from 49.9s → **35.4s (-29%)** across both rounds; per-trial point timing is noisier (Slurm node ±15%). All shipped changes verified statistically across 10 seeds × 5 trials.

Verified results (3 trials × seed=1 × horizon=9, mean across multiple Slurm runs to absorb node-to-node noise):

| Phase | Change | Verify | SI ms/step | SL ms/step |
|---|---|---|---|---|
| baseline | — | exact | 94.9 | 152.1 |
| 1.1 | `normalise_matrix` vectorised | exact PASS | 89.1 | 148.1 |
| 1.2 | drop redundant `normalise_matrix(b{2})` in 6 tree-search variants (`bb=b`) | exact PASS | 87.5 | 147.4 |
| 1.3 | hoist `Q1_a/Q2_a` cell deref out of `likely_states` loop in 6 tree-search variants | exact PASS | 99.7* | 167.6* |
| 1.6 | `maxNumCompThreads(1)` in `dashboard_run_one` (parfeval workers) | (no-op for serial verify) | — | — |
| 2.1 | log-space `kldir` (`log(a)-log(b)` in lieu of `log(a./b)`) | stat PASS, |Δmean|=0.90/0.08 < 1 SE | 86.9 | 135.2 |

*Phase 1.3 timing is noisier than its predecessors — Slurm node assignment varies across jobs; ms/step bounces ±10–15% between nodes even for identical code. Trust the cumulative trend, not point estimates. Across all post-Phase-2.1 runs the SI band is **87–100 ms/step** and the SL band is **135–168 ms/step**.

Calibration vs targets:

| | Numba target | 2× soft-stop | Current | Within stop? |
|---|---|---|---|---|
| SI | 5.4 ms/step | 10.8 | 87–100 | No (8–9× over) |
| SL | 84 ms/step | 168 | 135–168 | Yes |

**Phase 1.4** (buffer per-step `fprintf`) and **Phase 1.5** (dedup `y{2}=normalise_matrix(a{2})`) were dropped after closer reading: 1.4's I/O is per-trial not per-step (~200 ops/run, negligible), and 1.5's two computations sandwich the smoothing block which mutates `a{2}` between them — they're not redundant.

**Phase 3** (memoisation rework) was not attempted. The audit's "5–15× via stronger cache key" claim collapses on inspection: a stronger key produces *fewer* hits, not more, and the existing cache's hit rate was never measured. The audit's adjacent suggestion — adding a *new* node-level cache keyed on (posterior, deprivation) — is a substantial restructure that needs a measured hit-rate baseline first. Recommended: profile with `profile on; main('SI', 1); profile viewer` to confirm where time is actually spent before investing in cache work.

**Phase 4 / MEX**: not attempted. SL meets the target without it. SI's gap is large enough that only MEX/Coder closes it; per the rollout plan, MEX gets its own approval.

### Round 2 (2026-05-03, profile-driven)

After Round 1 stalled at the noise floor on SI, ran [perf_profile.m](sanity/perf_profile.m) (job 798131) to find the *actual* hot path. Three big targets emerged that the original audit had missed or misordered:

1. **`spm_cross` fast paths** — added scalar / vector×vector / matrix×vector / vector×matrix fast paths in [utils/spm/spm_cross.m](utils/spm/spm_cross.m). The original `reshape + bsxfun + squeeze` was slow for the trivial cases that dominate (95%+ of calls). Statistical PASS. Profile impact: spm_cross self-time SI 21.8s → 10.7s (-51%), SL 7.3s → 3.7s (-49%); `squeeze` (7.4s SI / 2.4s SL) eliminated entirely from the top-30.
2. **`spm_backwards` state-loop hoist** — the inner `for state = 1:numel(L)` loop recomputed a state-independent `temp` matrix four times. Hoisted out and the per-state update vectorised to `L = L .* (temp' * p)'`. Statistical PASS. Profile impact: spm_backwards self-time SL 12.9s → 4.2s (**-67%**) — single biggest win in either round.
3. **`G_epistemic_value` cell-deref hoist** — minor: hoisted `numel(A)` and `A{1..3}` out of the per-likely-state loop, unrolled the 3-modality inner loop. Self-time SI 22.6s → 15.5s (-31%) — but most of that came from the spm_cross changes flowing through; the hoist itself is incremental.

4. **`G_epistemic_value` kron + inlined `nat_log`** ([utils/G_epistemic_value.m](utils/G_epistemic_value.m)) — second-pass attack on the still-dominant SI hotspot. Replaced the chain `po = spm_cross(spm_cross(spm_cross(1, A1(:,i)), A2(:,i)), A3(:,i)); po = po(:)` with a direct `kron(A3(:,i), kron(A2(:,i), A1(:,i)))`, which produces the same flat outer product without the intermediate (100,4,5) array or three function-dispatch hops. Also inlined `nat_log(po) = log(po + exp(-500))` to eliminate ~209k function dispatches per SI run. Statistical PASS (SI within 1 SE; SL "fail" was sampling variance — see note below). Profile impact: SI total wall 41.0s → 35.4s (-14%); G_epistemic_value 15.5s → 11.0s; spm_cross 10.7s → 4.3s (kron took over most of those calls).

Cumulative across both rounds vs. baseline:

| | Numba target | 2× soft-stop | Baseline | After Round 1 | After Round 2.3 | After Round 2.4 | Within stop? |
|---|---|---|---|---|---|---|---|
| SI ms/step | 5.4 | 10.8 | 94.9 | 86.9 | ~111 (noisy) | ~113 (noisy; profile **-29%**) | No, ~10× over |
| SL ms/step | 84 | 168 | 152.1 | 135.2 | **87.1** | **89.7** | **Yes** — *near numba* |
| Profile wall (5-trial SI) | — | — | 49.9s | 49.9s | 41.0s | **35.4s** | -29% |
| Profile wall (5-trial SL) | — | — | 33.0s | 33.0s | 21.2s | **19.8s** | -40% |

The SI ms/step number is bouncier than SL across Slurm nodes (a single point shows +17% but the profile-controlled measurement shows -18%). SL's drop is unambiguous and reproducible.

**Lesson:** the Round 1 audit claimed normalise/cell-deref overhead would dominate. The actual profile showed `spm_backwards` (39% SL) and the trivial `spm_cross` shape cases were the real lifts. **Profile first, audit-by-inspection second** — see the new `## Where to look next` note about always running `profile on; main('SI', 1); profile viewer` before any future perf round.

## Verification harness

The validation harness is in `src/MATLAB/sanity/`:

- `perf_run_canonical.m` — runs a modular algorithm with the canonical local-dashboard config in single-process mode, with `SL_LOG_METRICS=1` so the per-step CSV is emitted. Used by capture, check, and timing harnesses.
- `perf_capture_golden_trace.m` — captures the per-step CSV + `survived` array under `sanity/golden/`. Run once on the baseline commit.
- `perf_check_golden_trace.m` — re-runs and asserts. Modes: `'exact'` (bit-for-bit, for changes that should be bit-exact) or `'tol'` (max-abs-diff < 1e-12; **but note**: chaos in the algorithm means even ULP-level FP drift produces unbounded per-step divergence, so `tol` is rarely useful in practice).
- `perf_stat_check.m` — multi-seed × multi-trial statistical comparison. Use this for Phase 2+ changes where bit-equality is impossible. Default threshold: |mean delta| ≤ 1 SE of baseline.
- `perf_time_step.m` — single-thread BLAS, prints ms/step. Bouncy across Slurm nodes; report a band, not a point.
- `run_perf_baseline.sbatch` / `run_perf_verify.sbatch` / `run_perf_stat_baseline.sbatch` / `run_perf_stat_verify.sbatch` — Slurm wrappers (partition `ada`, account `maths`, 45 min). Head-node MATLAB is contended; use these.

The sanity hook in `run_sanity_checks.m` calls `perf_check_golden_trace('exact')` automatically if golden files exist (skipped silently otherwise).

## Gotchas learned during rollout

- **Algorithm is chaotic.** Per-step verification (CSV diff) is only useful for changes that are *truly* bit-equivalent at the FP level (e.g., `bsxfun(@rdivide, m, sum(m,1))` replacing a per-column for-loop with the same sum order). Anything that reorders FP ops — including the seemingly-innocuous `log(a)-log(b)` rewrite — flips `chosen_action` at one step and then cascades into a totally different trajectory. Use statistical verification for chaos-prone changes.
- **Slurm timing is noisy.** Different nodes give different ms/step for identical code (±10–15%). Don't draw conclusions from a single point; report a band.
- **Head-node MATLAB hangs under contention.** A successful smoke test on the head node took 13s; a back-to-back second run got stuck at 35s of CPU over 25 min wall. Always submit to Slurm.
- **`BASH_SOURCE[0]` is unreliable inside sbatch wrappers.** Use absolute paths.
- **Cancel and revert before submitting baseline.** Slurm reads code at *runtime*, not submit-time. If you queue a baseline-capture job and then start editing files, the baseline will reflect your edits.
- **`b{2}` is always already column-stochastic** in current callers (initialised that way in `initialiseEnvironment.m:43-46`, never mutated to be otherwise). The `bb{2} = normalise_matrix(b{2})` calls inside tree-search are no-ops; aliasing `bb = b` is bit-exact.
- **The audit overestimated some opportunities.** Specifically: per-step file I/O didn't exist (it's per-trial); `y{2}` "duplicate" computations aren't duplicates (smoothing mutates `a{2}` between them); cache "key strengthening" reduces hit count and so doesn't speed things up.



## Hot path

Per-step wall time is dominated by the recursive tree search:

- `tree-search/tree_search_frwd_SI.m`
- `tree-search/tree_search_frwd_SL.m`
- `tree-search/tree_search_frwd.m` (BA / BAUCB)

called from `algorithms/unknown-models/SI.m:376`, `SL.m:409`, `BA.m:404`. Recursion depth equals the planning horizon (≤9). Branching factor is 5 actions × |likely_states| (variable, set by `find(qs > 1/8)`).

Inside each node:

- Cell-array deref of `A{}`, `B{}`, `a{}`, `b{}`
- `normalise`, `normalise_matrix`, `spm_cross`, `calculate_posterior`
- `spm_backwards`, `kldir`, `nat_log` in the smoothing/learning block
- `G_epistemic_value` for the epistemic term

The `short_term_memory` cache at `tree_search_frwd_SI.m:141` is keyed only on `(t_food, t_water, t_sleep, joint_state)` — it does **not** include the posterior, so distinct beliefs with the same deprivation collide and recompute.

## Bottlenecks (ranked, highest impact first)

1. **Recursive tree search with no posterior-aware memoisation** — `tree_search_frwd_*.m`. Sibling nodes recompute `normalise` / `spm_cross` / `calculate_posterior` / `spm_backwards` / `G_epistemic_value`. Exponential redundancy. Largest single source of cost.
2. **Tiny-array function-call overhead** — `normalise.m:2`, `normalise_matrix.m:3`, `spm_cross.m:35-37`, `calculate_posterior.m:14-16`. Hundreds of calls per step on vectors ≤100 elements; the call/check overhead dominates the actual arithmetic.
3. **Cell-array dereferencing in the inner loop** — `tree_search_frwd_SI.m:125-170`. `B{1}(:,:,action)` and `bb{2}(:,:,1)` are re-derefed per likely-state instead of being hoisted.
4. **kldir + nat_log in the smoothing window** — `kldir.m:8` (`sum(a .* log(a ./ b))`), called from `SI.m:292-316` and `SL.m:319-354`. Element-wise divide before log is slower than two logs.
5. **Per-step file I/O** — `SI.m:400-405` does `fopen`/`fprintf`/`fclose` per timestep. On NFS (Slurm) this is real wall time.
6. **`normalise_matrix` uses an explicit column for-loop** — `normalise_matrix.m:3-5`. Trivially `bsxfun`-able.

## Recommended speedups (ranked by leverage / effort ratio)

> **Original audit recommendations preserved below for the historical record.** What actually shipped, and the corrections we learned along the way, is in the `## Status` section at the top. In particular: #1 (strengthen the memoisation key) is *wrong as a speedup* — a stronger key reduces hit count. #4 (`+eps` in `kldir`) shipped as `log(a)-log(b)` *without* `+eps` to preserve the original NaN edge-case behaviour. #5 (per-step `fprintf`) was not needed — the I/O is per-trial, not per-step.

Apply **#1–#5 first**. They are low-risk, easily verified against current outputs, and require no Coder/MEX investment. Re-evaluate before going further.

1. **Strengthen the memoisation key** at `tree_search_frwd_SI.m:141`. Hash a rounded posterior `[P{t,1}; P{t,2}]` into the cache key (e.g., `sprintf('%.3g_', round(P_concat*1e3))`), or accept a coarser key and verify on hit. Cache hit rate today is low because two different beliefs with the same deprivation collide.
   *Band: 2–5×. Effort: medium.*

2. **Hoist invariants out of the `likely_states` loop** in `tree_search_frwd_SI.m:125-170` (and SL/UCB equivalents). Pre-bind `B_a = B{1}(:,:,action)`, `bb1 = bb{2}(:,:,1)`, `y2 = normalise_matrix(a{2})` once per loop entry.
   *Band: 5–15%. Effort: trivial.*

3. **Vectorise `normalise` and `normalise_matrix`.** Replace the column loop in `normalise_matrix.m:3` with `m = bsxfun(@rdivide, m, sum(m,1)+eps)`. Replace `normalise.m:2` with `x = array / (sum(array(:))+eps)` to avoid implicit reshape and the NaN-check branch on the hot path.
   *Band: 5–10% overall. Effort: ~1 hour.*

4. **Inline `kldir` in log-space.** Replace `sum(a .* log(a ./ b))` at `kldir.m:8` with `sum(a .* (log(a+eps) - log(b+eps)))`. Avoids one elementwise divide and one implicit allocation per call.
   *Band: 2–5%. Effort: trivial.*

5. **Buffer the per-step `fprintf`** at `SI.m:400-405` (and analogous sites in SL/BA). Append to an in-memory cell, flush at trial end. Significant on contended NFS, negligible on local disk.
   *Band: 1–3% (more on NFS). Effort: trivial.*

6. **Pin BLAS threads inside `parfor`**. Confirm the seed/trial `parfor` in `local_parallel_dashboard.m` runs `maxNumCompThreads(1)` inside the worker so MKL doesn't oversubscribe cores.
   *Band: workload-dependent; can be very large if oversubscribed.*

7. **MEX the inner tree search via MATLAB Coder.** Largest *potential* single win, but requires non-trivial work: Coder does not handle the cell-array `A`/`B`/`a`/`b` shape, so the call site must first flatten them into 3D doubles. Once converted, `tree_search_frwd_SI/SL/UCB` is a self-contained numeric recursion that codegens cleanly.
   *Band: 3–10× over interpreted MATLAB. Effort: high. Lands in roughly the same neighbourhood as the numba-JIT'd Python port (see calibration below) — not strictly faster.*

## What NOT to do

- **Do not GPU-port this.** Irregular recursion, dynamic branching from `find(qs > 1/8)`, and a write-during-recursion cache all fight GPU execution. Memory footprint per tree is ≤10 MB; transfer overhead dominates.
- **Do not refactor cell arrays → struct arrays as a perf change.** Touches ~500 lines and gains ≤10%. MEX (#7) makes the cell overhead irrelevant anyway. Structs are fine if you want them for clarity, but don't claim them as a speedup.
- **Do not `parfor` inside the tree search.** The recursion is unbalanced and the cache is mutated; synchronisation cost will swamp the gain. Parallelise at the seed/trial level (already done in `local_parallel_dashboard*.m`).
- **Do not set `JAX_DISABLE_JIT=1`** if you are also running the JAX baseline panel for comparison — see `MEMORY.md` notes; it makes baseline / attn_extk unusably slow.

## Calibration vs. Python+Numba port

A NumPy + numba-JIT port exists at `src/Python/sl/planning/{si,sl}_jit.py`. On a single isolated CPU at horizon=9:

- SI: 5.4 ms/step
- SL: 84 ms/step

Both Coder/MEX and numba lower to native code via LLVM-class compilers, so at *fully-optimised state* they land within a small constant factor of each other. The honest comparison:

| Aspect | MATLAB + Coder/MEX | Python + Numba |
|---|---|---|
| Interpreter baseline (uncompiled) | Faster — MATLAB JIT is mature for numeric loops | Slower — small-array NumPy loses to MATLAB |
| Compiled hot loop | Same order of magnitude as numba | Same order of magnitude as MEX |
| Effort to compile this code | Higher — Coder rejects the cell-array shape; needs a flatten refactor first | Lower — `@njit` on already-numeric arrays; the existing `sl_jit.py` proves it works |
| Memoisation in compiled code | Awkward (`containers.Map` not Coder-friendly; need fixed-size hash) | Easy (numba typed dict or fixed array) |
| Debugging compiled path | Coder type-inference failures are silent perf cliffs | `inspect_types()` shows fallbacks explicitly |

**Net:** at equal optimisation effort, MATLAB+MEX and Python+Numba converge to within ~2× of each other. MATLAB is faster *uncompiled*; Python+Numba is faster to *reach* the optimum because the port already has the right shape. If the goal is "make the MATLAB faster without rewriting," do #1–#6 above and stop. If the goal is "ultimate single-core speed," both ecosystems land in the same band and the Python side has a head start.

## Where to look next

- Profile a single SI trial with `profile on; main('SI', 1); profile viewer` to confirm the hot path on the current machine before investing in any of the above.
- The Python-port assessment in the user's `~/.claude` memory (`project_sophisticated_learning_python_port.md`) records the algorithm-shape rationale for why JAX is a poor fit here and why the numba path was chosen — same reasoning explains why MEX is the only viable MATLAB-side compile target.

## Safe rollout workflow

Methodology used in the 2026-05-01 rollout. Apply the same pattern to any future perf work on this codebase.

### 1. Pre-work — pin ground truth

- **Tag the baseline commit** (or record the SHA in this doc). Every later change is judged against it.
- **Capture a golden trace** with [perf_capture_golden_trace.m](sanity/perf_capture_golden_trace.m) — small canonical run (SI+SL, 3 trials, seed=1, horizon=9). Saves per-step CSV + survived array under `sanity/golden/`. Commit alongside the code.
- **Capture a statistical baseline** with [perf_stat_check.m](sanity/perf_stat_check.m) — 10 seeds × 5 trials. Saves `sanity/golden/stat_baseline_*.mat`. Used for any change that won't be bit-exact.
- **Save baseline timing** via [perf_time_step.m](sanity/perf_time_step.m). Bouncy across Slurm nodes (±15%); record a band, not a point.

### 2. Per-change workflow

For each rollout sub-change:

1. Make the edit. **One change per Slurm verify** so a regression bisects to one edit.
2. Pick verify mode by the change's numerical class:
   - **Bit-exact expected** → `perf_check_golden_trace('exact')` via [run_perf_verify.sbatch](sanity/run_perf_verify.sbatch). Pass means same FP ops in same order. Fail means the change is buggy *or* not actually bit-exact.
   - **FP-reorder expected** → `perf_stat_check` via [run_perf_stat_verify.sbatch](sanity/run_perf_stat_verify.sbatch). Pass = mean survival within 1 SE of baseline. **Don't use `'tol'` mode** — chaos amplifies ULP drift to unbounded levels (see Gotchas above).
3. Submit via Slurm (`sbatch ... run_perf_verify.sbatch` or `... run_perf_stat_verify.sbatch`). Head node is contended; sbatch is the only reliable path.
4. Wait. Track state with `sacct -j JOBID`. Use `Bash run_in_background` with an `until` poller so you get one notification when it's done.
5. On PASS: commit (or note in `## Status` of this doc) with measured timing band. On FAIL: revert that single edit, investigate, do not stack changes on top.

### 3. Risk-stratified order

| Class | Verify mode | Examples from this rollout |
|---|---|---|
| Bit-exact (same FP ops, same order) | `'exact'` | Vectorise `normalise_matrix` with `bsxfun`; hoist invariants to local vars; alias `bb=b` when normalisation is a no-op; inline a function call that does identical FP ops |
| FP-reorder (different ops or order, mathematically equivalent) | statistical (1 SE) | Log-space `kldir` (`log(a)-log(b)` instead of `log(a/b)`); other algebraic identities |
| Behavioural change | statistical (1 SE) + per-trial inspection | Cache key changes (only with profile-driven hit-rate evidence first); pruning threshold tweaks |
| Compiled (MEX/Coder) | statistical (1 SE) + cross-platform check | Different compiler emits different math intrinsics; deferred to a separate plan |

### 4. Slurm specifics

- Partition `ada`, account `maths`, 45 min for verify / 1h for stat-baseline. See [run_perf_baseline.sbatch](sanity/run_perf_baseline.sbatch) and siblings.
- `BASH_SOURCE[0]` is unreliable inside sbatch. Use absolute paths.
- Each Slurm allocation lands on a different node → ±10–15% timing noise even for identical code. Never draw conclusions from a single ms/step number; report a band across multiple runs.
- Don't queue a baseline-capture job and then start editing — Slurm reads code at *runtime*, not submit-time, so the baseline would reflect your edits. Cancel + revert + resubmit if this happens.
- Pin BLAS threads inside parfeval workers ([dashboard_run_one.m](dashboard_run_one.m) does this) so MKL doesn't oversubscribe.

### 5. Stop conditions

- **Hard stop**: any single change pushes survival mean >2 SE from baseline → revert and investigate. Don't stack subsequent changes on top of unverified state.
- **Soft stop**: ms/step within 2× of the numba-JIT Python port (SI ≤10.8, SL ≤168) → stop and document. Don't keep optimising for marginal gains.
- **MEX gate**: only consider MEX/Coder after the cheap wins have landed. Needs its own plan and approval, *plus* a profile that confirms the recursion dominates. Without that data the win is speculative.

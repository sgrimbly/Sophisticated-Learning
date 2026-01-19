# MATLAB ↔ Python Equivalence Notes (Modular Unknown‑Model Algorithms)

This note compares:

- **MATLAB reference (ground truth)**: modular algorithm entrypoints in `src/MATLAB/algorithms/unknown-models/modular_versions/` and shared utilities in `src/MATLAB/utils/` + `src/MATLAB/tree-search/`.
- **Python port attempt**: `src/Python/` (primarily `src/Python/algorithms/sophisticated_agent/sophisticated_agent.py`, `src/Python/agent_utils.py`, `src/Python/forward_tree_search.py`, `src/Python/active_inference_utils.py`, `src/Python/spm_utils.py`).

Goal: list **every observed mismatch / issue / modeling-choice difference** in the Python attempt, and give **explicit steps** to make the Python and MATLAB implementations essentially equivalent (for the same seed/config).

---

## 1) Canonical MATLAB Model (What the Modular Versions Implement)

### 1.1 Hidden state factors (unknown-model task)

The MATLAB “unknown model” gridworld has **two hidden-state factors**:

1. **Position**: `num_states = grid_size^2` (default 100 for a 10×10 grid).
2. **Context**: `num_contexts = 4` (a latent “season” / context that changes stochastically).

The **joint state** is the Cartesian product: `num_joint_states = num_states * num_contexts` (default 400).

### 1.2 Observation modalities

MATLAB uses **three observation modalities** (see `src/MATLAB/utils/initialiseEnvironment.m`):

1. **Modality 1 (position identity)**: outcomes = `num_states`. Deterministic `A{1}(i,i,ctx)=1`.
2. **Modality 2 (resource cue)**: outcomes = 4: `[empty, food, water, sleep]`. Deterministic in the *environment* `A{2}`, but **unknown to the agent** (agent learns `a{2}`; initialized to 0.1 everywhere).
3. **Modality 3 (hill cue)**: outcomes = 5: `[ctx1, ctx2, ctx3, ctx4, none]`. Deterministic and **known** to the agent (`a{3} = A{3}`).

### 1.3 Transition dynamics (B)

Two transition models (see `src/MATLAB/utils/initialiseEnvironment.m`):

- **B{1}** (position): 5 actions: stay + 4 moves on the grid (deterministic given action; edge behaviour determined by the exact `circshift` logic).
- **B{2}** (context): a fixed 4×4 stochastic matrix, applied independent of action: “mostly stay, sometimes advance context”, cyclic.

Agent beliefs `b` are set equal to true transitions in the current MATLAB code (i.e., context transitions are *not* learned in these runs).

### 1.4 Needs / survival constraints

In modular MATLAB loops (e.g. `src/MATLAB/algorithms/unknown-models/modular_versions/SI_modular.m`) the agent remains alive while:

- `time_since_food < 22`
- `time_since_water < 20`
- `time_since_sleep < 25`
- and `t < 100`

### 1.5 Shared inference + smoothing + learning logic (outside tree-search)

Across the modular algorithms, the core per-timestep pattern is:

1. Propagate hidden states with `src/MATLAB/utils/updateEnvironmentStates.m`.
2. Sample observations from `A`.
3. Compute a **predicted posterior** (via predicted observation distributions) and an **actual posterior** (via actual observations) using `src/MATLAB/utils/calculate_posterior.m`.
4. Run backward smoothing of **context beliefs** using only the hill modality with `src/MATLAB/utils/spm/spm_backwards.m`.
5. Update the learned likelihood parameters `a{2}` using the smoothed context posterior (and a specific penalised update rule).
6. Reset short-term memory if:
   - hill visited, or
   - context prediction error (predicted vs actual posterior, rounded to 1 decimal).

### 1.6 Algorithm family differences (what changes across BA / SI / SL / etc.)

The *main loop + learning update* is largely shared; the key differences are the **tree-search functions** called:

| MATLAB modular entrypoint | Tree search used | EFE terms | Simulates learning in planning? | Memory tensor shape |
|---|---|---|---|---|
| `BA_modular.m` | `tree_search_frwd.m` | extrinsic only | no | `(35,35,35,num_joint_states,5)` |
| `BAUCB_modular.m` | `tree_search_frwd_UCB.m` | extrinsic + UCB bonus | no | `(35,35,35,num_joint_states,5)` + visit counts `Nt` |
| `SI_modular.m` | `tree_search_frwd_SI.m` | extrinsic + epistemic + novelty | no | `(35,35,35,num_joint_states)` |
| `SI_smooth_modular.m` | `tree_search_frwd_SI_smooth.m` | extrinsic + epistemic + novelty (with smoothing window) | no | `(35,35,35,num_joint_states)` |
| `SL_modular.m` | `tree_search_frwd_SL.m` | extrinsic + epistemic + novelty | **yes** (updates `a` inside tree search) | `(35,35,35,num_joint_states)` |
| `SL_noSmooth_modular.m` | `tree_search_frwd_SL_noSmooth.m` | extrinsic + epistemic + novelty | **yes** (updates `a` inside tree search) | `(35,35,35,num_joint_states)` |

Key references:

- Tree-search functions: `src/MATLAB/tree-search/`
- Preferences: `src/MATLAB/utils/determineObservationPreference.m`
- Epistemic value: `src/MATLAB/utils/G_epistemic_value.m`
- Novelty (KL on Dirichlet updates): `src/MATLAB/utils/kldir.m`

---

## 2) Python Port Attempt: What Exists Today

### 2.1 Implemented algorithms

Python only exposes **SI** and **SL** via `src/Python/main.py` → `experiment()` in:

- `src/Python/algorithms/sophisticated_agent/sophisticated_agent.py`

The codebase explicitly states BA / BAUCB are not implemented.

There are also “variant” MATLAB algorithms (`SI_smooth`, `SL_noSmooth`) that are not selectable in Python.

### 2.2 Python module responsibilities (current intent)

- `src/Python/agent_utils.py`:
  - Initialize A/a/B/b/D (attempted equivalent to `initialiseEnvironment.m`).
  - Update environment and needs counters.
  - Compute observations.
  - Compute predicted/actual posterior.
  - Run backward smoothing + update `a`.
- `src/Python/spm_utils.py`:
  - `spm_cross` and `spm_backwards`.
- `src/Python/active_inference_utils.py`:
  - `calculate_posterior`, `calculate_novelty`, `G_epistemic_value`, `determine_observation_preference`.
- `src/Python/forward_tree_search.py`:
  - `forward_tree_search_SI` / `forward_tree_search_SL`.

---

## 3) Issues + Modeling Differences in the Python Version (vs MATLAB Modular Ground Truth)

This section is intentionally exhaustive and “nitpicky” because small differences change trajectories, learning, and RNG ordering.

### 3.1 Hard failures / correctness blockers

1. **`src/Python/algorithms/sophisticated_agent/sophisticated_agent.py` contains syntax errors**:
   - `logging.info(f"TRIAL {trial+1} COMPLETE バ"")`
   - `logging.info(f"EXPERIMENT COMPLETE バ".")`
   These prevent the Python experiment runner from executing.

2. **Missing/unused module path**:
   - `src/Python/algorithms/sophisticated_inference/` exists but is empty.
   - The notebook `src/Python/algorithms/sophisticated_agent/sophisticated_agent.ipynb` imports `from sophisticated_inference import agent_loop`, which cannot work with the current tree.

3. **Matplotlib backend hard-coded**:
   - `matplotlib.use('Qt5Agg')` inside `sophisticated_agent.py` can fail in headless / non-Qt environments and is unrelated to algorithm parity.

### 3.2 Generative process / environment mismatches

1. **B{1} (grid movement) does not match MATLAB**:
   - MATLAB movement is defined in `src/MATLAB/utils/initialiseEnvironment.m` using `circshift` and specific boundary checks.
   - Python reimplements movement in `src/Python/agent_utils.py` using `np.roll` and **row/column boundary sets**.
   - The up/down rolls in Python are the opposite sign of MATLAB’s `circshift` calls (and boundary conditions differ), meaning the *graph of reachable states under each action* can differ.

2. **Hard-coded 10×10 assumptions** in Python:
   - `agent_utils.set_resource_locations()` hard-codes A/a shapes as `(100,100,4)`, `(4,100,4)`, `(5,100,4)` instead of using `num_states`.
   - Movement exclusions (`exclude_left`, etc.) are built assuming 10×10 (step size 10).

3. **Indexing convention conversion is implicit and fragile**:
   - MATLAB is 1-based (defaults: start 51, hill 55, etc.).
   - Python uses 0-based indices (start 50, hill 54) in `experiment()`.
   - This is fine *only if* every internal function consistently assumes 0-based indices. Currently some functions sample with `np.argmax(np.cumsum(...) >= rand)` and others use indices as if they were 1-based in comments/tests.

### 3.3 Needs constraints + planning horizon off-by-one

1. **Different “constraint constants”**:
   - MATLAB: alive while `< [22,20,25]` (food/water/sleep).
   - Python uses `resource_constraints = {"Food":21,"Water":19,"Sleep":24}` and `is_alive()` checks `time_since > constraint`.

   This can be made equivalent, but it interacts with the horizon computation (next point).

2. **Tree-search horizon computed differently**:
   - MATLAB modular horizon: `horizon = min(9, min([22 - t_food, 20 - t_water, 25 - t_sleep]))` then clamp `horizon>=1`.
   - Python: `needs = constraint - time_since`, with constraints set to 21/19/24, then `horizon = max(1, min(9, needs["Food"], ...))`.

   This yields horizons that are typically **1 step shorter** than MATLAB when a need is limiting (i.e., when the min is < 9).

3. **Preference thresholds differ**:
   - MATLAB `determineObservationPreference.m` flips preferences to -500 when `t_food > 21`, `t_water > 19`, `t_sleep > 24` (i.e., at/after the lethal boundary).
   - Python `determine_observation_preference()` triggers on `>= constraint` using constraints that are already “minus 1”, so it triggers one step earlier.

### 3.4 Inference (`calculate_posterior`) is not equivalent

1. **Python `calculate_posterior()` samples from O again**:
   - In `src/Python/active_inference_utils.py`, it does:
     - `resource_observation = argmax(cumsum(O[1]) >= rand())`
     - `context_observation = argmax(cumsum(O[2]) >= rand())`
   - MATLAB `src/MATLAB/utils/calculate_posterior.m` does *not* sample; it treats `O{modal,t}` as a probability vector (can be one-hot or a distribution) and marginalizes properly.

   This single difference:
   - advances the RNG in Python where MATLAB does not,
   - makes predicted-posterior computations (where O is not one-hot) stochastic and wrong,
   - breaks parity of “prediction error” and memory resets.

2. **Python posterior math is a simplified derivation**:
   - The docstring explicitly says it is “kept relatively simple”.
   - The MATLAB version constructs a likelihood matrix `L(context, state)` from modalities 2..3 and integrates out the state factor.

To match MATLAB exactly, Python should implement the same tensor operations (see §4.3).

### 3.5 Backward smoothing (`spm_backwards`) is not equivalent

1. **Python `spm_backwards()` samples observations**:
   - In `src/Python/spm_utils.py`, it samples `obs` via CDF on `O[timestep][2]`.
   - MATLAB `src/MATLAB/utils/spm/spm_backwards.m` never samples. It multiplies by the *full observation distribution* (which matters in planning, when O is not one-hot).

2. **The modality used must remain hill-only**:
   - MATLAB uses only modality 3 in `spm_backwards` (hill cue), because modality 2 is unknown/learned and not used for smoothing in that function.
   - Python is currently structured similarly, but the sampling makes it diverge in imagined trajectories (where hill observations are distributions, not one-hot).

### 3.6 Learning update for `a` differs from MATLAB

In MATLAB (e.g. `BA_modular.m` / `SI_modular.m`), when updating `a{2}`:

- It subtracts a penalty **only from the first row** (“empty”) when that first-row entry is zero.

In Python (`src/Python/agent_utils.py:update_agent_likelihood`):

- When the first entry is zero, it subtracts the penalty from **all outcomes**:
  - `a_learning[:, j, i] -= amount_to_subtract`

This changes the sign/scale of the update and will produce different learned likelihoods.

### 3.7 Rounding semantics differ in key comparisons

MATLAB uses `round()` (ties away from zero).

Python uses:

- `round_half_up()` for some 1-decimal comparisons (good),
- but uses `np.round(..., 3)` for the context-change check in smoothing (ties-to-even), which can diverge from MATLAB edge cases.

### 3.8 Tree search (SI/SL) differs from MATLAB in multiple places

#### 3.8.1 Time-since update timing inside tree-search

In MATLAB tree-search functions (`tree_search_frwd_SI.m`, `tree_search_frwd_SL.m`):

- Preferences/extrinsic are computed from `t_food/t_water/t_sleep` **before** the update:
  - `t_food = round((t_food + 1) * (1 - O{2,t}(2)))` happens after adding the extrinsic term.

In Python forward tree search (`src/Python/forward_tree_search.py`):

- It increments the `time_since_resource` dictionary **before** computing preferences/extrinsic.

This shifts the magnitude of extrinsic reward and can change action selection.

#### 3.8.2 SL novelty + imagined learning uses the wrong observation/time index

In MATLAB `tree_search_frwd_SL.m`:

- The novelty (and imagined a-updates) iterate `timey = start:t` and use `O{2,timey}` and `P{timey,1}`.

In Python `forward_tree_search_SL()`:

- The novelty loop iterates `smoothing_t`, but uses:
  - `a_learning = imagined_O[1]` (current observation) for *all* smoothing_t values,
  - `smoothed_posterior = [P[0], smoothed_posterior]` (current position posterior) for *all* smoothing_t values.

This is not equivalent to MATLAB, which uses the observation and position posterior at each `timey`.

#### 3.8.3 Memory-access accounting is inverted

MATLAB increments `memory_accessed` on **cache hits** (when it *reads* from `short_term_memory`).

Python increments `memory_accessed` on **cache misses** (after it computes and stores a new value).

So the metric is not comparable and also the code-path differs.

#### 3.8.4 Action ordering and tie-breaking differs

- MATLAB loops `actions = 1:5` deterministically.
- Python uses `np.random.permutation(5)` (random action evaluation order), which can change tie-breaking and interacts with RNG parity.

#### 3.8.5 Confusing dimension naming (`num_resource_observations`)

Python sizes the joint state dimension as:

- `num_states * num_resource_observations`

But joint states are actually `num_states * num_context_states`.

This works only because both happen to be 4 in the default environment, but it is semantically wrong and will break as soon as those differ.

### 3.9 Missing algorithms / variants in Python

Python has no equivalents for:

- `BA_modular.m` / `tree_search_frwd.m`
- `BAUCB_modular.m` / `tree_search_frwd_UCB.m`
- `SI_smooth_modular.m` / `tree_search_frwd_SI_smooth.m`
- `SL_noSmooth_modular.m` / `tree_search_frwd_SL_noSmooth.m`

If “equivalent versions” means parity across the algorithm suite, these must be implemented or explicitly scoped out.

### 3.10 Reproducibility / RNG ordering differences

Even if the math is corrected, **RNG ordering** must match to get identical trajectories:

- MATLAB functions like `calculate_posterior()` and `spm_backwards()` do not call `rand`.
- Python currently calls `np.random` inside both.
- Python also uses both `random` and `numpy.random`; MATLAB uses one RNG stream.

To approach equivalence:

1. Remove all “extra sampling” (see §§3.4, 3.5).
2. Ensure only the same conceptual events consume RNG calls (true state transitions and observation sampling).
3. Consider centralizing RNG usage (pass a `np.random.Generator` instance everywhere).

---

## 4) Explicit Step‑By‑Step Plan to Make Python Equivalent to MATLAB Modular Code

The steps below are ordered to (a) get the Python code runnable, (b) lock down the generative process and inference, and (c) then align the tree-search policies.

### 4.1 Step 0: Decide the parity target

Be explicit about what “equivalent” means:

- **Option A (minimal)**: Match MATLAB’s SI_modular and SL_modular only.
- **Option B (full suite)**: Also implement BA, BAUCB, SI_smooth, SL_noSmooth.

Everything below assumes Option A, but includes Option B work as add-ons.

### 4.2 Step 1: Make the Python runner execute (no parity yet)

Edit `src/Python/algorithms/sophisticated_agent/sophisticated_agent.py`:

1. Fix the invalid f-strings / stray characters so the file parses.
2. Remove the hard-coded Qt backend selection (or guard it behind `if visualise:`).
3. Ensure imports are package-relative (avoid fragile `sys.path.append` hacks).

### 4.3 Step 2: Make the environment initialization byte‑for‑byte equivalent

Goal: `A`, `B`, and `D` in Python match MATLAB (modulo 0/1 indexing offsets).

1. Re-implement Python environment init directly from:
   - `src/MATLAB/utils/initialiseEnvironment.m`
2. Ensure the same:
   - A modality definitions,
   - initial `a{2} = 0.1`,
   - context transition matrix values,
   - **position transition tensor B{1} including its boundary/wrap behaviour**.

Practical approach:

- Write a small Python “constructor” that mirrors MATLAB literally, then compare against the existing Python construction.
- Add unit tests that check specific transitions for a handful of boundary states and actions (these should be deterministic).

### 4.4 Step 3: Replace `calculate_posterior()` with a faithful port

Edit `src/Python/active_inference_utils.py:calculate_posterior`:

1. Remove all sampling from `O`.
2. Accept `O[1]` and `O[2]` as probability vectors (one-hot or distributions).
3. Compute the context posterior exactly as MATLAB does in `src/MATLAB/utils/calculate_posterior.m`.

Vectorized MATLAB-equivalent sketch (Python/Numpy):

- Compute modality likelihoods as expectation under `obs_dist`:
  - `lik2(state,ctx) = Σ_o O2[o] * A2[o,state,ctx]`
  - `lik3(state,ctx) = Σ_o O3[o] * A3[o,state,ctx]`
- Multiply `L = lik2 * lik3`, then integrate out `state` using `P_state`.
- Update only the context factor `P_context`.

Add a new unit test where `O[2]` is **not one-hot** (a distribution). This is exactly where the current implementation diverges from MATLAB.

### 4.5 Step 4: Replace `spm_backwards()` with a faithful port

Edit `src/Python/spm_utils.py:spm_backwards`:

1. Remove all sampling.
2. Use only the hill modality (Python modality index 2).
3. Treat `O[timestep][2]` as a probability vector (one-hot or a distribution), just like MATLAB.
4. Ensure the function does not advance RNG state.

Again, add a unit test with non-one-hot hill observations (as occur in tree-search).

### 4.6 Step 5: Make smoothing + learning (`a` updates) match MATLAB exactly

Edits in `src/Python/agent_utils.py`:

1. In `smooth_beliefs()`:
   - Use MATLAB-equivalent rounding when checking posterior changes (use `round_half_up(..., 3)` rather than `np.round`).
2. In `update_agent_likelihood()`:
   - Apply the penalty only to the “empty” outcome row (row 0), matching MATLAB.
   - Keep scaling constants identical: `proportion = 0.3`, update scale `0.7`, floor `0.05`.

### 4.7 Step 6: Align needs thresholds + horizon computation

Edits in `src/Python/algorithms/sophisticated_agent/sophisticated_agent.py` (and related helpers):

1. Choose one consistent representation:
   - Either store the MATLAB thresholds directly `{Food:22, Water:20, Sleep:25}` and define:
     - alive iff `time_since < threshold`
     - horizon uses `threshold - time_since`
   - Or keep “max time_since” and fix all formulas accordingly.
2. Match MATLAB’s horizon exactly (including the `if horizon == 0, horizon = 1` behavior).
3. Match MATLAB’s preference threshold logic exactly (see `src/MATLAB/utils/determineObservationPreference.m`).

### 4.8 Step 7: Make SI tree-search match `tree_search_frwd_SI.m`

Edit `src/Python/forward_tree_search.py:forward_tree_search_SI`:

Required changes:

1. Match the timing of needs updates:
   - Do **not** pre-increment needs before computing the extrinsic term.
   - Update needs using the exact MATLAB formula after extrinsic is added.
2. Cache semantics:
   - Increment `memory_accessed` on cache hits, not misses.
3. Deterministic action loop:
   - Use action order `[0,1,2,3,4]` (Python) to match MATLAB’s `1:5`.
4. Ensure all calls use the corrected `calculate_posterior` and `spm_backwards` that do not sample.

### 4.9 Step 8: Make SL tree-search match `tree_search_frwd_SL.m`

Edit `src/Python/forward_tree_search.py:forward_tree_search_SL`:

Critical changes:

1. In the novelty/learning loop over `timey = start:t`:
   - Use `imagined_historical_agent_O[timey][1]` (resource modality at that timey), not `imagined_O[1]` for all timey.
   - Use the position posterior at that timey (from history), not `P[0]` at current time for all timey.
2. Apply MATLAB’s `a_learning(a_learning <= 0.2) = 0` threshold inside SL planning.
3. Update imagined `a` inside planning exactly as MATLAB does (`a = a + a_learning`, no 0.7 scaling / no 0.05 floor inside tree-search).
4. Keep the known MATLAB quirk:
   - MATLAB contains a TODO about whether `y` should be updated after `a` changes inside SL planning, but it does **not** update `y`. Match that if parity is the goal.

### 4.10 Step 9 (Option B): Implement missing MATLAB algorithms in Python

If you want the Python suite to cover the same set as the MATLAB modular folder:

1. **BA**:
   - Implement `tree_search_frwd` (extrinsic-only, action-indexed memory `(…,state,action)`).
2. **BAUCB**:
   - Implement `tree_search_frwd_UCB` including:
     - visit counts `Nt` update at root,
     - UCB bonus `ucb_scale * sqrt(log(total_visits+1) / Nt(state))`,
     - memory indexed by `(…,joint_state,action)`.
3. **SI_smooth** and **SL_noSmooth**:
   - Add algorithm flags that swap in the corresponding novelty computation differences.

### 4.11 Step 10: Add parity tests + a reference-trace workflow

To *prove* equivalence, add a workflow:

1. In MATLAB, dump a “reference trace” for a tiny run (e.g., 1 trial, 10 timesteps):
   - true_states, observations, Q/P context posteriors, chosen actions, and snapshots of `a{2}`.
2. Save as `.mat` or `.json`.
3. In Python, load that trace and assert that the same quantities match at each timestep.

This is the only reliable way to validate end-to-end parity because tiny mismatches in RNG ordering will otherwise hide inside long trajectories.

---

## 5) Quick Checklist (Parity Work Items)

- [ ] Fix Python syntax/runtime errors (`sophisticated_agent.py`)
- [ ] Match `B{1}` movement tensor exactly
- [ ] Rewrite `calculate_posterior` to be distribution-based (no sampling)
- [ ] Rewrite `spm_backwards` to be distribution-based (no sampling)
- [ ] Fix `update_agent_likelihood` “empty-row-only” penalty
- [ ] Use MATLAB-equivalent rounding everywhere comparisons occur
- [ ] Match needs thresholds + horizon formula exactly
- [ ] Rewrite SI tree-search to match MATLAB timing + caching semantics
- [ ] Rewrite SL tree-search novelty loop to use per-timey O/P (not current only)
- [ ] Optional: implement BA/BAUCB + variants for full suite parity


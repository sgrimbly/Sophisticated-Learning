"""Per-trial agent loop and per-experiment driver.

Faithful port of:
    src/MATLAB/algorithms/unknown-models/modular_versions/SI_modular.m
    src/MATLAB/algorithms/unknown-models/modular_versions/SL_modular.m
    src/MATLAB/algorithms/unknown-models/modular_versions/BA_modular.m
    src/MATLAB/algorithms/unknown-models/modular_versions/BAUCB_modular.m

The structure of one timestep ``t`` (matches MATLAB lines 263-501 of
SI_modular.m exactly):

    1. Normalise b_ctx into bb_ctx.
    2. If t > 0: env state transition + need-timer update.
       If t == 0: initial true state already populated.
    3. Sample observations from A.
    4. If t > 0: predicted-vs-actual context posterior comparison +
       backward-smoothed Dirichlet update for a_resource.
    5. y_resource = normalise(a_resource); compute horizon.
    6. Compute actual posterior P with calculate_posterior(Q, y, O).
    7. Memory reset on context-prediction-error or hill visit.
    8. Tree search → best_actions.
    9. chosen_action[t] = best_actions[0]; t += 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import GridConfig, RunOptions, Weights, resolve_algorithm
from .efe import determine_observation_preference  # noqa: F401  re-export
from .env import (
    GenerativeModel, initialise_environment, normalise_b_ctx,
    sample_observations, update_environment_states, update_needs,
)
from .inference import (
    calculate_posterior, joint_pos_ctx, normalise, normalise_matrix_columns,
    spm_backwards, spm_cross,
)
from .learning import real_dirichlet_update
from .planning.common import PlannerInputs
from .planning.si import tree_search_si
from .planning.sl import tree_search_sl
from .planning.ba import tree_search_ba
from .planning.baucb import tree_search_baucb
from .rng import make_rng, sample_categorical, select_from_posterior


@dataclass
class TrialResult:
    survived: bool
    t_terminal: int
    chosen_actions: List[int]
    true_states: List[tuple]            # list of (pos, ctx)
    observations: List[tuple]           # list of (pos_idx, res_idx, hill_idx)
    memory_resets: int
    pe_memory_resets: int
    hill_memory_resets: int
    horizons: List[int]
    a_resource_final: np.ndarray
    Q_pos_history: List[np.ndarray]     # predictive (pre-observation) posterior
    Q_ctx_history: List[np.ndarray]
    P_pos_history: List[np.ndarray]     # post-observation posterior
    P_ctx_history: List[np.ndarray]
    O_resource_history: List[np.ndarray]
    O_hill_history: List[np.ndarray]


@dataclass
class ExperimentResult:
    config: GridConfig
    options: RunOptions
    weights: Weights
    trials: List[TrialResult] = field(default_factory=list)
    survived_count: int = 0


# ---------------------------------------------------------------------------
# Per-trial main loop
# ---------------------------------------------------------------------------


def run_trial(
    grid: GridConfig,
    options: RunOptions,
    weights: Weights,
    model: GenerativeModel,
    rng: np.random.Generator,
) -> TrialResult:
    """Run one trial. Mutates ``model.a_resource`` in place (real learning)."""
    spec = resolve_algorithm(options.algorithm)
    family = spec["family"]

    S = grid.num_states
    C = grid.num_contexts
    JS = grid.num_joint_states

    # Per-trial state.
    true_states: List[tuple] = []
    observations: List[tuple] = []
    chosen_actions: List[int] = []
    horizons: List[int] = []

    Q_pos_history: List[np.ndarray] = []
    Q_ctx_history: List[np.ndarray] = []
    P_pos_history: List[np.ndarray] = []
    P_ctx_history: List[np.ndarray] = []
    O_pos_history: List[np.ndarray] = []
    O_resource_history: List[np.ndarray] = []
    O_hill_history: List[np.ndarray] = []

    # Initial predictive posterior (before any observation): D priors.
    Q_pos = model.D_pos.copy()
    Q_ctx = model.D_ctx.copy()

    # Initial true state: position is fixed, context sampled from D_ctx.
    true_pos = grid.start_position
    true_ctx = sample_categorical(model.D_ctx, rng)

    t_food = t_water = t_sleep = 0
    short_term_memory = _alloc_short_term_memory(family, JS)
    Nt = np.zeros(JS, dtype=np.float64) if family == "BAUCB" else None
    memory_resets = pe_resets = hill_resets = 0

    # Predicted-context-posterior buffer (filled on previous step's tree search input)
    predicted_P_ctx_for_t: Optional[np.ndarray] = None

    t = 0
    while options.is_alive(t, t_food, t_water, t_sleep):
        # ----- 1. normalise b_ctx -------------------------------------
        bb_ctx = normalise_b_ctx(model.b_ctx)

        # ----- 2. env transition + need timers ------------------------
        if t > 0:
            chosen = chosen_actions[-1]
            Q_pos, Q_ctx, true_pos, true_ctx = update_environment_states(
                Q_pos_history[-1], Q_ctx_history[-1],
                true_states[-1][0], true_states[-1][1],
                chosen,
                model.B_pos, model.B_ctx, bb_ctx,
                rng,
            )

        true_states.append((int(true_pos), int(true_ctx)))
        t_food, t_water, t_sleep = update_needs(
            grid, true_pos, true_ctx, t, t_food, t_water, t_sleep,
        )

        # ----- 3. observations ----------------------------------------
        O_pos, O_res, O_hill = sample_observations(
            model.A_pos, model.A_resource, model.A_hill,
            true_pos, true_ctx, rng,
        )
        observations.append((int(np.argmax(O_pos)), int(np.argmax(O_res)), int(np.argmax(O_hill))))
        O_pos_history.append(O_pos)
        O_resource_history.append(O_res)
        O_hill_history.append(O_hill)

        Q_pos_history.append(Q_pos.copy())
        Q_ctx_history.append(Q_ctx.copy())

        # ----- 4. smoothing-based learning (only when t > 0) ----------
        # MATLAB recomputes y twice in the per-step block:
        #   line 352:  y{2} = normalise_matrix(a{2});  -- pre-smoothing,
        #              used for predicted_observations_posterior /
        #              predicted_posterior.
        #   line 449:  y{2} = normalise_matrix(a{2});  -- POST-smoothing,
        #              used for the actual P calculation and the planner.
        # Python therefore takes y_resource_now (pre) inside the smoothing
        # block and y_resource (post) for steps 5+ below.
        if t > 0 and family in {"SI", "SL"}:
            y_resource_now = normalise_matrix_columns(model.a_resource)

            # 4a. predicted observations posterior (used for context-PE detection)
            qs_t = joint_pos_ctx(Q_pos, Q_ctx)  # column-major flatten
            # MATLAB: predictive_observations_posterior{2,t} = normalise(y{2}(:,:) * qs(:))'
            y2_flat = y_resource_now.reshape(y_resource_now.shape[0], -1, order="F")
            y3_flat = model.A_hill.reshape(model.A_hill.shape[0], -1, order="F")
            pred_O_res = normalise(y2_flat @ qs_t)
            pred_O_hill = normalise(y3_flat @ qs_t)

            _, predicted_P_ctx_for_t = calculate_posterior(
                Q_pos, Q_ctx, y_resource_now, model.A_hill,
                pred_O_res, pred_O_hill,
            )

            # 4b. backward smoothing + a-update (real_smoothing path).
            # MATLAB ``dashboard_run_one`` overrides ``options.real_smoothing``
            # with ``algorithm_spec.smoothing_on`` when ``is_unknown_model``
            # (which is true for SI/SL family). Mirror that here so that
            # algorithm name alone selects the canonical variant.
            spec_smoothing = spec.get("smoothing", options.real_smoothing)
            if spec_smoothing:
                start = max(0, t - 6)
                for timey in range(start, t + 1):
                    L_ctx = spm_backwards(
                        O_hill_history, Q_pos_history, Q_ctx_history[timey],
                        model.A_hill, bb_ctx, timey, t,
                    )
                    # Gate: update only if context belief changed (or terminal step)
                    pre_round = np.round(L_ctx, 3)
                    cmp_round = np.round(Q_ctx_history[timey], 3)
                    if (timey > start and not np.array_equal(pre_round, cmp_round)) or timey == t:
                        model.a_resource = real_dirichlet_update(
                            model.a_resource,
                            O_resource_history[timey],
                            Q_pos_history[timey],
                            L_ctx,
                            proportion=0.3,
                            scale=0.7,
                            floor=0.05,
                        )
            else:
                # SI_noSmooth / SL_noSmooth path (single-step learning at t)
                model.a_resource = real_dirichlet_update(
                    model.a_resource,
                    O_resource_history[t],
                    Q_pos_history[t],
                    Q_ctx_history[t],  # no smoothing → use predictive directly
                    proportion=0.3,
                    scale=0.7,
                    floor=0.05,
                )

        # ----- 5. y_resource (POST-smoothing) + horizon ---------------
        # Mirrors MATLAB SI_modular line 449 — recompute y AFTER the smoothing
        # block so the planner and posterior see the just-updated model.
        y_resource = normalise_matrix_columns(model.a_resource)
        horizon = options.horizon(t_food, t_water, t_sleep)
        horizons.append(horizon)

        # ----- 6. actual posterior P (post-observation) --------------
        P_pos, P_ctx = calculate_posterior(
            Q_pos, Q_ctx, y_resource, model.A_hill,
            O_res, O_hill,
        )
        P_pos_history.append(P_pos.copy())
        P_ctx_history.append(P_ctx.copy())

        # MATLAB invariant: after the planner runs, Q{t, factor} is overwritten
        # to the posterior at step t (tree_search_frwd_SI returns the modified
        # cell). update_environment_states at step t+1 then propagates from
        # this posterior, so the agent's predictive belief at every later step
        # encodes the cumulative observational information (most importantly,
        # hill-cue context concentration). Without this overwrite, Q stays at
        # the predictive (uniform) and posterior info is lost between steps.
        Q_pos_history[t] = P_pos.copy()
        Q_ctx_history[t] = P_ctx.copy()

        # ----- 7. memory reset on context-PE or hill visit -----------
        if t > 0 and predicted_P_ctx_for_t is not None:
            if not np.array_equal(np.round(predicted_P_ctx_for_t, 1), np.round(P_ctx, 1)):
                short_term_memory[...] = 0.0
                memory_resets += 1
                pe_resets += 1
        if true_pos == grid.hill_pos:
            short_term_memory[...] = 0.0
            memory_resets += 1
            hill_resets += 1

        # ----- 8. plan -----------------------------------------------
        planner_inputs = PlannerInputs(
            A_pos=model.A_pos, A_resource=model.A_resource, A_hill=model.A_hill,
            a_resource=model.a_resource,
            y_pos=model.A_pos, y_resource=y_resource, y_hill=model.A_hill,
            B_pos=model.B_pos, bb_ctx=bb_ctx,
            weights=weights,
        )

        if family == "SI":
            if options.use_jit_planner:
                from .planning.si_jit import tree_search_si_jit_run
                result = tree_search_si_jit_run(
                    short_term_memory,
                    O_pos, O_res, O_hill,
                    Q_pos, Q_ctx,
                    planner_inputs,
                    t=t, N=t + horizon,
                    t_food=t_food, t_water=t_water, t_sleep=t_sleep,
                    true_t=t,
                    novelty_on=spec.get("novelty", True) and weights.novelty != 0,
                    epistemic_on=weights.epistemic != 0,
                )
            else:
                result = tree_search_si(
                    short_term_memory,
                    O_pos, O_res, O_hill,
                    Q_pos, Q_ctx,
                    planner_inputs,
                    t=t, N=t + horizon,
                    t_food=t_food, t_water=t_water, t_sleep=t_sleep,
                    true_t=t,
                    novelty_on=spec.get("novelty", True) and weights.novelty != 0,
                    epistemic_on=weights.epistemic != 0,
                )
        elif family == "SL":
            sl_kwargs = dict(
                t=t, N=t + horizon,
                t_food=t_food, t_water=t_water, t_sleep=t_sleep,
                true_t=t,
                novelty_on=spec.get("novelty", True) and weights.novelty != 0,
                epistemic_on=weights.epistemic != 0,
                smoothing_on=spec.get("smoothing", True),
                adaptive_likelihood_in_plan=spec.get("adaptive_plan", False)
                                          or options.adaptive_likelihood_in_plan,
                learning_prune_threshold=options.learning_prune_threshold,
                history_O_resource=O_resource_history,
                history_O_hill=O_hill_history,
                history_P_pos=Q_pos_history,
                history_P_ctx=Q_ctx_history,
            )
            if options.use_jit_planner:
                from .planning.sl_jit import tree_search_sl_jit_run
                result = tree_search_sl_jit_run(
                    short_term_memory, O_pos, O_res, O_hill, Q_pos, Q_ctx,
                    planner_inputs, **sl_kwargs,
                )
            else:
                result = tree_search_sl(
                    short_term_memory, O_pos, O_res, O_hill, Q_pos, Q_ctx,
                    planner_inputs, **sl_kwargs,
                )
        elif family == "BA":
            result = tree_search_ba(
                short_term_memory,
                O_pos, O_res, O_hill,
                Q_pos, Q_ctx,
                planner_inputs,
                t=t, N=t + horizon,
                t_food=t_food, t_water=t_water, t_sleep=t_sleep,
                true_t=t,
                state_selection=options.state_selection,
                rng=rng,
            )
        elif family == "BAUCB":
            cur_joint = true_pos + S * true_ctx
            result = tree_search_baucb(
                short_term_memory, Nt,
                O_pos, O_res, O_hill,
                Q_pos, Q_ctx,
                planner_inputs,
                t=t, N=t + horizon,
                t_food=t_food, t_water=t_water, t_sleep=t_sleep,
                true_t=t,
                state_selection=options.state_selection,
                rng=rng,
                current_joint_state=cur_joint,
                ucb_scale=weights.ucb_scale,
            )
        else:
            raise ValueError(f"Unsupported algorithm family: {family}")

        if not result.best_actions:
            # horizon == 0 edge case shouldn't happen given options.horizon clamp;
            # guard with a default no-op (stay).
            chosen_action = 0
        else:
            chosen_action = result.best_actions[0]
        chosen_actions.append(int(chosen_action))

        t += 1

    survived = t >= options.max_steps_per_trial
    return TrialResult(
        survived=survived,
        t_terminal=t,
        chosen_actions=chosen_actions,
        true_states=true_states,
        observations=observations,
        memory_resets=memory_resets,
        pe_memory_resets=pe_resets,
        hill_memory_resets=hill_resets,
        horizons=horizons,
        a_resource_final=model.a_resource.copy(),
        Q_pos_history=Q_pos_history,
        Q_ctx_history=Q_ctx_history,
        P_pos_history=P_pos_history,
        P_ctx_history=P_ctx_history,
        O_resource_history=O_resource_history,
        O_hill_history=O_hill_history,
    )


def _alloc_short_term_memory(family: str, num_joint_states: int) -> np.ndarray:
    if family in {"SI", "SL"}:
        return np.zeros((35, 35, 35, num_joint_states), dtype=np.float64)
    if family in {"BA", "BAUCB"}:
        return np.zeros((35, 35, 35, num_joint_states, 5), dtype=np.float64)
    raise ValueError(f"Unknown family for STM allocation: {family}")


# ---------------------------------------------------------------------------
# Multi-trial driver
# ---------------------------------------------------------------------------


def run_experiment(
    grid: GridConfig,
    options: RunOptions,
    weights: Weights,
) -> ExperimentResult:
    """Run ``options.num_trials`` trials with a single ``a`` that persists
    across trials (matches MATLAB ``a_history`` accumulation).
    """
    rng = make_rng(options.seed)
    model = initialise_environment(grid)

    out = ExperimentResult(config=grid, options=options, weights=weights)
    for trial in range(options.num_trials):
        result = run_trial(grid, options, weights, model, rng)
        out.trials.append(result)
        if result.survived:
            out.survived_count += 1
    return out

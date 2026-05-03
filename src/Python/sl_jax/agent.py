"""sl_jax agent: outer trial loop in Python, JAX-batched per-step planner."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
from sl.env import sample_observations, update_environment_states, update_needs  # noqa: E402
from sl.inference import normalise_matrix_columns  # noqa: E402
from sl.learning import real_dirichlet_update  # noqa: E402
from sl.rng import make_rng, sample_categorical  # noqa: E402

from .config import GridConfig, JaxOptions
from .env import JaxModel, build_jax_model, update_a_resource
from .planning import PlannerInputsJax, plan_batched


@dataclass
class JaxTrialResult:
    seed: int
    trial: int
    t_terminal: int
    chosen_actions: List[int]
    true_states: List[tuple]


@dataclass
class JaxRunResult:
    config_seed: int
    trials: List[JaxTrialResult] = field(default_factory=list)


def _build_planner_inputs(model: JaxModel, opts: JaxOptions) -> PlannerInputsJax:
    y_resource = jnp.asarray(
        normalise_matrix_columns(np.asarray(model.a_resource))
    )
    return PlannerInputsJax(
        A_pos=model.A_pos,
        A_resource=model.A_resource,
        A_hill=model.A_hill,
        a_resource=model.a_resource,
        y_resource=y_resource,
        B_pos=model.B_pos,
        bb_ctx=model.bb_ctx,
        w_novelty=jnp.float64(opts.w_novelty),
        w_learning=jnp.float64(opts.w_learning),
        w_epistemic=jnp.float64(opts.w_epistemic),
        preference_inverse_precision=jnp.float64(opts.preference),  # 'inverse_precision' mode
    )


def run_trial(
    grid: GridConfig,
    opts: JaxOptions,
    model: JaxModel,
    rng: np.random.Generator,
    trial_idx: int = 0,
) -> tuple[JaxTrialResult, JaxModel]:
    """Single trial. Updates ``model.a_resource`` (returned)."""
    # --- True / belief state init ---
    Q_pos = np.asarray(model.D_pos).copy()
    Q_ctx = np.asarray(model.D_ctx).copy()
    true_pos = grid.start_position
    true_ctx = sample_categorical(np.array(model.D_ctx, dtype=np.float64), rng)

    t_food = t_water = t_sleep = 0
    chosen_actions: List[int] = []
    true_states: List[tuple] = []
    O_res_history: List[np.ndarray] = []
    Q_pos_history: List[np.ndarray] = []
    Q_ctx_history: List[np.ndarray] = []

    use_sl = opts.smoothing_on  # SL rollout in planner (currently same as SI in proto)

    # --- per-step loop ---
    A_pos_np = np.asarray(model.A_pos)
    A_resource_np = np.asarray(model.A_resource)
    A_hill_np = np.asarray(model.A_hill)
    B_pos_np = np.asarray(model.B_pos)
    B_ctx_np = np.asarray(model.B_ctx)
    bb_ctx_np = np.asarray(model.bb_ctx)
    a_resource_np = np.asarray(model.a_resource)

    t = 0
    while (t < opts.max_steps_per_trial
           and t_food < opts.food_threshold
           and t_water < opts.water_threshold
           and t_sleep < opts.sleep_threshold):

        # --- env transition (real) ---
        if t > 0:
            chosen = chosen_actions[-1]
            Q_pos, Q_ctx, true_pos, true_ctx = update_environment_states(
                Q_pos, Q_ctx, true_states[-1][0], true_states[-1][1],
                chosen, B_pos_np, B_ctx_np, bb_ctx_np, rng,
            )

        true_states.append((int(true_pos), int(true_ctx)))
        t_food, t_water, t_sleep = update_needs(
            grid, true_pos, true_ctx, t, t_food, t_water, t_sleep,
        )

        # --- observation ---
        O_pos, O_res, O_hill = sample_observations(
            A_pos_np, A_resource_np, A_hill_np, true_pos, true_ctx, rng,
        )
        O_res_history.append(O_res)
        Q_pos_history.append(Q_pos.copy())
        Q_ctx_history.append(Q_ctx.copy())

        # --- agent posterior over (pos, ctx) AT this step (for planner) ---
        from sl.inference import calculate_posterior
        y_resource_now = normalise_matrix_columns(a_resource_np)
        P_pos, P_ctx = calculate_posterior(
            Q_pos, Q_ctx, y_resource_now, A_hill_np, O_res, O_hill,
        )

        # MATLAB Q-overwrite invariant: future propagation uses posterior.
        Q_pos = P_pos
        Q_ctx = P_ctx

        # --- plan via batched policy enumeration ---
        inputs = PlannerInputsJax(
            A_pos=jnp.asarray(A_pos_np),
            A_resource=jnp.asarray(A_resource_np),
            A_hill=jnp.asarray(A_hill_np),
            a_resource=jnp.asarray(a_resource_np),
            y_resource=jnp.asarray(y_resource_now),
            B_pos=jnp.asarray(B_pos_np),
            bb_ctx=jnp.asarray(bb_ctx_np),
            w_novelty=jnp.float64(opts.w_novelty),
            w_learning=jnp.float64(opts.w_learning),
            w_epistemic=jnp.float64(opts.w_epistemic),
            preference_inverse_precision=jnp.float64(opts.preference),
        )
        best_action, best_G = plan_batched(
            jnp.asarray(P_pos),
            jnp.asarray(P_ctx),
            jnp.int32(t_food), jnp.int32(t_water), jnp.int32(t_sleep),
            inputs,
            horizon=opts.horizon,
            use_sl_rollout=use_sl,
        )
        chosen = int(best_action)
        chosen_actions.append(chosen)

        # --- Dirichlet a-update (real-trial learning) ---
        if t > 0:
            if opts.smoothing_on:
                from sl.inference import spm_backwards
                start = max(0, t - opts.smoothing_window)
                for timey in range(start, t + 1):
                    L_ctx = spm_backwards(
                        [np.asarray(o) for o in (Q_ctx_history[:t+1])],  # placeholder
                        Q_pos_history, Q_ctx_history[timey],
                        A_hill_np, bb_ctx_np, timey, t,
                    )
                    pre_round = np.round(L_ctx, 3)
                    cmp_round = np.round(Q_ctx_history[timey], 3)
                    if (timey > start and not np.array_equal(pre_round, cmp_round)) or timey == t:
                        a_resource_np = real_dirichlet_update(
                            a_resource_np, O_res_history[timey],
                            Q_pos_history[timey], L_ctx,
                            proportion=opts.learning_proportion,
                            scale=opts.learning_scale,
                            floor=opts.learning_floor,
                        )
            else:
                # single-step learning (SI)
                a_resource_np = real_dirichlet_update(
                    a_resource_np, O_res_history[t],
                    Q_pos_history[t], Q_ctx_history[t],
                    proportion=opts.learning_proportion,
                    scale=opts.learning_scale,
                    floor=opts.learning_floor,
                )

        t += 1

    # Save updated a_resource back into model.
    new_model = update_a_resource(model, jnp.asarray(a_resource_np))
    return JaxTrialResult(
        seed=opts.seed,
        trial=trial_idx,
        t_terminal=t,
        chosen_actions=chosen_actions,
        true_states=true_states,
    ), new_model


def run(grid: GridConfig, opts: JaxOptions) -> JaxRunResult:
    rng = make_rng(opts.seed)
    model = build_jax_model(grid)
    result = JaxRunResult(config_seed=opts.seed)
    t0 = time.time()
    for trial in range(opts.num_trials):
        tr, model = run_trial(grid, opts, model, rng, trial_idx=trial)
        result.trials.append(tr)
    print(f'sl_jax run: {opts.num_trials} trials in {time.time()-t0:.1f}s')
    return result

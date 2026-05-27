"""Generative environment: A, B, D and per-step state/observation updates.

Faithful port of:
    src/MATLAB/utils/initialiseEnvironment.m   (lines 1-78)
    src/MATLAB/utils/updateEnvironmentStates.m
    inline need-update + observation-sample blocks of SI_modular.m

Action numbering (0-based here, MATLAB +1 was 1-5):
    0 = stay         (identity)
    1 = left         (circshift -1)
    2 = right        (circshift +1)
    3 = up           (circshift +grid_size)   (per MATLAB semantics)
    4 = down         (circshift -grid_size)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import GridConfig
from .inference import normalise, normalise_matrix_columns
from .rng import sample_categorical


@dataclass
class GenerativeModel:
    """Bundle of generative tensors. All arrays are 0-based.

    Modality 0 = position identity, 1 = resource cue, 2 = hill cue.
    Factor 0 = position, factor 1 = context.
    """
    A_pos: np.ndarray       # (S, S, C)
    A_resource: np.ndarray  # (4, S, C)
    A_hill: np.ndarray      # (5, S, C)
    a_pos: np.ndarray
    a_resource: np.ndarray
    a_hill: np.ndarray
    B_pos: np.ndarray       # (S, S, 5)
    B_ctx: np.ndarray       # (C, C, 5) — same matrix replicated across actions
    b_pos: np.ndarray
    b_ctx: np.ndarray
    D_pos: np.ndarray       # (S,)
    D_ctx: np.ndarray       # (C,)


def _build_position_transition(grid_size: int, num_states: int) -> np.ndarray:
    """Construct B[1] for the grid. Replicates MATLAB circshift+mod logic.

    The MATLAB code uses 1-based indexing with:
      mod(i, grid_size) ~= 1 -> not leftmost column   (i.e. column != 1)
      mod(i, grid_size) ~= 0 -> not rightmost column  (i.e. column != grid_size)
      i > grid_size          -> not top row
      i <= num_states - grid_size -> not bottom row

    In 0-based terms (i' = i-1):
      column 0          : i' % grid_size == 0
      column grid_size-1: i' % grid_size == grid_size-1
      row 0             : i' < grid_size
      row grid_size-1   : i' >= num_states - grid_size

    MATLAB applies ``circshift`` on the column ``B{1}(:, i, action)`` of
    size num_states. The shift on the *flattened* column is by ±1 for
    horizontal moves and ±grid_size for vertical. Direction signs match
    MATLAB exactly.
    """
    B = np.zeros((num_states, num_states, 5), dtype=np.float64)
    # Action 0 = stay = identity for all states
    eye = np.eye(num_states, dtype=np.float64)
    for action in range(5):
        B[:, :, action] = eye

    for i in range(num_states):
        # action 1 (left): shift by -1, except leftmost column
        if i % grid_size != 0:
            B[:, i, 1] = np.roll(B[:, i, 1], -1)
        # action 2 (right): shift by +1, except rightmost column
        if i % grid_size != grid_size - 1:
            B[:, i, 2] = np.roll(B[:, i, 2], 1)
        # action 3 (up): MATLAB circshift +grid_size, except top row
        if i >= grid_size:
            B[:, i, 3] = np.roll(B[:, i, 3], grid_size)
        # action 4 (down): MATLAB circshift -grid_size, except bottom row
        if i < num_states - grid_size:
            B[:, i, 4] = np.roll(B[:, i, 4], -grid_size)
    return B


def _build_context_transition() -> np.ndarray:
    """Cyclic 4-context Markov transition (mostly stay, sometimes advance).

    Replicates the constant 4×4 matrix in initialiseEnvironment.m. Replicated
    across the 5 action axes (context evolves independently of action).
    """
    M = np.array([
        [0.95, 0.0,  0.0,  0.05],
        [0.05, 0.95, 0.0,  0.0],
        [0.0,  0.05, 0.95, 0.0],
        [0.0,  0.0,  0.05, 0.95],
    ], dtype=np.float64)
    return np.broadcast_to(M[:, :, None], (4, 4, 5)).copy()


def initialise_environment(grid: GridConfig) -> GenerativeModel:
    """Port of initialiseEnvironment.m. Returns a fully-built GenerativeModel."""
    S = grid.num_states
    C = grid.num_contexts

    # Modality 0: position identity, deterministic & known.
    A_pos = np.zeros((S, S, C), dtype=np.float64)
    a_pos = np.zeros_like(A_pos)
    for i in range(S):
        A_pos[i, i, :] = 1.0
        a_pos[i, i, :] = 1.0

    # Modality 1: resource cue, 4 outcomes.
    # Default outcome is 'empty' (index 0) for all (state, context); then for
    # each context i the food/water/sleep source flips index 0->{food,water,sleep}.
    A_resource = np.zeros((4, S, C), dtype=np.float64)
    A_resource[0, :, :] = 1.0
    for i in range(C):
        A_resource[1, grid.food_sources[i], i] = 1.0
        A_resource[0, grid.food_sources[i], i] = 0.0
        A_resource[2, grid.water_sources[i], i] = 1.0
        A_resource[0, grid.water_sources[i], i] = 0.0
        A_resource[3, grid.sleep_sources[i], i] = 1.0
        A_resource[0, grid.sleep_sources[i], i] = 0.0

    # Agent's prior on a_resource = 0.1 everywhere (Dirichlet concentration).
    a_resource = np.full_like(A_resource, 0.1)

    # Modality 2: hill cue, 5 outcomes [ctx1..ctx4, none].
    A_hill = np.zeros((5, S, C), dtype=np.float64)
    A_hill[4, :, :] = 1.0
    for i in range(C):
        A_hill[i, grid.hill_pos, i] = 1.0
        A_hill[4, grid.hill_pos, i] = 0.0
    a_hill = A_hill.copy()

    # Initial state priors.
    D_pos = np.zeros(S, dtype=np.float64)
    D_pos[grid.start_position] = 1.0
    D_pos = normalise(D_pos)
    D_ctx = np.full(C, 1.0 / C, dtype=np.float64)

    # Transitions.
    B_pos = _build_position_transition(grid.grid_size, S)
    B_ctx = _build_context_transition()

    return GenerativeModel(
        A_pos=A_pos, A_resource=A_resource, A_hill=A_hill,
        a_pos=a_pos, a_resource=a_resource, a_hill=a_hill,
        B_pos=B_pos, B_ctx=B_ctx,
        b_pos=B_pos.copy(), b_ctx=B_ctx.copy(),
        D_pos=D_pos, D_ctx=D_ctx,
    )


# ---------------------------------------------------------------------------
# Per-step update helpers (called from agent.run_trial)
# ---------------------------------------------------------------------------


def update_environment_states(
    Q_pos_prev: np.ndarray,
    Q_ctx_prev: np.ndarray,
    true_pos_prev: int,
    true_ctx_prev: int,
    chosen_action: int,
    B_pos: np.ndarray,
    B_ctx: np.ndarray,
    bb_ctx: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Port of updateEnvironmentStates.m for a single step.

    Returns updated (Q_pos, Q_ctx, true_pos, true_ctx). The agent's *belief*
    transitions use ``B_pos[:,:,chosen_action]`` and ``bb_ctx[:,:,chosen_action]``
    (column-normalised), while the *true* state transitions sample from the
    real ``B_pos[:, true_pos_prev, chosen_action]`` and ``B_ctx[:, true_ctx_prev, 0]``.
    """
    # Agent belief propagation
    Q_pos_new = B_pos[:, :, chosen_action] @ Q_pos_prev
    Q_ctx_new = bb_ctx[:, :, chosen_action] @ Q_ctx_prev

    # True state sampling (env)
    true_pos_new = sample_categorical(B_pos[:, true_pos_prev, chosen_action], rng)
    true_ctx_new = sample_categorical(B_ctx[:, true_ctx_prev, 0], rng)

    return Q_pos_new, Q_ctx_new, true_pos_new, true_ctx_new


def update_needs(
    grid: GridConfig,
    true_pos: int,
    true_ctx: int,
    t: int,
    t_food: int,
    t_water: int,
    t_sleep: int,
) -> tuple[int, int, int]:
    """Port of inline need-update block in SI_modular.m.

    Increments time-since for non-resource cells only when ``t > 0`` (first
    timestep is special: no increment).
    """
    on_food = grid.food_sources[true_ctx] == true_pos
    on_water = grid.water_sources[true_ctx] == true_pos
    on_sleep = grid.sleep_sources[true_ctx] == true_pos

    if on_food:
        new_food = 0
        new_water = t_water + (1 if t > 0 else 0)
        new_sleep = t_sleep + (1 if t > 0 else 0)
    elif on_water:
        new_food = t_food + (1 if t > 0 else 0)
        new_water = 0
        new_sleep = t_sleep + (1 if t > 0 else 0)
    elif on_sleep:
        new_food = t_food + (1 if t > 0 else 0)
        new_water = t_water + (1 if t > 0 else 0)
        new_sleep = 0
    else:
        if t > 0:
            new_food = t_food + 1
            new_water = t_water + 1
            new_sleep = t_sleep + 1
        else:
            new_food, new_water, new_sleep = t_food, t_water, t_sleep

    return new_food, new_water, new_sleep


def sample_observations(
    A_pos: np.ndarray,
    A_resource: np.ndarray,
    A_hill: np.ndarray,
    true_pos: int,
    true_ctx: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of inline observation block.

    Returns three one-hot observation vectors (positional, resource, hill).
    Each is sampled by inverse-CDF from the corresponding ``A`` column.
    """
    out = []
    for A in (A_pos, A_resource, A_hill):
        col = A[:, true_pos, true_ctx]
        idx = sample_categorical(col, rng)
        vec = np.zeros(A.shape[0], dtype=np.float64)
        vec[idx] = 1.0
        out.append(vec)
    return tuple(out)


def normalise_b_ctx(b_ctx: np.ndarray) -> np.ndarray:
    """Column-normalise the context belief transition (uses action 0).

    Mirrors MATLAB ``bb{2} = normalise_matrix(b{2})`` which acts on the
    full (C, C, num_actions) tensor.
    """
    out = np.empty_like(b_ctx)
    for action in range(b_ctx.shape[2]):
        out[:, :, action] = normalise_matrix_columns(b_ctx[:, :, action])
    return out

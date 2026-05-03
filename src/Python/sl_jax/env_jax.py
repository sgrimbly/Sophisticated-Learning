"""JAX-native env primitives.

Pure-JAX versions of the env transitions, observation sampling,
need-timer updates, calculate_posterior, and real Dirichlet update.
All take/return jnp arrays and are jit/vmap/scan-friendly.

Note on RNG: we use JAX's PRNG (jax.random) — does NOT match the
NumPy results bit-for-bit, but produces the same statistical
distribution. This is intentional for the GPU-native path.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


_LOG_FLOOR = 1e-16


def normalise_flat(x: jnp.ndarray) -> jnp.ndarray:
    s = x.sum()
    return jnp.where(s > 0, x / s, jnp.ones_like(x) / x.size)


def sample_categorical(key, p: jnp.ndarray) -> jnp.ndarray:
    """Single categorical draw from probabilities p (shape (K,)). Returns int32."""
    p_safe = jnp.where(jnp.isfinite(p), p, 0.0)
    p_safe = jnp.maximum(p_safe, 0.0)
    total = p_safe.sum()
    p_safe = jnp.where(total > 0, p_safe / total, jnp.ones_like(p_safe) / p_safe.size)
    return jax.random.categorical(key, jnp.log(jnp.maximum(p_safe, _LOG_FLOOR))).astype(jnp.int32)


def update_environment_states_jax(
    Q_pos_prev: jnp.ndarray,    # (S,)
    Q_ctx_prev: jnp.ndarray,    # (C,)
    true_pos_prev: jnp.ndarray, # scalar
    true_ctx_prev: jnp.ndarray, # scalar
    chosen_action: jnp.ndarray, # scalar
    B_pos: jnp.ndarray,         # (S, S, num_actions)
    B_ctx: jnp.ndarray,         # (C, C, num_actions)
    bb_ctx: jnp.ndarray,        # (C, C, num_actions)
    key: jax.random.KeyArray,
):
    """Returns (Q_pos_new, Q_ctx_new, true_pos_new, true_ctx_new, key)."""
    # Belief propagation
    Q_pos_new = B_pos[:, :, chosen_action] @ Q_pos_prev
    Q_ctx_new = bb_ctx[:, :, chosen_action] @ Q_ctx_prev

    # True transitions sampled from real B (matches MATLAB
    # updateEnvironmentStates.m). Position uses chosen action; context
    # uses action-0 column (cyclic transition).
    key, k1, k2 = jax.random.split(key, 3)
    pos_dist = B_pos[:, true_pos_prev, chosen_action]
    ctx_dist = B_ctx[:, true_ctx_prev, 0]
    true_pos_new = sample_categorical(k1, pos_dist)
    true_ctx_new = sample_categorical(k2, ctx_dist)
    return Q_pos_new, Q_ctx_new, true_pos_new, true_ctx_new, key


def update_needs_jax(
    food_sources: jnp.ndarray,   # (C,) per-context food positions
    water_sources: jnp.ndarray,  # (C,)
    sleep_sources: jnp.ndarray,  # (C,)
    true_pos: jnp.ndarray,
    true_ctx: jnp.ndarray,
    t: jnp.ndarray,
    t_food: jnp.ndarray,
    t_water: jnp.ndarray,
    t_sleep: jnp.ndarray,
):
    """Pure-JAX port of update_needs. Increment by 1 if t > 0 and not
    on the resource for the active context."""
    food_pos = food_sources[true_ctx]
    water_pos = water_sources[true_ctx]
    sleep_pos = sleep_sources[true_ctx]

    on_food = true_pos == food_pos
    on_water = true_pos == water_pos
    on_sleep = true_pos == sleep_pos

    inc = jnp.where(t > 0, 1, 0)

    # Default: increment all
    new_food = t_food + inc
    new_water = t_water + inc
    new_sleep = t_sleep + inc

    # Reset whichever resource we're on (overrides increment).
    new_food = jnp.where(on_food, 0, new_food)
    new_water = jnp.where(on_water, 0, new_water)
    new_sleep = jnp.where(on_sleep, 0, new_sleep)
    return new_food, new_water, new_sleep


def sample_observations_jax(
    A_pos: jnp.ndarray,        # (S, S, C)
    A_resource: jnp.ndarray,   # (4, S, C)
    A_hill: jnp.ndarray,       # (5, S, C)
    true_pos: jnp.ndarray,
    true_ctx: jnp.ndarray,
    key: jax.random.KeyArray,
):
    """Sample one-hot observations from A. Returns (O_pos, O_res, O_hill, key)."""
    key, k1, k2, k3 = jax.random.split(key, 4)
    p_pos = A_pos[:, true_pos, true_ctx]
    p_res = A_resource[:, true_pos, true_ctx]
    p_hill = A_hill[:, true_pos, true_ctx]
    o_pos = sample_categorical(k1, p_pos)
    o_res = sample_categorical(k2, p_res)
    o_hill = sample_categorical(k3, p_hill)
    O_pos = jax.nn.one_hot(o_pos, p_pos.size, dtype=p_pos.dtype)
    O_res = jax.nn.one_hot(o_res, p_res.size, dtype=p_res.dtype)
    O_hill = jax.nn.one_hot(o_hill, p_hill.size, dtype=p_hill.dtype)
    return O_pos, O_res, O_hill, key


def calculate_posterior_jax(
    P_pos: jnp.ndarray,
    P_ctx: jnp.ndarray,
    A_resource: jnp.ndarray,
    A_hill: jnp.ndarray,
    O_resource: jnp.ndarray,
    O_hill: jnp.ndarray,
):
    L_resource = jnp.einsum("o,osc->sc", O_resource, A_resource)
    L_hill = jnp.einsum("o,osc->sc", O_hill, A_hill)
    L = L_resource * L_hill
    LL = P_pos @ L
    y = LL * P_ctx
    return P_pos, normalise_flat(y)


def real_dirichlet_update_jax(
    a_resource: jnp.ndarray,    # (4, S, C)
    O_res: jnp.ndarray,         # (4,) one-hot
    P_pos: jnp.ndarray,         # (S,)
    P_ctx_smoothed: jnp.ndarray, # (C,)
    proportion: float = 0.3,
    scale: float = 0.7,
    floor: float = 0.05,
):
    """Pure-JAX port of sl.learning.real_dirichlet_update.

    Mirrors:
      a_learning = O_res ⊗ P_pos ⊗ P_ctx_smoothed
      a_learning *= (a_resource > 0)
      apply row-0 penalty: where row-0 of a_learning is 0, subtract
        proportion * max(a_learning[1:]) from row 0.
      a_new = a_resource + scale * a_learning
      clamp a_new at floor.
    """
    L = (O_res[:, None, None]
         * P_pos[None, :, None]
         * P_ctx_smoothed[None, None, :])  # (4, S, C)
    a_learning = L * (a_resource > 0)
    # row-0 penalty
    max_above = jnp.max(a_learning[1:], axis=0)             # (S, C)
    amount = proportion * max_above                          # (S, C)
    empty_zero = a_learning[0] == 0                          # (S, C)
    new_row0 = jnp.where(empty_zero,
                         a_learning[0] - amount,
                         a_learning[0])
    a_learning = a_learning.at[0].set(new_row0)

    out = a_resource + scale * a_learning
    out = jnp.where(out <= floor, floor, out)
    return out


def normalise_matrix_columns_jax(a: jnp.ndarray) -> jnp.ndarray:
    """Column-normalise across outcome axis (axis 0)."""
    s = a.sum(axis=0, keepdims=True)
    return jnp.where(s > 0, a / s, jnp.ones_like(a) / a.shape[0])

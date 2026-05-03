"""JAX-friendly environment primitives.

We re-use the canonical NumPy initialiser from ``sl.env``, then convert
the resulting tensors to JAX arrays (kept on default backend; will
automatically transfer to GPU when used inside a jit/vmap region).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
from sl.env import GenerativeModel, initialise_environment  # noqa: E402

from .config import GridConfig


@dataclass
class JaxModel:
    """JAX-array generative-model bundle.

    All arrays are jnp arrays (float64). Layout matches sl.env exactly
    so MATLAB / sl-package semantics are preserved.
    """

    A_pos: jnp.ndarray        # (S, S, C)  — identity-like
    A_resource: jnp.ndarray   # (4, S, C) — true resource likelihood
    A_hill: jnp.ndarray       # (5, S, C) — true hill likelihood
    a_resource: jnp.ndarray   # (4, S, C) — learned Dirichlet pseudocounts
    B_pos: jnp.ndarray        # (S, S, 5) — true position transition
    B_ctx: jnp.ndarray        # (C, C, 5) — true context transition
    bb_ctx: jnp.ndarray       # (C, C, 5) — agent's normalised b
    D_pos: jnp.ndarray        # (S,)
    D_ctx: jnp.ndarray        # (C,)


def build_jax_model(grid: GridConfig) -> JaxModel:
    np_model: GenerativeModel = initialise_environment(grid)
    return JaxModel(
        A_pos=jnp.asarray(np_model.A_pos),
        A_resource=jnp.asarray(np_model.A_resource),
        A_hill=jnp.asarray(np_model.A_hill),
        a_resource=jnp.asarray(np_model.a_resource),
        B_pos=jnp.asarray(np_model.B_pos),
        B_ctx=jnp.asarray(np_model.B_ctx),
        bb_ctx=jnp.asarray(_normalise_columns(np_model.b_ctx)),
        D_pos=jnp.asarray(np_model.D_pos),
        D_ctx=jnp.asarray(np_model.D_ctx),
    )


def _normalise_columns(b: np.ndarray) -> np.ndarray:
    """Column-normalise b along outcome axis (matches sl.env.normalise_b_ctx)."""
    out = np.empty_like(b)
    for a in range(b.shape[2]):
        s = b[:, :, a].sum(axis=0, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        out[:, :, a] = b[:, :, a] / s
    return out


def update_a_resource(model: JaxModel, new_a: jnp.ndarray) -> JaxModel:
    """Return a new JaxModel with a_resource replaced (immutable)."""
    return JaxModel(
        A_pos=model.A_pos,
        A_resource=model.A_resource,
        A_hill=model.A_hill,
        a_resource=new_a,
        B_pos=model.B_pos,
        B_ctx=model.B_ctx,
        bb_ctx=model.bb_ctx,
        D_pos=model.D_pos,
        D_ctx=model.D_ctx,
    )

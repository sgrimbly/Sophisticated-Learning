"""Optional Numba JIT acceleration for hot inner kernels.

Numba is *not* a hard dependency. If the import fails, ``HAS_NUMBA`` is
False and callers should fall back to the pure-NumPy implementations in
:mod:`sl.inference` / :mod:`sl.efe`.

This file targets the two real bottlenecks in profiling:

  * :func:`G_epistemic_value` — its inner loop over non-zero joint states
    is the largest single hot spot.
  * :func:`spm_cross_2factor` — called once per tree-search node.

We deliberately do **not** JIT the recursive tree-search functions: Numba
can do recursion, but the tree-search closures over Python lists of
arrays (history buffers in SL) which would force an object-mode fallback
and yield no speedup. Hot-path acceleration of the leaf math captures
most of the win.

To enable JIT in a venv:
    pip install numba

The JIT versions are *only* drop-in for the specific signatures below;
they do not replicate the public-API tolerances of the NumPy versions
(e.g. cell/list inputs).
"""
from __future__ import annotations

import numpy as np

try:
    import numba as _nb
    HAS_NUMBA = True
except Exception:  # pragma: no cover — exercised only when numba absent
    HAS_NUMBA = False


if HAS_NUMBA:

    _LOG_FLOOR = float(np.exp(-16.0))
    _REALMAX = float(np.finfo(np.float64).max)

    @_nb.njit(cache=True, fastmath=True)
    def _nat_log(x):  # type: ignore[misc]
        return np.log(x + _LOG_FLOOR)

    # ------------------------------------------------------------------
    # Inference primitives — replacements for the NumPy versions
    # ------------------------------------------------------------------

    @_nb.njit(cache=True, fastmath=True)
    def normalise_jit(x: np.ndarray) -> np.ndarray:
        """JIT version of inference.normalise (1-D)."""
        x = np.ascontiguousarray(x).ravel().astype(np.float64)
        n = x.shape[0]
        out = np.empty(n, dtype=np.float64)
        s = 0.0
        for i in range(n):
            v = x[i]
            if v != v or v < 0.0 or v == np.inf or v == -np.inf:
                v = 0.0
            x[i] = v
            s += v
        if s <= 0.0:
            for i in range(n):
                out[i] = 1.0 / n
            return out
        for i in range(n):
            out[i] = x[i] / s
        return out

    @_nb.njit(cache=True, fastmath=True)
    def calculate_posterior_jit(
        P_pos: np.ndarray, P_ctx: np.ndarray,
        A_resource: np.ndarray, A_hill: np.ndarray,
        O_resource: np.ndarray, O_hill: np.ndarray,
    ):
        """JIT inference.calculate_posterior. Returns (P_pos, P_ctx_new)."""
        n_o_r, n_s, n_c = A_resource.shape
        n_o_h = A_hill.shape[0]

        # Per-(state, context) likelihood from each modality.
        L = np.empty((n_s, n_c), dtype=np.float64)
        for s in range(n_s):
            for c in range(n_c):
                lr = 0.0
                for o in range(n_o_r):
                    lr += O_resource[o] * A_resource[o, s, c]
                lh = 0.0
                for o in range(n_o_h):
                    lh += O_hill[o] * A_hill[o, s, c]
                L[s, c] = lr * lh

        # LL[c] = sum_s P_pos[s] * L[s, c]
        LL = np.zeros(n_c, dtype=np.float64)
        for c in range(n_c):
            v = 0.0
            for s in range(n_s):
                v += P_pos[s] * L[s, c]
            LL[c] = v

        y = np.empty(n_c, dtype=np.float64)
        for c in range(n_c):
            y[c] = LL[c] * P_ctx[c]
        return P_pos, normalise_jit(y)

    @_nb.njit(cache=True, fastmath=True)
    def kldir_jit(a: np.ndarray, b: np.ndarray) -> float:
        """JIT efe.kldir."""
        n = a.size
        af = a.ravel()
        bf = b.ravel()
        kl = 0.0
        for i in range(n):
            av = af[i]; bv = bf[i]
            if av <= 0.0:
                continue
            if bv <= 0.0:
                return _REALMAX
            kl += av * np.log(av / bv)
        if kl != kl or kl == np.inf or kl == -np.inf:
            return _REALMAX
        return kl

    @_nb.njit(cache=True, fastmath=True)
    def joint_pos_ctx_v2(p_pos: np.ndarray, p_ctx: np.ndarray) -> np.ndarray:
        """Same as joint_pos_ctx_jit, kept for API parity."""
        n_p = p_pos.shape[0]
        n_c = p_ctx.shape[0]
        out = np.empty(n_p * n_c, dtype=np.float64)
        idx = 0
        for j in range(n_c):
            v = p_ctx[j]
            for i in range(n_p):
                out[idx] = p_pos[i] * v
                idx += 1
        return out

    # ------------------------------------------------------------------
    # Learning helpers
    # ------------------------------------------------------------------

    @_nb.njit(cache=True, fastmath=True)
    def build_a_learning_jit(
        O_res: np.ndarray, P_pos: np.ndarray, P_ctx: np.ndarray,
        a_mask: np.ndarray,
    ) -> np.ndarray:
        """4-way outer product (4 outcomes × S positions × C contexts) gated by mask>0.

        Equivalent to learning.build_a_learning for the canonical resource modality.
        """
        n_o = O_res.shape[0]
        n_s = P_pos.shape[0]
        n_c = P_ctx.shape[0]
        out = np.zeros((n_o, n_s, n_c), dtype=np.float64)
        for o in range(n_o):
            ov = O_res[o]
            if ov == 0.0:
                continue
            for s in range(n_s):
                sv = ov * P_pos[s]
                if sv == 0.0:
                    continue
                for c in range(n_c):
                    if a_mask[o, s, c] > 0.0:
                        out[o, s, c] = sv * P_ctx[c]
        return out

    @_nb.njit(cache=True, fastmath=True)
    def apply_row0_penalty_jit(a_learning: np.ndarray, proportion: float) -> np.ndarray:
        """JIT learning.apply_row0_penalty."""
        n_o, n_s, n_c = a_learning.shape
        out = a_learning.copy()
        for s in range(n_s):
            for c in range(n_c):
                if out[0, s, c] != 0.0:
                    continue
                # max over outcomes 1..end at (s, c)
                m = -1e308
                for o in range(1, n_o):
                    if out[o, s, c] > m:
                        m = out[o, s, c]
                amount = proportion * m
                out[0, s, c] = out[0, s, c] - amount
        return out

    @_nb.njit(cache=True, fastmath=True)
    def real_dirichlet_update_jit(
        a_resource: np.ndarray,
        O_res: np.ndarray,
        P_pos: np.ndarray,
        P_ctx: np.ndarray,
        proportion: float,
        scale: float,
        floor: float,
    ) -> np.ndarray:
        """JIT learning.real_dirichlet_update."""
        a_learning = build_a_learning_jit(O_res, P_pos, P_ctx, a_resource)
        a_learning = apply_row0_penalty_jit(a_learning, proportion)
        n_o, n_s, n_c = a_resource.shape
        out = np.empty_like(a_resource)
        for o in range(n_o):
            for s in range(n_s):
                for c in range(n_c):
                    v = a_resource[o, s, c] + scale * a_learning[o, s, c]
                    if v <= floor:
                        v = floor
                    out[o, s, c] = v
        return out

    @_nb.njit(cache=True, fastmath=True)
    def planning_dirichlet_update_jit(
        a_resource_imag: np.ndarray,
        O_res: np.ndarray,
        P_pos: np.ndarray,
        P_ctx: np.ndarray,
        learning_weight: float,
        prune_threshold: float,
    ):
        """JIT learning.planning_dirichlet_update.

        Returns (a_imag_new, a_learning_unweighted, a_learning_weighted).
        """
        a_learning = build_a_learning_jit(O_res, P_pos, P_ctx, a_resource_imag)
        if prune_threshold > 0.0:
            n_o, n_s, n_c = a_learning.shape
            for o in range(n_o):
                for s in range(n_s):
                    for c in range(n_c):
                        if a_learning[o, s, c] <= prune_threshold:
                            a_learning[o, s, c] = 0.0

        a_weighted = a_learning.copy()
        n_o, n_s, n_c = a_weighted.shape
        for o in range(1, n_o):
            for s in range(n_s):
                for c in range(n_c):
                    a_weighted[o, s, c] = learning_weight * a_learning[o, s, c]

        a_new = a_resource_imag + a_learning
        return a_new, a_learning, a_weighted

    # ------------------------------------------------------------------
    # spm_backwards — the inner loop is small but called many times in SL
    # ------------------------------------------------------------------

    @_nb.njit(cache=True, fastmath=True)
    def spm_backwards_step(
        L: np.ndarray,                # (C,) — current smoothed posterior, mutated
        p: np.ndarray,                # (C, C) — propagator, mutated
        B_ctx_step: np.ndarray,       # (C, C)
        O_hill_step: np.ndarray,      # (n_o,)
        Q_pos_step: np.ndarray,       # (S,)
        A_hill: np.ndarray,           # (n_o, S, C)
    ):
        """One step of spm_backwards integration. Mutates L and p in place."""
        n_o, n_s, n_c = A_hill.shape

        # p = B_ctx_step @ p
        new_p = np.zeros((n_c, n_c), dtype=np.float64)
        for i in range(n_c):
            for j in range(n_c):
                v = 0.0
                for k in range(n_c):
                    v += B_ctx_step[i, k] * p[k, j]
                new_p[i, j] = v
        for i in range(n_c):
            for j in range(n_c):
                p[i, j] = new_p[i, j]

        # L_modal[s, c] = sum_o O_hill_step[o] * A_hill[o, s, c]
        # temp_c = Q_pos_step @ L_modal  -> (n_c,)
        temp_c = np.zeros(n_c, dtype=np.float64)
        for c in range(n_c):
            v = 0.0
            for s in range(n_s):
                lm = 0.0
                for o in range(n_o):
                    lm += O_hill_step[o] * A_hill[o, s, c]
                v += Q_pos_step[s] * lm
            temp_c[c] = v

        # aaa = temp_c @ p   -> (n_c,)
        aaa = np.zeros(n_c, dtype=np.float64)
        for c in range(n_c):
            v = 0.0
            for cc in range(n_c):
                v += temp_c[cc] * p[cc, c]
            aaa[c] = v

        for c in range(n_c):
            L[c] = L[c] * aaa[c]


    @_nb.njit(cache=True, fastmath=True)
    def spm_cross_2factor(p_pos: np.ndarray, p_ctx: np.ndarray) -> np.ndarray:
        """Outer product of two 1-D arrays — equivalent to spm_cross(p_pos, p_ctx).

        Returns shape (len(p_pos), len(p_ctx)).
        """
        n_p = p_pos.shape[0]
        n_c = p_ctx.shape[0]
        out = np.empty((n_p, n_c), dtype=np.float64)
        for i in range(n_p):
            v = p_pos[i]
            for j in range(n_c):
                out[i, j] = v * p_ctx[j]
        return out

    @_nb.njit(cache=True, fastmath=True)
    def joint_pos_ctx_jit(p_pos: np.ndarray, p_ctx: np.ndarray) -> np.ndarray:
        """Column-major flatten of spm_cross_2factor (matches MATLAB qs(:))."""
        n_p = p_pos.shape[0]
        n_c = p_ctx.shape[0]
        out = np.empty(n_p * n_c, dtype=np.float64)
        idx = 0
        for j in range(n_c):
            v = p_ctx[j]
            for i in range(n_p):
                out[idx] = p_pos[i] * v
                idx += 1
        return out

    @_nb.njit(cache=True, fastmath=True)
    def G_epistemic_value_jit(
        A_pos_flat: np.ndarray,        # (S, S*C)  reshape(A_pos, S, -1, order='F')
        A_resource_flat: np.ndarray,    # (4, S*C)
        A_hill_flat: np.ndarray,        # (5, S*C)
        P_pos: np.ndarray,
        P_ctx: np.ndarray,
    ) -> float:
        """JIT version of G_epistemic_value for the canonical 3-modality model.

        Specialised to (pos, ctx) factor structure; A arrays must be
        F-order-flattened to (n_outcomes, S*C) before calling.
        """
        n_pos = P_pos.shape[0]
        n_ctx = P_ctx.shape[0]
        n_joint = n_pos * n_ctx

        n_o_pos = A_pos_flat.shape[0]
        n_o_res = A_resource_flat.shape[0]
        n_o_hill = A_hill_flat.shape[0]

        n_outcomes_total = n_o_pos * n_o_res * n_o_hill

        qx = np.empty(n_joint, dtype=np.float64)
        idx = 0
        for j in range(n_ctx):
            v = P_ctx[j]
            for i in range(n_pos):
                qx[idx] = P_pos[i] * v
                idx += 1

        qo = np.zeros(n_outcomes_total, dtype=np.float64)
        G = 0.0

        for k in range(n_joint):
            w = qx[k]
            if w <= _LOG_FLOOR:
                continue

            # build po = outer(A_pos_flat[:,k], A_resource_flat[:,k], A_hill_flat[:,k])
            # but in column-major sense for parity with spm_cross.
            # po has size n_o_pos * n_o_res * n_o_hill, indexed in F-order
            # (pos varies fastest, then resource, then hill).
            po = np.empty(n_outcomes_total, dtype=np.float64)
            idx = 0
            for hh in range(n_o_hill):
                vh = A_hill_flat[hh, k]
                for rr in range(n_o_res):
                    vrh = vh * A_resource_flat[rr, k]
                    for pp in range(n_o_pos):
                        po[idx] = A_pos_flat[pp, k] * vrh
                        idx += 1

            # accumulate
            log_po = np.log(po + _LOG_FLOOR)
            ent = 0.0
            for ii in range(n_outcomes_total):
                qo[ii] += w * po[ii]
                ent += po[ii] * log_po[ii]
            G += w * ent

        # subtract entropy of marginal qo
        ent_qo = 0.0
        log_qo = np.log(qo + _LOG_FLOOR)
        for ii in range(n_outcomes_total):
            ent_qo += qo[ii] * log_qo[ii]
        return G - ent_qo


    def G_epistemic_value_fast(A_modalities, P_factors) -> float:
        """Wrapper that flattens A modalities to F-order and calls JIT body.

        Drop-in for :func:`sl.efe.G_epistemic_value` when
        ``len(A_modalities) == 3`` and ``len(P_factors) == 2`` (the canonical
        sophisticated-learning case). For other shapes, raises ``NotImplementedError``;
        callers must fall back to the NumPy version.
        """
        if len(A_modalities) != 3 or len(P_factors) != 2:
            raise NotImplementedError("JIT path only supports 3 modalities × 2 factors")
        A_pos, A_res, A_hill = A_modalities
        P_pos, P_ctx = P_factors
        # F-order flatten so column-major linear index matches MATLAB.
        Apf = A_pos.reshape(A_pos.shape[0], -1, order="F")
        Arf = A_res.reshape(A_res.shape[0], -1, order="F")
        Ahf = A_hill.reshape(A_hill.shape[0], -1, order="F")
        return G_epistemic_value_jit(Apf, Arf, Ahf,
                                      np.ascontiguousarray(P_pos.ravel(), dtype=np.float64),
                                      np.ascontiguousarray(P_ctx.ravel(), dtype=np.float64))

else:  # pragma: no cover

    def spm_cross_2factor(p_pos, p_ctx):  # type: ignore[no-redef]
        return np.outer(p_pos, p_ctx)

    def joint_pos_ctx_jit(p_pos, p_ctx):  # type: ignore[no-redef]
        return np.outer(p_pos, p_ctx).ravel(order="F")

    def G_epistemic_value_fast(A_modalities, P_factors):  # type: ignore[no-redef]
        from .efe import G_epistemic_value
        return G_epistemic_value(A_modalities, P_factors)


if HAS_NUMBA:

    @_nb.njit(cache=True, fastmath=True)
    def spm_backwards_jit_array(
        L_init: np.ndarray,         # (C,) — initial L = P_history_ctx_t
        O_hill_stack: np.ndarray,   # (T_total, n_o) — observation history
        P_pos_stack: np.ndarray,    # (T_total, S)   — position posterior history
        A_hill: np.ndarray,         # (n_o, S, C)
        B_ctx_step: np.ndarray,     # (C, C) — single action slice
        timey: int,
        t: int,
    ) -> np.ndarray:
        """JIT version of inference.spm_backwards taking flat history arrays."""
        n_c = L_init.shape[0]
        L = L_init.copy()
        p = np.eye(n_c, dtype=np.float64)
        for timestep in range(timey + 1, t + 1):
            spm_backwards_step(L, p, B_ctx_step,
                                O_hill_stack[timestep], P_pos_stack[timestep],
                                A_hill)

        # spm_norm: column-normalise (here L is 1-D; treat as a single column)
        s = 0.0
        for i in range(n_c):
            v = L[i]
            if v != v or v == np.inf or v == -np.inf:
                v = 0.0
            L[i] = v
            s += v
        out = np.empty(n_c, dtype=np.float64)
        if s <= 0.0 or s != s:
            for i in range(n_c):
                out[i] = 1.0 / n_c
        else:
            for i in range(n_c):
                out[i] = L[i] / s
        return out


def _wrap_jit_real_dirichlet_update(jit_func):
    """Match the kwarg signature of learning.real_dirichlet_update."""
    def _wrapped(a_resource, O_res, P_pos, P_ctx,
                  proportion=0.3, scale=0.7, floor=0.05):
        a_resource = np.ascontiguousarray(a_resource, dtype=np.float64)
        O_res = np.ascontiguousarray(O_res, dtype=np.float64).ravel()
        P_pos = np.ascontiguousarray(P_pos, dtype=np.float64).ravel()
        P_ctx = np.ascontiguousarray(P_ctx, dtype=np.float64).ravel()
        return jit_func(a_resource, O_res, P_pos, P_ctx,
                        float(proportion), float(scale), float(floor))
    return _wrapped


def _wrap_jit_planning_dirichlet_update(jit_func):
    def _wrapped(a_resource_imag, O_res, P_pos, P_ctx,
                  learning_weight, prune_threshold=0.2):
        a = np.ascontiguousarray(a_resource_imag, dtype=np.float64)
        O = np.ascontiguousarray(O_res, dtype=np.float64).ravel()
        Pp = np.ascontiguousarray(P_pos, dtype=np.float64).ravel()
        Pc = np.ascontiguousarray(P_ctx, dtype=np.float64).ravel()
        return jit_func(a, O, Pp, Pc, float(learning_weight), float(prune_threshold))
    return _wrapped


def _wrap_jit_calculate_posterior(jit_func):
    def _wrapped(P_pos, P_ctx, A_resource, A_hill, O_resource, O_hill):
        return jit_func(
            np.ascontiguousarray(P_pos, dtype=np.float64).ravel(),
            np.ascontiguousarray(P_ctx, dtype=np.float64).ravel(),
            np.ascontiguousarray(A_resource, dtype=np.float64),
            np.ascontiguousarray(A_hill, dtype=np.float64),
            np.ascontiguousarray(O_resource, dtype=np.float64).ravel(),
            np.ascontiguousarray(O_hill, dtype=np.float64).ravel(),
        )
    return _wrapped


def _wrap_jit_spm_backwards(jit_func):
    def _wrapped(O_history_hill, P_history_pos, P_history_ctx_t, A_hill, B_ctx,
                  timey, t):
        # Stack heterogeneous-time history into contiguous arrays.
        # O_history_hill[k] has shape (n_o,); P_history_pos[k] shape (S,).
        n_o = A_hill.shape[0]
        n_s = A_hill.shape[1]
        T_total = max(t + 1, len(O_history_hill))
        O_stack = np.zeros((T_total, n_o), dtype=np.float64)
        P_stack = np.zeros((T_total, n_s), dtype=np.float64)
        for k in range(T_total):
            if k < len(O_history_hill) and O_history_hill[k] is not None:
                O_stack[k, :] = O_history_hill[k]
            if k < len(P_history_pos) and P_history_pos[k] is not None:
                P_stack[k, :] = P_history_pos[k]
        L_init = np.ascontiguousarray(P_history_ctx_t, dtype=np.float64).ravel()
        # Use action 0 of B_ctx (matches MATLAB; context evolves independently of action)
        B_step = B_ctx[:, :, 0] if B_ctx.ndim == 3 else B_ctx
        return jit_func(L_init, O_stack, P_stack,
                        np.ascontiguousarray(A_hill, dtype=np.float64),
                        np.ascontiguousarray(B_step, dtype=np.float64),
                        int(timey), int(t))
    return _wrapped


def maybe_swap_in_jit_efe(extended: bool = True) -> bool:
    """Monkey-patch the JIT versions into their NumPy host modules.

    Parameters
    ----------
    extended : bool
        If True (default), also swap calculate_posterior, kldir,
        normalise, real_dirichlet_update, planning_dirichlet_update,
        spm_backwards. If False, only G_epistemic_value (the original
        minimal patch).

    Returns True iff Numba was available and the patches were applied.
    """
    if not HAS_NUMBA:
        return False

    from . import efe as _efe
    _efe._G_epistemic_value_orig = _efe.G_epistemic_value
    _efe.G_epistemic_value = G_epistemic_value_fast

    if not extended:
        return True

    from . import inference as _inference
    from . import learning as _learning
    from . import agent as _agent
    from . import efe as _efe2
    from .planning import si as _si
    from .planning import sl as _sl
    from .planning import ba as _ba
    from .planning import baucb as _baucb
    from .planning import common as _common

    _inference._calculate_posterior_orig = _inference.calculate_posterior
    _inference.calculate_posterior = _wrap_jit_calculate_posterior(calculate_posterior_jit)
    # Replace in the modules that imported the symbol directly.
    _agent.calculate_posterior = _inference.calculate_posterior
    _si.calculate_posterior = _inference.calculate_posterior
    _sl.calculate_posterior = _inference.calculate_posterior
    _ba.calculate_posterior = _inference.calculate_posterior
    _baucb.calculate_posterior = _inference.calculate_posterior

    _inference._spm_backwards_orig = _inference.spm_backwards
    _inference.spm_backwards = _wrap_jit_spm_backwards(spm_backwards_jit_array)
    _agent.spm_backwards = _inference.spm_backwards
    _sl.spm_backwards = _inference.spm_backwards

    _efe2._kldir_orig = _efe2.kldir
    _efe2.kldir = kldir_jit
    _si.kldir = kldir_jit
    _sl.kldir = kldir_jit
    _learning._build_a_learning_orig = _learning.build_a_learning

    _learning._real_dirichlet_update_orig = _learning.real_dirichlet_update
    _learning.real_dirichlet_update = _wrap_jit_real_dirichlet_update(real_dirichlet_update_jit)
    _agent.real_dirichlet_update = _learning.real_dirichlet_update

    _learning._planning_dirichlet_update_orig = _learning.planning_dirichlet_update
    _learning.planning_dirichlet_update = _wrap_jit_planning_dirichlet_update(
        planning_dirichlet_update_jit
    )
    _sl.planning_dirichlet_update = _learning.planning_dirichlet_update
    return True

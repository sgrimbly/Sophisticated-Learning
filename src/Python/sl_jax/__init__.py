"""JAX-batched Sophisticated Inference / Learning prototype.

This is an *approximation* of the canonical MATLAB SI/SL semantics:

  * Planner replaces the recursive cache-pruned tree search with **fixed-
    horizon full policy enumeration** over the action space (5^H
    sequences). All policies are scored in one batched forward pass
    using JAX primitives, vmapped across seeds.
  * No ``short_term_memory`` cache (full enumeration eliminates the need).
  * No ``likely_states > 1/8`` pruning (we follow B-transitions
    deterministically along each policy chain — qs is implicit in the
    state propagation).
  * Outer trial loop and Dirichlet a-learning remain in Python/NumPy
    (mostly the same shape as the canonical port). Only the planner
    is JAX-batched.

This is intentionally a *separate* package from ``sl/``: we don't try
to maintain seed-for-seed parity with MATLAB; the goal is similar
qualitative survival behavior + GPU/multi-seed throughput.
"""

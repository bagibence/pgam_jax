"""
Microbenchmark for PenaltyHandler.compute_sqrt / compute_log_det_and_grad.

Reports JIT-compiled steady-state time, not first-call compile time. Used to
verify the always-vmap refactor doesn't regress vs. the dispatch-style handler.

Usage: uv run python benchmarks/microbench_penalty_handler.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


from pgam_jax._penalty_handler import PenaltyHandler, DROP_LAST_COL, IDENTITY


def _diff2_penalty(n):
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return jnp.array(D.T @ D, dtype=float)


def make_handler():
    """Singleton-heavy config (worst case for always-vmap).

    4 SINGLE at different q, 1 KRONECKER, 1 KRONECKER_WITH_NULL, 1 GENERAL.
    No two penalties share a _group_key — every group is size 1.
    """
    ph = PenaltyHandler()
    for q, id_fn in [
        (8, IDENTITY),
        (10, DROP_LAST_COL),
        (12, IDENTITY),
        (15, DROP_LAST_COL),
    ]:
        ph.add(_diff2_penalty(q), penalize_null_space=False, identifiability_fn=id_fn)
    ph.add_kron(
        [_diff2_penalty(8), _diff2_penalty(10)],
        penalize_null_space=False,
        identifiability_fn=IDENTITY,
    )
    ph.add_kron(
        [_diff2_penalty(8), _diff2_penalty(10)],
        penalize_null_space=True,
        identifiability_fn=IDENTITY,
    )
    S_kron_general = jnp.stack(
        [
            jnp.kron(_diff2_penalty(6), jnp.eye(6)),
            jnp.kron(jnp.eye(6), _diff2_penalty(6)),
        ]
    )
    ph.add(S_kron_general, penalize_null_space=True, identifiability_fn=IDENTITY)
    return ph


def _time_jit(fn, args, *, warmup=3, iters=100):
    jfn = jax.jit(fn)
    for _ in range(warmup):
        out = jfn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = jfn(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6  # μs per call


def main():
    ph = make_handler()
    rng = np.random.default_rng(0)
    rhos = [jnp.array(rng.standard_normal(p.rho_len)) for p in ph._penalties]
    cs, cl = ph.build()

    sqrt_us = _time_jit(lambda r: cs(r), (rhos,))
    ld_us = _time_jit(lambda r: cl(r), (rhos,))

    print(f"n_penalties      = {len(ph)}")
    print(f"compute_sqrt          = {sqrt_us:8.1f} us/call")
    print(f"compute_log_det_and_g = {ld_us:8.1f} us/call")


if __name__ == "__main__":
    main()

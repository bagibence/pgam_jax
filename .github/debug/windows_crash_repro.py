"""
Find the smallest piece of the vmapped GENERAL penalty path that crashes on Windows.

Run one case per process, because a crash kills the process:

    python .github/debug/windows_crash_repro.py <case>
    python .github/debug/windows_crash_repro.py --list

The workflow ``.github/workflows/windows-debug.yaml`` runs every case. Cases
whose name starts with ``jax_`` import only jax and numpy, so they can go into a
JAX bug report as they are. Delete this file together with the workflow when the
debugging is done.
"""

import faulthandler
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

# Same inputs as the S1 and S_kron fixtures in tests/test_penalty_handler.py.
RHOS = np.array([[0.5, -0.5], [-0.5, 0.5]])


def diff2_penalty(n):
    """n x n second-derivative penalty, rank n - 2."""
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i : i + 3] = [1.0, -2.0, 1.0]
    return D.T @ D


def kron_penalty_tensor(S):
    """Kronecker-sum tensor for S x S, shape (2, n * n, n * n)."""
    eye = np.eye(S.shape[0])
    return np.stack([np.kron(S, eye), np.kron(eye, S)])


def full_rank_tensor(S_tensor):
    """Numpy copy of _full_rank_precompute: drop the joint null space."""
    frobs = np.sqrt((S_tensor**2).sum(axis=(1, 2), keepdims=True))
    ev, U = np.linalg.eigh((S_tensor / frobs).sum(axis=0))
    U_keep = U[:, ev > ev[-1] * np.finfo(float).eps ** 0.8]
    return U_keep.T[None] @ S_tensor @ U_keep[None]


S1 = diff2_penalty(10)
FULL_RANK_S = full_rank_tensor(kron_penalty_tensor(S1))  # (2, 96, 96)


def case_pgam_test_setup():
    """
    Eager part of TestJIT::test_vmap_groups_all_methods.

    This is the baseline. It must crash, or the script does not reproduce the
    CI failure.
    """
    from pgam_jax._penalty_handler import PenaltyHandler
    from pgam_jax.penalty_utils import IDENTITY, ndim_tensor_product_basis_penalty

    S1_j = jnp.array(S1)
    S_kron = jnp.array(ndim_tensor_product_basis_penalty(S1, S1))
    ph = PenaltyHandler()
    for _ in range(2):
        ph.add(S1_j, penalize_null_space=False, identifiability_fn=IDENTITY)
    for _ in range(2):
        ph.add(S1_j, penalize_null_space=True, identifiability_fn=IDENTITY)
    for _ in range(2):
        ph.add_kron([S1_j, S1_j], penalize_null_space=False, identifiability_fn=IDENTITY)
    for _ in range(2):
        ph.add_kron([S1_j, S1_j], penalize_null_space=True, identifiability_fn=IDENTITY)
    for _ in range(2):
        ph.add(S_kron, penalize_null_space=True, identifiability_fn=IDENTITY)
    rhos = [
        jnp.array([0.5]),
        jnp.array([-0.5]),
        jnp.array([0.5, -0.5]),
        jnp.array([-0.5, 0.5]),
        jnp.array([0.5, -0.5]),
        jnp.array([-0.5, 0.5]),
        jnp.array([0.5, -0.5, 0.0]),
        jnp.array([-0.5, 0.5, 0.0]),
        jnp.array([0.5, -0.5]),
        jnp.array([-0.5, 0.5]),
    ]
    return ph.compute_sqrt(rhos)


def case_pgam_general_only():
    """Only the two GENERAL penalties, so only the vmapped _GeneralPenalty.sqrt runs."""
    from pgam_jax._penalty_handler import PenaltyHandler
    from pgam_jax.penalty_utils import IDENTITY, ndim_tensor_product_basis_penalty

    S_kron = jnp.array(ndim_tensor_product_basis_penalty(S1, S1))
    ph = PenaltyHandler()
    for _ in range(2):
        ph.add(S_kron, penalize_null_space=True, identifiability_fn=IDENTITY)
    return ph.compute_sqrt([jnp.array(r) for r in RHOS])


def case_slam_vmap():
    """vmap of transform_slam_with_Q, the while_loop inside _GeneralPenalty.sqrt."""
    from pgam_jax._slam_compute import transform_slam_with_Q

    S = jnp.stack([jnp.array(FULL_RANK_S)] * 2)
    return jax.vmap(transform_slam_with_Q)(S, jnp.exp(jnp.array(RHOS)))


def case_slam_twice():
    """transform_slam_with_Q on the same inputs, without vmap."""
    from pgam_jax._slam_compute import transform_slam_with_Q

    S = jnp.array(FULL_RANK_S)
    return [transform_slam_with_Q(S, jnp.exp(jnp.array(r))) for r in RHOS]


def _weighted_sums():
    return jnp.stack([jnp.einsum("i,ijk->jk", jnp.exp(r), FULL_RANK_S) for r in RHOS])


def case_jax_eigh_single():
    """eigh on one (96, 96) matrix. Control case for jax_eigh_vmap."""
    return jnp.linalg.eigh(_weighted_sums()[0])


def case_jax_eigh_vmap():
    """vmap of eigh on a (2, 96, 96) batch."""
    return jax.vmap(jnp.linalg.eigh)(_weighted_sums())


def case_jax_eigvalsh_vmap():
    """vmap of eigvalsh on a (2, 96, 96) batch."""
    return jax.vmap(jnp.linalg.eigvalsh)(_weighted_sums())


def case_jax_while_eigh_vmap():
    """
    vmap of a while_loop with eigvalsh, cond and eigh inside.

    This has the same structure as transform_slam_with_Q, without the pgam_jax
    details.
    """

    def run(S_tensor, lams):
        q = S_tensor.shape[1]

        def body(carry):
            Q, step = carry
            S = Q.T @ jnp.einsum("i,ijk->jk", lams, S_tensor) @ Q
            ev = jnp.linalg.eigvalsh(S)

            def rotate(_):
                _, U = jnp.linalg.eigh(S)
                return Q @ U[:, ::-1]

            Q = lax.cond(ev[-1] > 0, rotate, lambda _: Q, None)
            return Q, step + 1

        init = (jnp.eye(q, dtype=S_tensor.dtype), jnp.zeros((), dtype=jnp.int32))
        Q, _ = lax.while_loop(lambda c: c[1] < 5, body, init)
        return Q

    S = jnp.stack([jnp.asarray(FULL_RANK_S)] * 2)
    return jax.vmap(run)(S, jnp.exp(jnp.asarray(RHOS)))


CASES = {
    name.removeprefix("case_"): fn
    for name, fn in globals().items()
    if name.startswith("case_")
}


def main():
    if sys.argv[1:] == ["--list"]:
        print("\n".join(CASES))
        return
    if len(sys.argv) != 2 or sys.argv[1] not in CASES:
        raise SystemExit(f"usage: {sys.argv[0]} --list | {{{','.join(CASES)}}}")
    name = sys.argv[1]
    faulthandler.enable()
    print(f"{name}: jax {jax.__version__}, default float={jnp.array(1.0).dtype}", flush=True)
    jax.block_until_ready(CASES[name]())
    print(f"{name}: OK", flush=True)


if __name__ == "__main__":
    main()

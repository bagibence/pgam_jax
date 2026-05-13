"""
Tests for _slam_compute: transform_slam, log_det_slam, grad_log_det_slam,
hes_log_det_slam.

float64 is enabled globally in conftest.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgam_jax._slam_compute import (
    _compute_log_det_slam_factory,
    grad_log_det_slam,
    hes_log_det_slam,
    log_det_and_grad_slam,
    log_det_slam,
    transform_slam,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def psd_stack():
    """Three random PSD matrices, moderate rho."""
    rng = np.random.default_rng(42)
    q, M = 8, 3
    A = [rng.standard_normal((q, q)) for _ in range(M)]
    S = jnp.stack([jnp.array(a.T @ a) for a in A])
    rho = jnp.array(rng.uniform(-3, 3, M))
    return S, rho


# ---------------------------------------------------------------------------
# 1. Toy extreme-lambda log-det ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_logdet_extreme_rho_ground_truth(seed):
    """log|λ₁ S₁ + λ₂ S₂| == 3 ρ₁ + 2 ρ₂ when S₁,S₂ span complementary subspaces."""
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal((5, 5))
    _, U = np.linalg.eigh(rand.T @ rand)
    S = jnp.stack(
        [
            jnp.array(U[:, :3] @ U[:, :3].T),
            jnp.array(U[:, 3:] @ U[:, 3:].T),
        ]
    )
    rho = jnp.array([-20.0, 15.0])
    true_log_det = 3 * (-20.0) + 2 * 15.0

    S_i_out = jax.jit(transform_slam)(S, rho)
    ld = float(log_det_slam(rho, S_i_out))

    assert abs(ld - true_log_det) < 1e-8, f"seed={seed}: error={ld - true_log_det:.2e}"


# ---------------------------------------------------------------------------
# 2. S_i_out finite, no NaN/Inf
# ---------------------------------------------------------------------------


def test_transform_finite_moderate(psd_stack):
    S, rho = psd_stack
    S_i_out = transform_slam(S, rho)
    assert jnp.isfinite(S_i_out).all()


def test_transform_finite_extreme():
    rng = np.random.default_rng(7)
    q, M = 6, 4
    A = [rng.standard_normal((q, q)) for _ in range(M)]
    S = jnp.stack([jnp.array(a.T @ a) for a in A])
    rho = jnp.array([-20.0, 15.0, -5.0, 0.0])
    S_i_out = transform_slam(S, rho)
    assert jnp.isfinite(S_i_out).all()


# ---------------------------------------------------------------------------
# 3. Log-det consistent with naive eigvalsh (moderate rho)
# ---------------------------------------------------------------------------


def test_logdet_vs_naive_eigvalsh(psd_stack):
    S, rho = psd_stack
    lams = jnp.exp(rho)

    S_i_out = transform_slam(S, rho)
    ld_algo = float(log_det_slam(rho, S_i_out))

    Slam_naive = jnp.einsum("ijk,i->jk", S, lams)
    ev = np.linalg.eigvalsh(np.array(Slam_naive))
    ld_naive = float(np.log(ev[ev > ev[-1] * 1e-12]).sum())

    assert abs(ld_algo - ld_naive) < 1e-6


# ---------------------------------------------------------------------------
# 4. Gradient finite-difference check
# ---------------------------------------------------------------------------


def test_gradient_finite_difference(psd_stack):
    S, rho = psd_stack
    M = rho.shape[0]
    h = 1e-5

    S_i_out = jax.jit(transform_slam)(S, rho)
    g = np.array(grad_log_det_slam(rho, S_i_out))

    for j in range(M):
        rho_p = rho.at[j].add(h)
        rho_m = rho.at[j].add(-h)
        fd = (
            float(log_det_slam(rho_p, jax.jit(transform_slam)(S, rho_p)))
            - float(log_det_slam(rho_m, jax.jit(transform_slam)(S, rho_m)))
        ) / (2 * h)
        err = abs(g[j] - fd)
        assert err < 1e-5, f"grad[{j}]: analytic={g[j]:.6f} fd={fd:.6f} err={err:.2e}"


# ---------------------------------------------------------------------------
# 5. Hessian finite-difference check (gradient of gradient)
# ---------------------------------------------------------------------------


def test_hessian_finite_difference():
    rng = np.random.default_rng(7)
    q, M = 6, 3
    A = [rng.standard_normal((q, q)) for _ in range(M)]
    S = jnp.stack([jnp.array(a.T @ a) for a in A])
    rho = jnp.array(rng.uniform(-2, 2, M))
    h = 1e-4

    S_i_out = jax.jit(transform_slam)(S, rho)
    H = np.array(hes_log_det_slam(rho, S_i_out))

    for j in range(M):
        rho_p = rho.at[j].add(h)
        rho_m = rho.at[j].add(-h)
        fd_col = (
            np.array(grad_log_det_slam(rho_p, jax.jit(transform_slam)(S, rho_p)))
            - np.array(grad_log_det_slam(rho_m, jax.jit(transform_slam)(S, rho_m)))
        ) / (2 * h)
        err = np.max(np.abs(H[:, j] - fd_col))
        assert err < 1e-4, f"hess col {j}: max|err|={err:.2e}"


# ---------------------------------------------------------------------------
# 6. JIT compiles without error
# ---------------------------------------------------------------------------


def test_jit_compiles():
    rng = np.random.default_rng(99)
    q, M = 5, 2
    S = jnp.stack([jnp.array(rng.standard_normal((q, q))) for _ in range(M)])
    S = jnp.einsum("mij,mkj->mik", S, S)
    rho = jnp.array([-20.0, 15.0])
    jax.jit(transform_slam)(S, rho)  # must not raise


# ---------------------------------------------------------------------------
# 7. Empty-gamma termination — previously crashed with ValueError
# ---------------------------------------------------------------------------


def test_empty_gamma_no_crash():
    """All matrices dominant from iteration 1; gamma empties before r == Q."""
    rng = np.random.default_rng(99)
    v = rng.standard_normal(6)
    S = jnp.stack([jnp.array(np.outer(v, v)), jnp.array(np.outer(v, v) * 0.5)])
    rho = jnp.array([2.0, 1.0])
    S_i_out = transform_slam(S, rho)
    assert jnp.isfinite(S_i_out).all()


# ---------------------------------------------------------------------------
# 8. log_det_and_grad_slam agrees with separate log_det + grad calls
# ---------------------------------------------------------------------------


def test_log_det_and_grad_matches_separate(psd_stack):
    S, rho = psd_stack
    S_i_out = transform_slam(S, rho)

    ld_ref = float(log_det_slam(rho, S_i_out))
    g_ref = np.array(grad_log_det_slam(rho, S_i_out))

    ld, g = log_det_and_grad_slam(rho, S_i_out)

    assert abs(float(ld) - ld_ref) < 1e-12
    np.testing.assert_allclose(np.array(g), g_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. Factory: val+grad matches per-block loop
# ---------------------------------------------------------------------------


def _make_penalty_tree(rng, shapes):
    """Build a penalty_tree: list of (M_i, q_i, q_i) PSD tensors."""
    blocks = []
    rhos = []
    for M, q in shapes:
        A = rng.standard_normal((M, q, q))
        S = jnp.array(np.einsum("mij,mkj->mik", A, A))
        blocks.append(S)
        rhos.append(jnp.array(rng.uniform(-2, 2, M)))
    return blocks, rhos


@pytest.fixture(scope="module")
def multi_block_data():
    rng = np.random.default_rng(0)
    # mix of shapes: two blocks of (2,6,6), one of (3,8,8), one of (2,6,6) again
    shapes = [(2, 6), (3, 8), (2, 6), (4, 5)]
    return _make_penalty_tree(rng, shapes)


def test_factory_val_grad_matches_loop(multi_block_data):
    penalty_tree, rho_tree = multi_block_data
    compute_val_grad, _ = _compute_log_det_slam_factory(penalty_tree)
    ld_factory, g_factory = compute_val_grad(rho_tree)

    for i, (S, rho) in enumerate(zip(penalty_tree, rho_tree)):
        S_out = transform_slam(S, rho)
        ld_ref = float(log_det_slam(rho, S_out))
        g_ref = np.array(grad_log_det_slam(rho, S_out))
        assert abs(float(ld_factory[i]) - ld_ref) < 1e-6, f"block {i} log_det mismatch"
        np.testing.assert_allclose(
            np.array(g_factory[i]), g_ref, atol=1e-6, err_msg=f"block {i} grad mismatch"
        )


# ---------------------------------------------------------------------------
# 10. Factory: hess matches per-block loop
# ---------------------------------------------------------------------------


def test_factory_hess_matches_loop(multi_block_data):
    penalty_tree, rho_tree = multi_block_data
    _, compute_hess = _compute_log_det_slam_factory(penalty_tree)
    h_factory = compute_hess(rho_tree)

    for i, (S, rho) in enumerate(zip(penalty_tree, rho_tree)):
        S_out = transform_slam(S, rho)
        h_ref = np.array(hes_log_det_slam(rho, S_out))
        np.testing.assert_allclose(
            np.array(h_factory[i]), h_ref, atol=1e-6, err_msg=f"block {i} hess mismatch"
        )


# ---------------------------------------------------------------------------
# 11. Block-diagonal validation: sum/concat/block_diag vs naive eigvalsh + FD
# ---------------------------------------------------------------------------


def _naive_log_det_slam(penalty_tree, rho_flat, split_sizes):
    """Assemble full block-diagonal S_lam and return log|S_lam|_+ via eigvalsh."""
    rho_list = np.split(rho_flat, np.cumsum(split_sizes[:-1]))
    blocks = [
        np.einsum("k,kij->ij", np.exp(r), np.array(S))
        for r, S in zip(rho_list, penalty_tree)
    ]
    S_lam = np.zeros((sum(b.shape[0] for b in blocks),) * 2)
    offset = 0
    for b in blocks:
        n = b.shape[0]
        S_lam[offset : offset + n, offset : offset + n] = b
        offset += n
    ev = np.linalg.eigvalsh(S_lam)
    return float(np.sum(np.log(ev[ev > ev[-1] * 1e-10])))


@pytest.fixture(scope="module")
def block_diag_data():
    rng = np.random.default_rng(7)
    # moderate rho only — naive eigvalsh is reliable here
    shapes = [(2, 5), (3, 7), (2, 5)]
    blocks, rhos = _make_penalty_tree(rng, shapes)
    # keep rho moderate
    rhos = [jnp.array(rng.uniform(-1.5, 1.5, M)) for M, _ in shapes]
    return blocks, rhos


def test_block_diag_log_det(block_diag_data):
    penalty_tree, rho_tree = block_diag_data
    split_sizes = [len(r) for r in rho_tree]
    rho_flat = np.concatenate([np.array(r) for r in rho_tree])

    compute_val_grad, _ = _compute_log_det_slam_factory(penalty_tree)
    log_dets, _ = compute_val_grad(rho_tree)

    ld_factory = float(sum(float(ld) for ld in log_dets))
    ld_naive = _naive_log_det_slam(penalty_tree, rho_flat, split_sizes)

    assert (
        abs(ld_factory - ld_naive) < 1e-6
    ), f"log det: factory={ld_factory:.8f} naive={ld_naive:.8f}"


def test_block_diag_grad(block_diag_data):
    penalty_tree, rho_tree = block_diag_data
    split_sizes = [len(r) for r in rho_tree]
    rho_flat = np.concatenate([np.array(r) for r in rho_tree])
    h = 1e-6

    compute_val_grad, _ = _compute_log_det_slam_factory(penalty_tree)
    _, grads = compute_val_grad(rho_tree)
    grad_factory = np.concatenate([np.array(g) for g in grads])

    grad_fd = np.zeros_like(rho_flat)
    for j in range(len(rho_flat)):
        rp = rho_flat.copy()
        rp[j] += h
        rm = rho_flat.copy()
        rm[j] -= h
        grad_fd[j] = (
            _naive_log_det_slam(penalty_tree, rp, split_sizes)
            - _naive_log_det_slam(penalty_tree, rm, split_sizes)
        ) / (2 * h)

    np.testing.assert_allclose(
        grad_factory, grad_fd, atol=1e-8, err_msg="gradient vs FD"
    )


def test_block_diag_hess(block_diag_data):
    penalty_tree, rho_tree = block_diag_data
    split_sizes = [len(r) for r in rho_tree]
    rho_flat = np.concatenate([np.array(r) for r in rho_tree])
    h = 1e-4

    _, compute_hess = _compute_log_det_slam_factory(penalty_tree)
    hesses = compute_hess(rho_tree)

    # assemble block-diagonal hessian from factory blocks
    n_total = sum(split_sizes)
    H_factory = np.zeros((n_total, n_total))
    offset = 0
    for hb in hesses:
        hb = np.array(hb)
        k = hb.shape[0]
        H_factory[offset : offset + k, offset : offset + k] = hb
        offset += k

    # FD hessian via central differences on the gradient
    def grad_fd_at(rho_f):
        g = np.zeros(len(rho_f))
        for j in range(len(rho_f)):
            rp = rho_f.copy()
            rp[j] += h
            rm = rho_f.copy()
            rm[j] -= h
            g[j] = (
                _naive_log_det_slam(penalty_tree, rp, split_sizes)
                - _naive_log_det_slam(penalty_tree, rm, split_sizes)
            ) / (2 * h)
        return g

    H_fd = np.zeros((n_total, n_total))
    for j in range(n_total):
        rp = rho_flat.copy()
        rp[j] += h
        rm = rho_flat.copy()
        rm[j] -= h
        H_fd[:, j] = (grad_fd_at(rp) - grad_fd_at(rm)) / (2 * h)

    np.testing.assert_allclose(H_factory, H_fd, atol=1e-6, err_msg="hessian vs FD")

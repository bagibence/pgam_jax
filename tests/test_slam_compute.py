"""
Tests for _slam_compute: transform_slam, log_det_slam, grad_log_det_slam,
hes_log_det_slam.

float64 is enabled globally in conftest.py.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from pgam_jax._slam_compute import (
    transform_slam,
    log_det_slam,
    grad_log_det_slam,
    hes_log_det_slam,
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

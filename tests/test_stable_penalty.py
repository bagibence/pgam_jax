"""Tests for _stable_penalty.py — steps 1, 2, 3."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgam_jax.penalty_utils import (
    SqrtMethod,
    _make_single_fns,
    _precompute_single,
    sym_sqrt,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_spd(q, rng, rank=None):
    """Random symmetric positive-semidefinite matrix of size q x q."""
    rank = rank or q
    A = rng.standard_normal((q, rank))
    return A @ A.T


def _diff2(n):
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def _second_deriv_penalty(n):
    D = _diff2(n)
    return D.T @ D


# ---------------------------------------------------------------------------
# Step 1: sym_sqrt
# ---------------------------------------------------------------------------

class TestSymSqrt:
    def test_full_rank_round_trip(self):
        rng = np.random.default_rng(0)
        S = _random_spd(8, rng)
        S = jnp.array(S)
        B, rank, log_det = sym_sqrt(S)
        np.testing.assert_allclose(B.T @ B, S, atol=1e-10)
        assert int(rank) == 8

    def test_rank_deficient_round_trip(self):
        """B.T @ B should recover S even when S has a null space."""
        rng = np.random.default_rng(1)
        S = jnp.array(_random_spd(8, rng, rank=5))
        B, rank, log_det = sym_sqrt(S)
        np.testing.assert_allclose(B.T @ B, S, atol=1e-10)
        assert int(rank) == 5

    def test_log_det_correct(self):
        rng = np.random.default_rng(2)
        S = jnp.array(_random_spd(6, rng))
        _, _, log_det = sym_sqrt(S)
        _, ref = jnp.linalg.slogdet(S)
        np.testing.assert_allclose(float(log_det), float(ref), atol=1e-10)

    def test_second_deriv_penalty_null_space(self):
        """2nd-deriv penalty has a 2-dim null space; rank should be n-2."""
        n = 8
        S = jnp.array(_second_deriv_penalty(n))
        B, rank, log_det = sym_sqrt(S)
        np.testing.assert_allclose(B.T @ B, S, atol=1e-10)
        assert int(rank) == n - 2

    def test_preserves_small_eigenvalues_f64(self):
        """
        Old code used eps_f32 * max_eig as threshold, masking valid small
        eigenvalues when max/min ratio > 1/eps_f32 ~ 8e6.  With sqrt(eps_f64)
        the safe ratio is ~67M.  Test a ratio of 1e7 which breaks the old
        threshold but not the new one.

        Old threshold: eps_f32 * 1e4 = 1.2e-7 * 1e4 = 1.2e-3  > 1e-3  → masked
        New threshold: sqrt(eps_f64) * 1e4 ~ 1.5e-8 * 1e4 = 1.5e-4  < 1e-3  → kept
        """
        S = jnp.diag(jnp.array([1e5, 1.0, 1e-5]))
        B, rank, _ = sym_sqrt(S)
        assert int(rank) == 3, "all eigenvalues should survive at f64"
        np.testing.assert_allclose(B.T @ B, S, atol=1e-8)


# ---------------------------------------------------------------------------
# Step 3: SINGLE precompute + block functions
# ---------------------------------------------------------------------------

class TestSingleCase:
    @pytest.fixture()
    def penalty_tensor(self):
        n = 10
        S = jnp.array(_second_deriv_penalty(n))
        # shape (1, q, q) as the factory expects
        return S[None]

    def test_precompute_shapes(self, penalty_tensor):
        data = _precompute_single(penalty_tensor)
        q = penalty_tensor.shape[-1]
        assert data["B"].shape == (q, q)
        assert int(data["rank"]) == q - 2  # 2nd-deriv has 2-dim null space

    def test_sqrt_fn_round_trip(self, penalty_tensor):
        """B_scaled.T @ B_scaled should equal lambda * S."""
        data = _precompute_single(penalty_tensor)
        sqrt_fn, _ = _make_single_fns(data, apply_id=lambda x: x, positive_mon_func=jnp.exp)

        for rho_val in [-2.0, 0.0, 3.0, 10.0]:
            rho = jnp.array([rho_val])
            B = sqrt_fn(rho)
            lam = jnp.exp(rho[0])
            S_expected = lam * penalty_tensor[0]
            np.testing.assert_allclose(B.T @ B, S_expected, atol=1e-10, rtol=1e-8)

    def test_logdet_fn_value(self, penalty_tensor):
        """log|lambda * S|_+ should equal rank * log(lambda) + log|S|_+."""
        data = _precompute_single(penalty_tensor)
        _, logdet_fn = _make_single_fns(data, apply_id=lambda x: x, positive_mon_func=jnp.exp)

        # ground truth: apply the same masking as sym_sqrt to the unscaled matrix
        _, rank_ref, log_det_S_ref = sym_sqrt(penalty_tensor[0])

        for rho_val in [-2.0, 0.0, 3.0]:
            rho = jnp.array([rho_val])
            _, log_det, _ = logdet_fn(rho)
            ref = rank_ref * rho_val + log_det_S_ref
            np.testing.assert_allclose(float(log_det), float(ref), atol=1e-12)

    def test_logdet_grad_value(self, penalty_tensor):
        """Analytic grad should equal autodiff grad."""
        data = _precompute_single(penalty_tensor)
        _, logdet_fn = _make_single_fns(data, apply_id=lambda x: x, positive_mon_func=jnp.exp)

        def log_det_only(rho):
            _, ld, _ = logdet_fn(rho)
            return ld

        for rho_val in [-1.0, 0.0, 2.0]:
            rho = jnp.array([rho_val])
            analytic = logdet_fn(rho)[2]
            numeric = jax.grad(log_det_only)(rho)
            np.testing.assert_allclose(analytic, numeric, atol=1e-8)

    def test_apply_id_drops_column(self, penalty_tensor):
        data = _precompute_single(penalty_tensor)
        q = penalty_tensor.shape[-1]
        sqrt_fn, _ = _make_single_fns(
            data,
            apply_id=lambda x: x[..., :-1],
            positive_mon_func=jnp.exp,
        )
        B = sqrt_fn(jnp.array([0.0]))
        assert B.shape[1] == q - 1
"""Tests for _chol_deriv.py — JAX closed-form vs Wood's sequential recurrence and FD.

The JAX implementation uses the closed-form ``dR = Φ(R^{-T} dD R^{-1}) R``;
this test validates it against (a) the Wood 2017 Appendix B.7 sequential
recurrence (transcribed verbatim from the legacy PGAM `grad_cholesky`),
and (b) central differences of ``chol(D(x))``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import central_diff

from pgam_jax._chol_deriv import grad_cholesky, grad_chol_Vbeta

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Reference: legacy PGAM grad_cholesky (Wood 2017 Appendix B.7) — numpy loops.
# Kept here as the gold-standard ground truth for the new closed-form path.
# ---------------------------------------------------------------------------

def _legacy_grad_cholesky(grad_D: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Sequential upper-triangular Cholesky-derivative recurrence.

    Transcribed verbatim from the numpy reference (``der_wrt_smoothing.py``):

        R_ii dR_ii = ½ (dD_ii − Σ_{m<i} R_{mi} dR_{mi} − Σ_{m<i} dR_{mi} R_{mi})
        R_ii dR_ij = dD_ij − Σ_{m<i} dR_{mi} R_{mj} − Σ_{m<i} R_{mi} dR_{mj}
                            − R_{ij} dR_{ii}                    (j > i)
    """
    R = np.asarray(R)
    grad_D = np.asarray(grad_D)
    n = R.shape[0]
    dR = np.zeros((n, n))
    for i in range(n):
        Bii = (
            grad_D[i, i]
            - np.dot(dR[:i, i], R[:i, i])
            - np.dot(R[:i, i], dR[:i, i])
        )
        dR[i, i] = 0.5 * Bii / R[i, i]
        for j in range(i + 1, n):
            Bij = (
                grad_D[i, j]
                - np.dot(dR[:i, i], R[:i, j])
                - np.dot(R[:i, i], dR[:i, j])
            )
            dR[i, j] = (Bij - R[i, j] * dR[i, i]) / R[i, i]
    return dR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spd_matrix():
    """Well-conditioned SPD matrix used as D in R^T R = D."""
    rng = np.random.default_rng(7)
    p = 8
    A = rng.standard_normal((p, p))
    return jnp.asarray(A @ A.T + 0.5 * np.eye(p))


@pytest.fixture
def grad_D_sym():
    """Symmetric perturbation dD/dx."""
    rng = np.random.default_rng(11)
    p = 8
    B = rng.standard_normal((p, p))
    return jnp.asarray(B + B.T)


# ---------------------------------------------------------------------------
# grad_cholesky — single-parameter derivative
# ---------------------------------------------------------------------------

class TestGradCholesky:

    def test_matches_legacy_recurrence(self, spd_matrix, grad_D_sym):
        """Closed-form must equal Wood Appendix B.7 sequential recurrence."""
        R = jnp.linalg.cholesky(spd_matrix).T
        dR_jax = grad_cholesky(grad_D_sym, R)
        dR_legacy = _legacy_grad_cholesky(np.asarray(grad_D_sym), np.asarray(R))
        np.testing.assert_allclose(
            np.asarray(dR_jax), dR_legacy, atol=1e-10, rtol=1e-10
        )

    def test_matches_finite_difference(self, spd_matrix, grad_D_sym):
        """dR matches central difference of cholesky(D(x))."""
        D = np.asarray(spd_matrix)
        dD = np.asarray(grad_D_sym)
        R = jnp.linalg.cholesky(spd_matrix).T
        dR = np.asarray(grad_cholesky(grad_D_sym, R))

        def R_of_x(x):
            return np.linalg.cholesky(D + x[0] * dD).T.ravel()

        # central_diff returns Jacobian (p*p, 1); reshape to (p, p).
        dR_fd = central_diff(R_of_x, np.array([0.0]), rel_step=1e-6).reshape(R.shape)
        np.testing.assert_allclose(dR, dR_fd, rtol=1e-5, atol=1e-7)

    def test_dR_is_upper_triangular(self, spd_matrix, grad_D_sym):
        """The returned dR must respect the upper-triangular structure of R."""
        R = jnp.linalg.cholesky(spd_matrix).T
        dR = np.asarray(grad_cholesky(grad_D_sym, R))
        strict_lower = dR - np.triu(dR)
        np.testing.assert_allclose(strict_lower, 0.0, atol=1e-12)

    def test_satisfies_defining_identity(self, spd_matrix, grad_D_sym):
        """dR^T R + R^T dR = dD."""
        R = jnp.linalg.cholesky(spd_matrix).T
        dR = grad_cholesky(grad_D_sym, R)
        lhs = dR.T @ R + R.T @ dR
        np.testing.assert_allclose(
            np.asarray(lhs), np.asarray(grad_D_sym), atol=1e-10
        )

    def test_zero_perturbation(self, spd_matrix):
        """dR is zero when dD is zero."""
        R = jnp.linalg.cholesky(spd_matrix).T
        dR = grad_cholesky(jnp.zeros_like(spd_matrix), R)
        np.testing.assert_allclose(np.asarray(dR), 0.0, atol=1e-12)

    def test_jittable(self, spd_matrix, grad_D_sym):
        """grad_cholesky composes cleanly with jax.jit."""
        R = jnp.linalg.cholesky(spd_matrix).T
        dR_eager = grad_cholesky(grad_D_sym, R)
        dR_jit = jax.jit(grad_cholesky)(grad_D_sym, R)
        np.testing.assert_allclose(
            np.asarray(dR_jit), np.asarray(dR_eager), atol=1e-12
        )


# ---------------------------------------------------------------------------
# grad_chol_Vbeta — batched derivative of chol(V_β) wrt ρ  (AIC V'' term)
# ---------------------------------------------------------------------------

class TestGradCholVbeta:
    """``grad_chol_Vbeta(V_β, dV_β⁻¹/dρ) → dR/dρ`` with ``R^T R = V_β``.

    Mirrors the legacy ``grad_chol_Vb_rho`` (caller in ``compute_AIC``):
    V_β is the covariance, ``dV_β⁻¹/dρ_h = dH/dρ_h + λ_h S_h/φ`` is the
    precision derivative. The function internally re-inverts to the
    covariance derivative ``-V_β · dV_β⁻¹/dρ · V_β`` before calling
    grad_cholesky.
    """

    def test_matches_legacy_per_rho(self):
        """Batched (vmap) result matches per-ρ legacy recurrence."""
        rng = np.random.default_rng(3)
        p, M = 6, 3
        # Construct a covariance V_β and its-inverse-derivatives dV_β⁻¹/dρ_h.
        A = rng.standard_normal((p, p))
        Vb = jnp.asarray(A @ A.T + 0.4 * np.eye(p))
        dVb_inv = np.stack([
            (lambda B: B + B.T)(rng.standard_normal((p, p)))
            for _ in range(M)
        ])
        dVb_inv_j = jnp.asarray(dVb_inv)

        dR_batched = np.asarray(grad_chol_Vbeta(Vb, dVb_inv_j))

        # Reference: legacy applies grad_cholesky to dV_β/dρ = -V_β · dV_β⁻¹/dρ · V_β
        # against R = chol(V_β).T.
        Vb_np = np.asarray(Vb)
        R_np = np.linalg.cholesky(Vb_np).T
        dVb_np = -np.einsum("ij,hjk,kl->hil", Vb_np, dVb_inv, Vb_np)
        dR_ref = np.stack([
            _legacy_grad_cholesky(dVb_np[h], R_np) for h in range(M)
        ])
        np.testing.assert_allclose(dR_batched, dR_ref, atol=1e-10, rtol=1e-10)

    def test_matches_finite_difference(self):
        """dR[h] = d/dρ_h chol(V_β(ρ)) matches central differences.

        Parameterise the precision V_β⁻¹(ρ) = V_β⁻¹_0 + Σ_h ρ_h · S_h, so
        dV_β⁻¹/dρ_h = S_h, and at ρ=0 the covariance is V_β_0 = inv(V_β⁻¹_0).
        """
        rng = np.random.default_rng(5)
        p, M = 6, 2
        A = rng.standard_normal((p, p))
        Vbi0 = A @ A.T + 0.4 * np.eye(p)  # precision at ρ=0
        S_stack = np.stack([
            (lambda B: B + B.T)(rng.standard_normal((p, p)))
            for _ in range(M)
        ])

        Vb0 = np.linalg.inv(Vbi0)
        # dV_β⁻¹/dρ_h = S_h  (precision derivative — what grad_chol_Vbeta consumes)
        dVb_inv = S_stack

        dR_batched = np.asarray(
            grad_chol_Vbeta(jnp.asarray(Vb0), jnp.asarray(dVb_inv))
        )

        # FD reference: dR[h] = d/dρ_h chol(V_β(ρ))|_ρ=0 where
        # V_β(ρ) = inv(Vbi0 + Σ ρ_h S_h).
        def chol_at(rho):
            Vbi_p = Vbi0 + np.einsum("h,hij->ij", rho, S_stack)
            Vb_p = np.linalg.inv(Vbi_p)
            Vb_p = 0.5 * (Vb_p + Vb_p.T)  # symmetrise — Cholesky requires it
            return np.linalg.cholesky(Vb_p).T.ravel()

        dR_fd = central_diff(chol_at, np.zeros(M), rel_step=1e-6)
        dR_fd = dR_fd.reshape(p, p, M).transpose(2, 0, 1)
        np.testing.assert_allclose(dR_batched, dR_fd, rtol=1e-5, atol=1e-7)

    def test_jittable(self):
        rng = np.random.default_rng(13)
        p, M = 5, 2
        A = rng.standard_normal((p, p))
        Vb = jnp.asarray(A @ A.T + 0.3 * np.eye(p))
        dVb_inv = jnp.asarray(np.stack([
            (lambda B: B + B.T)(rng.standard_normal((p, p)))
            for _ in range(M)
        ]))
        dR_e = grad_chol_Vbeta(Vb, dVb_inv)
        dR_j = jax.jit(grad_chol_Vbeta)(Vb, dVb_inv)
        np.testing.assert_allclose(np.asarray(dR_j), np.asarray(dR_e), atol=1e-12)


# ---------------------------------------------------------------------------
# Discrepant-ρ stress: realistic Laplace-REML precision
# V_β⁻¹ = X^T W X + Σ_k exp(ρ_k) S_k, with ρ_k spanning many decades.
# ---------------------------------------------------------------------------

class TestDiscrepantRho:
    """Stress tests with smoothing parameters of wildly different magnitudes.

    In a real Laplace-REML fit some ρ_k diverge (heavily smoothed/effectively
    unused covariates, ``exp(20) ≈ 5e8``) while others shrink to keep wiggly
    components alive (``exp(-15) ≈ 3e-7``).  This spans ~10–15 orders of
    magnitude in λ_k — the regime where the naïve ``symmetric_sqrt(Σ λ S)``
    drops small-λ modes and PenaltyHandler's Wood-2011 stable construction is
    required.

    These tests verify :func:`grad_chol_Vbeta` survives the same regime: build
    the realistic Laplace-REML precision ``V_β⁻¹ = X^T W X + Σ_k exp(ρ_k) S_k``,
    pass the covariance ``V_β = (V_β⁻¹)⁻¹`` and the precision derivative
    ``dV_β⁻¹/dρ_k = λ_k S_k`` to ``grad_chol_Vbeta``, and check vs (a) the Wood
    Appendix B.7 recurrence and (b) central differences of ``chol(V_β(ρ))``.
    """

    @staticmethod
    def _make_precision(rng, p, n, rho, S_stack):
        """V_β⁻¹ = X^T W X + Σ_k exp(ρ_k) · S_k, fully SPD."""
        X = rng.standard_normal((n, p))
        W = np.abs(rng.standard_normal(n)) + 0.1
        XWX = X.T @ (W[:, None] * X)
        lams = np.exp(np.asarray(rho))
        return XWX + np.einsum("k,kij->ij", lams, S_stack), lams

    @staticmethod
    def _random_psd_stack(rng, p, M, jitter=1e-3):
        """Stack of (p, p) PSD penalty-like matrices with a touch of jitter."""
        return np.stack([
            (B := rng.standard_normal((p, p))) @ B.T + jitter * np.eye(p)
            for _ in range(M)
        ])

    @pytest.mark.parametrize(
        "rho_vals",
        [
            (10.0, -10.0, 0.0),     # 8 orders apart
            (15.0, -15.0, 0.0),     # 13 orders apart
            (20.0, -20.0, 5.0),     # 17 orders apart — extreme
            (-20.0, 20.0, -5.0),    # reversed sign pattern
        ],
    )
    def test_discrepant_matches_legacy(self, rho_vals):
        """Closed-form JAX path equals Wood B.7 recurrence under extreme ρ."""
        rng = np.random.default_rng(101)
        p, n, M = 8, 40, 3
        S_stack = self._random_psd_stack(rng, p, M)
        rho = np.array(rho_vals)
        Vbi, lams = self._make_precision(rng, p, n, rho, S_stack)
        Vb = np.linalg.inv(Vbi)
        Vb = 0.5 * (Vb + Vb.T)  # symmetrise — Cholesky requires it
        # dV_β⁻¹/dρ_k = λ_k S_k  (in this stripped-down setup, no W-derivative)
        dVb_inv = np.einsum("k,kij->kij", lams, S_stack)

        dR_jax = np.asarray(
            grad_chol_Vbeta(jnp.asarray(Vb), jnp.asarray(dVb_inv))
        )

        # Legacy reference: R = chol(V_β).T, dV_β/dρ = -V_β · dV_β⁻¹/dρ · V_β.
        R_np = np.linalg.cholesky(Vb).T
        dVb_np = -np.einsum("ij,hjk,kl->hil", Vb, dVb_inv, Vb)
        dR_legacy = np.stack([
            _legacy_grad_cholesky(dVb_np[h], R_np) for h in range(M)
        ])

        # Entries scale with V_β entries; rtol catches drift on dominant
        # entries, atol floor avoids spurious failures on near-zero ones.
        np.testing.assert_allclose(dR_jax, dR_legacy, rtol=1e-8, atol=1e-6)

    @pytest.mark.parametrize(
        "rho_vals",
        [
            (10.0, -10.0),
            (15.0, -8.0),
            (-15.0, 12.0),
        ],
    )
    def test_discrepant_matches_finite_difference(self, rho_vals):
        """Closed-form matches central differences of ``chol(V_β(ρ))``."""
        rng = np.random.default_rng(202)
        p, n, M = 6, 30, 2
        S_stack = self._random_psd_stack(rng, p, M)
        rho = np.array(rho_vals)

        X = rng.standard_normal((n, p))
        W = np.abs(rng.standard_normal(n)) + 0.1
        XWX = X.T @ (W[:, None] * X)

        def precision_at(rho_):
            lams = np.exp(np.asarray(rho_))
            return XWX + np.einsum("k,kij->ij", lams, S_stack)

        Vbi0 = precision_at(rho)
        Vb0 = np.linalg.inv(Vbi0)
        Vb0 = 0.5 * (Vb0 + Vb0.T)
        lams0 = np.exp(rho)
        dVb_inv = np.einsum("k,kij->kij", lams0, S_stack)

        dR_jax = np.asarray(
            grad_chol_Vbeta(jnp.asarray(Vb0), jnp.asarray(dVb_inv))
        )

        def chol_at(rho_):
            Vb_p = np.linalg.inv(precision_at(rho_))
            Vb_p = 0.5 * (Vb_p + Vb_p.T)
            return np.linalg.cholesky(Vb_p).T.ravel()

        dR_fd = central_diff(chol_at, rho, rel_step=1e-6)
        dR_fd = dR_fd.reshape(p, p, M).transpose(2, 0, 1)
        np.testing.assert_allclose(dR_jax, dR_fd, rtol=1e-3, atol=1e-4)

    def test_precision_dominated_by_huge_lambda(self):
        """One ρ → +∞: V_β picks up a tiny mode; dR stays finite, matches legacy."""
        rng = np.random.default_rng(303)
        p, n, M = 5, 25, 3
        S_stack = self._random_psd_stack(rng, p, M, jitter=1e-2)
        rho = np.array([18.0, -12.0, -12.0])

        X = rng.standard_normal((n, p))
        W = np.abs(rng.standard_normal(n)) + 0.1
        XWX = X.T @ (W[:, None] * X)
        lams = np.exp(rho)
        Vbi = XWX + np.einsum("k,kij->ij", lams, S_stack)
        Vb = np.linalg.inv(Vbi)
        Vb = 0.5 * (Vb + Vb.T)
        dVb_inv = np.einsum("k,kij->kij", lams, S_stack)

        dR_jax = np.asarray(
            grad_chol_Vbeta(jnp.asarray(Vb), jnp.asarray(dVb_inv))
        )
        assert np.all(np.isfinite(dR_jax))

        R_np = np.linalg.cholesky(Vb).T
        dVb_np = -np.einsum("ij,hjk,kl->hil", Vb, dVb_inv, Vb)
        dR_legacy = np.stack([
            _legacy_grad_cholesky(dVb_np[h], R_np) for h in range(M)
        ])
        np.testing.assert_allclose(dR_jax, dR_legacy)
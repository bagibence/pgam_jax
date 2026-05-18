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

from pgam_jax._chol_deriv import grad_cholesky, grad_U_Vbeta

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
# grad_U_Vbeta — batched derivative of U (where U U^T = V_β) wrt ρ  (AIC V''
# term, mgcv U-convention; see _chol_deriv.grad_U_Vbeta docstring for why we
# use V_β = U U^T rather than the textbook V_β = R^T R).
# ---------------------------------------------------------------------------


def _mgcv_dU_reference(V_beta_inv: np.ndarray, dVb_inv_stack: np.ndarray) -> np.ndarray:
    """Independent numpy reference for grad_U_Vbeta — mirrors mgcv Vb.corr.

    Steps:
      R̃ = chol(V_β⁻¹).T                                 (upper-tri)
      dR̃ = grad_cholesky(dV_β⁻¹/dρ_h, R̃)                (upper-chol deriv recurrence)
      dU = −R̃⁻¹ · dR̃ · R̃⁻¹                              (via explicit dense inv)
    """
    R_tilde = np.linalg.cholesky(V_beta_inv).T
    R_inv = np.linalg.solve(R_tilde, np.eye(R_tilde.shape[0]))  # upper-tri R̃⁻¹
    dU = np.zeros_like(dVb_inv_stack)
    for h in range(dVb_inv_stack.shape[0]):
        dR_tilde = _legacy_grad_cholesky(dVb_inv_stack[h], R_tilde)
        dU[h] = -R_inv @ dR_tilde @ R_inv
    return dU


class TestGradUVbeta:
    """``grad_U_Vbeta(V_β⁻¹, dV_β⁻¹/dρ) → dU/dρ`` with ``U U^T = V_β``.

    U is the upper-tri inverse of the Cholesky factor of V_β⁻¹.  Convention
    differs from Wood (2017) App. B.7 / legacy numpy PGAM (which use
    ``R^T R = V_β`` and assemble ``Σ V_ρ dR^T dR``); see the function
    docstring for the stability rationale.
    """

    def test_matches_mgcv_reference(self):
        """Batched (vmap) result matches the mgcv-recipe numpy reference."""
        rng = np.random.default_rng(3)
        p, M = 6, 3
        A = rng.standard_normal((p, p))
        Vbi = A @ A.T + 0.4 * np.eye(p)          # precision (well-conditioned)
        dVb_inv = np.stack([
            (lambda B: B + B.T)(rng.standard_normal((p, p)))
            for _ in range(M)
        ])

        dU_batched = np.asarray(
            grad_U_Vbeta(jnp.asarray(Vbi), jnp.asarray(dVb_inv))
        )
        dU_ref = _mgcv_dU_reference(Vbi, dVb_inv)
        np.testing.assert_allclose(dU_batched, dU_ref, atol=1e-10, rtol=1e-10)

    def test_satisfies_defining_identity(self):
        """dU U^T + U dU^T = dV_β/dρ_h, the U-convention defining relation."""
        rng = np.random.default_rng(7)
        p, M = 5, 3
        A = rng.standard_normal((p, p))
        Vbi = A @ A.T + 0.5 * np.eye(p)
        dVb_inv = np.stack([
            (lambda B: B + B.T)(rng.standard_normal((p, p)))
            for _ in range(M)
        ])

        Vb = np.linalg.inv(Vbi)
        R_tilde = np.linalg.cholesky(Vbi).T
        U = np.linalg.solve(R_tilde, np.eye(p))         # U = R̃⁻¹, upper-tri

        dU = np.asarray(
            grad_U_Vbeta(jnp.asarray(Vbi), jnp.asarray(dVb_inv))
        )
        for h in range(M):
            lhs = dU[h] @ U.T + U @ dU[h].T
            rhs = -Vb @ dVb_inv[h] @ Vb        # = dV_β/dρ_h
            np.testing.assert_allclose(lhs, rhs, atol=1e-10, rtol=1e-10)

    def test_matches_finite_difference(self):
        """dU[h] = d/dρ_h U(ρ) matches central differences of chol(V_β⁻¹(ρ))⁻¹."""
        rng = np.random.default_rng(5)
        p, M = 6, 2
        A = rng.standard_normal((p, p))
        Vbi0 = A @ A.T + 0.4 * np.eye(p)
        S_stack = np.stack([
            (lambda B: B + B.T)(rng.standard_normal((p, p)))
            for _ in range(M)
        ])
        dVb_inv = S_stack                                  # dV_β⁻¹/dρ_h = S_h

        dU_batched = np.asarray(
            grad_U_Vbeta(jnp.asarray(Vbi0), jnp.asarray(dVb_inv))
        )

        def U_at(rho):
            Vbi_p = Vbi0 + np.einsum("h,hij->ij", rho, S_stack)
            R_tilde = np.linalg.cholesky(Vbi_p).T
            return np.linalg.solve(R_tilde, np.eye(p)).ravel()

        dU_fd = central_diff(U_at, np.zeros(M), rel_step=1e-6)
        dU_fd = dU_fd.reshape(p, p, M).transpose(2, 0, 1)
        np.testing.assert_allclose(dU_batched, dU_fd, rtol=1e-5, atol=1e-7)

    def test_dU_is_upper_triangular(self):
        """Returned dU[h] must respect the upper-triangular structure of U."""
        rng = np.random.default_rng(9)
        p, M = 7, 2
        A = rng.standard_normal((p, p))
        Vbi = jnp.asarray(A @ A.T + 0.5 * np.eye(p))
        dVb_inv = jnp.asarray(np.stack([
            (lambda B: B + B.T)(rng.standard_normal((p, p)))
            for _ in range(M)
        ]))
        dU = np.asarray(grad_U_Vbeta(Vbi, dVb_inv))
        for h in range(M):
            strict_lower = dU[h] - np.triu(dU[h])
            np.testing.assert_allclose(strict_lower, 0.0, atol=1e-12)

    def test_jittable(self):
        rng = np.random.default_rng(13)
        p, M = 5, 2
        A = rng.standard_normal((p, p))
        Vbi = jnp.asarray(A @ A.T + 0.3 * np.eye(p))
        dVb_inv = jnp.asarray(np.stack([
            (lambda B: B + B.T)(rng.standard_normal((p, p)))
            for _ in range(M)
        ]))
        dU_e = grad_U_Vbeta(Vbi, dVb_inv)
        dU_j = jax.jit(grad_U_Vbeta)(Vbi, dVb_inv)
        np.testing.assert_allclose(np.asarray(dU_j), np.asarray(dU_e), atol=1e-12)


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

    These tests verify :func:`grad_U_Vbeta` survives the same regime: build
    the realistic Laplace-REML precision ``V_β⁻¹ = X^T W X + Σ_k exp(ρ_k) S_k``,
    pass it and the precision derivative ``dV_β⁻¹/dρ_k = λ_k S_k`` to
    ``grad_U_Vbeta``, and check vs (a) the mgcv-recipe numpy reference and (b)
    central differences of ``chol(V_β⁻¹(ρ))⁻¹``.  In contrast to the textbook
    R-convention path (which would Cholesky-factor V_β directly and so amplify
    its small eigenvalues in the strong-penalty directions), the U-convention
    only ever factors V_β⁻¹, which the penalty *regularises* — so we expect
    accurate results even at the extreme corners of ρ-space.
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
    def test_discrepant_matches_mgcv_reference(self, rho_vals):
        """JAX path equals mgcv-recipe reference under extreme ρ."""
        rng = np.random.default_rng(101)
        p, n, M = 8, 40, 3
        S_stack = self._random_psd_stack(rng, p, M)
        rho = np.array(rho_vals)
        Vbi, lams = self._make_precision(rng, p, n, rho, S_stack)
        # dV_β⁻¹/dρ_k = λ_k S_k  (in this stripped-down setup, no W-derivative)
        dVb_inv = np.einsum("k,kij->kij", lams, S_stack)

        dU_jax = np.asarray(
            grad_U_Vbeta(jnp.asarray(Vbi), jnp.asarray(dVb_inv))
        )
        dU_ref = _mgcv_dU_reference(Vbi, dVb_inv)

        # Entries scale with U entries; rtol catches drift on dominant
        # entries, atol floor avoids spurious failures on near-zero ones.
        np.testing.assert_allclose(dU_jax, dU_ref, rtol=1e-8, atol=1e-6)

    @pytest.mark.parametrize(
        "rho_vals",
        [
            (10.0, -10.0),
            (15.0, -8.0),
            (-15.0, 12.0),
        ],
    )
    def test_discrepant_matches_finite_difference(self, rho_vals):
        """Matches central differences of ``U(ρ) = chol(V_β⁻¹(ρ))⁻¹``."""
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
        lams0 = np.exp(rho)
        dVb_inv = np.einsum("k,kij->kij", lams0, S_stack)

        dU_jax = np.asarray(
            grad_U_Vbeta(jnp.asarray(Vbi0), jnp.asarray(dVb_inv))
        )

        def U_at(rho_):
            R_tilde = np.linalg.cholesky(precision_at(rho_)).T
            return np.linalg.solve(R_tilde, np.eye(p)).ravel()

        dU_fd = central_diff(U_at, rho, rel_step=1e-6)
        dU_fd = dU_fd.reshape(p, p, M).transpose(2, 0, 1)
        np.testing.assert_allclose(dU_jax, dU_fd, rtol=1e-3, atol=1e-4)

    def test_precision_dominated_by_huge_lambda(self):
        """One ρ → +∞: V_β picks up a tiny mode; dU stays finite, matches reference."""
        rng = np.random.default_rng(303)
        p, n, M = 5, 25, 3
        S_stack = self._random_psd_stack(rng, p, M, jitter=1e-2)
        rho = np.array([18.0, -12.0, -12.0])

        X = rng.standard_normal((n, p))
        W = np.abs(rng.standard_normal(n)) + 0.1
        XWX = X.T @ (W[:, None] * X)
        lams = np.exp(rho)
        Vbi = XWX + np.einsum("k,kij->ij", lams, S_stack)
        dVb_inv = np.einsum("k,kij->kij", lams, S_stack)

        dU_jax = np.asarray(
            grad_U_Vbeta(jnp.asarray(Vbi), jnp.asarray(dVb_inv))
        )
        assert np.all(np.isfinite(dU_jax))
        dU_ref = _mgcv_dU_reference(Vbi, dVb_inv)
        np.testing.assert_allclose(dU_jax, dU_ref, rtol=1e-8, atol=1e-6)
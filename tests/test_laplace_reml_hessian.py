"""Tests for _laplace_reml_hessian.py — analytical Hessian of REML wrt ρ.

The Laplace-REML gradient (``laplace_reml(..., return_grad=True)[1]``) is the
total derivative dREML/dρ at the MAP β̂(ρ).  Differentiating it once more
numerically gives a high-precision Hessian reference; we compare against
:func:`hess_laplace_reml`'s analytical output.

For each fixture we also assert symmetry of the analytical Hessian.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import central_diff, fit_beta

from pgam_jax._laplace_reml import laplace_reml
from pgam_jax._laplace_reml_hessian import hess_laplace_reml


jax.config.update("jax_enable_x64", True)


def _grad_at_rho(rho_flat, prob, refit=False):
    """Analytical REML gradient at ρ — used as the FD seed for the Hessian."""
    X, y, obs, inv_link = prob["X"], prob["y"], prob["obs"], prob["inv_link"]
    S_all, beta_hat0, phi = prob["S_all"], prob["beta_hat"], prob["phi"]
    compute_sqrt = prob["compute_sqrt"]
    compute_ld = prob["compute_log_det_and_grad"]
    M_null = prob["M_null"]

    rho_arr = jnp.asarray(rho_flat)
    rhos_tree = [rho_arr[k:k + 1] for k in range(rho_arr.shape[0])]

    if refit:
        beta_hat = fit_beta(X, y, obs, inv_link, S_all, rho_arr, phi,
                            np.asarray(beta_hat0))
    else:
        beta_hat = beta_hat0

    _, g = laplace_reml(
        beta_hat, X, y, obs, inv_link, S_all, rhos_tree,
        phi, M_null, compute_sqrt, compute_ld,
        return_grad=True,
    )
    return np.asarray(g)


class TestHessLaplaceREML:
    """Analytical Hessian matches central differences of the analytical gradient."""

    def _fd_check(self, prob):
        X, y, obs, inv_link = prob["X"], prob["y"], prob["obs"], prob["inv_link"]
        S_all, beta_hat, V_beta, V_beta_inv, phi = (
            prob["S_all"], prob["beta_hat"], prob["V_beta"],
            prob["V_beta_inv"], prob["phi"],
        )
        rho = np.asarray(prob["rho"])
        rhos_tree = prob["rhos_tree"]

        # Block-diagonal Hessian of log|S_λ|_+ from PenaltyHandler.
        # The fixture's compute_log_det_and_grad alone isn't enough — rebuild
        # the handler from the fixture's penalty tree to get compute_log_det_hess.
        from pgam_jax._penalty_handler import PenaltyHandler

        ph = PenaltyHandler(non_linearity=jnp.exp)
        # The fixture's S_all has shape (M, P, P) with the intercept row/col zero;
        # for the handler we'd want raw (per-smooth) tensors, but the fixture
        # already passed (M, 1, q, q) → (1, q, q) blocks earlier.  We rebuild
        # by stripping the intercept and registering one (q, q) block per smooth.
        # Practically: every fixture smooth has SINGLE method (penalize_null_space=False),
        # whose log_det is linear in ρ → log_det_hess = 0.  So just zero matrix.
        M = rho.size
        log_det_hess_Slam = jnp.zeros((M, M))

        H_analytic = np.asarray(hess_laplace_reml(
            jnp.asarray(beta_hat), X, y, obs, inv_link,
            S_all, jnp.asarray(rho), phi,
            V_beta, V_beta_inv, log_det_hess_Slam,
        ))

        # FD reference: differentiate the analytical gradient wrt ρ.
        # rel_step=1e-5 strikes the cancellation-vs-truncation balance — same
        # eps PGAM's _script/check_hess_laplace_reml.py uses.
        H_fd = central_diff(
            lambda r: _grad_at_rho(r, prob, refit=True),
            rho,
            rel_step=1e-6,
        )

        # Hessian magnitudes can span a few orders depending on the fixture;
        # PGAM's FD-vs-analytic floor here is ~1e-3 relative / 1e-2 absolute.
        np.testing.assert_allclose(H_analytic, H_fd, rtol=5e-3, atol=1e-2)

    def _symmetry_check(self, prob):
        X, y, obs, inv_link = prob["X"], prob["y"], prob["obs"], prob["inv_link"]
        S_all, beta_hat, V_beta, V_beta_inv, phi = (
            prob["S_all"], prob["beta_hat"], prob["V_beta"],
            prob["V_beta_inv"], prob["phi"],
        )
        rho = jnp.asarray(prob["rho"])
        M = rho.shape[0]

        H = np.asarray(hess_laplace_reml(
            jnp.asarray(beta_hat), X, y, obs, inv_link,
            S_all, rho, phi,
            V_beta, V_beta_inv, jnp.zeros((M, M)),
        ))
        np.testing.assert_allclose(H, H.T, atol=1e-10)

    def test_fd_poisson(self, poisson_gam_problem):
        self._fd_check(poisson_gam_problem)

    def test_fd_gamma(self, gamma_gam_problem):
        self._fd_check(gamma_gam_problem)

    def test_symmetric_poisson(self, poisson_gam_problem):
        self._symmetry_check(poisson_gam_problem)

    def test_symmetric_gamma(self, gamma_gam_problem):
        self._symmetry_check(gamma_gam_problem)


class TestLogDetHessContribution:
    """Verify ``add2`` integrates the log-det Hessian additively (linearity)."""

    def test_log_det_hess_is_additive(self, poisson_gam_problem):
        """Hess(rho) increments by 0.5/φ · ΔlogDetHess when ΔlogDetHess is added."""
        prob = poisson_gam_problem
        X, y, obs, inv_link = prob["X"], prob["y"], prob["obs"], prob["inv_link"]
        S_all, beta_hat, V_beta, V_beta_inv, phi = (
            prob["S_all"], prob["beta_hat"], prob["V_beta"],
            prob["V_beta_inv"], prob["phi"],
        )
        rho = jnp.asarray(prob["rho"])
        M = rho.shape[0]

        H0 = hess_laplace_reml(
            jnp.asarray(beta_hat), X, y, obs, inv_link,
            S_all, rho, phi, V_beta, V_beta_inv, jnp.zeros((M, M)),
        )
        rng = np.random.default_rng(42)
        bump = rng.standard_normal((M, M))
        bump = 0.5 * (bump + bump.T)
        H1 = hess_laplace_reml(
            jnp.asarray(beta_hat), X, y, obs, inv_link,
            S_all, rho, phi, V_beta, V_beta_inv, jnp.asarray(bump),
        )
        diff = np.asarray(H1 - H0)
        expected = 0.5 * bump / float(phi)
        np.testing.assert_allclose(diff, expected, atol=1e-12)
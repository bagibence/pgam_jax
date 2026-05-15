"""Tests for _laplace_reml_beta_derivs.py."""

import jax.numpy as jnp
import numpy as np
from conftest import central_diff, fit_beta

from pgam_jax._laplace_reml_beta_derivs import d2beta_hat, dH_drho, dbeta_hat
from pgam_jax._pirls_weights import _make_w_fn


class TestDbetaHat:
    """J = d beta_hat / d rho, shape (M, p): central-FD of beta_hat(rho)."""

    def _fd_check(self, prob):
        rho = np.asarray(prob["rho"])
        X, y, obs, inv_link = prob["X"], prob["y"], prob["obs"], prob["inv_link"]
        S_all, beta_hat, V_beta, phi = (
            prob["S_all"], prob["beta_hat"], prob["V_beta"], prob["phi"]
        )

        J = dbeta_hat(beta_hat, V_beta, S_all, jnp.asarray(rho), phi)

        # scipy returns Jacobian of shape (p, M) for f: R^M -> R^p; we want (M, p).
        # rel_step=1e-4 matches the PGAM reference FD eps.
        J_fd = central_diff(
            lambda r: np.asarray(fit_beta(X, y, obs, inv_link, S_all,
                                          jnp.asarray(r), phi,
                                          np.asarray(beta_hat))),
            rho,
            rel_step=1e-4,
        ).T

        # FD precision floor on this design is ~1e-4 absolute (matches PGAM —
        # see _script/check_dbeta_hat.py); rtol catches drift on large entries.
        np.testing.assert_allclose(np.asarray(J), J_fd, rtol=1e-3, atol=1e-4)

    def test_fd_poisson(self, poisson_gam_problem):
        self._fd_check(poisson_gam_problem)

    def test_fd_gamma(self, gamma_gam_problem):
        self._fd_check(gamma_gam_problem)


class TestD2betaHat:
    """d²β̂/(dρ_h dρ_k), shape (M, M, p): central-FD of J(rho)."""

    def _fd_check(self, prob):
        rho = np.asarray(prob["rho"])
        X, y, obs, inv_link = prob["X"], prob["y"], prob["obs"], prob["inv_link"]
        S_all, beta_hat, V_beta, phi = (
            prob["S_all"], prob["beta_hat"], prob["V_beta"], prob["phi"]
        )

        J = dbeta_hat(beta_hat, V_beta, S_all, jnp.asarray(rho), phi)
        dH = dH_drho(beta_hat, X, y, obs, inv_link, J, phi)
        d2B = d2beta_hat(beta_hat, V_beta, dH, S_all, jnp.asarray(rho), phi, J)

        # FD reference: differentiate J(ρ) = -V_β(ρ) (λ S_λ/φ) β̂(ρ) once more.
        # Re-fit β̂ at perturbed ρ, recompute V_β, and assemble J via dbeta_hat —
        # this is what d2beta_hat differentiates analytically.
        from pgam_jax._laplace_reml_vbeta import vbeta_and_logdet

        compute_sqrt = prob["compute_sqrt"]

        def _vbeta_at(r, b):
            w = _make_w_fn(y, obs, inv_link)(X @ b)
            R = jnp.linalg.qr(jnp.sqrt(w)[:, None] * X, mode="r")
            # rho_flat is M-dim; the GAM fixture uses one-rho-per-smooth, so
            # the tree mirrors that layout.
            rhos_tree = [r[k:k + 1] for k in range(r.shape[0])]
            sqrt_pen = compute_sqrt(rhos_tree)
            sqrt_pen = jnp.hstack((jnp.zeros((sqrt_pen.shape[0], 1)), sqrt_pen))
            Vb, _, _ = vbeta_and_logdet(R, sqrt_pen, phi)
            return Vb

        def J_flat_at_rho(r):
            r_j = jnp.asarray(r)
            b = fit_beta(X, y, obs, inv_link, S_all, r_j, phi,
                         np.asarray(beta_hat))
            Vb = _vbeta_at(r_j, b)
            return np.asarray(
                dbeta_hat(b, Vb, S_all, r_j, phi)
            ).ravel()

        M = rho.size
        p = X.shape[1]
        d2B_fd = central_diff(J_flat_at_rho, rho, rel_step=1e-4)
        # scipy returns (M*p, M) — reshape to (M, p, M) and transpose to (M, M, p)
        d2B_fd = d2B_fd.reshape(M, p, M).transpose(0, 2, 1)

        np.testing.assert_allclose(np.asarray(d2B), d2B_fd, rtol=1e-2, atol=1e-4)

    def test_fd_poisson(self, poisson_gam_problem):
        self._fd_check(poisson_gam_problem)

    def test_fd_gamma(self, gamma_gam_problem):
        self._fd_check(gamma_gam_problem)


class TestDHdrho:
    """dH/d rho, shape (M, p, p): central-FD of H(beta_hat(rho))."""

    def _fd_check(self, prob):
        rho = np.asarray(prob["rho"])
        X, y, obs, inv_link = prob["X"], prob["y"], prob["obs"], prob["inv_link"]
        S_all, beta_hat, V_beta, phi = (
            prob["S_all"], prob["beta_hat"], prob["V_beta"], prob["phi"]
        )

        J = dbeta_hat(beta_hat, V_beta, S_all, jnp.asarray(rho), phi)
        dH = dH_drho(beta_hat, X, y, obs, inv_link, J, phi)             # (M, p, p)

        p = X.shape[1]
        M = rho.size
        w_fn = _make_w_fn(y, obs, inv_link)

        def H_flat_at_rho(r):
            b = fit_beta(X, y, obs, inv_link, S_all, jnp.asarray(r), phi,
                         np.asarray(beta_hat))
            w = w_fn(X @ b)
            return np.asarray(X.T @ (w[:, None] * X) / phi).ravel()

        # scipy returns Jacobian of shape (p*p, M); reshape to (M, p, p)
        dH_fd = central_diff(H_flat_at_rho, rho, rel_step=1e-4).T.reshape(M, p, p)

        np.testing.assert_allclose(np.asarray(dH), dH_fd, rtol=1e-3, atol=1e-4)

    def test_fd_poisson(self, poisson_gam_problem):
        self._fd_check(poisson_gam_problem)

    def test_fd_gamma(self, gamma_gam_problem):
        self._fd_check(gamma_gam_problem)
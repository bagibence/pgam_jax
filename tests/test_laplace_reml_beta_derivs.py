"""Tests for _laplace_reml_beta_derivs.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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


class TestD2betaHatFactorization:
    """Pin the U.transpose-based factorization against a transpose-free reference.

    The current ``d2beta_hat`` shares one ``V_β · (dSlam @ J)`` contraction across
    ``add2[h, k] = −V_β·dSlam[k]·J[h]`` and ``add1``'s dSlam-part ``[h, k]
    = −V_β·dSlam[h]·J[k] = −U[k, h]`` by reading ``U`` with a transposed layout.
    See ``_script/bench_d2beta_hat.py`` — the transpose has a real cost in JAX
    (XLA materialises a permuted buffer) but saves an entire ``V_β·dSlam·J``
    pass, so it wins at all tested sizes.

    This test guards against regressions if anyone rewrites the factorization
    or swaps to a different einsum spelling.  Parameterised over the bench's
    size sweep so the asserted equivalence covers tiny → high-D regimes.
    """

    @staticmethod
    def _d2beta_hat_no_transpose(beta_hat, V_beta, dH, S_all, rho, phi, J):
        """Reference: two independent ``V_β·dSlam·J`` einsums, no transpose."""
        M = rho.shape[0]
        dSlam = S_all * jnp.exp(rho)[:, None, None] / phi
        Vb_dSlam = jnp.einsum("il,klm->kim", V_beta, dSlam)
        part_a = -jnp.einsum("kim,hm->hki", Vb_dSlam, J)   # add2
        part_b = -jnp.einsum("him,km->hki", Vb_dSlam, J)   # add1's dSlam-part
        Vb_dH = jnp.einsum("ij,hjl->hil", V_beta, dH)
        add1_dH = -jnp.einsum("hil,kl->hki", Vb_dH, J)
        hes = add1_dH + part_a + part_b
        idx = jnp.arange(M)
        return hes.at[idx, idx].add(J)

    @staticmethod
    def _make_problem(p, M, seed=0):
        keys = jax.random.split(jax.random.key(seed), 5)
        A = jax.random.normal(keys[0], (p, p))
        V_beta = A @ A.T + 0.5 * jnp.eye(p)
        Bs = jax.random.normal(keys[1], (M, p, p))
        S_all = jnp.einsum("kij,klj->kil", Bs, Bs) + 1e-3 * jnp.eye(p)
        rho = jax.random.normal(keys[2], (M,))
        beta_hat = jax.random.normal(keys[3], (p,))
        J = jax.random.normal(keys[4], (M, p))
        Cs = jax.random.normal(jax.random.fold_in(keys[0], 1), (M, p, p))
        dH = Cs + jnp.swapaxes(Cs, -1, -2)
        return beta_hat, V_beta, dH, S_all, rho, 1.0, J

    @pytest.mark.parametrize(
        "p,M",
        [(15, 2), (30, 3), (60, 4), (120, 6), (200, 8), (300, 10)],
    )
    def test_matches_no_transpose_reference(self, p, M):
        args = self._make_problem(p, M)
        out_current = d2beta_hat(*args)
        out_ref = self._d2beta_hat_no_transpose(*args)
        # Tolerance grows with p (more FP accumulations in V_β·dSlam·J);
        # both reorderings are O(M·p³)+O(M²·p²), differences are pure
        # FP-summation-order noise.  Floor matches the bench's max|Δ|.
        np.testing.assert_allclose(
            np.asarray(out_current), np.asarray(out_ref),
            rtol=1e-8, atol=1e-8,
        )


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
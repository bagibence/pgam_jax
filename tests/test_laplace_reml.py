"""Tests for _laplace_reml.py — value and gradient of the Laplace REML objective."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import nemos.observation_models as nmo_obs
import numpy as np
import pytest
from conftest import central_diff, fit_beta
from nemos.inverse_link_function_utils import exp

from pgam_jax._laplace_reml import laplace_reml, laplace_reml_compute_factory
from pgam_jax._penalty_handler import PenaltyHandler

DATA_DIR = Path(__file__).parent / "data"

_OBS_MODELS = {
    "Poisson": nmo_obs.PoissonObservations,
    "Gamma": nmo_obs.GammaObservations,
}


def _laplace_reml_kwargs(prob):
    """Common kwargs threading the PenaltyHandler callables and rho tree."""
    return dict(
        beta_hat=prob["beta_hat"], X=prob["X"], y=prob["y"],
        obs_model=prob["obs"], inverse_link_fn=prob["inv_link"],
        S_all=prob["S_all"], rhos_tree=prob["rhos_tree"],
        phi=prob["phi"], M_null=prob["M_null"],
        compute_sqrt=prob["compute_sqrt"],
        compute_log_det_and_grad=prob["compute_log_det_and_grad"],
    )


def _reml_value_only(prob):
    return float(laplace_reml(**_laplace_reml_kwargs(prob), return_grad=False))


class TestLaplaceRemlValue:
    """Sanity checks on the REML value."""

    def test_value_is_finite_poisson(self, poisson_gam_problem):
        val = _reml_value_only(poisson_gam_problem)
        assert np.isfinite(val), f"REML value not finite: {val}"

    def test_value_is_finite_gamma(self, gamma_gam_problem):
        val = _reml_value_only(gamma_gam_problem)
        assert np.isfinite(val), f"REML value not finite: {val}"

    def test_value_matches_value_and_grad_poisson(self, poisson_gam_problem):
        """laplace_reml(return_grad=False) == laplace_reml(return_grad=True)[0]."""
        val_only = _reml_value_only(poisson_gam_problem)
        val, _ = laplace_reml(**_laplace_reml_kwargs(poisson_gam_problem),
                              return_grad=True)
        np.testing.assert_allclose(float(val), val_only, rtol=1e-12)


class TestLaplaceRemlGrad:
    """grad REML wrt rho: FD of REML(rho) at re-optimised beta_hat(rho).

    By the envelope theorem (∂L/∂β = 0 at the MAP), the total derivative
    d/drho L(rho, β̂(rho)) equals the partial ∂L/∂rho evaluated at fixed β̂.
    So FD of REML re-optimised at perturbed rho equals our analytical gradient.
    """

    def _fd_check(self, prob):
        rho = np.asarray(prob["rho"])
        X, y, obs, inv_link = prob["X"], prob["y"], prob["obs"], prob["inv_link"]
        S_all, beta_hat, phi, M_null = (
            prob["S_all"], prob["beta_hat"], prob["phi"], prob["M_null"],
        )
        compute_sqrt = prob["compute_sqrt"]
        compute_ld = prob["compute_log_det_and_grad"]

        _, grad = laplace_reml(**_laplace_reml_kwargs(prob), return_grad=True)

        def reml_at_rho(r):
            r_j = jnp.asarray(r)
            rhos_tree = [r_j[k:k + 1] for k in range(r_j.size)]
            b = fit_beta(X, y, obs, inv_link, S_all, r_j, phi, np.asarray(beta_hat))
            return float(laplace_reml(
                b, X, y, obs, inv_link, S_all, rhos_tree, phi, M_null,
                compute_sqrt, compute_ld, return_grad=False,
            ))

        grad_fd = central_diff(reml_at_rho, rho, rel_step=1e-4)

        np.testing.assert_allclose(np.asarray(grad), grad_fd, rtol=1e-3, atol=1e-4)

    def test_fd_poisson(self, poisson_gam_problem):
        self._fd_check(poisson_gam_problem)

    def test_fd_gamma(self, gamma_gam_problem):
        self._fd_check(gamma_gam_problem)


class TestLaplaceRemlComputeFactory:
    """The custom_vjp objective: jax.value_and_grad must equal the analytical pair.

    Confirms that any optimiser calling jax.value_and_grad on the factory
    objective gets exactly laplace_reml's analytical (value, gradient).
    """

    def _check(self, prob):
        objective = laplace_reml_compute_factory(
            prob["obs"], prob["inv_link"], prob["phi"], prob["M_null"],
            prob["compute_sqrt"], prob["compute_log_det_and_grad"],
            prob["rhos_tree"],
        )
        value_ad, grad_ad = jax.value_and_grad(objective)(
            prob["rhos_tree"], prob["beta_hat"], prob["X"], prob["y"], prob["S_all"]
        )
        value_an, grad_an = laplace_reml(**_laplace_reml_kwargs(prob), return_grad=True)

        np.testing.assert_allclose(float(value_ad), float(value_an), rtol=1e-12)
        # grad_ad is a pytree matching rhos_tree; grad_an is the flat (M,) vector
        grad_ad_flat = np.concatenate([np.atleast_1d(np.asarray(g)) for g in grad_ad])
        np.testing.assert_allclose(grad_ad_flat, np.asarray(grad_an), rtol=1e-12)

    def test_poisson(self, poisson_gam_problem):
        self._check(poisson_gam_problem)

    def test_gamma(self, gamma_gam_problem):
        self._check(gamma_gam_problem)


# ─── Regression vs numpy PGAM reml_objective ─────────────────────────────────


def _load_pgam_fixture(name):
    path = DATA_DIR / f"laplace_reml_{name}.json"
    if not path.exists():
        pytest.skip(f"Fixture {path.name} not generated; run "
                    "_script/generate_laplace_reml_fixtures.py")
    return json.loads(path.read_text())


class TestLaplaceRemlRegression:
    """laplace_reml at PGAM's beta_hat must match PGAM's reml_objective value+grad.

    Fixtures produced by _script/generate_laplace_reml_fixtures.py using
    PGAM's reml_objective(..., return_type="eval_grad", null_dim=M_null).
    Tolerance is tight because both implementations use the same beta_hat —
    no FD, no re-optimisation; only the analytical formulae differ.
    """

    @pytest.mark.parametrize("family_file", ["poisson", "gamma"])
    def test_matches_pgam(self, family_file):
        data = _load_pgam_fixture(family_file)

        X = jnp.asarray(data["X"])
        y = jnp.asarray(data["y"])
        S_all_np = np.asarray(data["S_all"])           # (M, P, P)
        S_all = jnp.asarray(S_all_np)
        rho = jnp.asarray(data["rho"])
        beta_hat = jnp.asarray(data["beta_hat"])
        phi = float(data["phi"])
        M_null = int(data["M_null"])
        obs = _OBS_MODELS[data["family"]]()

        # Each S_all[k] is sparse in its smooth's coef block; extract the small
        # block (drop the zero intercept row/col + zero rows/cols of other smooths)
        # and feed to PenaltyHandler so it picks the stable per-block sqrt path.
        ph = PenaltyHandler(non_linearity=jnp.exp)
        for k in range(S_all_np.shape[0]):
            S = S_all_np[k]
            keep = np.any(S != 0, axis=0) | np.any(S != 0, axis=1)
            small = S[np.ix_(keep, keep)]
            ph.add(jnp.asarray(small), penalize_null_space=False)
        compute_sqrt, compute_ld = ph.build()
        rhos_tree = [rho[k:k + 1] for k in range(rho.size)]

        value, grad = laplace_reml(
            beta_hat, X, y, obs, exp, S_all, rhos_tree, phi, M_null,
            compute_sqrt, compute_ld, return_grad=True,
        )

        # Value bit-tight; gradient looser because PGAM computes IRLS weights
        # via closed-form and JAX via autodiff over nemos' log_likelihood —
        # same quantity, different FP operation order → ~1e-9 accumulated noise.
        np.testing.assert_allclose(float(value), data["reml_value"], rtol=1e-10)
        np.testing.assert_allclose(np.asarray(grad), data["reml_grad"],
                                   rtol=1e-7, atol=1e-9)
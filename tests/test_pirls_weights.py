"""Tests for _pirls_weights.py — Component A of Laplace-REML.

float64 is enabled globally in conftest.py.

Validation strategy
-------------------
1. Statsmodels analytical ground truth for w, dw/dmu, small_h.
   All four nemos families use canonical links so alpha = 1 and the
   formula for w collapses to 1 / (g'(μ)² V(μ)).  statsmodels provides
   g', g'', V, V' — sufficient for w, dw/dmu, and small_h.

2. scipy.optimize.check_grad for gradient relationships:
     sum w(η)       gradient wrt η  →  small_h   (h = dw/dη)
     sum h(η)       gradient wrt η  →  deriv_small_h · (dμ/dη)

   d²w/dμ² follows from the quotient-rule combination of already-validated
   terms (dw/deta, d2w/deta2, dmu/deta, d2mu/deta2), so no separate check
   is needed beyond the check_grad tests above.

3. Closed-form ground truths for Poisson and Gaussian.
"""

import numpy as np
import pytest
from scipy.optimize import check_grad
import statsmodels.genmod.families as smf

import jax.numpy as jnp
import nemos.observation_models as nmo_obs
from nemos.inverse_link_function_utils import exp, identity, logistic
from nemos.utils import one_over_x

from pgam_jax._pirls_weights import (
    d2w_dmu2,
    deriv_small_h,
    dw_dmu,
    pirls_weight,
    small_h,
)
from pgam_jax._utils import elementwise_derivative

# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------

N = 30
rng = np.random.default_rng(0)

# (nemos obs_model, nemos inv_link, statsmodels family, eta_np, y_np)
FAMILIES = {
    "poisson": (
        nmo_obs.PoissonObservations(),
        exp,
        smf.Poisson(),
        rng.uniform(0.2, 2.0, N),
        rng.poisson(1.5, N).astype(float),
    ),
    "gaussian": (
        nmo_obs.GaussianObservations(),
        identity,
        smf.Gaussian(),
        rng.uniform(-2.0, 2.0, N),
        rng.normal(0.0, 1.0, N),
    ),
    "gamma": (
        nmo_obs.GammaObservations(),
        one_over_x,
        smf.Gamma(),            # default: reciprocal link — matches nemos
        rng.uniform(0.5, 3.0, N),
        rng.gamma(2.0, 0.5, N),
    ),
    "bernoulli": (
        nmo_obs.BernoulliObservations(),
        logistic,
        smf.Binomial(),         # default: logit link — matches nemos
        rng.uniform(-2.0, 2.0, N),
        rng.integers(0, 2, N).astype(float),
    ),
}


def _jax_inputs(name):
    obs, inv_link, sm_fam, eta_np, y_np = FAMILIES[name]
    return obs, inv_link, sm_fam, jnp.array(eta_np), jnp.array(y_np)


# ---------------------------------------------------------------------------
# Statsmodels analytical references (canonical link ⇒ alpha = 1)
# ---------------------------------------------------------------------------

def _sm_w(mu, sm_fam):
    gp = sm_fam.link.deriv(mu)
    return 1.0 / (gp ** 2 * sm_fam.variance(mu))


def _sm_dw_dmu(mu, sm_fam):
    gp  = sm_fam.link.deriv(mu)
    gpp = sm_fam.link.deriv2(mu)
    V   = sm_fam.variance(mu)
    Vp  = sm_fam.variance.deriv(mu)
    return -(2 * gpp * V + Vp * gp) / (gp ** 3 * V ** 2)


def _sm_small_h(mu, sm_fam):
    return _sm_dw_dmu(mu, sm_fam) / sm_fam.link.deriv(mu)


# ---------------------------------------------------------------------------
# 1. Statsmodels analytical checks — w, dw/dmu, small_h
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["poisson", "gaussian", "gamma", "bernoulli"])
def test_w_vs_statsmodels(family):
    obs, inv_link, sm_fam, eta_vec, y_vec = _jax_inputs(family)
    mu = np.array(inv_link(eta_vec))
    np.testing.assert_allclose(
        pirls_weight(eta_vec, y_vec, obs, inv_link), _sm_w(mu, sm_fam), rtol=1e-10)


@pytest.mark.parametrize("family", ["poisson", "gaussian", "gamma", "bernoulli"])
def test_dw_dmu_vs_statsmodels(family):
    obs, inv_link, sm_fam, eta_vec, y_vec = _jax_inputs(family)
    mu = np.array(inv_link(eta_vec))
    np.testing.assert_allclose(
        dw_dmu(eta_vec, y_vec, obs, inv_link), _sm_dw_dmu(mu, sm_fam), rtol=1e-10)


@pytest.mark.parametrize("family", ["poisson", "gaussian", "gamma", "bernoulli"])
def test_small_h_vs_statsmodels(family):
    obs, inv_link, sm_fam, eta_vec, y_vec = _jax_inputs(family)
    mu = np.array(inv_link(eta_vec))
    np.testing.assert_allclose(
        small_h(eta_vec, y_vec, obs, inv_link), _sm_small_h(mu, sm_fam), rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. check_grad: gradient relationships in η-space
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["poisson", "gaussian", "gamma", "bernoulli"])
def test_small_h_is_dw_deta(family):
    """small_h = dw/dη: gradient of Σ w(η) wrt η."""
    obs, inv_link, _sm, eta_vec, y_vec = _jax_inputs(family)
    assert check_grad(
        lambda eta: jnp.sum(pirls_weight(eta, y_vec, obs, inv_link)),
        lambda eta: small_h(eta, y_vec, obs, inv_link),
        np.array(eta_vec),
    ) < 1e-4


@pytest.mark.parametrize("family", ["poisson", "gaussian", "gamma", "bernoulli"])
def test_deriv_small_h_is_d2w_deta2_over_dmu_deta(family):
    """deriv_small_h · (dμ/dη) = d²w/dη²: gradient of Σ h(η) wrt η."""
    obs, inv_link, _sm, eta_vec, y_vec = _jax_inputs(family)
    assert check_grad(
        lambda eta: jnp.sum(small_h(eta, y_vec, obs, inv_link)),
        lambda eta: deriv_small_h(eta, y_vec, obs, inv_link) * elementwise_derivative(inv_link)(eta),
        np.array(eta_vec),
    ) < 1e-4


# ---------------------------------------------------------------------------
# 3. Closed-form ground truths — Poisson and Gaussian
# ---------------------------------------------------------------------------

def test_poisson_analytical():
    obs, inv_link, _sm, eta_vec, y_vec = _jax_inputs("poisson")
    mu = np.array(inv_link(eta_vec))

    np.testing.assert_allclose(pirls_weight(eta_vec, y_vec, obs, inv_link),  mu,          rtol=1e-10)
    np.testing.assert_allclose(dw_dmu(eta_vec, y_vec, obs, inv_link),        np.ones(N),  atol=1e-10)
    np.testing.assert_allclose(d2w_dmu2(eta_vec, y_vec, obs, inv_link),      np.zeros(N), atol=1e-10)
    np.testing.assert_allclose(small_h(eta_vec, y_vec, obs, inv_link),       mu,          rtol=1e-10)
    np.testing.assert_allclose(deriv_small_h(eta_vec, y_vec, obs, inv_link), np.ones(N),  atol=1e-10)


def test_gaussian_analytical():
    obs, inv_link, _sm, eta_vec, y_vec = _jax_inputs("gaussian")

    np.testing.assert_allclose(pirls_weight(eta_vec, y_vec, obs, inv_link),  np.ones(N),  rtol=1e-10)
    np.testing.assert_allclose(dw_dmu(eta_vec, y_vec, obs, inv_link),        np.zeros(N), atol=1e-10)
    np.testing.assert_allclose(d2w_dmu2(eta_vec, y_vec, obs, inv_link),      np.zeros(N), atol=1e-10)
    np.testing.assert_allclose(small_h(eta_vec, y_vec, obs, inv_link),       np.zeros(N), atol=1e-10)
    np.testing.assert_allclose(deriv_small_h(eta_vec, y_vec, obs, inv_link), np.zeros(N), atol=1e-10)
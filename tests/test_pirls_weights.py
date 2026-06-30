"""Tests for _pirls_weights.py — Component A of Laplace-REML.

float64 is enabled globally in conftest.py.

The implementation computes w = −d²ℓ/dη² (observed Hessian), which is the
correct weight for Laplace-REML.  For canonical links g''V + g'V' = 0, so
this reduces to the Fisher information 1/(g'²V).  For non-canonical links a
y-dependent correction survives:

    w = 1/(g'²V) + (y−μ)(g''V + g'V')/(g'³V²)

Validation strategy
-------------------
1a. Statsmodels analytical ground truth for w — ALL link functions.
    The formula above is computable from statsmodels link/variance objects.

1b. Statsmodels analytical ground truth for dw/dμ and small_h —
    CANONICAL links only.  The correction's μ-derivatives require g''' and
    V'' which statsmodels does not expose; for non-canonical links the
    check_grad tests in section 2 provide gradient-level validation instead.

2.  scipy.optimize.check_grad for gradient relationships — ALL link functions:
      Σ w(η)     gradient wrt η  →  small_h       (h = dw/dη)
      Σ h(η)     gradient wrt η  →  deriv_small_h · (dμ/dη)

3.  Closed-form ground truths for Poisson (log link) and Gaussian (identity).
"""

import jax.numpy as jnp
import nemos.observation_models as nmo_obs
import numpy as np
import pytest
import statsmodels.genmod.families as smf
import statsmodels.genmod.families.links as sm_links
from conftest import central_diff
from nemos.inverse_link_function_utils import exp, identity, logistic
from nemos.utils import one_over_x

from pgam_jax._pirls_weights import (
    _make_w_fn,
    d2w_dmu2,
    deriv_small_h,
    dw_dmu,
    small_h,
)
from pgam_jax._utils import elementwise_derivative

N = 30

# ---------------------------------------------------------------------------
# Family/link registry
#
# Each raw tuple: (obs_model, inv_link, sm_fam, eta_lo, eta_hi, y_fn, seed)
# eta_lo/hi: uniform draw range for η, chosen so μ = inv_link(η) stays in
#            the family's natural domain.
# y_fn(rng): draws a response vector of length N.
# seed: fixed per case → tests are deterministic and xdist-safe.
# ---------------------------------------------------------------------------


def _poisson_y(rng):
    return rng.poisson(1.5, N).astype(float)


def _gaussian_y(rng):
    return rng.normal(0.0, 1.0, N)


def _gamma_y(rng):
    return rng.gamma(2.0, 0.5, N)


def _bernoulli_y(rng):
    return rng.integers(0, 2, N).astype(float)


# Raw tuples — defined here so analytical tests can reference them directly
# without duplicating the case definitions.
_POISSON_LOG = (
    nmo_obs.PoissonObservations(),
    exp,
    smf.Poisson(),
    0.2,
    2.0,
    _poisson_y,
    0,
)
_POISSON_ID = (
    nmo_obs.PoissonObservations(),
    identity,
    smf.Poisson(link=sm_links.Identity()),
    0.8,
    2.0,
    _poisson_y,
    1,
)  # eta_lo=0.8 keeps check_grad gradients tractable
_GAUSSIAN_ID = (
    nmo_obs.GaussianObservations(),
    identity,
    smf.Gaussian(),
    -2.0,
    2.0,
    _gaussian_y,
    2,
)
_GAUSSIAN_LOG = (
    nmo_obs.GaussianObservations(),
    exp,
    smf.Gaussian(link=sm_links.Log()),
    -2.0,
    2.0,
    _gaussian_y,
    3,
)
_GAMMA_RECIPROCAL = (
    nmo_obs.GammaObservations(),
    one_over_x,
    smf.Gamma(),
    0.5,
    3.0,
    _gamma_y,
    4,
)
_GAMMA_LOG = (
    nmo_obs.GammaObservations(),
    exp,
    smf.Gamma(link=sm_links.Log()),
    0.5,
    3.0,
    _gamma_y,
    5,
)
_BERNOULLI_LOGIT = (
    nmo_obs.BernoulliObservations(),
    logistic,
    smf.Binomial(),
    -2.0,
    2.0,
    _bernoulli_y,
    6,
)

# Canonical links satisfy g''V + g'V' = 0, so the observed Hessian equals
# the Fisher information.  Non-canonical links add a y-dependent correction.
CANONICAL_CASES = [
    pytest.param(_POISSON_LOG, id="poisson-log"),
    pytest.param(_GAUSSIAN_ID, id="gaussian-identity"),
    pytest.param(_GAMMA_RECIPROCAL, id="gamma-reciprocal"),
    pytest.param(_BERNOULLI_LOGIT, id="bernoulli-logit"),
]

ALL_CASES = CANONICAL_CASES + [
    pytest.param(_POISSON_ID, id="poisson-identity"),
    pytest.param(_GAUSSIAN_LOG, id="gaussian-log"),
    pytest.param(_GAMMA_LOG, id="gamma-log"),
]


def _unpack(case):
    """Return (obs, inv_link, sm_fam, eta_jax, y_jax) with per-case rng."""
    obs, inv_link, sm_fam, eta_lo, eta_hi, y_fn, seed = case
    rng = np.random.default_rng(seed)
    eta = jnp.array(rng.uniform(eta_lo, eta_hi, N))
    y = jnp.array(y_fn(rng))
    return obs, inv_link, sm_fam, eta, y


# ---------------------------------------------------------------------------
# Statsmodels analytical references
# ---------------------------------------------------------------------------


def _sm_w(mu, y, sm_fam):
    """w = −d²ℓ/dη² = Fisher information + non-canonical correction.

    For canonical links g''V + g'V' = 0 and the correction vanishes.
    """
    gp = sm_fam.link.deriv(mu)
    gpp = sm_fam.link.deriv2(mu)
    V = sm_fam.variance(mu)
    Vp = sm_fam.variance.deriv(mu)
    fisher = 1.0 / (gp**2 * V)
    correction = (y - mu) * (gpp * V + gp * Vp) / (gp**3 * V**2)
    return fisher + correction


def _sm_dw_dmu(mu, sm_fam):
    """d(Fisher)/dμ — valid only for canonical links where dw/dμ = d(Fisher)/dμ."""
    gp = sm_fam.link.deriv(mu)
    gpp = sm_fam.link.deriv2(mu)
    V = sm_fam.variance(mu)
    Vp = sm_fam.variance.deriv(mu)
    return -(2 * gpp * V + Vp * gp) / (gp**3 * V**2)


def _sm_small_h(mu, sm_fam):
    """dw/dη = (dw/dμ) / g′(μ) — valid only for canonical links."""
    return _sm_dw_dmu(mu, sm_fam) / sm_fam.link.deriv(mu)


# ---------------------------------------------------------------------------
# 1a. Analytical reference for w — ALL link functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASES)
def test_w_vs_statsmodels(case):
    obs, inv_link, sm_fam, eta_vec, y_vec = _unpack(case)
    mu = np.array(inv_link(eta_vec))
    y = np.array(y_vec)
    w_fn = _make_w_fn(y_vec, obs, inv_link)
    np.testing.assert_allclose(w_fn(eta_vec), _sm_w(mu, y, sm_fam), rtol=1e-10)


# ---------------------------------------------------------------------------
# 1b. Analytical references for dw/dμ and small_h — CANONICAL links only
#     (the correction term's μ-derivative requires g''' and V'' which
#      statsmodels does not expose; non-canonical links are covered by §2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CANONICAL_CASES)
def test_dw_dmu_vs_statsmodels(case):
    obs, inv_link, sm_fam, eta_vec, y_vec = _unpack(case)
    mu = np.array(inv_link(eta_vec))
    np.testing.assert_allclose(
        dw_dmu(eta_vec, y_vec, obs, inv_link), _sm_dw_dmu(mu, sm_fam), rtol=1e-10
    )


@pytest.mark.parametrize("case", CANONICAL_CASES)
def test_small_h_vs_statsmodels(case):
    obs, inv_link, sm_fam, eta_vec, y_vec = _unpack(case)
    mu = np.array(inv_link(eta_vec))
    np.testing.assert_allclose(
        small_h(eta_vec, y_vec, obs, inv_link), _sm_small_h(mu, sm_fam), rtol=1e-10
    )


# ---------------------------------------------------------------------------
# 2. check_grad: gradient relationships in η-space — ALL link functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ALL_CASES)
def test_small_h_is_dw_deta(case):
    """small_h = dw/dη: gradient of Σ w(η) wrt η."""
    obs, inv_link, _sm, eta_vec, y_vec = _unpack(case)
    w_fn = _make_w_fn(y_vec, obs, inv_link)
    grad_fd = central_diff(lambda eta: jnp.sum(w_fn(eta)), eta_vec)
    np.testing.assert_allclose(
        grad_fd, small_h(eta_vec, y_vec, obs, inv_link), atol=1e-7
    )


@pytest.mark.parametrize("case", ALL_CASES)
def test_deriv_small_h_is_d2w_deta2_over_dmu_deta(case):
    """deriv_small_h · (dμ/dη) = d²w/dη²: gradient of Σ h(η) wrt η."""
    obs, inv_link, _sm, eta_vec, y_vec = _unpack(case)
    grad_fd = central_diff(
        lambda eta: jnp.sum(small_h(eta, y_vec, obs, inv_link)), eta_vec
    )
    np.testing.assert_allclose(
        grad_fd,
        deriv_small_h(eta_vec, y_vec, obs, inv_link)
        * elementwise_derivative(inv_link)(eta_vec),
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# 3. Closed-form ground truths — Poisson (log link) and Gaussian (identity)
# ---------------------------------------------------------------------------


def test_poisson_log_analytical():
    obs, inv_link, _sm, eta_vec, y_vec = _unpack(_POISSON_LOG)
    mu = np.array(inv_link(eta_vec))
    w_fn = _make_w_fn(y_vec, obs, inv_link)

    np.testing.assert_allclose(w_fn(eta_vec), mu, rtol=1e-10)
    np.testing.assert_allclose(
        dw_dmu(eta_vec, y_vec, obs, inv_link), np.ones(N), atol=1e-10
    )
    np.testing.assert_allclose(
        d2w_dmu2(eta_vec, y_vec, obs, inv_link), np.zeros(N), atol=1e-10
    )
    np.testing.assert_allclose(small_h(eta_vec, y_vec, obs, inv_link), mu, rtol=1e-10)
    np.testing.assert_allclose(
        deriv_small_h(eta_vec, y_vec, obs, inv_link), np.ones(N), atol=1e-10
    )


def test_gaussian_identity_analytical():
    obs, inv_link, _sm, eta_vec, y_vec = _unpack(_GAUSSIAN_ID)
    w_fn = _make_w_fn(y_vec, obs, inv_link)

    np.testing.assert_allclose(w_fn(eta_vec), np.ones(N), rtol=1e-10)
    np.testing.assert_allclose(
        dw_dmu(eta_vec, y_vec, obs, inv_link), np.zeros(N), atol=1e-10
    )
    np.testing.assert_allclose(
        d2w_dmu2(eta_vec, y_vec, obs, inv_link), np.zeros(N), atol=1e-10
    )
    np.testing.assert_allclose(
        small_h(eta_vec, y_vec, obs, inv_link), np.zeros(N), atol=1e-10
    )
    np.testing.assert_allclose(
        deriv_small_h(eta_vec, y_vec, obs, inv_link), np.zeros(N), atol=1e-10
    )

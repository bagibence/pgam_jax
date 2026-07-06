"""Tests for _laplace_reml_vbeta.py — Component B of Laplace-REML.

float64 is enabled globally in conftest.py.
"""

import jax.numpy as jnp
import nemos.observation_models as nmo_obs
import numpy as np
import pytest
from nemos.glm.initialize_parameters import INVERSE_FUNCS
from nemos.inverse_link_function_utils import exp

from pgam_jax._laplace_reml_vbeta import vbeta_and_logdet
from pgam_jax.iterative_optim import model_constructors_for_weights_and_pseudo_data

N, P = 30, 5

_OBS = nmo_obs.PoissonObservations()
_VARIANCE_FUNC = lambda mu: mu  # Poisson: V(μ) = μ
_LINK_FUNC = INVERSE_FUNCS[exp]  # forward link: log


def _make_R(beta_hat, X, y, fisher_scoring=False):
    """Compute R = QR(sqrt(W) X) using model_constructors."""
    get_pw = model_constructors_for_weights_and_pseudo_data(
        _VARIANCE_FUNC, _LINK_FUNC, fisher_scoring=fisher_scoring
    )
    mu = exp(X @ beta_hat)
    _, w = get_pw(y, mu)
    _, R = jnp.linalg.qr(jnp.sqrt(w)[:, None] * X)
    return R, w


def _make_problem(rho_scale=1.0, phi=1.0, seed=0, full_rank_penalty=False):
    rng = np.random.default_rng(seed)
    X = jnp.array(rng.standard_normal((N, P)))
    beta_hat = jnp.array(rng.standard_normal(P) * 0.3)
    y = jnp.array(rng.poisson(np.exp(np.array(X @ beta_hat))).astype(float))
    if full_rank_penalty:
        sqrt_penalty = jnp.array(rho_scale * np.eye(P))
    else:
        B = np.zeros((P - 1, P))
        for i in range(P - 1):
            B[i, i] = 1.0
            B[i, i + 1] = -1.0
        sqrt_penalty = jnp.array(B * rho_scale)
    return beta_hat, X, y, sqrt_penalty, phi


def test_vbeta_is_inverse_of_vbeta_inv():
    """V_beta @ V_beta_inv ≈ I (atol 1e-10)."""
    beta_hat, X, y, sqrt_penalty, phi = _make_problem()
    R, _ = _make_R(beta_hat, X, y)
    V_beta, V_beta_inv, _ = vbeta_and_logdet(R, sqrt_penalty, phi)
    np.testing.assert_allclose(V_beta @ V_beta_inv, np.eye(P), atol=1e-10)


def test_logdet_matches_slogdet():
    """log_det_HpS matches slogdet(H + S_lam)[1] for phi=1, full-rank penalty."""
    beta_hat, X, y, sqrt_penalty, phi = _make_problem(full_rank_penalty=True)
    R, w = _make_R(beta_hat, X, y)
    _, _, log_det_HpS = vbeta_and_logdet(R, sqrt_penalty, phi)
    H_plus_S = X.T @ (w[:, None] * X) + sqrt_penalty.T @ sqrt_penalty
    _, logdet_direct = jnp.linalg.slogdet(H_plus_S)
    np.testing.assert_allclose(float(log_det_HpS), float(logdet_direct), rtol=1e-10)


@pytest.mark.parametrize("rho_scale", [1e-6, 1.0, 1e6])
def test_finite_for_extreme_rho(rho_scale):
    """No NaN or Inf in outputs for extreme penalty magnitudes."""
    beta_hat, X, y, sqrt_penalty, phi = _make_problem(rho_scale=rho_scale)
    R, _ = _make_R(beta_hat, X, y)
    V_beta, V_beta_inv, log_det_HpS = vbeta_and_logdet(R, sqrt_penalty, phi)
    assert jnp.all(jnp.isfinite(V_beta))
    assert jnp.all(jnp.isfinite(V_beta_inv))
    assert jnp.isfinite(log_det_HpS)

"""Observation-model weight derivatives for Laplace-REML (Component A).

All quantities are derived from the per-observation log-likelihood
    log f(y; η)  where  η = X @ β  (linear predictor)
via forward-mode AD (JVP) applied to elementwise functions.  No vmap is used:
every function here maps arrays to arrays, and each output[i] depends only on
the corresponding input[i], so jax.jvp with a ones-tangent extracts the full
diagonal Jacobian in a single pass.

Public functions accept:
    eta_vec         : shape (n,), linear predictor X @ beta
    y_vec           : shape (n,), observations
    obs_model       : nemos Observations instance (for log_likelihood only)
    inverse_link_fn : callable g⁻¹: η → μ, from glm.inverse_link_function

The basic IRLS weight w = -d²logf/dη² is NOT exposed here; use
model_constructors_for_weights_and_pseudo_data from iterative_optim instead.

mu-space derivatives are recovered via the chain rule:
    dw/dμ   = (dw/dη) / (dμ/dη)
    d²w/dμ² = (d²w/dη² · dμ/dη − dw/dη · d²μ/dη²) / (dμ/dη)³
    h       = (dw/dμ) / g′(μ)  =  dw/dη          [g′ factors cancel]
    dh/dμ   = d²w/dη² / (dμ/dη)
"""

from typing import Callable

import jax.numpy as jnp

from ._utils import elementwise_derivative


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_lik_vec(
    eta_vec: jnp.ndarray,
    y_vec: jnp.ndarray,
    obs_model,
    inverse_link_fn: Callable,
) -> jnp.ndarray:
    """Per-observation log-likelihoods log f(y_i; η_i), shape (n,)."""
    mu_vec = inverse_link_fn(eta_vec)
    return obs_model.log_likelihood(y_vec, mu_vec, aggregate_sample_scores=lambda x: x)


def _make_w_fn(y_vec, obs_model, inverse_link_fn):
    """Return w(η) = -d²logf/dη² as a callable, for internal differentiation."""
    nll = lambda eta: -_log_lik_vec(eta, y_vec, obs_model, inverse_link_fn)
    return elementwise_derivative(elementwise_derivative(nll))


# ---------------------------------------------------------------------------
# Public API — higher-order weight derivatives consumed by Components D, F, G
# ---------------------------------------------------------------------------

def dw_dmu(
    eta_vec: jnp.ndarray,
    y_vec: jnp.ndarray,
    obs_model,
    inverse_link_fn: Callable,
) -> jnp.ndarray:
    """dw/dμ evaluated at μ = g⁻¹(η), shape (n,)."""
    w_fn = _make_w_fn(y_vec, obs_model, inverse_link_fn)
    dw_deta  = elementwise_derivative(w_fn)(eta_vec)
    dmu_deta = elementwise_derivative(inverse_link_fn)(eta_vec)
    return dw_deta / dmu_deta


def d2w_dmu2(
    eta_vec: jnp.ndarray,
    y_vec: jnp.ndarray,
    obs_model,
    inverse_link_fn: Callable,
) -> jnp.ndarray:
    """d²w/dμ² evaluated at μ = g⁻¹(η), shape (n,)."""
    w_fn  = _make_w_fn(y_vec, obs_model, inverse_link_fn)
    dw_fn = elementwise_derivative(w_fn)
    dw_deta    = dw_fn(eta_vec)
    d2w_deta2  = elementwise_derivative(dw_fn)(eta_vec)
    dmu_deta   = elementwise_derivative(inverse_link_fn)(eta_vec)
    d2mu_deta2 = elementwise_derivative(elementwise_derivative(inverse_link_fn))(eta_vec)
    return (d2w_deta2 * dmu_deta - dw_deta * d2mu_deta2) / dmu_deta ** 3


def small_h(
    eta_vec: jnp.ndarray,
    y_vec: jnp.ndarray,
    obs_model,
    inverse_link_fn: Callable,
) -> jnp.ndarray:
    """h = (dw/dμ) / g′(μ) = dw/dη, shape (n,)."""
    w_fn = _make_w_fn(y_vec, obs_model, inverse_link_fn)
    return elementwise_derivative(w_fn)(eta_vec)


def deriv_small_h(
    eta_vec: jnp.ndarray,
    y_vec: jnp.ndarray,
    obs_model,
    inverse_link_fn: Callable,
) -> jnp.ndarray:
    """dh/dμ = d²w/dη² / (dμ/dη), shape (n,)."""
    w_fn  = _make_w_fn(y_vec, obs_model, inverse_link_fn)
    dw_fn = elementwise_derivative(w_fn)
    d2w_deta2 = elementwise_derivative(dw_fn)(eta_vec)
    dmu_deta  = elementwise_derivative(inverse_link_fn)(eta_vec)
    return d2w_deta2 / dmu_deta
"""PIRLS weight functions for Laplace-REML.

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

The forward link is never needed.  mu-space derivatives are recovered via the
chain rule:
    dw/dμ   = (dw/dη) / (dμ/dη)
    d²w/dμ² = (d²w/dη² · dμ/dη − dw/dη · d²μ/dη²) / (dμ/dη)³
    h       = (dw/dμ) / g′(μ)  =  dw/dη          [g′ factors cancel]
    dh/dμ   = d²w/dη² / (dμ/dη)
"""

from typing import Callable

import jax.numpy as jnp

from ._utils import elementwise_derivative


# ---------------------------------------------------------------------------
# Batched log-likelihood in η-space
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


# ---------------------------------------------------------------------------
# Public API — all return shape (n,)
# ---------------------------------------------------------------------------

def pirls_weight(
    eta_vec: jnp.ndarray,
    y_vec: jnp.ndarray,
    obs_model,
    inverse_link_fn: Callable,
) -> jnp.ndarray:
    """PIRLS weights w = −d²logf/dη², shape (n,)."""
    nll = lambda eta: -_log_lik_vec(eta, y_vec, obs_model, inverse_link_fn)
    return elementwise_derivative(elementwise_derivative(nll))(eta_vec)


def dw_dmu(
    eta_vec: jnp.ndarray,
    y_vec: jnp.ndarray,
    obs_model,
    inverse_link_fn: Callable,
) -> jnp.ndarray:
    """dw/dμ evaluated at μ = g⁻¹(η), shape (n,)."""
    nll   = lambda eta: -_log_lik_vec(eta, y_vec, obs_model, inverse_link_fn)
    w_fn  = elementwise_derivative(elementwise_derivative(nll))
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
    nll    = lambda eta: -_log_lik_vec(eta, y_vec, obs_model, inverse_link_fn)
    w_fn   = elementwise_derivative(elementwise_derivative(nll))
    dw_fn  = elementwise_derivative(w_fn)
    dw_deta   = dw_fn(eta_vec)
    d2w_deta2 = elementwise_derivative(dw_fn)(eta_vec)
    dmu_deta  = elementwise_derivative(inverse_link_fn)(eta_vec)
    d2mu_deta2 = elementwise_derivative(
        elementwise_derivative(inverse_link_fn)
    )(eta_vec)
    return (d2w_deta2 * dmu_deta - dw_deta * d2mu_deta2) / dmu_deta ** 3


def small_h(
    eta_vec: jnp.ndarray,
    y_vec: jnp.ndarray,
    obs_model,
    inverse_link_fn: Callable,
) -> jnp.ndarray:
    """h = (dw/dμ) / g′(μ) = dw/dη, shape (n,)."""
    nll  = lambda eta: -_log_lik_vec(eta, y_vec, obs_model, inverse_link_fn)
    w_fn = elementwise_derivative(elementwise_derivative(nll))
    return elementwise_derivative(w_fn)(eta_vec)


def deriv_small_h(
    eta_vec: jnp.ndarray,
    y_vec: jnp.ndarray,
    obs_model,
    inverse_link_fn: Callable,
) -> jnp.ndarray:
    """dh/dμ = d²w/dη² / (dμ/dη), shape (n,)."""
    nll    = lambda eta: -_log_lik_vec(eta, y_vec, obs_model, inverse_link_fn)
    w_fn   = elementwise_derivative(elementwise_derivative(nll))
    dw_fn  = elementwise_derivative(w_fn)
    d2w_deta2 = elementwise_derivative(dw_fn)(eta_vec)
    dmu_deta  = elementwise_derivative(inverse_link_fn)(eta_vec)
    return d2w_deta2 / dmu_deta
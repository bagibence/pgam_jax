"""Derivatives of the MAP estimate and observed Hessian wrt log-smoothing parameters."""

import jax.numpy as jnp

from ._pirls_weights import small_h


def dbeta_hat(
    beta_hat: jnp.ndarray,
    V_beta: jnp.ndarray,
    S_all: jnp.ndarray,
    rho: jnp.ndarray,
    phi: float,
) -> jnp.ndarray:
    """Gradient of the MAP estimate beta_hat wrt log-smoothing parameters rho.

    From implicit differentiation of the penalised score equation at the MAP:

        J[k] = d beta_hat / d rho_k
             = -V_beta @ (lambda_k S_k / phi) @ beta_hat

    where lambda_k = exp(rho_k).

    Parameters
    ----------
    beta_hat : shape (p,)
        MAP coefficient estimate.
    V_beta : shape (p, p)
        Posterior covariance (H + S_lambda/phi)^{-1}, from vbeta_and_logdet.
    S_all : shape (M, p, p)
        Stack of raw penalty matrices, one per smoothing parameter.
    rho : shape (M,)
        Log-smoothing parameters.
    phi :
        Dispersion parameter (positive scalar).

    Returns
    -------
    J : shape (M, p)
        J[k] = d beta_hat / d rho_k.
    """
    lams = jnp.exp(rho)  # (M,)
    P1 = jnp.einsum("kij,j->ki", S_all * lams[:, None, None], beta_hat) / phi  # (M, p)
    return jnp.einsum("ij,kj->ki", -V_beta, P1)  # (M, p)


def dH_drho(
    beta_hat: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    obs_model,
    inverse_link_fn,
    J: jnp.ndarray,
    phi: float,
) -> jnp.ndarray:
    """Gradient of the observed Hessian H = X^T W X / phi wrt log-smoothing parameters.

    Differentiating through the IRLS weight w(eta) and using
    dw_l/drho_k = (dw/deta)_l * (X J[k])_l = h_l * (X J[k])_l gives

        dH/drho_k = (1/phi) * X^T diag(h * (X J[k])) X

    Implementation note
    -------------------
    Expressed as ``jnp.einsum("oi,oj,ok->kij", X, X, weights)``.  In numpy
    opt_einsum refuses to decompose this (no FLOP-reducing pair) and falls
    back to a generic nested-loop kernel — there the hand-fused
    ``X^T @ V`` over a flattened ``(n, M*p)`` is required (see PGAM's
    ``grad_H_chunked_fused`` and ``_scripts/bench_hes_H.py``).  XLA does model
    broadcasts as first-class HLO ops and lowers this einsum to the same
    fused-broadcast-matmul; ``_script/bench_dH_drho.py`` shows einsum at ~30%
    faster than the explicit fused version on CPU at production sizes.
    """
    eta = X @ beta_hat  # (n,)
    h = small_h(eta, y, obs_model, inverse_link_fn)  # (n,)
    XJ = X @ J.T  # (n, M)
    weights = h[:, None] * XJ  # (n, M)
    return jnp.einsum("oi,oj,ok->kij", X, X, weights) / phi  # (M, p, p)

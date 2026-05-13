"""Derivatives of the MAP estimate and observed Hessian wrt log-smoothing parameters."""

import jax.numpy as jnp


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
    lams = jnp.exp(rho)                                                        # (M,)
    P1 = jnp.einsum("kij,j->ki", S_all * lams[:, None, None], beta_hat) / phi  # (M, p)
    return jnp.einsum("ij,kj->ki", -V_beta, P1)                                # (M, p)
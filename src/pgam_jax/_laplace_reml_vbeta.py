"""Posterior covariance V_β and log|H + S_λ| for Laplace-REML.

Mirrors the numpy reference (Vbeta_rho_all in der_wrt_smoothing.py) using the
thin SVD of A = [R ; B] where

    R  = upper-triangular QR factor of sqrt(W) X,   R^T R = X^T W X
    B  = sqrt_penalty,                               B^T B = S_λ    (no phi)

From A = U Σ V^T:   A^T A = V^T^T Σ² V^T = X^T W X + S_λ

so, with H = X^T W X / phi:

    phi * (H + S_λ) = V^T^T Σ² V^T
    V_β             = phi * V^T^T Σ^{-2} V^T   = (H + S_λ)^{-1}
    V_β^{-1}        = V^T^T Σ² V^T / phi        = H + S_λ
    log|H + S_λ|    = 2 Σ_k log σ_k − r log φ   r = rank(H + S_λ)

Singular-value masking mirrors _pql_reml.py: near-zero rows of V_T are zeroed
(JAX-friendly substitute for numpy's row-deletion).

R must be precomputed by the caller using model_constructors_for_weights_and_pseudo_data
from iterative_optim, followed by jnp.linalg.qr(sqrt(W)[:, None] * X, mode="r").
"""

import jax.numpy as jnp

from ._pql_gcv import FLOAT_EPS, _vmap_where


def vbeta_and_logdet(
    R: jnp.ndarray,
    sqrt_penalty: jnp.ndarray,
    phi: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute V_β, V_β^{-1}, and log|H + S_λ| via a single SVD.

    Parameters
    ----------
    R : shape (p, p)
        Upper-triangular QR factor of sqrt(W) X, so R^T R = X^T W X.
        Precomputed by the caller from IRLS weights.
    sqrt_penalty : shape (q, p)
        Square root of S_λ, i.e. B s.t. B^T B = S_λ.  Not scaled by phi.
    phi :
        Dispersion parameter (positive scalar).

    Returns
    -------
    V_beta : shape (p, p)
        Posterior covariance phi * (H + S_λ)^{-1}.
    V_beta_inv : shape (p, p)
        Posterior precision (H + S_λ) / phi.
    log_det_HpS : scalar
        log|H + S_λ|.
    """
    U, s, V_T = jnp.linalg.svd(jnp.vstack([R, sqrt_penalty]), full_matrices=False)
    del U

    low_vals = s < FLOAT_EPS * s.max()
    s = jnp.where(low_vals, 0.0, s)
    V_T = _vmap_where(low_vals, 0, V_T.T).T

    s_safe = jnp.where(low_vals, 1.0, s)
    s_inv = jnp.where(low_vals, 0.0, 1.0 / s_safe)

    V_beta = phi * (V_T.T * s_inv**2) @ V_T
    V_beta_inv = (V_T.T * s**2) @ V_T / phi

    r = jnp.sum(jnp.where(low_vals, 0.0, 1.0))
    log_det_HpS = 2.0 * jnp.sum(
        jnp.where(low_vals, 0.0, jnp.log(s_safe))
    ) - r * jnp.log(phi)

    return V_beta, V_beta_inv, log_det_HpS

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
"""

from typing import Callable

import jax.numpy as jnp

from ._pql_gcv import FLOAT_EPS, _vmap_where
from ._pirls_weights import pirls_weight


def vbeta_and_logdet(
    beta_hat: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    obs_model,
    inverse_link_fn: Callable,
    sqrt_penalty: jnp.ndarray,
    phi: float,
    R: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute V_β, V_β^{-1}, and log|H + S_λ| via a single QR + SVD.

    Parameters
    ----------
    beta_hat : shape (p,)
        MAP estimate of β.
    X : shape (n, p)
        Design matrix.
    y : shape (n,)
        Observations.
    obs_model :
        Nemos observation model.
    inverse_link_fn :
        Inverse link function μ = g^{-1}(η).
    sqrt_penalty : shape (q, p)
        Square root of S_λ, i.e. B s.t. B^T B = S_λ.  Not scaled by phi.
    phi :
        Dispersion parameter (positive scalar).
    R : shape (p, p), optional
        Precomputed upper-triangular QR factor of sqrt(W) X.  Pass to
        avoid recomputing W and QR inside a fit iteration.

    Returns
    -------
    V_beta : shape (p, p)
        Posterior covariance (H + S_λ)^{-1}.
    V_beta_inv : shape (p, p)
        Posterior precision H + S_λ.
    log_det_HpS : scalar
        log|H + S_λ|.
    """
    if R is None:
        w = pirls_weight(X @ beta_hat, y, obs_model, inverse_link_fn)
        _, R = jnp.linalg.qr(jnp.sqrt(w)[:, None] * X)

    U, s, V_T = jnp.linalg.svd(jnp.vstack([R, sqrt_penalty]), full_matrices=False)
    del U

    low_vals = s < FLOAT_EPS * s.max()
    s     = jnp.where(low_vals, 0.0, s)
    V_T   = _vmap_where(low_vals, 0, V_T.T).T

    s_safe = jnp.where(low_vals, 1.0, s)           # substitute 1.0 to guard log / division
    s_inv  = jnp.where(low_vals, 0.0, 1.0 / s_safe)

    # V_beta = phi * V_T^T diag(s^{-2}) V_T
    # V_beta_inv = V_T^T diag(s^2) V_T / phi
    # Compact form: scale rows of V_T then hit with V_T^T on the left.
    V_beta     = phi * (V_T.T * s_inv**2) @ V_T
    V_beta_inv = (V_T.T * s**2) @ V_T / phi

    r           = jnp.sum(jnp.where(low_vals, 0.0, 1.0))
    log_det_HpS = 2.0 * jnp.sum(jnp.where(low_vals, 0.0, jnp.log(s_safe))) - r * jnp.log(phi)

    return V_beta, V_beta_inv, log_det_HpS
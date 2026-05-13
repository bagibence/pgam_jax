"""Laplace-approximated REML objective and its gradient wrt log-smoothing parameters."""

import jax.numpy as jnp

from ._laplace_reml_beta_derivs import dbeta_hat
from ._laplace_reml_vbeta import vbeta_and_logdet
from ._pirls_weights import _make_w_fn, small_h
from ._slam_compute import log_det_and_grad_slam, log_det_slam, transform_slam
from .penalty_utils import symmetric_sqrt


def laplace_reml(
    beta_hat: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    obs_model,
    inverse_link_fn,
    S_all: jnp.ndarray,
    rho: jnp.ndarray,
    phi: float,
    M_null: int,
    return_grad: bool = False,
):
    """Laplace-approximated REML at the MAP estimate.

    Wood (2017) eq. 6.18:

        ℓ(ρ) = l(β̂) - (1/2φ) β̂^T S_λ β̂
               + (1/2φ) log|S_λ|_+
               - (1/2) log|H + S_λ/φ|
               + (M_0/2) log(2π)

    where H = X^T W X / φ is the observed Fisher information at β̂ and M_0 is
    the null-space dimension of S_λ (static — depends only on S_all, not ρ).

    The gradient wrt ρ is the sum of three contributions:

        add1 = -(1/2φ) λ_r β̂^T S_r β̂
        add2 = +(1/2φ) ∂ log|S_λ|_+ / ∂ρ_r
        add3 = -(1/2) tr(V_β (∂H/∂ρ_r + λ_r S_r / φ))

    The add3 trace splits into (J[r]·q)/φ + λ_r tr(V_β S_r)/φ where
    q = X^T (h · diag(X V_β X^T)) and h = dw/dη; the first term is a
    drop-in for `dH_drho` (Component D) but reused via the precomputed J.

    Parameters
    ----------
    beta_hat : shape (p,)
        MAP coefficient estimate at ρ.
    X : shape (n, p)
    y : shape (n,)
    obs_model : nemos observation model
    inverse_link_fn : callable g^{-1}: η → μ
    S_all : shape (M, p, p)
        Stack of raw penalty matrices.
    rho : shape (M,)
        Log-smoothing parameters.
    phi :
        Positive scalar dispersion.
    M_null : int
        Null-space dimension of S_λ (static).
    return_grad : bool
        If True, also return the gradient wrt ρ.

    Returns
    -------
    value : scalar
    grad : shape (M,), only when ``return_grad=True``
    """
    eta = X @ beta_hat
    lams = jnp.exp(rho)
    S_lam = jnp.einsum("kij,k->ij", S_all, lams)

    # log-likelihood
    mu = inverse_link_fn(eta)
    ll = obs_model.log_likelihood(y, mu, aggregate_sample_scores=jnp.sum)

    # penalty: -0.5/phi * beta^T S_lam beta
    pen = -0.5 * beta_hat @ S_lam @ beta_hat / phi

    # log|H + S_lam/phi| via SVD of [R; sqrt(S_lam)]
    w = _make_w_fn(y, obs_model, inverse_link_fn)(eta)
    _, R = jnp.linalg.qr(jnp.sqrt(w)[:, None] * X, mode="reduced")
    sqrt_pen = symmetric_sqrt(S_lam)
    V_beta, _, log_det_HpS = vbeta_and_logdet(R, sqrt_pen, phi)

    # log|S_lam|_+ (and gradient)
    S_i_out = transform_slam(S_all, rho)
    if return_grad:
        log_det_Slam, grad_log_det_Slam = log_det_and_grad_slam(rho, S_i_out)
    else:
        log_det_Slam = log_det_slam(rho, S_i_out)

    value = (
        ll + pen
        + 0.5 * log_det_Slam / phi
        - 0.5 * log_det_HpS
        + 0.5 * M_null * jnp.log(2 * jnp.pi)
    )

    if not return_grad:
        return value

    # ── Gradient ─────────────────────────────────────────────────────────────
    # add1 = -(0.5/phi) * lam_r * beta^T S_r beta
    add1 = -0.5 * jnp.einsum(
        "i,rij,j->r", beta_hat, S_all * lams[:, None, None], beta_hat
    ) / phi

    # add2 = +(0.5/phi) * d log|S_lam|+/d rho
    add2 = 0.5 * grad_log_det_Slam / phi

    # add3 = -0.5 * tr(V_beta * (dH/drho_r + lam_r * S_r / phi))
    J = dbeta_hat(beta_hat, V_beta, S_all, rho, phi)              # (M, p)
    h = small_h(eta, y, obs_model, inverse_link_fn)               # (n,)
    diag_XVX = jnp.sum(X * (V_beta @ X.T).T, axis=1)              # (n,)
    q = X.T @ (h * diag_XVX)                                      # (p,)
    tr_dH = J @ q / phi                                           # (M,)
    tr_VS = jnp.einsum("ij,kij->k", V_beta, S_all) / phi          # (M,) — S symmetric
    add3 = -0.5 * (tr_dH + lams * tr_VS)

    return value, add1 + add2 + add3
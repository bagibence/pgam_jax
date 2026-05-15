"""Analytical Hessian of the Laplace-REML objective wrt log-smoothing parameters.

JAX port of ``der_wrt_smoothing.hess_laplace_appr_REML``.  Three additive
pieces, all evaluated at the MAP β̂(ρ):

    Hess[h, k] = add1[h, k] + add2[h, k] + add3[h, k]

with

    add1[h, k] = J[h]^T V_β⁻¹ J[k]
                 − δ_{hk} · (β̂^T λ_h S_h β̂) / (2 φ)

    add2[h, k] = (1 / (2 φ)) · ∂² log|S_λ|_+ / (∂ρ_h ∂ρ_k)

    add3[h, k] = ½ · ( tr_AhAr[h, k]  −  tr_d2Vb[h, k] )
       tr_AhAr[h, k] = tr( V_β · dV_β⁻¹/dρ_h · V_β · dV_β⁻¹/dρ_k )
       tr_d2Vb[h, k] = tr( V_β · d²V_β⁻¹/(dρ_h dρ_k) )
                     = tr( V_β · d²H/(dρ_h dρ_k) )
                       + δ_{hk} · λ_h · tr(V_β · S_h) / φ

The d²H trace is rewritten in O(n · M) / O(n · p) work by hoisting
``diag(X V_β X^T)`` and reusing it for both d²w / dη² and d²β / dρ² terms —
the *streaming-trace* form that avoids materialising the
(M, M, p, p) tensor d²H/dρ².

All inputs / derivatives are taken at the converged MAP; no inner solve is
re-run.  The caller is responsible for evaluating ``V_β``, ``V_β⁻¹``, ``J``,
``dH``, ``d²β``, and the ``compute_log_det_hess`` callable that returns the
block-diagonal Hessian of ``log|S_λ|_+`` wrt the flattened ρ.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._laplace_reml_beta_derivs import _dSlam_drho, d2beta_hat, dH_drho, dbeta_hat
from ._pirls_weights import _make_w_fn, small_h
from ._utils import elementwise_derivative


def hess_laplace_reml(
    beta_hat: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    obs_model,
    inverse_link_fn,
    S_all: jnp.ndarray,
    rho: jnp.ndarray,
    phi: float,
    V_beta: jnp.ndarray,
    V_beta_inv: jnp.ndarray,
    log_det_hess_Slam: jnp.ndarray,
) -> jnp.ndarray:
    """Hessian of the Laplace-REML objective wrt log-smoothing parameters.

    Parameters
    ----------
    beta_hat : (p,)
        MAP coefficient estimate at ρ.
    X : (n, p), y : (n,)
        Training data (X is the full design matrix incl. intercept).
    obs_model, inverse_link_fn :
        Nemos observation model and inverse link.
    S_all : (M, p, p)
        Stacked raw penalty matrices padded into the full coef space.
    rho : (M,)
        Flat log-smoothing parameter vector.
    phi :
        Dispersion (positive scalar).
    V_beta : (p, p)
        Posterior covariance ``φ (H + S_λ/φ)⁻¹`` (from :func:`vbeta_and_logdet`).
    V_beta_inv : (p, p)
        Posterior precision ``(H + S_λ/φ) / φ`` (from :func:`vbeta_and_logdet`).
    log_det_hess_Slam : (M, M)
        Hessian of ``log|S_λ|_+`` wrt the flattened ρ — block-diagonal, from
        ``PenaltyHandler.build()[2]``.

    Returns
    -------
    H : (M, M)
        Analytical Hessian ``d²REML/dρ²`` at β̂.
    """
    M = rho.shape[0]
    idx = jnp.arange(M)
    lams = jnp.exp(rho)
    S_tensor = S_all * lams[:, None, None]                    # (M, p, p)
    dSlam = S_tensor / phi                                    # (M, p, p)

    # ── J = dβ̂/dρ ────────────────────────────────────────────────
    J = dbeta_hat(beta_hat, V_beta, S_all, rho, phi)          # (M, p)

    # ── dV_β⁻¹/dρ = dH/dρ + λ S/φ ────────────────────────────────
    dH = dH_drho(beta_hat, X, y, obs_model, inverse_link_fn, J, phi)
    dVb_inv_drho = dH + dSlam                                 # (M, p, p)

    # ── add1: J^T V_β⁻¹ J  minus diagonal penalty correction ─────
    add1 = jnp.einsum("hj,ji,ki->hk", J, V_beta_inv, J)       # (M, M)
    pen_diag = 0.5 * jnp.einsum(
        "i,hij,j->h", beta_hat, S_tensor, beta_hat,
    ) / phi
    add1 = add1.at[idx, idx].add(-pen_diag)

    # ── add2: 0.5/φ · Hessian of log|S_λ|_+ ──────────────────────
    add2 = 0.5 * log_det_hess_Slam / phi                      # (M, M)

    # ── add3: 0.5 (tr_AhAr − tr_d2Vb) ─────────────────────────────
    # tr_AhAr[h, k] = tr( V_β · dV_β⁻¹/dρ_h · V_β · dV_β⁻¹/dρ_k )
    #              = tr( T_h · T_k ),   T_h := V_β · dV_β⁻¹/dρ_h
    T = jnp.einsum("ij,hjk->hik", V_beta, dVb_inv_drho)        # (M, p, p)
    tr_AhAr = jnp.einsum("hij,kji->hk", T, T)                  # (M, M)

    # tr( V_β · d²V_β⁻¹/(dρ_h dρ_k) ): split d²V_β⁻¹ = d²H + δ_{hk} λ_h S_h / φ.
    #
    # d²H = X^T · diag( d²w/dη² · (X J[h])·(X J[k])  +  (dw/dη) · X d²β[h,k] ) · X / φ
    # so tr(V_β · d²H) decomposes into two O(n · M²) terms via
    # diag_XVX[n] = X[n] · V_β · X[n]^T.
    eta = X @ beta_hat
    w_fn = _make_w_fn(y, obs_model, inverse_link_fn)
    d2w_deta2 = elementwise_derivative(elementwise_derivative(w_fn))(eta)  # (n,)
    h_eta = small_h(eta, y, obs_model, inverse_link_fn)                    # (n,)

    diag_XVX = jnp.sum(X * (V_beta @ X.T).T, axis=1)           # (n,)

    # Term 1 — d²w/dη² piece.  A[n, h] = (X J[h])[n].
    A = X @ J.T                                                # (n, M)
    w1 = d2w_deta2 * diag_XVX                                  # (n,)
    tr_d2H_t1 = (A * w1[:, None]).T @ A                        # (M, M)

    # Term 2 — d²β/dρ² piece. d2B = d²β/dρ².
    d2B = d2beta_hat(beta_hat, V_beta, dH, S_all, rho, phi, J) # (M, M, p)
    q = X.T @ (h_eta * diag_XVX)                               # (p,)
    tr_d2H_t2 = jnp.einsum("l,hrl->hr", q, d2B)                # (M, M)

    tr_d2H = (tr_d2H_t1 + tr_d2H_t2) / phi                     # (M, M)

    # Diagonal penalty: δ_{hk} · λ_h · tr(V_β · S_h) / φ
    tr_VS = jnp.einsum("ij,hji->h", V_beta, S_all) / phi       # (M,)
    tr_d2Vb = tr_d2H.at[idx, idx].add(lams * tr_VS)

    add3 = 0.5 * (tr_AhAr - tr_d2Vb)

    return add1 + add2 + add3
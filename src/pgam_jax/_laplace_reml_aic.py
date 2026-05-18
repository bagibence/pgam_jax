"""Corrected-AIC assembly (Wood 2017 eq. 6.32).

Given a converged ``(ρ̂, β̂(ρ̂))`` and the standard Laplace-REML state
(``V_β``, ``V_β⁻¹``, ``log_det_hess_Slam``), this module assembles the
corrected AIC

    AIC  =  −2·ℓ(β̂)  +  2·τ₂
    τ₂   =  tr( V'_β · I̥(β̂) )

with the second-order corrected posterior covariance

    V'_β = V_β + Jᵀ · V_ρ · J + Σ_{h,k} (∂_h R)ᵀ · V_ρ[h,k] · (∂_k R)

where ``R^T R = V_β`` and ``V_ρ = (−H_REML)⁻¹``.

The β-side Laplace approximation is valid for any path that returns β̂(ρ̂) at
the MAP — IRLS converges to the MAP for fixed ρ in all three methods
(``laplace_reml``, ``pql_reml``, ``pql_gcv``), so this assembly is reused by
each fit path.  V_ρ is plugged in from the Laplace-REML Hessian in every
case; for ``pql_gcv`` this is a documented approximation (ρ̂ wasn't selected
under REML asymptotics).

Mirrors the legacy numpy ``GAM_library.compute_AIC`` + the supporting
``hess_laplace_appr_REML`` / ``grad_chol_Vb_rho`` machinery in PGAM.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._chol_deriv import grad_U_Vbeta
from ._laplace_reml_beta_derivs import dbeta_hat, dH_drho
from ._laplace_reml_hessian import hess_laplace_reml
from ._pirls_weights import _make_w_fn


def compute_aic_corrected(
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Assemble Wood (2017) eq. 6.32 corrected AIC and the matching EDF.

    Parameters
    ----------
    beta_hat : (p,)
        MAP coefficient estimate at ρ.
    X : (n, p)
        Full design matrix including the intercept column.
    y : (n,)
        Response variable.
    obs_model, inverse_link_fn :
        Nemos observation model and inverse link.
    S_all : (M, p, p)
        Stacked raw penalty matrices, padded into the full coef space.
    rho : (M,)
        Flat log-smoothing parameter vector at the converged outer solution.
    phi :
        Dispersion parameter (positive scalar).
    V_beta : (p, p)
        Posterior covariance ``φ · (H + S_λ)⁻¹``.
    V_beta_inv : (p, p)
        Posterior precision ``(H + S_λ) / φ``.
    log_det_hess_Slam : (M, M)
        Hessian of ``log|S_λ|_+`` wrt ρ — from ``PenaltyHandler.build()[2]``.

    Returns
    -------
    aic : scalar
        Corrected AIC ``-2 ℓ(β̂) + 2 τ₂``.
    edf2 : scalar
        Effective degrees of freedom ``τ₂ = tr(V'_β · I̥(β̂))`` used in AIC.
    """
    # 1. REML Hessian → V_ρ as a linear operator.  -H_REML is symmetric and
    # asymptotically PSD at the REML maximum, so a thresholded eigendecomp
    # gives the same pseudoinverse semantics as ``pinv`` at half the FLOPs
    # (one eigh vs. one SVD + one V_ρ reconstruction).  We never materialise
    # V_ρ as a dense (M, M) — it is only used to multiply ``J`` and
    # ``dR.reshape(M, p²)``.
    H_reml = hess_laplace_reml(
        beta_hat, X, y, obs_model, inverse_link_fn,
        S_all, rho, phi, V_beta, V_beta_inv, log_det_hess_Slam,
    )
    w_eig, U_eig = jnp.linalg.eigh(-H_reml)
    rtol = 1e-8
    keep = w_eig > rtol * jnp.maximum(w_eig.max(), 0.0)
    w_inv = jnp.where(keep, 1.0 / jnp.where(keep, w_eig, 1.0), 0.0)

    def apply_V_rho(Z):
        """V_ρ · Z = U diag(w_inv) (Uᵀ · Z), threshold-regularised."""
        return U_eig @ (w_inv[:, None] * (U_eig.T @ Z))

    # 2. J = dβ̂/dρ and dV_β⁻¹/dρ — both used by the V_corr assembly.
    J = dbeta_hat(beta_hat, V_beta, S_all, rho, phi)                  # (M, p)
    dH = dH_drho(beta_hat, X, y, obs_model, inverse_link_fn, J, phi)  # (M, p, p)
    dSlam = S_all * jnp.exp(rho)[:, None, None] / phi                 # (M, p, p)
    dVb_inv = dH + dSlam                                              # (M, p, p)
    # dU stack for the U-convention V_β = U U^T (mgcv-style; see _chol_deriv
    # for the stability rationale).  Operates on V_β⁻¹ throughout — never
    # factors V_β directly.
    dU = grad_U_Vbeta(V_beta_inv, dVb_inv)                            # (M, p, p)

    # 3. V'_β = V_β + Jᵀ V_ρ J + Σ_{h,k} V_ρ[h,k] · dU_h · dU_k^T.
    # The transpose on the *second* factor matches mgcv's vcorr (trans=FALSE
    # branch, ``src/mat.c:1815``).  See _chol_deriv.grad_U_Vbeta docstring.
    V_rho_J = apply_V_rho(J)                                          # (M, p)
    V_prime = J.T @ V_rho_J                                            # (p, p)

    M = rho.shape[0]
    p = V_beta.shape[0]
    V_rho_dU = apply_V_rho(dU.reshape(M, p * p)).reshape(M, p, p)
    V_2prime = jnp.einsum("hia,hja->ij", dU, V_rho_dU)                 # (p, p)
    V_corr = V_beta + V_prime + V_2prime

    # 4. τ₂ = tr(V'_β · I̥).  I̥ = Xᵀ diag(w) X / φ, with w = -d²logf/dη² at β̂.
    eta = X @ beta_hat
    w = _make_w_fn(y, obs_model, inverse_link_fn)(eta)
    I_obs = (X.T * w) @ X / phi
    edf2 = jnp.sum(V_corr * I_obs)                                     # = tr(V'_β I̥)

    # 5. AIC = -2 ℓ(β̂) + 2 τ₂.  Use the unpenalised log-likelihood at β̂.
    mu = inverse_link_fn(eta)
    ll = obs_model.log_likelihood(y, mu, aggregate_sample_scores=jnp.sum)
    aic = -2.0 * ll + 2.0 * edf2
    return aic, edf2

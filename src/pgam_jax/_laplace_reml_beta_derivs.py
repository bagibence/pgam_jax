"""Derivatives of the MAP estimate and observed Hessian wrt log-smoothing parameters."""

import jax.numpy as jnp

from ._pirls_weights import small_h


def _dSlam_drho(S_all, rho, phi):
    """∂(λ_k S_k)/∂ρ = λ_k S_k / φ stack — shape (M, p, p)."""
    return S_all * jnp.exp(rho)[:, None, None] / phi


def d2beta_hat(
    beta_hat: jnp.ndarray,
    V_beta: jnp.ndarray,
    dH_drho_arr: jnp.ndarray,
    S_all: jnp.ndarray,
    rho: jnp.ndarray,
    phi: float,
    J: jnp.ndarray,
) -> jnp.ndarray:
    """Second derivative of the MAP estimate β̂ wrt log-smoothing parameters.

    Differentiating ``J[k] = -V_β (λ_k S_k / φ) β̂`` a second time and using
    ``dV_β/dρ_h = -V_β · (dH/dρ_h + λ_h S_h / φ) · V_β`` gives

        d²β̂/(dρ_h dρ_k) = -V_β · (dH/dρ_h + λ_h S_h/φ) · J[k]
                          + δ_{hk} · J[k]
                          - V_β · (λ_k S_k/φ) · J[h].

    The book (Wood 2017) contains a sign error here; this implementation matches
    the numpy reference (which encodes the corrected derivation in the project
    Overleaf notes).  Cross-validated against the legacy implementation by the
    AIC FD-vs-analytical test.

    Parameters
    ----------
    beta_hat : (p,)
        MAP coefficient estimate at ρ.
    V_beta : (p, p)
        Posterior covariance (output of :func:`vbeta_and_logdet`).
    dH_drho_arr : (M, p, p)
        Pre-computed ``dH/dρ`` from :func:`dH_drho`.
    S_all : (M, p, p)
        Stack of raw penalty matrices (full coef space).
    rho : (M,)
        Log-smoothing parameters.
    phi :
        Dispersion parameter (positive scalar).
    J : (M, p)
        Pre-computed ``dβ̂/dρ`` from :func:`dbeta_hat`.

    Returns
    -------
    d2B : (M, M, p)
        ``d2B[h, k, :] = d²β̂/(dρ_h dρ_k)``.
    """
    M = rho.shape[0]
    dSlam = _dSlam_drho(S_all, rho, phi)                       # (M, p, p)

    # Both add1's dSlam-part and add2 hit the SAME V_β contraction of
    # ``T[h, k] := dSlam[k] · J[h]`` — by symmetry of dSlam, ``T[k, h]`` equals
    # ``dSlam[h] · J[k]``, so the V_β post-multiplication is shared:
    #   add2[h, k]                = -V_β · T[h, k]
    #   add1's dSlam-part[h, k]   = -V_β · T[k, h]
    # The dH-part of add1 needs its own V_β · dH · J pass.
    # See `_script/bench_d2beta_hat.py` for the timing study justifying this
    # factorisation (10–40 % faster than the naïve dH+dSlam-then-multiply form
    # at realistic GAM sizes; bit-identical results to 1e-9 absolute).

    # T[h, k, i] = (dSlam[k] · J[h])[i]
    T = jnp.einsum("kij,hj->hki", dSlam, J)                    # (M, M, p)
    # U[h, k] = V_β · T[h, k]
    U = jnp.einsum("il,hkl->hki", V_beta, T)                   # (M, M, p)
    # add2 = -U;  add1's dSlam-part = -U.transpose(1, 0, 2)
    add_shared = -(U + U.transpose(1, 0, 2))                    # (M, M, p)

    # add1's dH-part: -V_β · dH[h] · J[k]
    Vb_dH = jnp.einsum("ij,hjl->hil", V_beta, dH_drho_arr)      # (M, p, p)
    add1_dH = -jnp.einsum("hil,kl->hki", Vb_dH, J)              # (M, M, p)

    hes = add1_dH + add_shared
    # Diagonal correction δ_{hk} · J[k]
    idx = jnp.arange(M)
    hes = hes.at[idx, idx].add(J)
    return hes


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
    eta = X @ beta_hat                                                # (n,)
    h = small_h(eta, y, obs_model, inverse_link_fn)                   # (n,)
    XJ = X @ J.T                                                      # (n, M)
    weights = h[:, None] * XJ                                         # (n, M)
    return jnp.einsum("oi,oj,ok->kij", X, X, weights) / phi           # (M, p, p)
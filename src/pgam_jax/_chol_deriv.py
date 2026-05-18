"""Derivative of an upper-triangular Cholesky factor wrt a scalar parameter.

For ``R^T R = D`` with ``R`` upper-triangular, differentiating gives
``R^{-T} dD R^{-1} = (dR R^{-1})^T + (dR R^{-1})``.  Because ``dR R^{-1}`` is
upper-triangular (product of two upper-triangular factors), the right-hand
side recovers it as

    dR R^{-1} = Φ(R^{-T} · dD · R^{-1}),
    Φ(A) := triu(A) − ½ diag(diag(A))           (upper triangle, half diagonal)

so

    dR = Φ(R^{-T} · dD · R^{-1}) · R.

Two triangular solves and one matmul — same O(p³) cost as the sequential
recurrence in Wood (2017) Appendix B.7, but expressed as BLAS-level ops so it
is jit/vmap-friendly and differentiates cleanly.  This is the path used by
the AIC computation; we avoid relying on ``jax.linalg.cholesky``'s autodiff,
which is numerically delicate at the near-rank-deficient precisions we hit
on the penalised Laplace-REML side.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def grad_cholesky(grad_D: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """Derivative of upper-triangular Cholesky factor ``R`` wrt a scalar param.

    Parameters
    ----------
    grad_D :
        Shape ``(p, p)`` — symmetric ``dD/dx``.
    R :
        Shape ``(p, p)`` upper-triangular factor satisfying ``R^T R = D``.

    Returns
    -------
    dR :
        Shape ``(p, p)`` upper-triangular — ``dR/dx``.
    """
    # Two triangular solves: A = R^{-T} dD, then M = A R^{-1} = R^{-T} dD R^{-1}.
    # Both use ``R.T`` (lower-triangular): the first directly, the second on
    # ``A.T`` since ``X R = A`` ⇔ ``R^T X^T = A^T``.
    A = solve_triangular(R.T, grad_D, lower=True)
    M = solve_triangular(R.T, A.T, lower=True).T
    # Φ_upper(M) — keep strict upper, halve diagonal, zero strict lower.
    upper_half = jnp.triu(M) - 0.5 * jnp.diag(jnp.diag(M))
    return upper_half @ R


def grad_U_Vbeta(
    V_beta_inv: jnp.ndarray,
    dVb_inv_drho: jnp.ndarray,
) -> jnp.ndarray:
    """Derivative of the ``U`` factor of ``V_β`` wrt log-smoothing parameters.

    The corrected-AIC ``V''_β`` term (Wood 2017 eq. 6.32) needs the derivative
    of a square-root factor of ``V_β``.  This implementation uses the
    ``U U^T`` factorisation rather than the textbook ``R^T R`` one,

        V_β  =  U U^T,    U upper-triangular,
        U   =  R̃⁻¹  where  R̃^T R̃ = V_β⁻¹  (R̃ also upper-triangular),

    for numerical stability: in penalised regression ``V_β⁻¹ = X^T W X + S_λ``
    is the well-conditioned object (the penalty ``S_λ`` regularises it), while
    ``V_β`` can have small eigenvalues in directions of strong smoothing.
    Operating only on R̃ via triangular solves never amplifies those small
    eigenvalues; factoring ``V_β`` directly via ``chol(V_β)`` does.

    The derivative formula mirrors mgcv's ``Vb.corr`` (R wrapper in
    ``R/gam.fit3.r``, C kernel ``dchol`` in ``src/mat.c:1845``):

        dR̃/dρ_h  =  dchol(dV_β⁻¹/dρ_h, R̃)              (standard upper-chol deriv)
        dU/dρ_h  = −R̃⁻¹ · dR̃/dρ_h · R̃⁻¹                (two triangular solves)

    The matching V_2prime assembly is

        V_2prime  =  Σ_{h,k} V_ρ[h,k] · dU_h · dU_k^T

    — note the transpose on the *second* factor, in contrast to the textbook
    ``Σ V_ρ dR^T dR`` form (Wood 2017 App. B.7) that the legacy numpy PGAM
    uses with ``R^T R = V_β``.  Both are valid leading-order Wood corrections
    of the marginal covariance ``Cov(β̂)``; we adopt mgcv's convention for the
    stability advantage and document the convention break here.

    Parameters
    ----------
    V_beta_inv : ``(p, p)``
        Posterior precision ``(H + S_λ)/φ`` (from :func:`vbeta_and_logdet`).
        Symmetric positive-definite.
    dVb_inv_drho : ``(M, p, p)``
        Stack of ``dV_β⁻¹/dρ_h``.  In the Laplace-REML pipeline this equals
        ``dH/dρ_h + λ_h S_h / φ`` — note no inversion is needed, the precision
        derivative is the natural quantity produced by the fit.

    Returns
    -------
    dU : ``(M, p, p)``
        ``dU[h]`` is upper-triangular and satisfies the U-convention defining
        identity ``dU[h] U^T + U dU[h]^T = dV_β/dρ_h``.
    """
    R_tilde = jnp.linalg.cholesky(V_beta_inv).T  # upper-tri, R̃^T R̃ = V_β⁻¹

    def per_smooth(dVbinv_h):
        dR_tilde = grad_cholesky(dVbinv_h, R_tilde)
        # dU = -R̃⁻¹ · dR̃ · R̃⁻¹ via two triangular solves.
        Z = solve_triangular(R_tilde, dR_tilde, lower=False)        # Z = R̃⁻¹ dR̃
        return -solve_triangular(R_tilde.T, Z.T, lower=True).T      # = Z · R̃⁻¹

    return jax.vmap(per_smooth)(dVb_inv_drho)

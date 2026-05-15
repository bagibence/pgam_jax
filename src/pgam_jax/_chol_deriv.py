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


def grad_chol_Vbeta(
    V_beta: jnp.ndarray,
    dVb_inv_drho: jnp.ndarray,
) -> jnp.ndarray:
    """Derivative of the upper-Cholesky factor of ``V_β`` wrt log-smoothing parameters.

    The corrected-AIC ``V''_β`` term (Wood 2017 eq. 6.32; legacy
    ``grad_chol_Vb_rho``) uses ``R^T R = V_β`` — the **covariance**, not the
    precision.  Given ``dV_β⁻¹/dρ_h`` (which the Laplace-REML pipeline
    produces directly as ``dH/dρ_h + λ_h S_h/φ``), the covariance derivative
    follows from the matrix-inverse identity

        dV_β/dρ_h = −V_β · dV_β⁻¹/dρ_h · V_β,

    and we ``vmap`` :func:`grad_cholesky` across the smoothing-parameter axis.

    Parameters
    ----------
    V_beta : ``(p, p)``
        Posterior covariance ``φ (H + S_λ/φ)⁻¹`` (from :func:`vbeta_and_logdet`).
    dVb_inv_drho : ``(M, p, p)``
        Stack ``dV_β⁻¹/dρ`` — typically ``dH/dρ + λ_k S_k/φ``.

    Returns
    -------
    dR : ``(M, p, p)``
        ``dR[h]`` is the derivative of the upper-triangular Cholesky factor of
        ``V_β`` wrt ``ρ_h``.
    """
    R = jnp.linalg.cholesky(V_beta).T  # upper triangular, R^T R = V_β
    # dV_β/dρ_h = -V_β · dV_β⁻¹/dρ_h · V_β
    dVb = -jnp.einsum("ij,hjk,kl->hil", V_beta, dVb_inv_drho, V_beta)
    return jax.vmap(lambda dD: grad_cholesky(dD, R))(dVb)
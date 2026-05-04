"""
_slam_compute.py
-----------------
JAX port of log|Σᵢ λᵢ Sᵢ|_+ and its gradient / Hessian w.r.t. log-λ.

Reference: Demo_PGAM/PGAM/src/PGAM/deriv_det_Slam.py

Key design decisions vs. the numpy reference
---------------------------------------------
1. transform_slam uses lax.scan with a fixed iteration budget of q steps.
   Each step after convergence (gamma_mask empty) is an exact no-op.

2. The Cholesky + try/except fallback in the reference is replaced by a
   single unconditional eigh path (_eigh_log_det_and_inv).  try/except cannot
   be traced by JAX JIT; eigh handles the singular/semi-definite case
   uniformly via jnp.where masking of non-positive eigenvalues.

3. Gradients and Hessian are computed analytically (not via autodiff).
   lax.scan through eigh does not reliably support reverse-mode AD, and the
   analytical expressions are cheap once S_lam^{-1} is available.

Public API
----------
    S_i_out               = transform_slam(S_tensor, rho)
    log_det               = log_det_slam(rho, S_i_out)
    grad   : (M,)         = grad_log_det_slam(rho, S_i_out)
    hessian: (M, M)       = hes_log_det_slam(rho, S_i_out)
"""

import warnings
import numpy as np
import jax.numpy as jnp
from jax import lax


def _warn_if_not_f64(fname: str) -> None:
    if jnp.finfo(float).dtype != np.dtype("float64"):
        warnings.warn(
            f"{fname}: JAX is operating in {jnp.finfo(float).dtype} precision. "
            "The Wood (2011) Appendix B algorithm uses machine-epsilon thresholds "
            "for the dominant/subdominant split and rank determination that are "
            "calibrated for float64 and will produce inaccurate results in lower "
            "precision.  Enable float64 with "
            "jax.config.update('jax_enable_x64', True).",
            UserWarning,
            stacklevel=3,
        )


# ===========================================================================
# Wood (2011) Appendix B — block-diagonal transform via lax.scan
# ===========================================================================


def _make_scan_body(lams, q):
    """
    Build the lax.scan body function for the Appendix B transform.

    Parameters
    ----------
    lams :
        Shape (M,) JAX array of smoothing parameters (not log-scale).
        Captured in the closure; may be a traced value.
    q :
        Static Python int — number of columns/rows in each penalty matrix.

    Returns
    -------
    :
        A callable ``body(state, _) -> (state, None)`` suitable for
        ``lax.scan``.  State tuple is
        ``(S_bar, S_i_out, gamma_mask, K)``:

        - ``S_bar``      : (M, q, q) working matrices; active block is [K:, K:]
        - ``S_i_out``    : (M, q, q) individually transformed output matrices
        - ``gamma_mask`` : (M,) bool — True for indices still in gamma
        - ``K``          : () int32 — accumulated block offset
    """
    M = lams.shape[0]

    _eps = jnp.finfo(float).eps
    _eps_split = float(_eps ** (1.0 / 3.0))
    _eps_rank = float(_eps**0.8)
    _big = float(jnp.finfo(float).max) * 1e-10

    I_q = jnp.eye(q)
    outer = jnp.arange(q, dtype=jnp.int32)

    def body(state, _):
        S_bar, S_i_out, gamma_mask, K = state

        act2d = (outer[:, None] >= K) & (outer[None, :] >= K)
        S_bar_act = S_bar * act2d[None, :, :]
        frobs = jnp.sqrt((S_bar_act**2).sum(axis=(1, 2)))
        omegas = frobs * lams * gamma_mask
        max_omega = jnp.where(gamma_mask, omegas, 0.0).max()

        alpha = (omegas >= _eps_split * max_omega) & gamma_mask
        gamma_prime = (omegas < _eps_split * max_omega) & gamma_mask

        safe_norms = jnp.where(alpha & (frobs > 0), frobs, 1.0)
        weights = jnp.where(alpha, 1.0 / safe_norms, 0.0)
        norm_S_act = jnp.einsum("ijk,i->jk", S_bar_act, weights)
        ev3 = jnp.linalg.eigvalsh(norm_S_act)
        r = jnp.sum(ev3 > ev3[-1] * _eps_rank, dtype=jnp.int32)

        terminate = r == q - K

        S_lam_alpha = jnp.einsum("ijk,i->jk", S_bar_act, jnp.where(alpha, lams, 0.0))
        S_eigh = S_lam_alpha - _big * jnp.diag((outer < K).astype(lams.dtype))
        _, U = jnp.linalg.eigh(S_eigh)
        U = U[:, ::-1]

        V = jnp.where(act2d, U, I_q)

        keep = outer < K + r
        T_alpha = jnp.where(keep[None, :], V, 0.0)
        S_from_alpha = T_alpha.T[None] @ S_i_out @ T_alpha[None]
        S_from_gamma = V.T[None] @ S_i_out @ V[None]
        S_i_out_new = jnp.where(
            alpha[:, None, None],
            S_from_alpha,
            jnp.where(gamma_prime[:, None, None], S_from_gamma, S_i_out),
        )

        dom_cols = (outer >= K) & (outer < K + r)
        V_n = jnp.where(dom_cols[None, :], I_q, V)
        S_bar_gp = V_n.T[None] @ S_bar @ V_n[None]
        S_bar_new = jnp.where(gamma_prime[:, None, None], S_bar_gp, S_bar)

        S_i_out_out = jnp.where(terminate, S_i_out, S_i_out_new)
        S_bar_out = jnp.where(terminate, S_bar, S_bar_new)
        K_out = jnp.where(terminate, K, K + r)
        gm_out = jnp.where(terminate, jnp.zeros(M, dtype=bool), gamma_prime)

        return (S_bar_out, S_i_out_out, gm_out, K_out), None

    return body


def transform_slam(S_tensor, rho):
    """
    Stable block-diagonal transform of Σᵢ λᵢ Sᵢ (Wood 2011, Appendix B).

    Runs ``lax.scan`` for exactly ``q`` iterations.  Iterations after
    convergence (``gamma_mask`` empty or ``r == Q``) are exact no-ops, so the
    budget is safe.

    Parameters
    ----------
    S_tensor :
        Shape (M, q, q) JAX array of positive semi-definite penalty matrices.
    rho :
        Shape (M,) JAX array of log-smoothing parameters
        (``lams = exp(rho)``).

    Returns
    -------
    :
        Shape (M, q, q) array ``S_i_out``.  Each ``S_i_out[i]`` has support
        concentrated in a single diagonal block, so ``Σᵢ λᵢ S_i_out[i]`` has
        no cross-scale cancellation.  Pass to :func:`log_det_slam`,
        :func:`grad_log_det_slam`, and :func:`hes_log_det_slam`.
    """
    _warn_if_not_f64("transform_slam")
    lams = jnp.exp(rho)
    M, q = S_tensor.shape[0], S_tensor.shape[1]

    body = _make_scan_body(lams, q)
    init = (
        S_tensor,
        S_tensor,
        jnp.ones(M, dtype=bool),
        jnp.zeros((), dtype=jnp.int32),
    )
    (_, S_i_out, _, _), _ = lax.scan(body, init, None, length=q)
    return S_i_out


# ===========================================================================
# Shared helper: log|S_lam|_+ and S_lam^{-1} via eigh
# ===========================================================================


def _eigh_log_det_and_inv(S_i_out, lams):
    """
    Compute ``log|S_lam|_+`` and the Moore-Penrose pseudo-inverse of S_lam.

    Uses ``eigh`` unconditionally — no ``try/except`` needed.  Non-positive
    eigenvalues are masked with ``jnp.where`` so both outputs are
    numerically well-defined for singular / semi-definite inputs.

    Parameters
    ----------
    S_i_out :
        Shape (M, q, q) transformed penalty matrices (output of
        :func:`transform_slam`).
    lams :
        Shape (M,) smoothing parameters (not log-scale).

    Returns
    -------
    log_det :
        Scalar — ``log|S_lam|_+``.
    Sinv :
        Shape (q, q) — ``S_lam^{-1}`` (pseudo-inverse in the range space).
    """
    Slam = jnp.einsum("ijk,i->jk", S_i_out, lams)
    Slam = 0.5 * (Slam + Slam.T)

    ev, U = jnp.linalg.eigh(Slam)
    pos = ev > jnp.finfo(float).eps
    ev_safe = jnp.where(pos, ev, 1.0)

    log_det = jnp.sum(jnp.where(pos, jnp.log(ev_safe), 0.0))
    Sinv = (U * jnp.where(pos, 1.0 / ev_safe, 0.0)[None, :]) @ U.T

    return log_det, Sinv


# ===========================================================================
# Log-det, gradient, Hessian
# ===========================================================================


def log_det_slam(rho, S_i_out):
    """
    Compute ``log|Σᵢ exp(ρᵢ) Sᵢ_out|_+``.

    Parameters
    ----------
    rho :
        Shape (M,) log-smoothing parameters.
    S_i_out :
        Shape (M, q, q) output of :func:`transform_slam`.

    Returns
    -------
    :
        Scalar log-determinant.
    """
    _warn_if_not_f64("log_det_slam")
    lams = jnp.exp(rho)
    log_det, _ = _eigh_log_det_and_inv(S_i_out, lams)
    return log_det


def grad_log_det_slam(rho, S_i_out):
    """
    Gradient of ``log|S_lam|`` w.r.t. ``ρ``.

    ``∂log|S_lam|/∂ρⱼ = λⱼ tr(S_lam⁻¹ Sⱼ_out)``

    Parameters
    ----------
    rho :
        Shape (M,) log-smoothing parameters.
    S_i_out :
        Shape (M, q, q) output of :func:`transform_slam`.

    Returns
    -------
    :
        Shape (M,) gradient vector.
    """
    _warn_if_not_f64("grad_log_det_slam")
    lams = jnp.exp(rho)
    _, Sinv = _eigh_log_det_and_inv(S_i_out, lams)
    grad = lams * jnp.einsum("kl,jlk->j", Sinv, S_i_out)
    return grad


def hes_log_det_slam(rho, S_i_out):
    """
    Hessian of ``log|S_lam|`` w.r.t. ``ρ``.

    ``H[i,j] = -λᵢ λⱼ tr(S_lam⁻¹ Sᵢ_out S_lam⁻¹ Sⱼ_out) + δᵢⱼ λᵢ tr(S_lam⁻¹ Sᵢ_out)``

    The diagonal term equals the gradient component, so no separate trace
    computation is needed beyond what is required for the off-diagonal part.

    Parameters
    ----------
    rho :
        Shape (M,) log-smoothing parameters.
    S_i_out :
        Shape (M, q, q) output of :func:`transform_slam`.

    Returns
    -------
    :
        Shape (M, M) Hessian matrix.
    """
    _warn_if_not_f64("hes_log_det_slam")
    lams = jnp.exp(rho)
    _, Sinv = _eigh_log_det_and_inv(S_i_out, lams)

    Sinv_S = jnp.einsum("kl,ilj->ikj", Sinv, S_i_out)
    H_off = jnp.einsum("ikl,jlk->ij", Sinv_S, Sinv_S)
    grad_diag = lams * jnp.einsum("kl,jlk->j", Sinv, S_i_out)

    return -jnp.outer(lams, lams) * H_off + jnp.diag(grad_diag)

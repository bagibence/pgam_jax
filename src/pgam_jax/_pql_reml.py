from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from numpy.typing import NDArray

from ._pql_gcv import FLOAT_EPS, _vmap_symm_mult, _vmap_trace, _vmap_where
from .penalty_utils import compute_penalty_blocks, prepend_zeros_for_intercept


@partial(
    jax.jit,
    static_argnames=("compute_log_det_and_grad", "compute_sqrt"),
)
def _compute_reml_and_states(
    regularization_strength: Any,
    X: NDArray,
    Q: NDArray,
    R: NDArray,
    y: NDArray,
    compute_log_det_and_grad: Callable,
    compute_sqrt: Callable,
):
    """
    Forward pass for linearized REML.

    Mirrors _compute_gcv_and_states in structure: same SVD of [R; sqrt_penalty],
    same singular-value masking.  Computes the three REML terms:

        reml_val = 0.5 * RSS_reml
                 + 0.5 * log|X'X + S_lam|
                 - 0.5 * log|S_lam|_+

    and returns all intermediate quantities needed by the backward pass.

    Parameters
    ----------
    regularization_strength :
        Pytree of log-smoothing parameters (one leaf per penalty block).
    X, Q, R :
        Whitened design matrix and its QR factors (sqrt(W) X, Q, R).
    y :
        Augmented working response (length >= n_obs; only first n_obs rows used).
    compute_log_det_and_grad :
        Static callable (from PenaltyHandler.build()) that receives
        regularization_strength and returns (log_dets list, grads list).
    compute_sqrt :
        Static callable (from PenaltyHandler.build()) that receives
        regularization_strength and returns the block-diagonal sqrt penalty matrix.

    Returns
    -------
    reml_val :
        Scalar REML objective.
    RSS_reml :
        Scalar penalised RSS (stored for diagnostics).
    log_det_XtXpSl :
        Scalar log|X'X + S_lam| (stored for diagnostics).
    y1 :
        Shape (k, 1) — U1ᵀ Qᵀ y, needed for the RSS gradient.
    s_inv :
        Shape (k,) — masked reciprocal singular values, needed for M_j.
    V_T :
        Shape (k, p) — right singular vectors, needed for M_j.
    log_det_sl_grads :
        Pytree of per-block log|S_lam|_+ gradients (same structure as
        regularization_strength), computed alongside the log-det in the
        forward pass and stored to avoid recomputation in the backward.
    n_obs :
        Number of observations (scalar int).
    """
    sqrt_penalty = compute_sqrt(regularization_strength)
    sqrt_penalty = prepend_zeros_for_intercept(sqrt_penalty)

    n_obs = X.shape[0]
    U, s, V_T = jnp.linalg.svd(jnp.vstack((R, sqrt_penalty)), full_matrices=False)

    # mask near-zero singular values (identical to GCV)
    low_vals = s < FLOAT_EPS * s.max()
    s = jnp.where(low_vals, 0.0, s)
    U = _vmap_where(low_vals, 0, U)
    V_T = _vmap_where(low_vals, 0, V_T.T).T

    U1 = U[: R.shape[0]]
    s_inv = jnp.where(low_vals, 0.0, 1.0 / jnp.where(low_vals, 1.0, s))

    # --- REML RSS = ||y||^2 - ||U1^T Q^T y||^2 ---
    y_obs = y[:n_obs].reshape(n_obs, -1)
    y1 = U1.T @ (Q.T @ y_obs)  # (k, 1)
    RSS_reml = jnp.sum(y_obs**2) - jnp.sum(y1**2)

    # --- log|X'X + S_lam| = 2 * sum log(s_k) over positive singular values ---
    log_s_safe = jnp.where(low_vals, 0.0, jnp.log(jnp.where(low_vals, 1.0, s)))
    log_det_XtXpSl = 2.0 * jnp.sum(log_s_safe)

    # --- log|S_lam|_+ and its gradient (shared eigh, no extra cost) ---
    log_dets, log_det_sl_grads = compute_log_det_and_grad(regularization_strength)
    log_det_Sl = jtu.tree_reduce(lambda a, b: a + b, log_dets)

    reml_val = 0.5 * RSS_reml + 0.5 * log_det_XtXpSl - 0.5 * log_det_Sl

    return reml_val, RSS_reml, log_det_XtXpSl, y1, s_inv, V_T, log_det_sl_grads, n_obs


@partial(
    jax.jit,
    static_argnames=("apply_identifiability",),
)
def _reml_grad_compute_from_states(
    regularization_strength: Any,
    penalty_tree: Any,
    y1: jnp.ndarray,
    s_inv: jnp.ndarray,
    V_T: jnp.ndarray,
    log_det_sl_grads: Any,
    apply_identifiability: Callable | None,
):
    """
    Gradient of the REML objective w.r.t. regularization_strength.

    Given states from _compute_reml_and_states, computes three gradient
    contributions per leaf j:

        grad_j = 0.5 * lam_j * (comp @ y1)^T Sj (comp @ y1)   [RSS]
               + 0.5 * lam_j * tr(comp^T Sj comp)              [log|X'X+Slam|]
               - 0.5 * d log|Slam|_+ / d rho_j                 [log|Slam|]

    where comp = V_T^T * s_inv  and  Mj = comp^T Sj comp.
    """
    comp = V_T.T * s_inv  # (n_features, k)
    comp_y1 = comp @ y1.ravel()  # (n_features,)

    blocks = compute_penalty_blocks(
        penalty_tree,
        apply_identifiability=apply_identifiability,
        shift_by=1,
    )

    lams = jtu.tree_map(jnp.exp, regularization_strength)

    # lam_j * (comp @ y1)^T Sj (comp @ y1)
    rss_grad = jtu.tree_map(
        lambda s_block, lam: lam * _vmap_symm_mult(s_block, comp_y1),
        blocks,
        lams,
    )

    # lam_j * tr(Mj) = lam_j * tr(comp^T Sj comp)
    logdet_xtx_grad = jtu.tree_map(
        lambda s_block, lam: lam * _vmap_trace(_vmap_symm_mult(s_block, comp)),
        blocks,
        lams,
    )

    return jtu.tree_map(
        lambda rss_g, ld_xtx_g, ld_sl_g: 0.5 * rss_g + 0.5 * ld_xtx_g - 0.5 * ld_sl_g,
        rss_grad,
        logdet_xtx_grad,
        log_det_sl_grads,
    )


def reml_compute_factory(
    compute_sqrt: Callable,
    compute_log_det_and_grad: Callable,
    apply_identifiability_columns: Callable | None,
    apply_identifiability: Callable | None,
):
    """
    Build a differentiable REML objective with a custom VJP.

    Parameters
    ----------
    compute_sqrt :
        Static callable from ``PenaltyHandler.build()`` computing the block-diagonal
        sqrt penalty matrix from regularization_strength.
    compute_log_det_and_grad :
        Static callable from ``PenaltyHandler.build()`` computing log|S_lam|_+ and
        its gradient w.r.t. rho from regularization_strength.
        Same semantics as in gcv_compute_factory.

    Returns
    -------
    :
        A jax.custom_vjp function
        _reml_compute(regularization_strength, penalty_tree, X, Q, R, y)
        returning the scalar REML objective with analytic reverse-mode gradient.
    """

    @jax.custom_vjp
    def _reml_compute(
        regularization_strength: Any,
        penalty_tree: Any,
        X: NDArray,
        Q: NDArray,
        R: NDArray,
        y: NDArray,
    ):
        return _compute_reml_and_states(
            regularization_strength,
            X,
            Q,
            R,
            y,
            compute_log_det_and_grad=compute_log_det_and_grad,
            compute_sqrt=compute_sqrt,
        )[0]

    def _reml_compute_fwd(regularization_strength, penalty_tree, X, Q, R, y):
        reml_val, RSS_reml, log_det_XtXpSl, y1, s_inv, V_T, log_det_sl_grads, n_obs = (
            _compute_reml_and_states(
                regularization_strength,
                X,
                Q,
                R,
                y,
                compute_log_det_and_grad=compute_log_det_and_grad,
                compute_sqrt=compute_sqrt,
            )
        )
        return reml_val, (
            regularization_strength,
            penalty_tree,
            y1,
            s_inv,
            V_T,
            log_det_sl_grads,
        )

    def _reml_compute_bwd(res, reml_bar):
        regularization_strength, penalty_tree, y1, s_inv, V_T, log_det_sl_grads = res
        reml_grad = _reml_grad_compute_from_states(
            regularization_strength,
            penalty_tree,
            y1,
            s_inv,
            V_T,
            log_det_sl_grads,
            apply_identifiability=apply_identifiability,
        )
        return (jtu.tree_map(lambda g: reml_bar * g, reml_grad),) + (None,) * 5

    _reml_compute.defvjp(_reml_compute_fwd, _reml_compute_bwd)
    return _reml_compute

"""Laplace-approximated REML objective and its gradient wrt log-smoothing parameters."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree

from ._laplace_reml_beta_derivs import dbeta_hat
from ._laplace_reml_vbeta import vbeta_and_logdet
from ._pirls_weights import _make_w_fn, small_h


def laplace_reml(
    beta_hat: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    obs_model,
    inverse_link_fn,
    S_all: jnp.ndarray,
    rhos_tree,
    phi: float,
    M_null: int,
    compute_sqrt,
    compute_log_det_and_grad,
    return_grad: bool = False,
):
    """Laplace-approximated REML at the MAP estimate.

    Wood (2017) eq. 6.18:

        ℓ(ρ) = l(β̂) - (1/2φ) β̂^T S_λ β̂
               + (1/2φ) log|S_λ|_+
               - (1/2) log|H + S_λ/φ|
               + (M_0/2) log(2π)

    Stability note: the sqrt and log-det of S_λ both go through
    ``PenaltyHandler.build()`` callables, which use the Wood (2011) stable
    block-disjoint parameterisation.  Direct ``symmetric_sqrt(Σ_k λ_k S_k)``
    would lose small-λ modes when the λ_k span different scales.

    Parameters
    ----------
    beta_hat : (p,)
        MAP coefficient estimate at ρ.
    X : (n, p), y : (n,)
    obs_model, inverse_link_fn :
        Nemos observation model and inverse link.
    S_all : (M, p, p)
        Stack of raw penalty matrices padded into the full coef space; used by
        the gradient formulas (add1, J, tr_VS).  Its leading axis must align
        with the flattened ``rhos_tree``.
    rhos_tree :
        Pytree of per-smooth rho arrays — the canonical representation,
        consistent with the PenaltyHandler that produced ``compute_sqrt`` /
        ``compute_log_det_and_grad``.  The flat ρ used by the S_all einsums is
        derived internally via ``ravel_pytree``.
    phi : positive scalar.
    M_null : int
        Null-space dimension of S_λ (static).
    compute_sqrt, compute_log_det_and_grad :
        Callables returned by ``PenaltyHandler.build()``.
    return_grad : bool

    Returns
    -------
    value : scalar
    grad : (M,) flat gradient wrt the raveled ρ, only when ``return_grad=True``
    """
    rho, _ = ravel_pytree(rhos_tree)
    eta = X @ beta_hat
    lams = jnp.exp(rho)

    # log-likelihood
    mu = inverse_link_fn(eta)
    ll = obs_model.log_likelihood(y, mu, aggregate_sample_scores=jnp.sum)

    # penalty: -0.5/phi * beta^T S_lam beta
    pen = -0.5 * jnp.einsum("i,kij,k,j->", beta_hat, S_all, lams, beta_hat) / phi

    # log|H + S_lam/phi| via SVD of [R; sqrt_penalty], where sqrt_penalty comes
    # from PenaltyHandler (Wood 2011 stable construction) with a zero column
    # prepended for the unpenalized intercept.
    w = _make_w_fn(y, obs_model, inverse_link_fn)(eta)
    R = jnp.linalg.qr(jnp.sqrt(w)[:, None] * X, mode="r")
    sqrt_pen = compute_sqrt(rhos_tree)
    sqrt_pen = jnp.hstack((jnp.zeros((sqrt_pen.shape[0], 1)), sqrt_pen))
    V_beta, _, log_det_HpS = vbeta_and_logdet(R, sqrt_pen, phi)

    # log|S_lam|_+ (and gradient) from PenaltyHandler
    log_dets, log_det_grads = compute_log_det_and_grad(rhos_tree)
    log_det_Slam = jtu.tree_reduce(lambda a, b: a + b, log_dets)

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
    grad_log_det, _ = ravel_pytree(log_det_grads)
    add2 = 0.5 * grad_log_det / phi

    # add3 = -0.5 * tr(V_beta * (dH/drho_r + lam_r * S_r / phi))
    J = dbeta_hat(beta_hat, V_beta, S_all, rho, phi)              # (M, p)
    h = small_h(eta, y, obs_model, inverse_link_fn)               # (n,)
    diag_XVX = jnp.sum(X * (V_beta @ X.T).T, axis=1)              # (n,)
    q = X.T @ (h * diag_XVX)                                      # (p,)
    tr_dH = J @ q / phi                                           # (M,)
    tr_VS = jnp.einsum("ij,kij->k", V_beta, S_all) / phi          # (M,) — S symmetric
    add3 = -0.5 * (tr_dH + lams * tr_VS)

    return value, add1 + add2 + add3


def laplace_reml_compute_factory(
    obs_model,
    inverse_link_fn,
    phi: float,
    M_null: int,
    compute_sqrt,
    compute_log_det_and_grad,
    rhos_tree_example,
):
    """Build a ``jax.custom_vjp`` Laplace-REML objective: ``rhos_tree → value``.

    Mirrors ``reml_compute_factory`` in ``_pql_reml.py``.  The custom VJP feeds
    the analytical gradient from ``laplace_reml(..., return_grad=True)`` back
    through autodiff, so any optimiser that calls ``jax.value_and_grad`` (e.g.
    ``optimistix.minimise``) gets the analytical gradient transparently —
    without differentiating through the inner β̂ solve.  ``rhos_tree`` stays a
    pytree throughout (optimistix optimises pytrees natively); the flat ρ is
    only an internal detail of ``laplace_reml``'s S_all einsums.

    Parameters
    ----------
    obs_model, inverse_link_fn :
        Nemos observation model and inverse link.
    phi :
        Positive scalar dispersion (φ ≡ 1 for Poisson).
    M_null : int
        Null-space dimension of S_λ (static).
    compute_sqrt, compute_log_det_and_grad :
        Callables from ``PenaltyHandler.build()``.
    rhos_tree_example :
        Any rho pytree with the structure used by this fit.  The ρ-structure is
        fixed for a fit, so the flat→tree ``unravel`` is built once here rather
        than per ``_fwd`` call.

    Returns
    -------
    objective : callable
        ``objective(rhos_tree, beta_hat, X, y, S_all) -> scalar`` with an
        analytic reverse-mode gradient wrt ``rhos_tree`` (β̂, X, y, S_all are
        treated as constants — β̂'s ρ-dependence is already folded into the
        analytical gradient by the envelope theorem).
    """
    _, unravel = ravel_pytree(rhos_tree_example)

    @jax.custom_vjp
    def _objective(rhos_tree, beta_hat, X, y, S_all):
        return laplace_reml(
            beta_hat, X, y, obs_model, inverse_link_fn, S_all, rhos_tree,
            phi, M_null, compute_sqrt, compute_log_det_and_grad, return_grad=False,
        )

    def _fwd(rhos_tree, beta_hat, X, y, S_all):
        value, grad_flat = laplace_reml(
            beta_hat, X, y, obs_model, inverse_link_fn, S_all, rhos_tree,
            phi, M_null, compute_sqrt, compute_log_det_and_grad, return_grad=True,
        )
        return value, unravel(grad_flat)

    def _bwd(grad_tree, value_bar):
        return (
            jtu.tree_map(lambda g: value_bar * g, grad_tree),
            None, None, None, None,  # beta_hat, X, y, S_all are not differentiated
        )

    _objective.defvjp(_fwd, _bwd)
    return _objective
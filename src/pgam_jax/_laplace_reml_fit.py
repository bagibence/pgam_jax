"""Inner MAP solver and outer rho-optimisation loop for the Laplace-REML GAM path.

Note
----
This is a *patched* approach: the smoothing penalty is baked directly into the
loss handed to a nemos solver, with the regularizer forced to ``UnRegularized``.
The cleaner long-term design — for the nemos port — is a dedicated smoothing
``Regularizer`` so the MAP step is an ordinary GLM fit and the solver can be
built once instead of rebuilt per rho.
"""

import jax
import jax.numpy as jnp
import nemos.solvers
from jax.flatten_util import ravel_pytree
from nemos.regularizer import UnRegularized

from ._laplace_reml import laplace_reml_compute_factory

# Inner MAP solve must converge tightly: the Laplace-REML gradient (and the FD
# checks against it) are sensitive to β̂ accuracy.
_DEFAULT_INNER_SOLVER = "LBFGS"
_DEFAULT_INNER_SOLVER_KWARGS = {"tol": 1e-12, "maxiter": 1000}

# Outer rho optimisation can run looser than the inner MAP solve.
_DEFAULT_OUTER_SOLVER = "LBFGS"
_DEFAULT_OUTER_SOLVER_KWARGS = {"tol": 1e-6, "maxiter": 200}


def make_inner_solver(solver_name=_DEFAULT_INNER_SOLVER, solver_kwargs=None):
    """Build a ``solve(loss, beta0) -> params`` closure from the nemos registry.

    The GAM object builds this once at fit setup — the user picks
    ``solver_name`` from ``nemos.solvers`` — and threads it into ``fit_beta``.
    The regularizer is forced to ``UnRegularized``: the smoothing penalty is
    already inside ``loss``.

    Parameters
    ----------
    solver_name :
        Any algorithm name in ``nemos.solvers`` (e.g. "LBFGS", "GradientDescent").
    solver_kwargs :
        Forwarded to the solver constructor (e.g. ``tol``, ``maxiter``).  If
        None, a tight default is used so the inner solve is accurate enough for
        the Laplace-REML gradient.

    Returns
    -------
    solve : callable
        ``solve(loss, beta0) -> params``, where ``loss(beta, *args) -> scalar``.
    """
    impl = nemos.solvers.get_solver(solver_name).implementation
    kwargs = _DEFAULT_INNER_SOLVER_KWARGS if solver_kwargs is None else solver_kwargs

    def solve(loss, beta0):
        # The loss is rho-dependent, so the solver is rebuilt per call — the
        # cost of the patched approach; a smoothing Regularizer would let the
        # solver be constructed once.
        solver = impl(loss, UnRegularized(), None, has_aux=False, **kwargs)
        return solver.run(beta0)[0]

    return solve


_default_inner_solve = make_inner_solver()


def fit_beta(
    X,
    y,
    obs_model,
    inverse_link_fn,
    S_all,
    rho,
    phi,
    beta0=None,
    solve=None,
):
    """MAP estimate of beta: minimise the penalised negative log-likelihood.

        loss(beta) = -log L(beta) + 0.5 beta^T S_lam beta

    Parameters
    ----------
    X : (n, p), y : (n,)
    obs_model, inverse_link_fn :
        Nemos observation model and inverse link.
    S_all : (M, p, p)
        Stacked raw penalty matrices padded into the full coef space.
    rho :
        Pytree or (M,) array of log-smoothing parameters; raveled internally to
        align with the leading axis of ``S_all``.
    phi :
        Positive scalar dispersion. Accepted for a uniform call signature but not used here.
        Would scale both the NLL and the penalty, but the MAP's location doesn't depend on it.
    beta0 : (p,) or None
        Warm-start; zeros if None.
    solve : callable or None
        ``solve(loss, beta0) -> params`` closure from ``make_inner_solver``.
        If None, a default LBFGS solve is used.

    Returns
    -------
    beta_hat : (p,)
        MAP estimate.
    """
    rho_flat, _ = ravel_pytree(rho)
    S_lam = jnp.einsum("kij,k->ij", S_all, jnp.exp(rho_flat))

    def penalized_nll(beta, *args):
        eta = X @ beta
        # passing scale=phi and scaling the penalty term would give the same
        nll = -obs_model.log_likelihood(
            y, inverse_link_fn(eta), aggregate_sample_scores=jnp.sum
        )
        return nll + 0.5 * jnp.dot(beta, S_lam @ beta)

    if beta0 is None:
        beta0 = jnp.zeros(X.shape[1])
    if solve is None:
        solve = _default_inner_solve

    return solve(penalized_nll, beta0)


def laplace_reml_outer_iteration(
    init_rhos_tree,
    init_beta,
    X,
    y,
    obs_model,
    inverse_link_fn,
    S_all,
    phi,
    M_null,
    compute_sqrt,
    compute_log_det_and_grad,
    inner_solve=None,
    outer_solver_name: str = _DEFAULT_OUTER_SOLVER,
    outer_solver_kwargs: dict | None = None,
):
    """Optimise log-smoothing parameters rho via the Laplace-REML objective.

    Outer loop: a nemos-registry solver minimises ``-laplace_reml`` over
    ``rhos_tree``.  Inner loop: at each rho evaluation ``fit_beta`` re-fits
    β̂ to the MAP.  The ``custom_vjp`` objective from
    ``laplace_reml_compute_factory`` supplies the analytical gradient, so the
    outer solver never differentiates through the inner solve — β̂ is
    ``stop_gradient``-ed and its ρ-dependence is folded into the analytical
    gradient by the envelope theorem.

    Parameters
    ----------
    init_rhos_tree :
        Initial log-smoothing parameters (pytree matching the PenaltyHandler).
    init_beta : (p,)
        Initial coefficients — also the warm-start for every inner MAP solve.
    X : (n, p), y : (n,)
    obs_model, inverse_link_fn :
        Nemos observation model and inverse link.
    S_all : (M, p, p)
        Stacked raw penalty matrices padded into the full coef space.
    phi :
        Positive scalar dispersion (φ ≡ 1 for Poisson).
    M_null : int
        Null-space dimension of S_λ (static).
    compute_sqrt, compute_log_det_and_grad :
        Callables from ``PenaltyHandler.build()``.
    inner_solve :
        ``solve(loss, beta0)`` closure for the inner MAP solve (see
        ``make_inner_solver``).  Default LBFGS if None.
    outer_solver_name, outer_solver_kwargs :
        Outer solver selection from ``nemos.solvers`` and its kwargs.

    Returns
    -------
    rhos_tree :
        Optimised log-smoothing parameters (same structure as init).
    beta_hat : (p,)
        MAP coefficients at the optimised rho.
    n_iter : int
        Number of outer solver steps.
    """
    objective = laplace_reml_compute_factory(
        obs_model,
        inverse_link_fn,
        phi,
        M_null,
        compute_sqrt,
        compute_log_det_and_grad,
        init_rhos_tree,
    )
    if inner_solve is None:
        inner_solve = _default_inner_solve

    def neg_reml(rhos_tree, *args):
        beta_hat = fit_beta(
            X,
            y,
            obs_model,
            inverse_link_fn,
            S_all,
            rhos_tree,
            phi,
            beta0=init_beta,
            solve=inner_solve,
        )
        return -objective(rhos_tree, jax.lax.stop_gradient(beta_hat), X, y, S_all)

    kwargs = (
        _DEFAULT_OUTER_SOLVER_KWARGS
        if outer_solver_kwargs is None
        else outer_solver_kwargs
    )
    impl = nemos.solvers.get_solver(outer_solver_name).implementation
    solver = impl(neg_reml, UnRegularized(), None, has_aux=False, **kwargs)
    rhos_opt, outer_state, _ = solver.run(init_rhos_tree)
    n_iter = int(outer_state.stats.num_steps)

    beta_opt = fit_beta(
        X,
        y,
        obs_model,
        inverse_link_fn,
        S_all,
        rhos_opt,
        phi,
        beta0=init_beta,
        solve=inner_solve,
    )
    return rhos_opt, beta_opt, n_iter

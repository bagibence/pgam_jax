"""Implement the PQL iteration.

Statsmodels terminology:
1. Link, as usual. link(mean) = jnp.dot(X, w)
2. fitted: inverse link
3. variance: the variance function of the observation model
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxopt import LBFGS, LBFGSB, ScipyBoundedMinimize, ScipyMinimize
from nemos.glm.initialize_parameters import INVERSE_FUNCS
from nemos.tree_utils import pytree_map_and_reduce

from ._utils import elementwise_derivative as _elementwise_derivative

FLOAT_EPS = jnp.finfo(float).eps


def tree_concat(tree1, tree2, axis):
    return jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=axis), tree1, tree2)


def model_constructors_for_weights_and_pseudo_data(
    variance_func, link_func, fisher_scoring=False
):
    """
    Compute the IRLS weights and pseudo-data.

    See Chapter 3, pp. 106–107 of:

    Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC Press.

    Parameters
    ----------
    variance_func:
        The variance function of the exponential family distribution.
    link_func:
        The link function, which maps the mean to the linear combination of the weights.

    Returns
    -------
        The IRLS weights and pseudo-data computing function.
    """

    variance_der = _elementwise_derivative(variance_func)
    link_func_der = _elementwise_derivative(link_func)
    link_func_der2 = _elementwise_derivative(link_func_der)

    @jax.jit
    def compute_alpha(y, rate):
        dy = y - rate
        corr = variance_der(rate) / variance_func(rate) + link_func_der2(
            rate
        ) / link_func_der(rate)
        return 1.0 + dy * jnp.clip(corr, FLOAT_EPS, jnp.inf)

    @jax.jit
    def compute_z(y, rate, alpha):
        rate = jnp.asarray(rate)
        lin_pred = link_func(rate)
        return lin_pred + link_func_der(rate) * (y - rate) / alpha

    @jax.jit
    def weight_compute(rate, alpha):
        dmu_deta = jnp.clip(1.0 / link_func_der(rate), FLOAT_EPS, jnp.inf)
        w = alpha * dmu_deta**2 / variance_func(rate)
        return w

    def compute_pseudo_data_and_weights(y, rate):
        alpha = (
            jtu.tree_map(jnp.ones_like, rate)
            if fisher_scoring
            else jtu.tree_map(compute_alpha, y, rate)
        )
        z = jtu.tree_map(compute_z, y, rate, alpha)
        w = jtu.tree_map(weight_compute, rate, alpha)
        return z, w

    return compute_pseudo_data_and_weights


def unflatten_coeffs(coeffs_flat, leaf_shapes):
    """Slice flat coeffs into a list of arrays matching leaf_shapes."""
    indices = jnp.cumsum(jnp.array([0] + leaf_shapes))
    slices = [coeffs_flat[indices[i] : indices[i + 1]] for i in range(len(leaf_shapes))]
    return slices


def weighted_least_squares(X, y, weights):
    """
    Robust WLS via QR decomposition.

    X: (n, d)
    y: (n,) or (n, k)
    weights: (n,)
    """
    sqrt_w = jnp.sqrt(weights)  # (n,)
    Xw = X * sqrt_w[:, None]  # (n, d)
    yw = y * sqrt_w if y.ndim == 1 else y * sqrt_w[:, None]  # (n,) or (n, k)

    Q, R = jnp.linalg.qr(Xw, mode="reduced")  # Q: (n, d), R: (d, d)
    beta = jnp.linalg.solve(R, Q.T @ yw)  # (d,) or (d, k)
    return beta, Xw, yw


def _tree_max_leaf_l2_delta(tree1, tree2):
    """Return the max L2 distance across matching leaves of two pytrees."""
    return pytree_map_and_reduce(lambda x, y: jnp.linalg.norm(x - y), max, tree1, tree2)


def _tree_max_leaf_l2(tree):
    """Return the max L2 norm across leaves of a pytree."""
    return pytree_map_and_reduce(jnp.linalg.norm, max, tree)


VALID_CONVERGENCE_CRITERIA = ("coef", "coef_and_reg", "gcv")


def check_pql_convergence(
    criterion: str,
    iteration: int,
    tol: float,
    old_params,
    new_params,
    old_reg_strength,
    new_reg_strength,
    old_score=None,
    new_score=None,
):
    """Check outer-loop convergence using the requested monitor."""
    if criterion == "coef":
        if iteration < 1:
            return False
        delta = _tree_max_leaf_l2_delta(old_params, new_params)
        return delta < tol * _tree_max_leaf_l2(new_params)

    if criterion == "coef_and_reg":
        if iteration < 1:
            return False
        coef_ok = _tree_max_leaf_l2_delta(
            old_params, new_params
        ) < tol * _tree_max_leaf_l2(new_params)
        reg_ok = _tree_max_leaf_l2_delta(
            old_reg_strength, new_reg_strength
        ) < tol * _tree_max_leaf_l2(new_reg_strength)
        return coef_ok & reg_ok

    if criterion == "gcv":
        if iteration <= 3:
            return False
        if old_score is None or new_score is None:
            return False
        return jnp.abs(new_score - old_score) < tol * jnp.abs(new_score)

    raise ValueError(
        f"convergence_criterion must be one of {VALID_CONVERGENCE_CRITERIA}, "
        f"got {criterion!r}."
    )


def pql_outer_iteration(
    reg_strength,
    init_pars,
    X,
    y,
    penalty_tree,
    obs_model,
    variance_func,
    inner_func,
    compute_sqrt,
    fisher_scoring=False,
    max_iter=100,
    tol_optim=10**-10,
    tol_update=10**-5,
    use_scipy=False,
    convergence_criterion: str = "gcv",
):
    """

    Parameters
    ----------
    reg_strength
    init_pars
    X:
        Tree of 2d arrays.
    y:
        Array, 1d or 2d.
    penalty_tree:
        List of penalty tensors trees (num_pen, n, n).
        num_pen is >1 when penalizing null-space or for multi-dim penalties.
    obs_model:
        A nemos observation model.
    variance_func
    inner_func:
        PQL inner loop function
    fisher_scoring
    max_iter
    use_scipy:
        If True, use scipy's L-BFGS-B for both the inner GCV minimization
        and the initial GLM fit instead of jaxopt's. Often faster on CPU.
        Defaults to False.
    convergence_criterion:
        Outer-loop convergence monitor. ``"coef"`` checks only coefficient
        movement, ``"coef_and_reg"`` checks both coefficient and log-regularizer
        movement, and ``"gcv"`` checks relative change in the optimized inner
        GCV score, matching the legacy PGAM convention.
        Defaults to ``"gcv"``.
    Returns
    -------

    """
    if convergence_criterion not in VALID_CONVERGENCE_CRITERIA:
        raise ValueError(
            f"convergence_criterion must be one of {VALID_CONVERGENCE_CRITERIA}, "
            f"got {convergence_criterion!r}."
        )
    # TODO: variance_func can be determined based on the observation model?
    inv_link_func = obs_model.default_inverse_link_function
    if use_scipy:
        solver = ScipyBoundedMinimize(method="l-bfgs-b", fun=inner_func, tol=tol_optim)
    else:
        solver = LBFGSB(inner_func, tol=tol_optim)
    # make sure everything is float
    reg_strength = jtu.tree_map(lambda x: jnp.asarray(x.astype(float)), reg_strength)
    y = jnp.asarray(y.astype(float))
    X = jtu.tree_map(lambda x: jnp.asarray(x.astype(float)), X)
    init_pars = jtu.tree_map(lambda x: jnp.asarray(x.astype(float)), init_pars)

    lower_bnd = jtu.tree_map(lambda x: jnp.full(x.shape, -12.0), reg_strength)
    upper_bnd = jtu.tree_map(lambda x: jnp.full(x.shape, 25.0), reg_strength)
    bounds = (lower_bnd, upper_bnd)

    def _solve_inner(reg_strength, X_inner, Q, R, y_inner):
        return solver.run(
            reg_strength,
            bounds=bounds,
            penalty_tree=penalty_tree,
            X=X_inner,
            Q=Q,
            R=R,
            y=y_inner,
        )

    # Wrap jaxopt's LBFGSB in our own jit so its `jit(while)` cache hits across
    # outer iterations (without this it recompiled once per iter). Scipy's path
    # cannot be jitted: ScipyBoundedMinimize uses custom_vjp internally and then
    # does np.asarray(...) on its inputs to hand them to host-side Fortran
    # L-BFGS-B; under jit those inputs are tracers and the conversion errors.
    if not use_scipy:
        _solve_inner = jax.jit(_solve_inner)

    link_func = INVERSE_FUNCS[inv_link_func]
    get_pseudo_data_and_weight = model_constructors_for_weights_and_pseudo_data(
        variance_func, link_func, fisher_scoring=fisher_scoring
    )
    # use nemos par struct
    coef, intercept = init_pars
    struct = jtu.tree_structure(X)

    n_obs = jtu.tree_leaves(X)[0].shape[0]
    leaf_shapes = [leaf.shape[1] for leaf in jtu.tree_leaves(X)]  # dims of each leaf

    def loss_unp(p):
        return obs_model._negative_log_likelihood(
            y,
            inv_link_func(X.dot(p[0]) + p[1]),
            aggregate_sample_scores=jnp.sum,
        )

    # TODO: Does this loop need to be converted to a jax loop? Was it even worth it?
    i = 0
    old_inner_score = None
    for i in range(max_iter):
        sqrt_penalty = compute_sqrt(reg_strength)

        # add a zero corresponding to not-penalizing the intercept
        sqrt_penalty = jnp.hstack((jnp.zeros((sqrt_penalty.shape[0], 1)), sqrt_penalty))

        # TODO: Lift this out into an initialization step?
        # initialize coefficients by fitting a GLM
        if i == 0:
            pen = jtu.tree_map(lambda x: x[:, 1:].T.dot(x[:, 1:]), sqrt_penalty)

            def loss(p):
                return loss_unp(p) + 0.5 * p[0].dot(pen).dot(p[0])

            if use_scipy:
                init_solver = ScipyMinimize(method="l-bfgs-b", fun=loss, tol=1e-8)
            else:
                init_solver = LBFGS(loss, tol=1e-8)
            (coef, intercept), state = init_solver.run(init_pars)

        # compute weights
        rate = pytree_map_and_reduce(
            lambda x, c, inter: inv_link_func(x.dot(c) + inter), sum, X, coef, intercept
        )
        pseudo_y, weights = get_pseudo_data_and_weight(y, rate)

        # attach intercept and concatenate
        X_agu = jnp.concatenate(
            [jnp.hstack([jnp.ones((n_obs, 1))] + jtu.tree_leaves(X)), sqrt_penalty],
            axis=0,
        )
        pseudo_y_agu = jnp.concatenate(
            [pseudo_y, jnp.zeros((sqrt_penalty.shape[0], *pseudo_y.shape[1:]))], axis=0
        )
        weights_agu = jnp.concatenate(
            [weights, jnp.ones((sqrt_penalty.shape[0], *weights.shape[1:]))], axis=0
        )

        # Wls coefficients and QR decomposition of weighted X
        coeffs, Xw, yw = weighted_least_squares(X_agu, pseudo_y_agu, weights_agu)
        Q, R = jnp.linalg.qr(Xw[:n_obs], mode="reduced")
        new_coef = jtu.tree_unflatten(struct, unflatten_coeffs(coeffs[1:], leaf_shapes))
        new_intercept = coeffs[0]

        # full optimization for regularizer strength
        new_reg_strength, state = _solve_inner(
            reg_strength,
            Xw[:n_obs],
            Q,
            R,
            yw[:n_obs],
        )
        new_reg_strength = jtu.tree_map(
            lambda x: jnp.clip(x, -25, 30),
            new_reg_strength,
        )
        new_inner_score = state.fun_val if use_scipy else state.value

        # convergence check
        converged = check_pql_convergence(
            convergence_criterion,
            iteration=i,
            tol=tol_update,
            old_params=(coef, intercept),
            new_params=(new_coef, new_intercept),
            old_reg_strength=reg_strength,
            new_reg_strength=new_reg_strength,
            old_score=old_inner_score,
            new_score=new_inner_score,
        )

        # update
        reg_strength = new_reg_strength
        coef, intercept = new_coef, new_intercept
        old_inner_score = new_inner_score

        if converged:
            break

    return (coef, intercept), reg_strength, i

"""Shared test fixtures and utilities."""

import jax
import jax.numpy as jnp
import nemos as nmo
import nemos.observation_models as nmo_obs
import numpy as np
import pytest
from nemos.inverse_link_function_utils import exp
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative

from pgam_jax._identifiable_features import compute_features_identifiable
from pgam_jax._laplace_reml_vbeta import vbeta_and_logdet
from pgam_jax._pirls_weights import _make_w_fn
from pgam_jax.penalty_utils import symmetric_sqrt

jax.config.update("jax_enable_x64", True)


# ─── Numerical derivative ─────────────────────────────────────────────────────


def central_diff(fn, x, **kwargs):
    """Central-difference Jacobian of ``fn`` at ``x`` via scipy's 3-point rule.

    Thin wrapper over ``scipy.optimize._numdiff.approx_derivative`` so we have
    one stable name in the test suite.  For scalar-valued ``fn`` returns a 1-D
    gradient of shape ``(n,)``; for vector-valued ``fn: R^n -> R^m`` returns
    the Jacobian of shape ``(m, n)``.

    Extra kwargs are forwarded to ``approx_derivative`` (e.g. ``rel_step``).
    """
    return approx_derivative(fn, np.asarray(x), method="3-point", **kwargs)


# ─── Penalty builders ─────────────────────────────────────────────────────────


def diff2_penalty(n):
    """n×n second-derivative penalty, rank n-2, null space = {constants, linear}."""
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return jnp.array(D.T @ D, dtype=float)


def gam_design_and_penalty(K_per_smooth, x_covs):
    """B-spline AdditiveBasis design with intercept, and matching diff2 penalties.

    For M smooths each with K basis functions (after identifiability), returns:
        X      : shape (N, 1 + M*K) — [intercept | smooth_1 | smooth_2 | ...]
        S_all  : shape (M, 1+M*K, 1+M*K) — diff2 penalty per smooth, intercept unpenalized

    Mirrors the PGAM reference test geometry (B-splines on uniform covariates,
    leading intercept column, difference-based penalty per smooth).
    """
    M = len(x_covs)
    assert all(len(x) == len(x_covs[0]) for x in x_covs)
    N = len(x_covs[0])

    bases = [nmo.basis.BSplineEval(n_basis_funcs=K_per_smooth + 1, order=4,
                                   bounds=(float(x.min()), float(x.max())))
             for x in x_covs]
    basis = bases[0]
    for b in bases[1:]:
        basis = basis + b   # AdditiveBasis

    X_smooth = np.asarray(compute_features_identifiable(
        basis, *x_covs, drop_conv_basis_col=False
    ))
    assert X_smooth.shape == (N, M * K_per_smooth), X_smooth.shape

    # Sum-to-zero (mean-center) each smooth's columns so they are orthogonal to
    # the intercept.  This is the same identifiability transform PGAM applies
    # in additive_model_preprocessing; without it H + S_λ/φ is poorly conditioned
    # and the FD-vs-analytic floor inflates roughly 4x.
    for k in range(M):
        s = k * K_per_smooth
        e = s + K_per_smooth
        X_smooth[:, s:e] -= X_smooth[:, s:e].mean(axis=0, keepdims=True)

    P = 1 + M * K_per_smooth
    X = jnp.asarray(np.hstack([np.ones((N, 1)), X_smooth]))

    S_block = np.asarray(diff2_penalty(K_per_smooth))
    S_all = np.zeros((M, P, P))
    for k in range(M):
        s = 1 + k * K_per_smooth
        e = s + K_per_smooth
        S_all[k, s:e, s:e] = S_block
    return X, jnp.asarray(S_all)


# ─── Penalized MAP optimization ───────────────────────────────────────────────


def fit_beta(X, y, obs_model, inv_link, S_all, rho, phi, beta0=None):
    """Minimize the penalized negative log-likelihood wrt beta.

        loss(beta) = -log L(beta) + 0.5/phi * beta^T S_lambda beta

    Uses scipy trust-ncg with JAX-supplied gradient and Hessian — converges
    to ~machine precision, which is required for finite-difference checks
    of d beta_hat / d rho to hit rtol≈1e-3.
    """
    S_lam = jnp.einsum("kij,k->ij", S_all, jnp.exp(rho))

    def loss_fn(b):
        return (
            -obs_model.log_likelihood(y, inv_link(X @ b), aggregate_sample_scores=jnp.sum)
            + 0.5 * jnp.dot(b, S_lam @ b) / phi
        )

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    hess = jax.jit(jax.hessian(loss_fn))

    def scipy_fun(b_np):
        val, grad = value_and_grad(jnp.asarray(b_np))
        return float(val), np.asarray(grad)

    def scipy_hess(b_np):
        return np.asarray(hess(jnp.asarray(b_np)))

    if beta0 is None:
        beta0 = np.zeros(X.shape[1])
    result = minimize(
        fun=scipy_fun, x0=np.asarray(beta0), jac=True, hess=scipy_hess,
        method="Newton-CG", tol=1e-14, options={"maxiter": 10000},
    )
    return jnp.asarray(result.x)


def make_vbeta(beta_hat, X, y, obs_model, inv_link, S_all, rho, phi):
    """V_beta, V_beta_inv, log|H+S_lam/phi| from observed Hessian weights at beta_hat."""
    w = _make_w_fn(y, obs_model, inv_link)(X @ beta_hat)
    _, R = jnp.linalg.qr(jnp.sqrt(w)[:, None] * X, mode="reduced")
    sqrt_pen = symmetric_sqrt(jnp.einsum("kij,k->ij", S_all, jnp.exp(rho)))
    return vbeta_and_logdet(R, sqrt_pen, phi)


# ─── Small GAM problems for Laplace-REML tests ────────────────────────────────
#
# Mirrors the PGAM reference test geometry (test_derivatives.py::gam_problem):
# two B-spline smooths on uniform [0, 10] covariates, intercept=0.5,
# normal(0, 0.3) smooth coefficients.  Well-conditioned, so the FD-vs-analytic
# precision floor stays well below the J magnitudes.

_N = 60
_K = 7  # basis functions per smooth after identifiability constraint


def _build_gam_problem(rng_seed, obs_model, y_draw, phi=1.0):
    rng = np.random.default_rng(rng_seed)
    x1 = rng.uniform(0, 10, _N)
    x2 = rng.uniform(0, 10, _N)
    X, S_all = gam_design_and_penalty(_K, [x1, x2])

    P = X.shape[1]
    beta_true = rng.normal(0, 0.3, P)
    beta_true[0] = 0.5  # intercept
    y = jnp.asarray(y_draw(rng, np.array(exp(X @ beta_true))))

    rho = jnp.array([0.5, 0.5])
    beta_hat = fit_beta(X, y, obs_model, exp, S_all, rho, phi)
    V_beta, V_beta_inv, log_det_HpS = make_vbeta(
        beta_hat, X, y, obs_model, exp, S_all, rho, phi
    )
    # M_null = unpenalized intercept (1) + diff2 null space (2) per smooth.
    # Analytical for this fixture; avoids relying on numerical matrix_rank.
    M_null = 1 + 2 * S_all.shape[0]
    return dict(
        X=X, y=y, obs=obs_model, inv_link=exp,
        S_all=S_all, rho=rho, phi=phi,
        beta_hat=beta_hat, V_beta=V_beta, V_beta_inv=V_beta_inv,
        log_det_HpS=log_det_HpS, M_null=M_null,
    )


@pytest.fixture(scope="module")
def poisson_gam_problem():
    return _build_gam_problem(
        rng_seed=7,
        obs_model=nmo_obs.PoissonObservations(),
        y_draw=lambda rng, mu: rng.poisson(mu).astype(float),
    )


@pytest.fixture(scope="module")
def gamma_gam_problem():
    return _build_gam_problem(
        rng_seed=7,
        obs_model=nmo_obs.GammaObservations(),
        y_draw=lambda rng, mu: rng.gamma(2.0, mu / 2.0),
    )
"""Shared test fixtures and utilities."""

import jax
import jax.numpy as jnp
import nemos as nmo
import nemos.observation_models as nmo_obs
import numpy as np
import pytest
from nemos.inverse_link_function_utils import exp
from scipy.optimize._numdiff import approx_derivative

from pgam_jax._identifiable_features import compute_features_identifiable
from pgam_jax._laplace_reml_fit import fit_beta  # noqa: F401 — re-exported for tests
from pgam_jax._laplace_reml_vbeta import vbeta_and_logdet
from pgam_jax._penalty_handler import PenaltyHandler
from pgam_jax._pirls_weights import _make_w_fn
from pgam_jax.penalty_utils import IDENTITY

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
    """B-spline AdditiveBasis design with intercept, with both flat and structured penalties.

    Returns
    -------
    X : (N, 1 + M*K)
        [intercept | smooth_1 | smooth_2 | ...].
    S_all : (M, 1+M*K, 1+M*K)
        Stack of full-coef-space penalty matrices (intercept row/col zero) —
        used by the derivative formulas (dbeta_hat, dH_drho, add1, tr_VS).
    compute_sqrt, compute_log_det_and_grad : callables
        From ``PenaltyHandler.build()``.  These provide the Wood-2011 stable
        sqrt(S_λ) and log|S_λ|_+ — required because the naïve
        ``symmetric_sqrt(Σ_k λ_k S_k)`` loses small-λ modes when the λ_k span
        different scales.  ``compute_sqrt`` returns the (rows, M*K) sqrt over
        the smooth coefs only — the caller is responsible for prepending a
        zero column for the unpenalized intercept.
    """
    M = len(x_covs)
    assert all(len(x) == len(x_covs[0]) for x in x_covs)
    N = len(x_covs[0])

    bases = [
        nmo.basis.BSplineEval(
            n_basis_funcs=K_per_smooth + 1,
            order=4,
            bounds=(float(x.min()), float(x.max())),
        )
        for x in x_covs
    ]
    basis = bases[0]
    for b in bases[1:]:
        basis = basis + b  # AdditiveBasis

    X_smooth = np.asarray(
        compute_features_identifiable(basis, *x_covs, drop_conv_basis_col=False)
    )
    assert X_smooth.shape == (N, M * K_per_smooth), X_smooth.shape

    # Sum-to-zero (mean-center) each smooth's columns so they are orthogonal to
    # the intercept.  Same identifiability transform PGAM applies in
    # additive_model_preprocessing; without it H + S_λ/φ is poorly conditioned
    # and the FD-vs-analytic floor inflates ~4x.
    for k in range(M):
        s = k * K_per_smooth
        e = s + K_per_smooth
        X_smooth[:, s:e] -= X_smooth[:, s:e].mean(axis=0, keepdims=True)

    P = 1 + M * K_per_smooth
    X = jnp.asarray(np.hstack([np.ones((N, 1)), X_smooth]))

    S_block_small = diff2_penalty(K_per_smooth)
    ph = PenaltyHandler()
    for _ in range(M):
        # TODO: Add an identifiability_fn here
        ph.add(S_block_small, penalize_null_space=False, identifiability_fn=IDENTITY)
    compute_sqrt, compute_log_det_and_grad = ph.build()

    S_all = np.zeros((M, P, P))
    for k in range(M):
        s = 1 + k * K_per_smooth
        e = s + K_per_smooth
        S_all[k, s:e, s:e] = np.asarray(S_block_small)
    return jnp.asarray(X), jnp.asarray(S_all), compute_sqrt, compute_log_det_and_grad


# ─── Penalized MAP optimization ───────────────────────────────────────────────
# fit_beta is the production inner MAP solver, imported above from
# pgam_jax._laplace_reml_fit so tests and production share one implementation.


def make_vbeta(beta_hat, X, y, obs_model, inv_link, compute_sqrt, rhos_tree, phi):
    """V_beta, V_beta_inv, log|H+S_lam/phi| using PenaltyHandler's stable sqrt.

    ``compute_sqrt`` and ``rhos_tree`` come from a built PenaltyHandler — they
    encode the Wood-2011 stable sqrt(Σ_k λ_k S_k) construction.  The result is
    prepended with a zero column for the unpenalized intercept, matching the
    convention used in _pql_reml.py.
    """
    w = _make_w_fn(y, obs_model, inv_link)(X @ beta_hat)
    R = jnp.linalg.qr(jnp.sqrt(w)[:, None] * X, mode="r")  # skip DORGQR; Q unused
    sqrt_pen = compute_sqrt(rhos_tree)
    sqrt_pen = jnp.hstack((jnp.zeros((sqrt_pen.shape[0], 1)), sqrt_pen))
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
    X, S_all, compute_sqrt, compute_log_det_and_grad = gam_design_and_penalty(
        _K, [x1, x2]
    )

    P = X.shape[1]
    M = S_all.shape[0]
    beta_true = rng.normal(0, 0.3, P)
    beta_true[0] = 0.5  # intercept
    y = jnp.asarray(y_draw(rng, np.array(exp(X @ beta_true))))

    rho_flat = jnp.array([0.5, 0.5])
    rhos_tree = [rho_flat[k : k + 1] for k in range(M)]  # one (1,) array per smooth
    beta_hat = fit_beta(X, y, obs_model, exp, S_all, rho_flat, phi)
    V_beta, V_beta_inv, log_det_HpS = make_vbeta(
        beta_hat, X, y, obs_model, exp, compute_sqrt, rhos_tree, phi
    )
    # M_null = unpenalized intercept (1) + diff2 null space (2) per smooth.
    # Analytical for this fixture; avoids relying on numerical matrix_rank.
    M_null = 1 + 2 * M
    return dict(
        X=X,
        y=y,
        obs=obs_model,
        inv_link=exp,
        S_all=S_all,
        rho=rho_flat,
        rhos_tree=rhos_tree,
        phi=phi,
        compute_sqrt=compute_sqrt,
        compute_log_det_and_grad=compute_log_det_and_grad,
        beta_hat=beta_hat,
        V_beta=V_beta,
        V_beta_inv=V_beta_inv,
        log_det_HpS=log_det_HpS,
        M_null=M_null,
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


@pytest.fixture(scope="module")
def gamma_gam_problem_phi2():
    """
    Same geometry as ``gamma_gam_problem`` but with dispersion ``phi=2``.

    Exercises the phi != 1 path of the Laplace-REML formulas.
    The phi=1 fixtures cannot catch a dispersion-scaling error because every
    1/phi factor collapses to 1 there.
    """
    return _build_gam_problem(
        rng_seed=7,
        obs_model=nmo_obs.GammaObservations(),
        y_draw=lambda rng, mu: rng.gamma(2.0, mu / 2.0),
        phi=2.0,
    )

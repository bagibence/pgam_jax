import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest
from nemos.observation_models import GammaObservations

from pgam_jax import GAM

jax.config.update("jax_enable_x64", True)


def _basis():
    return nmo.basis.BSplineEval(n_basis_funcs=10, order=4, bounds=(-1.0, 1.0))


def _additive_basis():
    return nmo.basis.BSplineEval(
        n_basis_funcs=8, order=4, bounds=(0.0, 10.0)
    ) + nmo.basis.BSplineEval(n_basis_funcs=8, order=4, bounds=(0.0, 10.0))


@pytest.mark.parametrize("method", ["pql_gcv", "pql_reml", "laplace_reml"])
def test_valid_method_does_not_raise(method):
    GAM(_basis(), method=method)


# Old names ("gcv", "reml") are now invalid — the rename is a hard break.
@pytest.mark.parametrize(
    "bad_method", ["gcv", "reml", "GCV", "REML", "ml", "", "bad_value"]
)
def test_invalid_method_raises(bad_method):
    with pytest.raises(ValueError, match="method must be one of"):
        GAM(_basis(), method=bad_method)


def test_laplace_reml_requires_poisson():
    """laplace_reml with a non-Poisson family raises with the phi-coupling note."""
    with pytest.raises(NotImplementedError, match="laplace_reml"):
        GAM(
            _basis(),
            observation_model=GammaObservations(),
            method="laplace_reml",
        )


def test_laplace_reml_fit_smoke():
    """GAM(method='laplace_reml') fits synthetic Poisson data end-to-end."""
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    eta = 0.5 + np.sin(0.6 * x1) + 0.3 * np.cos(0.4 * x2)
    y = rng.poisson(np.exp(eta)).astype(float)

    gam = GAM(_additive_basis(), method="laplace_reml")
    gam.fit((x1, x2), y)

    assert np.all(np.isfinite(gam.coef_))
    assert np.all(np.isfinite(np.asarray(gam.intercept_)))
    for r in gam.regularizer_strength_:
        assert np.all(np.isfinite(np.asarray(r)))
    assert gam.n_iter_ >= 0
    assert np.all(np.isfinite(gam.cov_beta_))
    assert np.isfinite(gam.scale_)


def _poisson_data(seed=0, n=500):
    """Single smooth Poisson regression problem with a clear signal."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    eta = np.sin(3.0 * x)
    y = rng.poisson(np.exp(eta - eta.mean()))
    return x, y, eta


def test_compute_cov_beta_edf_matches_u1_calculation():
    gam = GAM(_basis())
    X = jnp.array(
        [
            [-1.0, 0.2, 0.5],
            [-0.6, -0.3, 0.8],
            [-0.2, 0.7, -0.4],
            [0.2, -0.5, -0.7],
            [0.6, 0.4, 0.1],
            [1.0, -0.1, -0.6],
        ]
    )
    y = jnp.ones(X.shape[0])
    params = (jnp.zeros(X.shape[1]), jnp.array([0.0]))
    sqrt_penalty = jnp.array(
        [
            [0.8, -0.2, 0.1],
            [0.0, 0.6, 0.3],
            [0.0, 0.0, 1.1],
        ]
    )

    _, _ = gam._compute_cov_beta_from_fit_state(
        X,
        y,
        params,
        [jnp.array([0.0])],
        lambda _: sqrt_penalty,
    )
    edf1_from_F = jnp.sum(gam._edf1_by_coef)
    edf_from_F = jnp.sum(gam._edf_by_coef)

    X_full = jnp.column_stack((jnp.ones(X.shape[0]), X))
    R = jnp.linalg.qr(X_full, mode="r")
    sqrt_penalty_full = jnp.column_stack(
        (jnp.zeros(sqrt_penalty.shape[0]), sqrt_penalty)
    )
    U, _, _ = jnp.linalg.svd(
        jnp.vstack((R, sqrt_penalty_full)),
        full_matrices=False,
    )

    # U1 = U[:k] (first k=R.shape[0] rows) encodes the hat matrix via A = Q_xw U1 U1' Q_xw'
    U1 = U[: R.shape[0]]

    # EDF: edf1 = 2·tr(F) − tr(F²) where F = (X'WX + S_λ)⁻¹ X'WX
    # Expressed via U1: tr(F) = ‖U1‖²_F, tr(F²) = ‖U1'U1‖²_F (Wood 2017 eq. 6.13)
    edf_from_U1 = jnp.sum(U1**2)
    edf1_from_U1 = 2 * edf_from_U1 - jnp.sum((U1.T @ U1) ** 2)

    np.testing.assert_allclose(edf_from_F, edf_from_U1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(edf1_from_F, edf1_from_U1, rtol=1e-12, atol=1e-12)


def test_smooth_significance_requires_fitted_model():
    gam = GAM(_basis())

    with pytest.raises(AttributeError, match="not fitted"):
        gam.test_smooth_significance(0)


def test_smooth_significance_delegates_and_warns(monkeypatch):
    gam = GAM(_basis())
    gam.cov_beta_ = jnp.eye(1)
    selected_components = []

    def compute_p_value(component_index):
        selected_components.append(component_index)
        return 0.25

    monkeypatch.setattr(
        gam,
        "_smooth_pval_unpenalized",
        compute_p_value,
    )

    with pytest.warns(
        UserWarning,
        match="approximate.*High concurvity",
    ):
        p_value = gam.test_smooth_significance("BSplineEval")

    assert p_value == 0.25
    assert selected_components == ["BSplineEval"]


def test_null_smooth_is_not_significant():
    """Regression test based on the smooth p-value simulation."""
    rng = np.random.default_rng(np.random.SeedSequence([20260817, 0, 504]))
    n_samples = 300
    x = rng.uniform(-1.0, 1.0, n_samples)
    eta = np.full(n_samples, 0.2)
    y = rng.poisson(np.exp(eta)).astype(float)

    basis = nmo.basis.BSplineEval(
        n_basis_funcs=8,
        order=4,
        bounds=(-1.0, 1.0),
    )
    gam = GAM(
        basis,
        method="pql_reml",
        maxiter=50,
    )
    gam.fit((x,), y)

    with pytest.warns(UserWarning, match="approximate"):
        p_value_by_index = gam.test_smooth_significance(0)

    with pytest.warns(UserWarning, match="approximate"):
        p_value_by_label = gam.test_smooth_significance("BSplineEval")

    assert p_value_by_index == pytest.approx(p_value_by_label)
    assert 0.1 < p_value_by_index <= 1.0


@pytest.mark.parametrize("method", ["pql_gcv", "pql_reml"])
def test_fit_runs_end_to_end(method):
    """End-to-end fit for each smoothing-parameter method, including REML."""
    x, y, eta = _poisson_data()
    basis = nmo.basis.BSplineEval(n_basis_funcs=8, order=4, bounds=(-1.0, 1.0))
    gam = GAM(basis, method=method, maxiter=20)

    gam.fit((x,), y)

    # the identifiability constraint drops one column, so 8 -> 7 coefficients
    assert gam.coef_.shape == (7,)
    assert np.all(np.isfinite(np.asarray(gam.coef_)))
    assert np.all(np.isfinite(np.asarray(gam.intercept_)))
    assert all(np.all(np.isfinite(np.asarray(r))) for r in gam.regularizer_strength_)

    pred = np.asarray(gam.predict((x,)))
    assert pred.shape == (len(y),)
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0)  # Poisson mean

    # sanity: the fitted smooth tracks the true signal rather than being flat
    assert np.corrcoef(pred, np.exp(eta))[0, 1] > 0.5


def test_initialize_params_warm_starts_intercept_at_link_of_mean():
    """coef starts at 0 and the intercept at link(mean y), not 0.

    Regression: a zero intercept starts a sparse-count model far from the
    optimum, and the GLM warm-start solver could overshoot into the flat,
    rate-floored tail of the Poisson NLL and stall at a garbage intercept
    (``exp(eta) = 0`` -> NaNs downstream).
    """
    model = GAM(_basis())  # PoissonObservations, log link
    rng = np.random.default_rng(0)
    y = rng.poisson(0.03, size=3000).astype(float)
    X = rng.standard_normal((3000, 7))

    coef, intercept = model.initialize_params(X, y)

    assert coef.shape == (7,)
    assert np.all(np.asarray(coef) == 0.0)
    np.testing.assert_allclose(np.asarray(intercept), np.log(y.mean()), rtol=1e-6)


def test_initialize_params_raises_for_degenerate_response():
    """All-zero counts -> log(mean) = -inf -> raise ValueError"""
    model = GAM(_basis())
    y = np.zeros(500)
    X = np.zeros((500, 5))

    with pytest.raises(ValueError, match="Failed to initialize"):
        _, intercept = model.initialize_params(X, y)


def test_use_glm_init_reaches_same_solution():
    """Regardless of use_glm_init, the converged fit must converge to the same result."""
    x, y, _ = _poisson_data()
    basis = nmo.basis.BSplineEval(n_basis_funcs=8, order=4, bounds=(-1.0, 1.0))

    coefs = {}
    for use_glm_init in (True, False):
        gam = GAM(basis, method="pql_gcv", maxiter=50, use_glm_init=use_glm_init)
        gam.fit((x,), y)
        coefs[use_glm_init] = np.asarray(gam.coef_)

    np.testing.assert_allclose(coefs[True], coefs[False], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("method", ["pql_gcv", "pql_reml"])
def test_tensor_product_fit_runs_end_to_end(method):
    """End-to-end tensor-product fit through the KRONECKER_WITH_NULL penalty route."""
    rng = np.random.default_rng(1)
    n = 350
    x1 = rng.uniform(-1.0, 1.0, size=n)
    x2 = rng.uniform(-1.0, 1.0, size=n)
    eta = 0.8 * np.sin(2.0 * x1) + 0.5 * np.cos(2.5 * x2)
    y = rng.poisson(np.exp(eta - eta.mean()))
    basis = nmo.basis.BSplineEval(
        n_basis_funcs=6, order=4, bounds=(-1.0, 1.0)
    ) * nmo.basis.BSplineEval(n_basis_funcs=5, order=4, bounds=(-1.0, 1.0))
    gam = GAM(basis, method=method, maxiter=10)

    gam.fit((x1, x2), y)

    # the identifiability constraint drops one column from the full tensor product
    assert gam.coef_.shape == (6 * 5 - 1,)
    assert len(gam.regularizer_strength_) == 1
    assert gam.regularizer_strength_[0].shape == (3,)
    assert np.all(np.isfinite(np.asarray(gam.coef_)))
    assert np.all(np.isfinite(np.asarray(gam.intercept_)))
    assert np.all(np.isfinite(np.asarray(gam.regularizer_strength_[0])))

    pred = np.asarray(gam.predict((x1, x2)))
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0)

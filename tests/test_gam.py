import jax
import nemos as nmo
import numpy as np
import pytest

from pgam_jax import GAM

jax.config.update("jax_enable_x64", True)


def _basis():
    return nmo.basis.BSplineEval(n_basis_funcs=10, order=4, bounds=(-1.0, 1.0))


@pytest.mark.parametrize("method", ["gcv", "reml"])
def test_valid_method_does_not_raise(method):
    GAM(_basis(), method=method)


@pytest.mark.parametrize("bad_method", ["GCV", "REML", "ml", "", "bad_value"])
def test_invalid_method_raises(bad_method):
    with pytest.raises(ValueError, match=r'method must be one of \["gcv", "reml"\]'):
        GAM(_basis(), method=bad_method)


def _poisson_data(seed=0, n=500):
    """Single smooth Poisson regression problem with a clear signal."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    eta = np.sin(3.0 * x)
    y = rng.poisson(np.exp(eta - eta.mean()))
    return x, y, eta


@pytest.mark.parametrize("method", ["gcv", "reml"])
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


@pytest.mark.parametrize("method", ["gcv", "reml"])
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

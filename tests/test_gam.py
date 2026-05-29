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

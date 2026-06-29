import nemos as nmo
import numpy as np
import pytest
from nemos.observation_models import GammaObservations

from pgam_jax import GAM


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

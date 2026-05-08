import nemos as nmo
import pytest

from pgam_jax import GAM


def _basis():
    return nmo.basis.BSplineEval(n_basis_funcs=10, order=4, bounds=(-1.0, 1.0))


@pytest.mark.parametrize("method", ["gcv", "reml"])
def test_valid_method_does_not_raise(method):
    GAM(_basis(), method=method)


@pytest.mark.parametrize("bad_method", ["GCV", "REML", "ml", "", "bad_value"])
def test_invalid_method_raises(bad_method):
    with pytest.raises(ValueError, match=r'method must be one of \["gcv", "reml"\]'):
        GAM(_basis(), method=bad_method)

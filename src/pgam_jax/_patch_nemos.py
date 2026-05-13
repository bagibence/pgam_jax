import nemos as nmo
import numpy as np
from nemos.basis._basis import check_transform_input, min_max_rescale_samples
from nemos.basis._spline_basis import bspline
from nemos.type_casting import support_pynapple
from nemos.utils import row_wise_kron
from numpy.typing import ArrayLike

from ._nemos_compat import get_n_inputs


@support_pynapple(conv_type="numpy")
@check_transform_input
def _bspline_derivative(self, sample_pts: np.ndarray, der: int = 2):
    """
    Compute the basis derivative.

    Parameters
    ----------
    sample_pts:
        Sample points over which computing the derivative.
    der:
        Order of the derivative.

    Returns
    -------
        The derivative at the sample points.
    """
    bounds = getattr(self, "bounds", None)
    sample_pts, _ = min_max_rescale_samples(sample_pts, bounds)
    knot_locs = self._generate_knots(is_cyclic=False)
    shape = sample_pts.shape
    X = bspline(sample_pts, knot_locs, order=self.order, der=der, outer_ok=False)
    X = X.reshape(*shape, X.shape[1])
    if bounds is not None:
        scale = 1 / (bounds[1] - bounds[0])
        X = X * scale**der
    return X


@support_pynapple("numpy")
def _additive_derivative(self, *xi: ArrayLike):
    n1 = get_n_inputs(self.basis1)
    return np.hstack(
        self.basis1.derivative(*xi[:n1]),
        self.basis2.derivative(*xi[n1:]),
    )


def _multiplicative_derivative(self, *xi: ArrayLike):
    kron = support_pynapple(conv_type="numpy")(row_wise_kron)

    n1 = get_n_inputs(self.basis1)
    return kron(
        self.basis1.derivative(*xi[:n1]),
        self.basis2.derivative(*xi[n1:]),
        transpose=False,
    )


# apply_identifiability is not required where this is used
nmo.basis.BSplineEval.derivative = _bspline_derivative
nmo.basis.BSplineConv.derivative = _bspline_derivative
nmo.basis.AdditiveBasis.derivative = _additive_derivative
nmo.basis.MultiplicativeBasis.derivative = _multiplicative_derivative

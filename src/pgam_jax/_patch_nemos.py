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
        Sample points at which to compute the derivative.
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


@support_pynapple(conv_type="numpy")
@check_transform_input
def _cyclic_bspline_derivative(self, sample_pts: ArrayLike, der: int = 2):
    """
    Compute the basis derivative.

    Parameters
    ----------
    sample_pts :
        Sample points at which to compute the derivative.
    der:
        Order of the derivative.

    Returns
    -------
        The derivative at the sample points.

    Notes
    -----
    Evaluation uses NumPy and SciPy outside JIT. Concrete JAX arrays are
    accepted, but JAX transformations of this method are not supported.
    """
    if not 0 <= der < self.order:
        raise ValueError("Require 0 <= der < spline order")

    bounds = getattr(self, "bounds", None)

    # TODO: When integrating into nemos, support JAX tracing like .evaluate() does on development
    sample_pts, _ = min_max_rescale_samples(sample_pts, bounds, use_jax=False)
    original_shape = sample_pts.shape
    sample_pts = sample_pts.reshape(-1)

    # for cyclic, do not repeat knots
    knot_locs = np.unique(self._generate_knots(is_cyclic=True))
    # make sure knots are sorted
    knot_locs.sort()
    nk = knot_locs.size

    # extend knots
    xc = knot_locs[nk - self.order]
    knots = np.hstack(
        (
            knot_locs[0] - knot_locs[-1] + knot_locs[nk - self.order : nk - 1],
            knot_locs,
        )
    )

    right_idx = sample_pts > xc
    deriv_eval = bspline(sample_pts, knots, order=self.order, der=der, outer_ok=True)
    wrapped_pts = sample_pts - knots.max() + knot_locs[0]

    if np.any(right_idx):
        deriv_eval[right_idx] += bspline(
            wrapped_pts[right_idx], knots, order=self.order, der=der, outer_ok=True
        )

    deriv_eval = deriv_eval.reshape(*original_shape, deriv_eval.shape[1])

    if bounds is not None:
        scale = 1 / (bounds[1] - bounds[0])
        deriv_eval *= scale**der

    return deriv_eval


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
nmo.basis.CyclicBSplineEval.derivative = _cyclic_bspline_derivative
nmo.basis.CyclicBSplineConv.derivative = _cyclic_bspline_derivative
nmo.basis.AdditiveBasis.derivative = _additive_derivative
nmo.basis.MultiplicativeBasis.derivative = _multiplicative_derivative

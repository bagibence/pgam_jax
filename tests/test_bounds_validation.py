"""Regression tests for the explicit-bounds requirement on eval-mode bases.

1. Without explicit bounds, nemos rescales each input array to [0, 1] using
   per-batch min/max, so the same physical x maps to different normalized
   coordinates across fit and predict batches.

2. The chain-rule correction 1/(b-a)^r in the patched BSplineEval.derivative
   is only applied when bounds is not None, so penalty derivatives are in
   normalized-coordinate units instead of physical units when bounds is absent.
   So we require evaluation bases to have explicit bounds.
"""

import nemos as nmo
import numpy as np
import pytest

from pgam_jax import GAM

# ---------------------------------------------------------------------------
# 1. Validator: GAM.__init__ must reject eval bases without bounds
# ---------------------------------------------------------------------------


def test_validator_raises_for_eval_basis_without_bounds():
    basis = nmo.basis.BSplineEval(n_basis_funcs=10, order=4)
    with pytest.raises(ValueError, match="bounds"):
        GAM(basis)


def test_validator_raises_for_composite_with_missing_bounds():
    b1 = nmo.basis.BSplineEval(n_basis_funcs=10, order=4, bounds=(-1.0, 1.0))
    b2 = nmo.basis.BSplineEval(n_basis_funcs=10, order=4)  # missing bounds
    with pytest.raises(ValueError, match="bounds"):
        GAM(b1 + b2)


def test_validator_raises_for_multiplicative_with_missing_bounds():
    b1 = nmo.basis.BSplineEval(n_basis_funcs=10, order=4, bounds=(-1.0, 1.0))
    b2 = nmo.basis.BSplineEval(n_basis_funcs=10, order=4)  # missing bounds
    with pytest.raises(ValueError, match="bounds"):
        GAM(b1 * b2)


def test_validator_does_not_raise_for_conv_basis():
    # BSplineConv has no bounds attribute and must never trigger the validator
    basis = nmo.basis.BSplineConv(n_basis_funcs=10, window_size=51)
    GAM(basis)


def test_validator_does_not_raise_when_all_eval_bases_have_bounds():
    b1 = nmo.basis.BSplineEval(n_basis_funcs=10, order=4, bounds=(-1.0, 1.0))
    b2 = nmo.basis.BSplineEval(n_basis_funcs=8, order=4, bounds=(0.0, 5.0))
    GAM(b1 + b2)


# ---------------------------------------------------------------------------
# 2. Derivative chain-rule scaling: d^r B / dx^r = d^r B / du^r * (1/(b-a))^r
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("der", [1, 2])
def test_bspline_derivative_scales_with_bounds(der):
    """Patched derivative must apply the 1/(b-a)^r chain-rule correction.

    Bases A (bounds=(0,1)) and B (bounds=(0,L)) evaluated at physically
    corresponding points u in [0,1] and L*u in [0,L] must satisfy:
        B.derivative(L*u) == A.derivative(u) * (1/L)**der
    """
    import pgam_jax  # noqa: F401 — ensures the monkey-patch is applied

    L = 10.0
    n = 50
    u = np.linspace(0.05, 0.95, n)

    basis_a = nmo.basis.BSplineEval(n_basis_funcs=12, order=4, bounds=(0.0, 1.0))
    basis_b = nmo.basis.BSplineEval(n_basis_funcs=12, order=4, bounds=(0.0, L))

    da = basis_a.derivative(u, der=der)
    db = basis_b.derivative(L * u, der=der)

    np.testing.assert_allclose(db, da * (1.0 / L) ** der, rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. Predict-batch invariance: same physical x must predict the same value
#    regardless of what other points are in the batch
# ---------------------------------------------------------------------------


def test_predict_is_batch_invariant():
    """Prediction at a given x must not depend on other points in the batch.

    This catches the smooth-offset bug: without explicit bounds, nemos rescales
    each input array by its own min/max, so a singleton [0.0] and a linspace
    that contains 0 map 0 to different normalized coordinates and produce
    different predictions.
    """
    rng = np.random.default_rng(42)
    n = 300
    x = np.linspace(-3.0, 3.0, n)
    y = rng.poisson(0.1, size=n)

    basis = nmo.basis.BSplineEval(n_basis_funcs=10, order=4, bounds=(-5.0, 5.0))
    gam = GAM(basis, use_scipy=True, maxiter=3)
    gam.fit((x,), y)

    # predict on the full batch
    pred_full = np.asarray(gam.predict((x,)))

    # predict each point individually and compare
    for i in range(0, n, 30):
        x_single = np.array([x[i]])
        pred_single = np.asarray(gam.predict((x_single,))).item()
        np.testing.assert_allclose(
            pred_single,
            pred_full[i],
            rtol=1e-5,
            err_msg=f"predict at x={x[i]:.3f} differs between singleton and full batch",
        )

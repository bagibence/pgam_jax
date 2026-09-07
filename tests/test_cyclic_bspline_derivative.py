"""Numerical checks for cyclic spline derivatives and their energy penalties."""

import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest

from pgam_jax import GAM
from pgam_jax.penalty_utils import compute_energy_penalty_factors


@pytest.mark.parametrize("order", [3, 4, 5])
def test_derivative_order_zero_matches_evaluation(order):
    basis = nmo.basis.CyclicBSplineEval(10, order=order, bounds=(-np.pi, np.pi))
    x = np.linspace(-np.pi, np.pi, 1001)

    np.testing.assert_allclose(
        basis.derivative(x, der=0), basis.evaluate(x), atol=1e-12
    )


@pytest.mark.parametrize("order", [3, 4, 5])
@pytest.mark.parametrize("der", [1, 2])
def test_derivative_matches_finite_differences(order, der):
    basis = nmo.basis.CyclicBSplineEval(10, order=order, bounds=(-np.pi, np.pi))
    # Sample inside every knot interval. The highest derivative can jump at knots.
    u = (np.arange(10)[:, None] + np.array([0.2, 0.4, 0.6, 0.8])) / 10
    x = -np.pi + 2 * np.pi * u.ravel()
    h = 1e-4
    if der == 1:
        expected = (basis.evaluate(x + h) - basis.evaluate(x - h)) / (2 * h)
    elif der == 2:
        expected = (
            basis.evaluate(x + h) - 2 * basis.evaluate(x) + basis.evaluate(x - h)
        ) / h**2
    else:
        raise NotImplementedError(f"Unsupported derivative order: {der}")

    np.testing.assert_allclose(
        basis.derivative(x, der=der), expected, atol=5e-7, rtol=1e-6
    )


@pytest.mark.parametrize("der", [0, 1, 2])
def test_derivative_scales_with_bounds(der):
    unit_basis = nmo.basis.CyclicBSplineEval(10, bounds=(0.0, 1.0))
    scaled_basis = nmo.basis.CyclicBSplineEval(10, bounds=(-3.0, 7.0))
    u = np.linspace(0.0, 1.0, 101)

    np.testing.assert_allclose(
        scaled_basis.derivative(-3.0 + 10.0 * u, der=der),
        unit_basis.derivative(u, der=der) / 10.0**der,
        atol=1e-12,
        rtol=1e-10,
    )


@pytest.mark.parametrize("order", [3, 4, 5])
def test_continuous_derivatives_agree_at_periodic_endpoints(order):
    basis = nmo.basis.CyclicBSplineEval(10, order=order, bounds=(-np.pi, np.pi))
    # An order-k spline is continuous through derivative order k-2.
    for der in range(order - 1):
        endpoints = basis.derivative(np.array([-np.pi, np.pi]), der=der)
        np.testing.assert_allclose(endpoints[0], endpoints[1], atol=1e-10)


def test_derivative_preserves_multidimensional_input_shape():
    basis = nmo.basis.CyclicBSplineEval(10, bounds=(-np.pi, np.pi))
    x = np.linspace(-np.pi, np.pi, 24).reshape(4, 3, 2)

    actual = basis.derivative(x)

    assert actual.shape == (4, 3, 2, 10)
    np.testing.assert_allclose(actual.reshape(24, 10), basis.derivative(x.ravel()))


@pytest.mark.parametrize("order", [3, 4, 5])
def test_energy_penalty_is_positive_semidefinite_with_constant_null_space(order):
    basis = nmo.basis.CyclicBSplineEval(10, order=order, bounds=(-np.pi, np.pi))

    (penalty,) = compute_energy_penalty_factors(basis, n_simpson_samples=1001)
    penalty = np.asarray(penalty)

    assert penalty.shape == (10, 10)
    assert np.isfinite(penalty).all()
    np.testing.assert_allclose(penalty, penalty.T, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(penalty)
    assert eigenvalues[0] >= -1e-10
    assert np.count_nonzero(np.abs(eigenvalues) < 1e-10) == 1
    np.testing.assert_allclose(penalty @ np.ones(10), 0.0, atol=1e-10)


def test_convolution_energy_penalty_matches_unit_interval_evaluation():
    conv_basis = nmo.basis.CyclicBSplineConv(10, window_size=51)
    eval_basis = nmo.basis.CyclicBSplineEval(10, bounds=(0.0, 1.0))
    expected = compute_energy_penalty_factors(eval_basis, n_simpson_samples=1001)

    actual = compute_energy_penalty_factors(conv_basis, n_simpson_samples=1001)

    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("der", [0, 1, 2])
def test_eager_jax_input_matches_numpy_input(der):
    basis = nmo.basis.CyclicBSplineEval(10, bounds=(-np.pi, np.pi))
    x = np.linspace(-np.pi, np.pi, 101)
    expected = basis.derivative(x, der=der)

    actual = basis.derivative(jnp.asarray(x), der=der)

    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("mode", ["eval", "conv"])
def test_cyclic_basis_gam_fit_and_predict(mode):
    rng = np.random.default_rng(42)
    x = rng.uniform(-np.pi, np.pi, size=150)
    y = rng.poisson(np.exp(0.2 * np.cos(x)))
    if mode == "eval":
        basis = nmo.basis.CyclicBSplineEval(6, bounds=(-np.pi, np.pi))
    elif mode == "conv":
        basis = nmo.basis.CyclicBSplineConv(6, window_size=11)
    else:
        raise NotImplementedError(f"Unsupported basis mode: {mode}")
    model = GAM(basis, use_scipy=True, maxiter=3)

    model.fit((x,), y)
    prediction = np.asarray(model.predict((x,)))

    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.intercept_).all()
    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()

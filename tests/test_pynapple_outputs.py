"""Outputs of GAM.predict and smooth_compute for pynapple inputs."""

import jax
import nemos as nmo
import numpy as np
import pynapple as nap
import pytest

from pgam_jax import GAM

N_SAMPLES = 200
TIME_BIN = 0.01
TIME = np.arange(N_SAMPLES) * TIME_BIN


def _basis():
    return nmo.basis.BSplineEval(6, bounds=(-1.0, 1.0), label="a") + (
        nmo.basis.BSplineEval(6, bounds=(-1.0, 1.0), label="b")
    )


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    x1 = rng.uniform(-1.0, 1.0, N_SAMPLES)
    x2 = rng.uniform(-1.0, 1.0, N_SAMPLES)
    y = rng.poisson(np.exp(0.5 * np.sin(3.0 * x1) + 0.5 * x2)).astype(float)
    return (x1, x2), y


@pytest.fixture(scope="module")
def fitted_gam(data):
    xi, y = data
    return GAM(_basis()).fit(xi, y)


@pytest.fixture(scope="module")
def support():
    # Not the default support of a Tsd, so that the test sees if it is kept.
    return nap.IntervalSet(start=TIME[0], end=TIME[-1] + TIME_BIN)


def _assert_tsd_like(out, reference, expected_values):
    """Check that out is a Tsd with the times and support of reference."""
    assert isinstance(out, nap.Tsd)
    np.testing.assert_array_equal(out.t, reference.t)
    np.testing.assert_array_equal(
        out.time_support.values, reference.time_support.values
    )
    np.testing.assert_allclose(out.d, expected_values)


def test_predict_returns_an_array_for_array_inputs(fitted_gam, data):
    xi, _ = data
    assert isinstance(fitted_gam.predict(xi), jax.Array)


def test_predict_returns_a_tsd_for_pynapple_inputs(fitted_gam, data, support):
    xi, _ = data
    xi_tsd = tuple(nap.Tsd(t=TIME, d=x, time_support=support) for x in xi)

    out = fitted_gam.predict(xi_tsd)

    _assert_tsd_like(out, xi_tsd[0], fitted_gam.predict(xi))


@pytest.mark.parametrize("tsd_index", [0, 1])
def test_predict_returns_a_tsd_for_mixed_inputs(tsd_index, fitted_gam, data, support):
    """One pynapple input is enough, wherever it is in xi."""
    xi, _ = data
    tsd = nap.Tsd(t=TIME, d=xi[tsd_index], time_support=support)
    xi_mixed = tuple(tsd if i == tsd_index else x for i, x in enumerate(xi))

    out = fitted_gam.predict(xi_mixed)

    _assert_tsd_like(out, tsd, fitted_gam.predict(xi))


def test_predict_keeps_the_epochs_of_the_inputs():
    """
    The Tsd from predict has the epochs of the inputs as its time support.

    With a conv basis and nan_handling="drop", each epoch starts with
    window_size NaN rows, because the convolution restarts in each epoch.
    """
    window_size = 5
    n_per_epoch = 50
    # Two epochs of 50 bins, with a gap of 50 bins between them.
    t = (
        np.concatenate(
            [np.arange(n_per_epoch), 2 * n_per_epoch + np.arange(n_per_epoch)]
        )
        * TIME_BIN
    )
    epochs = nap.IntervalSet(
        start=[t[0], t[n_per_epoch]],
        end=[t[n_per_epoch - 1], t[-1]],
    )
    rng = np.random.default_rng(2)
    x = nap.Tsd(t=t, d=rng.uniform(-1.0, 1.0, t.size), time_support=epochs)
    counts = nap.Tsd(t=t, d=rng.poisson(0.3, t.size), time_support=epochs)
    y = nap.Tsd(t=t, d=rng.poisson(1.0, t.size), time_support=epochs)
    basis = nmo.basis.BSplineEval(6, bounds=(-1.0, 1.0)) + (
        nmo.basis.BSplineConv(4, window_size=window_size)
    )
    gam = GAM(basis, nan_handling="drop").fit((x, counts), y)

    out = gam.predict((x, counts))

    assert isinstance(out, nap.Tsd)
    np.testing.assert_array_equal(out.t, t)
    np.testing.assert_array_equal(out.time_support.values, epochs.values)
    for start, end in epochs.values:
        in_epoch = out.restrict(nap.IntervalSet(start=start, end=end)).d
        assert np.isnan(in_epoch[:window_size]).all()
        assert not np.isnan(in_epoch[window_size:]).any()


def test_smooth_compute_returns_arrays_for_an_array_input(fitted_gam, data):
    xi, _ = data
    out = fitted_gam.smooth_compute(xi[0], "a")
    assert len(out) == 3
    assert all(isinstance(o, jax.Array) for o in out)


@pytest.mark.parametrize("wrap_in_tuple", [False, True])
def test_smooth_compute_returns_tsds_for_a_pynapple_input(
    wrap_in_tuple, fitted_gam, data, support
):
    """mean, lower and upper are each a Tsd with the times of the input."""
    xi, _ = data
    tsd = nap.Tsd(t=TIME, d=xi[0], time_support=support)
    expected = fitted_gam.smooth_compute(xi[0], "a")

    out = fitted_gam.smooth_compute((tsd,) if wrap_in_tuple else tsd, "a")

    assert len(out) == 3
    for out_part, expected_part in zip(out, expected, strict=True):
        _assert_tsd_like(out_part, tsd, expected_part)

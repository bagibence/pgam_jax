"""Input checks of GAM.fit, predict, score, concurvity and smooth_compute."""

import warnings

import jax
import nemos as nmo
import numpy as np
import pynapple as nap
import pytest

from pgam_jax import GAM

N_SAMPLES = 200
TIME_BIN = 0.01
TIME = np.arange(N_SAMPLES) * TIME_BIN
# Far above the pynapple time precision (1 ns), below the bin width.
TIME_SHIFT = 0.003


def _basis():
    return nmo.basis.BSplineEval(6, bounds=(-1.0, 1.0), label="a") + (
        nmo.basis.BSplineEval(6, bounds=(-1.0, 1.0), label="b")
    )


def _tsd(d, shift=0.0, time_support=None):
    return nap.Tsd(t=TIME + shift, d=d, time_support=time_support)


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


# fit gets a new GAM, so that the tests do not change the shared fitted one.
CALLS_WITH_XI = {
    "fit": lambda gam, xi, y: GAM(_basis()).fit(xi, y),
    "predict": lambda gam, xi, y: gam.predict(xi),
    "score": lambda gam, xi, y: gam.score(xi, y),
    "concurvity": lambda gam, xi, y: gam.concurvity(xi),
}
CALLS_WITH_Y = {name: CALLS_WITH_XI[name] for name in ("fit", "score")}
# smooth_compute takes only the inputs of one component.
ALL_CALLS = {
    **CALLS_WITH_XI,
    "smooth_compute": lambda gam, xi, y: gam.smooth_compute(xi[0], "a"),
}


@pytest.mark.parametrize("method", CALLS_WITH_XI)
def test_xi_must_be_a_tuple(method, fitted_gam, data):
    xi, y = data
    with pytest.raises(TypeError, match="wrapped in a tuple"):
        CALLS_WITH_XI[method](fitted_gam, list(xi), y)


@pytest.mark.parametrize("method", CALLS_WITH_XI)
def test_xi_of_different_lengths_raises(method, fitted_gam, data):
    (x1, x2), y = data
    with pytest.raises(ValueError, match="All inputs need to have the same number"):
        CALLS_WITH_XI[method](fitted_gam, (x1, x2[:-5]), y)


@pytest.mark.parametrize("method", CALLS_WITH_Y)
def test_y_of_different_length_raises(method, fitted_gam, data):
    xi, y = data
    with pytest.raises(ValueError, match="All inputs and the output need"):
        CALLS_WITH_Y[method](fitted_gam, xi, y[:-5])


@pytest.mark.parametrize("method", CALLS_WITH_XI)
def test_xi_with_different_time_axes_raises(method, fitted_gam, data):
    (x1, x2), y = data
    xi = (_tsd(x1), _tsd(x2, shift=TIME_SHIFT))
    with pytest.raises(ValueError, match="Inputs have mismatching time axis"):
        CALLS_WITH_XI[method](fitted_gam, xi, _tsd(y))


@pytest.mark.parametrize("method", CALLS_WITH_Y)
def test_y_with_different_time_axis_raises(method, fitted_gam, data):
    (x1, x2), y = data
    xi = (_tsd(x1), _tsd(x2))
    with pytest.raises(ValueError, match="Inputs and outputs have mismatching"):
        CALLS_WITH_Y[method](fitted_gam, xi, _tsd(y, shift=TIME_SHIFT))


@pytest.mark.parametrize("method", CALLS_WITH_Y)
def test_y_with_same_times_but_other_time_support_raises(method, fitted_gam, data):
    """
    The check also compares the time support, not only the timestamps.

    A y built as ``nap.Tsd(t=xi[0].t, d=y)`` gets the default support, which
    ends at the last timestamp. Here the inputs have a support that ends one
    bin later.
    """
    (x1, x2), y = data
    support = nap.IntervalSet(start=TIME[0], end=TIME[-1] + TIME_BIN)
    xi = (_tsd(x1, time_support=support), _tsd(x2, time_support=support))
    y_tsd = nap.Tsd(t=xi[0].t, d=y)
    with pytest.raises(ValueError, match="Inputs and outputs have mismatching"):
        CALLS_WITH_Y[method](fitted_gam, xi, y_tsd)


def test_pynapple_inputs_give_the_same_results_as_arrays(fitted_gam, data):
    xi, y = data
    xi_tsd = tuple(_tsd(x) for x in xi)
    y_tsd = _tsd(y)

    gam_tsd = GAM(_basis()).fit(xi_tsd, y_tsd)

    np.testing.assert_allclose(gam_tsd.coef_, fitted_gam.coef_)
    np.testing.assert_allclose(gam_tsd.intercept_, fitted_gam.intercept_)
    np.testing.assert_allclose(fitted_gam.predict(xi_tsd), fitted_gam.predict(xi))
    np.testing.assert_allclose(fitted_gam.score(xi_tsd, y_tsd), fitted_gam.score(xi, y))
    jax.tree_util.tree_map(
        np.testing.assert_allclose,
        fitted_gam.concurvity(xi_tsd),
        fitted_gam.concurvity(xi),
    )


def test_integer_inputs_do_not_warn_about_float64():
    """Counts are exact as integers, so they need no float64 warning."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, N_SAMPLES)
    counts = rng.poisson(0.3, N_SAMPLES)
    y = rng.poisson(1.0, N_SAMPLES)
    basis = nmo.basis.BSplineEval(6, bounds=(-1.0, 1.0)) + nmo.basis.BSplineConv(
        4, window_size=5
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*not float64")
        gam = GAM(basis).fit((x, counts), y)
        gam.predict((x, counts))
        gam.score((x, counts), y)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("method", CALLS_WITH_XI)
def test_float32_input_warning_names_that_input(method, index, fitted_gam, data):
    xi, y = data
    xi = tuple(x.astype(np.float32) if i == index else x for i, x in enumerate(xi))

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        CALLS_WITH_XI[method](fitted_gam, xi, y)

    messages = [str(w.message) for w in record if "not float64" in str(w.message)]
    assert messages == [
        f"GAM.{method}: xi[{index}] has dtype float32, not float64. "
        "pgam_jax relies on float64 precision and can give inaccurate results "
        "with float32 data. Cast the inputs to float64."
    ]


@pytest.mark.parametrize("method", ALL_CALLS)
def test_float32_warnings_point_to_the_caller(method, fitted_gam, data):
    xi, y = data
    xi = (xi[0].astype(np.float32), xi[1])
    y = y.astype(np.float32)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ALL_CALLS[method](fitted_gam, xi, y)

    precision_warnings = [w for w in record if "not float64" in str(w.message)]
    assert precision_warnings
    assert all(w.filename == __file__ for w in precision_warnings)


def test_predict_convolves_each_epoch_separately():
    """
    predict must pass the pynapple inputs to the basis unchanged.

    The conv basis then restarts the convolution in each epoch, as in fit. If
    predict converts the inputs to arrays first, the convolution runs across
    the gap between the epochs, and only the first epoch starts with NaN rows.
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

    def basis():
        return nmo.basis.BSplineEval(6, bounds=(-1.0, 1.0)) + (
            nmo.basis.BSplineConv(4, window_size=window_size)
        )

    gam = GAM(basis(), nan_handling="drop").fit((x, counts), y)

    design = np.asarray(basis().compute_features(x, counts))
    nan_rows_design = np.isnan(design).any(axis=1)
    nan_rows_predict = np.isnan(np.asarray(gam.predict((x, counts))))

    assert nan_rows_design.sum() == len(epochs) * window_size
    np.testing.assert_array_equal(nan_rows_predict, nan_rows_design)

"""Warnings for float32 precision: x64 mode off, or float32 inputs to fit."""

import warnings

import jax
import nemos as nmo
import numpy as np
import pytest

from pgam_jax import GAM
from pgam_jax._utils import warn_if_not_float64, warn_if_x64_disabled


@pytest.fixture
def x64_disabled():
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", True)


def test_warn_if_x64_disabled_warns_when_off(x64_disabled):
    with pytest.warns(UserWarning, match="x64 mode is disabled"):
        warn_if_x64_disabled("f")


def test_warn_if_x64_disabled_silent_when_on():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_x64_disabled("f")


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_warn_if_not_float64_warns_for_low_precision_floats(dtype):
    with pytest.warns(UserWarning, match=f"y has dtype {np.dtype(dtype)}.*Do X."):
        warn_if_not_float64("f", {"y": np.ones(3, dtype=dtype)}, advice="Do X.")


@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.int64, bool])
def test_warn_if_not_float64_silent_for_float64_and_non_floats(dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_not_float64("f", {"y": np.ones(3, dtype=dtype)}, advice="Do X.")


def test_warn_if_not_float64_silent_when_x64_off(x64_disabled):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_not_float64("f", {"y": np.ones(3, dtype=np.float32)}, advice="Do X.")


def test_fit_warns_and_casts_float32_response():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=200)
    y = rng.poisson(np.exp(np.sin(3.0 * x))).astype(np.float32)
    gam = GAM(nmo.basis.BSplineEval(n_basis_funcs=10, order=4, bounds=(-1.0, 1.0)))

    with pytest.warns(UserWarning, match="y has dtype float32.*casts y to float64"):
        gam.fit((x,), y)

    assert gam.coef_.dtype == np.float64
    assert np.all(np.isfinite(gam.coef_))

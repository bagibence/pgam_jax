import jax.numpy as jnp
import numpy as np
import pytest

import pgam_jax._nan_policy as nan_policy
from pgam_jax._nan_policy import (
    _rows_with_nan,
    apply_nan_policy_for_fit,
    apply_nan_policy_for_transform,
    get_valid_y_rows,
    validate_nan_handling,
)


def test_validate_nan_handling():
    assert validate_nan_handling("zero") == "zero"
    assert validate_nan_handling("drop") == "drop"
    with pytest.raises(ValueError, match="nan_handling.*unknown"):
        validate_nan_handling("unknown")


def test_get_valid_y_rows_handles_single_and_multi_output():
    one_dimensional = jnp.array([1.0, jnp.nan, 3.0])
    two_dimensional = jnp.array(
        [
            [1.0, 2.0],
            [jnp.nan, 4.0],
            [5.0, jnp.nan],
        ]
    )

    np.testing.assert_array_equal(
        get_valid_y_rows(one_dimensional, n_rows=3),
        [True, False, True],
    )
    np.testing.assert_array_equal(
        get_valid_y_rows(two_dimensional, n_rows=3),
        [True, False, False],
    )


@pytest.mark.parametrize(
    "y",
    [jnp.array(1.0), jnp.ones(2)],
    ids=["no-sample-axis", "wrong-row-count"],
)
def test_get_valid_y_rows_rejects_misaligned_responses(y):
    with pytest.raises(ValueError, match="same number of rows"):
        get_valid_y_rows(y, n_rows=3)


def test_rows_with_nan_rejects_scalar_input():
    with pytest.raises(ValueError, match="Expected an array with a sample axis"):
        _rows_with_nan(jnp.array(1.0))


def test_fit_zero_removes_nan_responses_before_zero_fill_and_centering():
    X_raw = jnp.array(
        [
            [1.0, jnp.nan],
            [3.0, 5.0],
            [7.0, 9.0],
        ]
    )
    y = jnp.array([10.0, jnp.nan, 30.0])

    X, y, feature_mean = apply_nan_policy_for_fit(X_raw, y, "zero")

    np.testing.assert_allclose(y, [10.0, 30.0])
    np.testing.assert_allclose(feature_mean, [4.0, 4.5])
    np.testing.assert_allclose(
        X,
        [[-3.0, -4.5], [3.0, 4.5]],
    )


def test_fit_drop_intersects_valid_design_and_response_rows():
    X_raw = jnp.array(
        [
            [1.0, jnp.nan],
            [3.0, 5.0],
            [7.0, 9.0],
            [11.0, 13.0],
        ]
    )
    y = jnp.array([10.0, 20.0, jnp.nan, 40.0])

    X, y, feature_mean = apply_nan_policy_for_fit(X_raw, y, "drop")

    np.testing.assert_allclose(y, [20.0, 40.0])
    np.testing.assert_allclose(feature_mean, [7.0, 9.0])
    np.testing.assert_allclose(X, [[-4.0, -4.0], [4.0, 4.0]])


def test_fit_drop_does_not_zero_fill_selected_rows(monkeypatch):
    def fail_if_called(_):
        raise AssertionError("drop fit must not zero-fill")

    monkeypatch.setattr(nan_policy, "_zero_fill", fail_if_called)
    X, _, _ = apply_nan_policy_for_fit(
        jnp.array([[1.0, jnp.nan], [3.0, 5.0], [7.0, 9.0]]),
        None,
        "drop",
    )

    assert X.shape[0] == 2


def test_fit_policy_without_response_supports_prefit_design_diagnostics():
    X_raw = jnp.array([[1.0, jnp.nan], [3.0, 5.0], [7.0, 9.0]])

    X_zero, y_zero, _ = apply_nan_policy_for_fit(X_raw, None, "zero")
    X_drop, y_drop, _ = apply_nan_policy_for_fit(X_raw, None, "drop")

    assert y_zero is None
    assert y_drop is None
    assert X_zero.shape[0] == 3
    assert X_drop.shape[0] == 2
    np.testing.assert_allclose(X_zero.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(X_drop.mean(axis=0), 0.0, atol=1e-7)


@pytest.mark.parametrize(
    ("X_raw", "mode", "y"),
    [
        (jnp.ones((2, 2)), "zero", jnp.array([jnp.nan, jnp.nan])),
        (jnp.full((2, 2), jnp.nan), "drop", None),
    ],
)
def test_fit_policy_rejects_no_surviving_rows(X_raw, mode, y):
    with pytest.raises(ValueError, match="No rows remain"):
        apply_nan_policy_for_fit(X_raw, y, mode)


def test_transform_handlers_preserve_rows_and_expose_policy_eligibility():
    X_raw = jnp.array([[1.0, jnp.nan], [3.0, 5.0]])
    feature_mean = jnp.array([2.0, 4.0])

    zero = apply_nan_policy_for_transform(X_raw, feature_mean, "zero")
    drop = apply_nan_policy_for_transform(X_raw, feature_mean, "drop")

    np.testing.assert_allclose(zero, [[-1.0, -4.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        drop,
        [[-1.0, np.nan], [1.0, 1.0]],
    )
    np.testing.assert_array_equal(~_rows_with_nan(zero), [True, True])
    np.testing.assert_array_equal(~_rows_with_nan(drop), [False, True])


def test_policy_helpers_validate_design_and_feature_shapes():
    with pytest.raises(ValueError, match="2-dimensional"):
        apply_nan_policy_for_fit(jnp.ones(3), None, "zero")

    with pytest.raises(ValueError, match="feature_mean must have shape"):
        apply_nan_policy_for_transform(jnp.ones((3, 2)), jnp.ones(3), "zero")

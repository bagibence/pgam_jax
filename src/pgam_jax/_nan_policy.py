"""Pure array helpers for GAM design-matrix NaN policies."""

from __future__ import annotations

from typing import Literal, assert_never, cast, get_args

import jax
import jax.numpy as jnp

NanHandling = Literal["zero", "drop"]
VALID_NAN_HANDLING: tuple[NanHandling, ...] = get_args(NanHandling)


def validate_nan_handling(value: str) -> NanHandling:
    """Validate and narrow a user-provided NaN-handling mode."""
    if value not in VALID_NAN_HANDLING:
        raise ValueError(
            f"nan_handling must be one of {VALID_NAN_HANDLING}, got {value!r}."
        )
    return cast(NanHandling, value)


def _as_design_matrix(X_raw) -> jax.Array:
    X_raw = jnp.asarray(X_raw)
    if X_raw.ndim != 2:
        raise ValueError(
            f"Design matrix must be 2-dimensional, got shape {X_raw.shape}."
        )
    return X_raw


def _rows_with_nan(values: jax.Array) -> jax.Array:
    """Return one boolean per sample indicating a NaN in that sample."""
    if values.ndim == 0:
        raise ValueError("Expected an array with a sample axis.")
    if values.ndim == 1:
        return jnp.isnan(values)

    return jnp.any(jnp.isnan(values), axis=tuple(range(1, values.ndim)))


def get_valid_y_rows(y, *, n_rows: int) -> jax.Array:
    """Return response rows without NaNs after validating sample alignment."""
    y = jnp.asarray(y)
    if y.ndim == 0 or y.shape[0] != n_rows:
        raise ValueError(
            "Response and design matrix must contain the same number of rows; "
            f"got response shape {y.shape} and {n_rows} design rows."
        )
    return ~_rows_with_nan(y)


def _zero_fill(X: jax.Array) -> jax.Array:
    """Replace NaNs with zeros in X."""
    return jnp.where(jnp.isnan(X), jnp.zeros_like(X), X)


def _raise_on_no_rows_kept(
    kept_rows: jax.Array,
    *,
    nan_handling: NanHandling,
) -> None:
    if not bool(jnp.any(kept_rows)):
        raise ValueError(
            f"No rows remain after applying nan_handling={nan_handling!r} "
            "and any response-row filtering."
        )


def _center(X: jax.Array) -> tuple[jax.Array, jax.Array]:
    feature_mean = X.mean(axis=0)
    return X - feature_mean, feature_mean


def _fit_zero(
    X_raw: jax.Array,
    valid_y_rows: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Keep response-valid rows and zero-fill their design NaNs."""
    # all rows of X will be valid after zero-filling
    kept_rows = valid_y_rows
    _raise_on_no_rows_kept(kept_rows, nan_handling="zero")
    X, feature_mean = _center(_zero_fill(X_raw[kept_rows]))
    return X, feature_mean, kept_rows


def _fit_drop(
    X_raw: jax.Array,
    valid_y_rows: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Keep only rows valid in both the response and raw design."""
    valid_X_rows = ~_rows_with_nan(X_raw)
    kept_rows = valid_y_rows & valid_X_rows
    _raise_on_no_rows_kept(kept_rows, nan_handling="drop")
    X, feature_mean = _center(X_raw[kept_rows])
    return X, feature_mean, kept_rows


def apply_nan_policy_for_fit(
    X_raw,
    y,
    nan_handling: NanHandling,
) -> tuple[jax.Array, jax.Array | None, jax.Array]:
    """Apply a design NaN policy for GAM.fit and optionally align a response to its rows."""
    X_raw = _as_design_matrix(X_raw)
    nan_handling = validate_nan_handling(nan_handling)

    if y is None:
        y_array = None
        valid_y_rows = jnp.ones(X_raw.shape[0], dtype=bool)
    else:
        y_array = jnp.asarray(y)
        valid_y_rows = get_valid_y_rows(y_array, n_rows=X_raw.shape[0])

    if nan_handling == "zero":
        X, feature_mean, kept_rows = _fit_zero(X_raw, valid_y_rows)
    elif nan_handling == "drop":
        X, feature_mean, kept_rows = _fit_drop(X_raw, valid_y_rows)
    else:
        assert_never(nan_handling)

    y = None if y_array is None else y_array[kept_rows]
    return X, y, feature_mean


def _transform_zero(
    X_raw: jax.Array,
    feature_mean: jax.Array,
) -> jax.Array:
    """Zero-fill and retain every transformed design row."""
    return _zero_fill(X_raw) - feature_mean


def _transform_drop(
    X_raw: jax.Array,
    feature_mean: jax.Array,
) -> jax.Array:
    """Preserve design NaNs while applying the fitted feature mean."""
    return X_raw - feature_mean


def apply_nan_policy_for_transform(
    X_raw,
    feature_mean,
    nan_handling: NanHandling,
) -> jax.Array:
    """Apply a design policy without changing the number of rows."""
    X_raw = _as_design_matrix(X_raw)
    feature_mean = jnp.asarray(feature_mean)
    nan_handling = validate_nan_handling(nan_handling)

    expected_mean_shape = (X_raw.shape[1],)
    if feature_mean.shape != expected_mean_shape:
        raise ValueError(
            f"feature_mean must have shape {expected_mean_shape}, "
            f"got {feature_mean.shape}."
        )

    if nan_handling == "zero":
        return _transform_zero(X_raw, feature_mean)
    if nan_handling == "drop":
        return _transform_drop(X_raw, feature_mean)
    assert_never(nan_handling)

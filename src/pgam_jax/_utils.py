"""General JAX utilities shared across modules."""

import warnings
from functools import wraps
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from nemos.observation_models import (
    GammaObservations,
    GaussianObservations,
    Observations,
    PoissonObservations,
)

from ._typing import JaxFloatMatrix


def elementwise_derivative(f):
    """Derivative of an elementwise function via forward-mode AD.

    If f maps an array x to an array of the same shape where f(x)[i] depends
    only on x[i], then this returns df/dx[i] at each i with a single JVP call
    (cheaper than vmap + grad).
    """

    @wraps(f)
    def df(x):
        _, grad = jax.jvp(f, (x,), (jnp.ones_like(x),))
        return grad

    return df


def prepend_ones_for_intercept(X: jnp.ndarray) -> jnp.ndarray:
    """
    Prepend a column of ones (the intercept) to a 2D design matrix.

    Raises if `X` is not 2D, so a design matrix that has accidentally
    collapsed to 1D fails here rather than silently promoting to a column.
    """
    intercept_col = jnp.ones((X.shape[0], 1))
    return jnp.concatenate([intercept_col, X], axis=1)


def scale_estimated(obs_model: Observations) -> bool:
    """Determine if the observation model's scale is estimated or fixed."""

    if isinstance(obs_model, PoissonObservations):
        return False

    if isinstance(obs_model, (GammaObservations, GaussianObservations)):
        return True

    raise ValueError("`obs_model` has to be one of Poisson, Gamma, Gaussian.")


def stack_block_diag(
    submatrices: Sequence[JaxFloatMatrix],
    size: int,
) -> JaxFloatMatrix:
    for sm in submatrices:
        if sm.shape[0] != sm.shape[1]:
            raise ValueError("All submatrices should be square.")

    total = sum(sm.shape[0] for sm in submatrices)
    if total > size:
        raise ValueError("All submatrices should fit inside the requested size.")

    B = jax.scipy.linalg.block_diag(*submatrices)
    pad = size - total
    return jnp.pad(B, ((0, pad), (0, pad)))


def warn_if_x64_disabled(fname: str) -> None:
    """
    Warn that ``fname`` runs in float32 because JAX x64 mode is off.

    pgam_jax enables x64 on import, so this only fires if the user turned it off.
    Call it from public entry points, outside jit, so it fires on every call.
    """
    if not jax.config.jax_enable_x64:
        warnings.warn(
            f"{fname}: JAX x64 mode is disabled, so JAX computes in float32. "
            "pgam_jax relies on float64 precision and can give inaccurate results "
            "in float32. pgam_jax enables x64 on import unless JAX_ENABLE_X64 is "
            "set to a false value. Unset JAX_ENABLE_X64, or call "
            "jax.config.update('jax_enable_x64', True).",
            UserWarning,
            stacklevel=3,
        )


def warn_if_not_float64(
    fname: str, arrays: Mapping[str, ArrayLike], *, advice: str
) -> None:
    """
    Warn about each floating-point array whose dtype is not float64.

    Arrays that are not floating-point (for example integer counts) are skipped.
    Nothing is checked when x64 mode is off. In that case no array can be float64,
    and ``warn_if_x64_disabled`` already reports the cause.
    ``advice`` is appended to the message and tells the user what happens next.
    """
    if not jax.config.jax_enable_x64:
        return
    for name, arr in arrays.items():
        dtype = jnp.asarray(arr).dtype
        if jnp.issubdtype(dtype, jnp.floating) and dtype != np.float64:
            warnings.warn(
                f"{fname}: {name} has dtype {dtype}, not float64. "
                "pgam_jax relies on float64 precision and can give inaccurate "
                f"results with {dtype} data. {advice}",
                UserWarning,
                stacklevel=3,
            )


def to_zero_dim_jax_array(x: ArrayLike) -> jax.Array:
    """Turn `x` into a 0-dimensional jax array."""
    return jnp.reshape(jnp.asarray(x), ())


def singular_value_keep_mask(
    singular_values: jax.Array, matrix_shape: tuple
) -> jax.Array:
    """
    Return a boolean mask of numerically retained singular values.

    Uses JAX's default pseudoinverse rank convention.
    """
    rtol = 10.0 * max(matrix_shape) * jnp.finfo(singular_values.dtype).eps
    return singular_values > rtol * singular_values.max()

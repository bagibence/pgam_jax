"""General JAX utilities shared across modules."""

from functools import wraps
from typing import Sequence

import jax
import jax.numpy as jnp
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


def to_zero_dim_jax_array(x: ArrayLike) -> jax.Array:
    """Turn `x` into a 0-dimensional jax array."""
    return jnp.reshape(jnp.asarray(x), ())

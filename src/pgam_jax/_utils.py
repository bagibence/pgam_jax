"""General JAX utilities shared across modules."""

from functools import wraps

import jax
import jax.numpy as jnp


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

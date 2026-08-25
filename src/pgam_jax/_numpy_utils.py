"""
Small NumPy helpers shared across modules.

Nothing here touches JAX, so a NumPy-only module can use these helpers without
importing JAX. The JAX helpers live in :mod:`pgam_jax._utils`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _divide_with_fallback(
    numerator: FloatArray,
    denominator: FloatArray,
    fallback: float,
) -> FloatArray:
    """
    Divide elementwise, and use ``fallback`` where the denominator is zero.

    This is the vectorized form of an ``if denominator != 0`` at every element.
    NumPy skips the division where the mask is false, so a division by zero is
    never evaluated and no warning is raised. Those positions keep the fallback
    value that ``result`` starts with.

    Parameters
    ----------
    numerator, denominator : numpy.ndarray
        Arrays of the same shape.
    fallback : float
        The value to use where the denominator is zero.

    Returns
    -------
    numpy.ndarray
        The elementwise ratio, with ``fallback`` at every zero denominator.
    """
    result = np.full_like(numerator, fallback)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0.0,
    )
    return result


def _broadcast(values: ArrayLike, size: int, name: str) -> FloatArray:
    """
    Return ``values`` as a 1-D float array of length ``size``.

    A scalar is repeated ``size`` times. An array that is already that long
    passes through. Anything else raises, and ``name`` goes into the message so
    that the caller learns which argument was wrong.

    Parameters
    ----------
    values : array_like
        A scalar, or a sequence of length ``size``.
    size : int
        The required length.
    name : str
        The argument name to use in the error message.

    Returns
    -------
    numpy.ndarray
        A 1-D float array of length ``size``.

    Raises
    ------
    ValueError
        If ``values`` has neither length 1 nor length ``size``.
    """
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if arr.size == 1:
        arr = np.repeat(arr, size)
    if arr.size != size:
        raise ValueError(f"'{name}' must have length 1 or {size}, got {arr.size}")
    return arr

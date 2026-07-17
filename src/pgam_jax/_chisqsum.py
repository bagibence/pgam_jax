r"""Distribution of a linear combination of chi-squared random variables.

Computes the (survival) probability of

.. math::

    Q = \sum_j w_j X_j + \sigma Z,

where :math:`X_j \sim \chi^2_{\nu_j}(\delta_j^2)` are independent (possibly
non-central) chi-squared variables and :math:`Z \sim N(0, 1)`.  Weights may be
of either sign.  This is the null distribution used to obtain covariate-inclusion
p-values for penalised GAM smooth terms.

The probability is obtained by inverting the characteristic function of ``Q``
(Gil-Pelaez / Imhof, 1961):

.. math::

    \Pr(Q \le q) = \frac{1}{2}
        - \frac{1}{\pi} \int_0^\infty
        \frac{\sin\!\big(\phi(u) - q u\big)}{u}\, e^{-\psi(u)}\, \mathrm{d}u,

where, writing :math:`x_j = 2 w_j u`,

.. math::

    \phi(u) &= \frac{1}{2} \sum_j \left[
        \nu_j \arctan x_j + \frac{\delta_j^2\, x_j}{1 + x_j^2} \right], \\
    \psi(u) &= \frac{1}{2}\sigma^2 u^2
        + \frac{1}{4} \sum_j \nu_j \log\!\big(1 + x_j^2\big)
        + \frac{1}{2} \sum_j \frac{\delta_j^2\, x_j^2}{1 + x_j^2}.

The integral is evaluated with SciPy's adaptive quadrature: a short
non-oscillatory head on :math:`[0, a]`, plus the semi-infinite oscillatory tail
handled by the Fourier integrators (``weight="cos"``/``"sin"``), which converge
even for the slowly decaying tails produced by low degrees of freedom.

References
----------
Imhof, J.P. (1961) "Computing the distribution of quadratic forms in normal
variables." *Biometrika* 48, 419-426.

Davies, R.B. (1980) "The distribution of a linear combination of :math:`\chi^2`
random variables." *J. R. Statist. Soc. C* 29, 323-333.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

__all__ = ["psum_chisq"]

FloatArray = NDArray[np.float64]


def _quad(
    func: Callable[..., float],
    a: float,
    b: float,
    args: tuple[object, ...],
    weight: str | None,
    wvar: float | None,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> float:
    """Thin typed wrapper around :func:`scipy.integrate.quad`.

    SciPy ships no type stubs, so its return value is untyped; isolating the call
    here keeps that "unknown" type from propagating through the module.  When
    ``weight`` is ``None`` this is ordinary adaptive quadrature; when it is
    ``"cos"``/``"sin"`` SciPy uses its oscillatory Fourier integrator with angular
    frequency ``wvar``.  Only the integral value is returned (the error estimate
    is discarded).
    """
    result = quad(  # type: ignore
        func,
        a,
        b,
        args=args,
        weight=weight,
        wvar=wvar,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    return result[0]


def _phase_and_envelope(
    u: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    sigma_sq: float,
) -> tuple[float, float]:
    r"""Phase and envelope of the inversion integrand at frequency ``u``.

    The integrand of the characteristic-function inversion is
    :math:`\sin\!\big(\phi(u) - q u\big)\, e^{-\psi(u)} / u`.  This returns the
    tuple :math:`(\phi(u),\, e^{-\psi(u)} / u)`, with (writing
    :math:`x_j = 2 w_j u`)

    .. math::

        \phi(u) &= \frac{1}{2} \sum_j \left[
            \nu_j \arctan x_j + \frac{\delta_j^2\, x_j}{1 + x_j^2} \right], \\
        \psi(u) &= \frac{1}{2}\sigma^2 u^2
            + \frac{1}{4} \sum_j \nu_j \log\!\big(1 + x_j^2\big)
            + \frac{1}{2} \sum_j \frac{\delta_j^2\, x_j^2}{1 + x_j^2}.

    The reduction over the terms :math:`j` is vectorised over NumPy arrays.

    Parameters
    ----------
    u : float
        Frequency at which the characteristic function is evaluated.
    weights, df, noncentrality : numpy.ndarray
        The per-term weights :math:`w_j`, degrees of freedom :math:`\nu_j`, and
        non-centrality parameters :math:`\delta_j^2`.
    sigma_sq : float
        Variance :math:`\sigma^2` of the additive normal term.

    Returns
    -------
    phase : float
        The phase :math:`\phi(u)`.
    envelope : float
        The positive envelope :math:`e^{-\psi(u)} / u`.
    """
    x = 2.0 * weights * u
    x_sq = x**2
    ncp = noncentrality * x / (1.0 + x_sq)
    phase = 0.5 * np.sum(df * np.arctan(x) + ncp)
    log_modulus = (
        -0.5 * sigma_sq * u**2
        - 0.25 * np.sum(df * np.log1p(x_sq))
        - 0.5 * np.sum(x * ncp)
    )
    return phase, np.exp(log_modulus) / u


def _head_integrand(
    u: float,
    q: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    sigma_sq: float,
) -> float:
    """Full integrand on the non-oscillatory head ``[0, a]``."""
    phase, envelope = _phase_and_envelope(u, weights, df, noncentrality, sigma_sq)
    return np.sin(phase - q * u) * envelope


def _tail_cos_coefficient(
    u: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    sigma_sq: float,
) -> float:
    """Coefficient of ``cos(q u)`` in the oscillatory tail integrand."""
    phase, envelope = _phase_and_envelope(u, weights, df, noncentrality, sigma_sq)
    return envelope * np.sin(phase)


def _tail_sin_coefficient(
    u: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    sigma_sq: float,
) -> float:
    """Coefficient of ``sin(q u)`` in the oscillatory tail integrand."""
    phase, envelope = _phase_and_envelope(u, weights, df, noncentrality, sigma_sq)
    return envelope * np.cos(phase)


def _cdf_single(
    q: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    sigma_sq: float,
    split: float,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> float:
    """``Pr(Q <= q)`` for a scalar ``q`` via characteristic-function inversion."""
    params: tuple[object, ...] = (weights, df, noncentrality, sigma_sq)
    head = _quad(
        _head_integrand, 0.0, split, (q, *params), None, None, epsabs, epsrel, limit
    )
    tail_cos = _quad(
        _tail_cos_coefficient, split, np.inf, params, "cos", q, epsabs, epsrel, limit
    )
    tail_sin = _quad(
        _tail_sin_coefficient, split, np.inf, params, "sin", q, epsabs, epsrel, limit
    )
    return 0.5 - (head + tail_cos - tail_sin) / np.pi


def _broadcast(values: ArrayLike, size: int, name: str) -> FloatArray:
    """Return ``values`` as a 1-D float array of length ``size``."""
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if arr.size == 1:
        arr = np.repeat(arr, size)
    if arr.size != size:
        raise ValueError(f"'{name}' must have length 1 or {size}, got {arr.size}")
    return arr


def psum_chisq(
    q: ArrayLike,
    weights: ArrayLike,
    df: ArrayLike = 1.0,
    noncentrality: ArrayLike = 0.0,
    sigma: float = 0.0,
    lower_tail: bool = False,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 200,
) -> float | FloatArray:
    r"""Distribution function of a weighted sum of chi-squared variables.

    Evaluates :math:`\Pr(Q \le q)` (or the upper tail) for
    :math:`Q = \sum_j w_j X_j + \sigma Z`, where
    :math:`X_j \sim \chi^2_{\nu_j}(\delta_j^2)` and :math:`Z \sim N(0, 1)`, by
    numerically inverting the characteristic function of ``Q``.

    Parameters
    ----------
    q : array_like
        Point(s) at which to evaluate the distribution.
    weights : array_like
        The weights :math:`w_j`; may be positive or negative.  At least one must
        be non-zero.
    df : array_like, optional
        Degrees of freedom :math:`\nu_j` (must be positive).  A scalar is applied
        to every term.  Defaults to ``1``.
    noncentrality : array_like, optional
        Non-centrality parameters :math:`\delta_j^2` (must be non-negative).  A
        scalar is applied to every term.  Defaults to ``0`` (central).
    sigma : float, optional
        Standard deviation of the additive normal term (negative values are
        treated as ``0``).  Defaults to ``0``.
    lower_tail : bool, optional
        If ``True`` return :math:`\Pr(Q \le q)`; otherwise return the survival
        function :math:`\Pr(Q > q)`.  Defaults to ``False`` (upper tail), the
        convention used for p-values.
    epsabs, epsrel : float, optional
        Absolute and relative accuracy targets passed to the quadrature.
    limit : int, optional
        Maximum number of quadrature subintervals.

    Returns
    -------
    float or numpy.ndarray
        The (survival) probability, clipped to ``[0, 1]``.  A Python ``float`` is
        returned for scalar ``q``; otherwise an array matching ``q``.

    Notes
    -----
    The integrand is split into a non-oscillatory head on ``[0, 1/sd]`` (where
    ``sd`` is the standard deviation of ``Q``) and a semi-infinite oscillatory
    tail integrated with SciPy's Fourier quadrature, which remains accurate for
    the slowly decaying tails produced by low degrees of freedom.
    """
    weight_arr = np.atleast_1d(np.asarray(weights, dtype=float))
    n_terms = int(weight_arr.size)
    df_arr = _broadcast(df, n_terms, "df")
    ncp_arr = _broadcast(noncentrality, n_terms, "noncentrality")

    if np.any(df_arr <= 0):
        raise ValueError("'df' must be positive")
    if np.any(ncp_arr < 0):
        raise ValueError("'noncentrality' must be non-negative")
    if not np.any(weight_arr != 0.0):
        raise ValueError("at least one weight must be non-zero")
    sigma_sq = max(sigma, 0.0) ** 2

    variance = sigma_sq + np.sum(weight_arr**2 * (2.0 * df_arr + 4.0 * ncp_arr))
    split = 1.0 / np.sqrt(variance)

    q_arr = np.atleast_1d(np.asarray(q, dtype=float))
    out = np.empty(q_arr.shape, dtype=float)
    for i in range(q_arr.size):
        cdf = _cdf_single(
            float(q_arr.flat[i]),
            weight_arr,
            df_arr,
            ncp_arr,
            sigma_sq,
            split,
            epsabs,
            epsrel,
            limit,
        )
        out.flat[i] = cdf if lower_tail else 1.0 - cdf
    np.clip(out, 0.0, 1.0, out=out)

    if np.isnan(out).any():
        warnings.warn("psum_chisq: quadrature produced NaN", stacklevel=2)

    if np.ndim(q) == 0:
        return out.item()
    return out.reshape(np.shape(q))

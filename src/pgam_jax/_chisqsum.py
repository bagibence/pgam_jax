r"""Distribution of a linear combination of chi-squared random variables.

Computes the (survival) probability of

.. math::

    Q = \sum_j w_j X_j,

where :math:`X_j \sim \chi^2_{\nu_j}(\delta_j^2)` are independent (possibly
non-central) chi-squared variables.  Weights may be of either sign.  This is the
null distribution used to obtain covariate-inclusion p-values for penalised GAM
smooth terms.

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
    \psi(u) &= \frac{1}{4} \sum_j \nu_j \log\!\big(1 + x_j^2\big)
        + \frac{1}{2} \sum_j \frac{\delta_j^2\, x_j^2}{1 + x_j^2}.

Before any of this is evaluated the problem is nondimensionalised.  Writing
:math:`\mathrm{sd}` for the standard deviation of :math:`Q`, the substitution
:math:`t = \mathrm{sd}\, u` maps the integral onto itself, since
:math:`x_j = 2 w_j u = 2 (w_j/\mathrm{sd})\, t`, :math:`q u = (q/\mathrm{sd})\, t`
and :math:`\mathrm{d}u / u = \mathrm{d}t / t`.  The quadrature therefore runs on
normalised weights :math:`w_j/\mathrm{sd}` and standardised evaluation point
:math:`z = q/\mathrm{sd}`, with the split point at 1.  Every threshold and
frequency the method uses is then unit-free, so the result is invariant under a
common rescaling of ``q`` and the weights, as it must be.

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
from scipy.stats import chi2
from scipy.stats import f as f_dist
from scipy.stats import ncx2

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
) -> tuple[float, float]:
    """Thin typed wrapper around :func:`scipy.integrate.quad`.

    SciPy ships no type stubs, so its return value is untyped; isolating the call
    here keeps that "unknown" type from propagating through the module.  When
    ``weight`` is ``None`` this is ordinary adaptive quadrature; when it is
    ``"cos"``/``"sin"`` SciPy uses its oscillatory Fourier integrator with angular
    frequency ``wvar``.

    Returns the integral and its estimated absolute error.  Raises
    :class:`RuntimeError` if QUADPACK reports non-convergence or returns a
    non-finite value or invalid error estimate.
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
        full_output=True,
    )
    value, abs_error, _info, *diagnostics = result
    name = getattr(func, "__name__", "integrand")

    if diagnostics:
        raise RuntimeError(f"{name} quadrature failed: {diagnostics[0]}")

    value = float(value)
    abs_error = float(abs_error)
    if not np.isfinite(value):
        raise RuntimeError(f"{name} quadrature returned a non-finite value")
    if not np.isfinite(abs_error) or abs_error < 0.0:
        raise RuntimeError(
            f"{name} quadrature returned an invalid error estimate: {abs_error}"
        )
    return value, abs_error


def _phase_and_envelope(
    u: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> tuple[float, float]:
    r"""Phase and envelope of the inversion integrand at frequency ``u``.

    The integrand of the characteristic-function inversion is
    :math:`\sin\!\big(\phi(u) - q u\big)\, e^{-\psi(u)} / u`.  This returns the
    tuple :math:`(\phi(u),\, e^{-\psi(u)} / u)`, with (writing
    :math:`x_j = 2 w_j u`)

    .. math::

        \phi(u) &= \frac{1}{2} \sum_j \left[
            \nu_j \arctan x_j + \frac{\delta_j^2\, x_j}{1 + x_j^2} \right], \\
        \psi(u) &= \frac{1}{4} \sum_j \nu_j \log\!\big(1 + x_j^2\big)
            + \frac{1}{2} \sum_j \frac{\delta_j^2\, x_j^2}{1 + x_j^2}.

    The reduction over the terms :math:`j` is vectorised over NumPy arrays.

    Parameters
    ----------
    u : float
        Frequency at which the characteristic function is evaluated.
    weights, df, noncentrality : numpy.ndarray
        The per-term weights :math:`w_j`, degrees of freedom :math:`\nu_j`, and
        non-centrality parameters :math:`\delta_j^2`.

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
    log_modulus = -0.25 * np.sum(df * np.log1p(x_sq)) - 0.5 * np.sum(x * ncp)
    return phase, np.exp(log_modulus) / u


def _head_integrand(
    u: float,
    q: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> float:
    """Full integrand on the non-oscillatory head ``[0, a]``."""
    phase, envelope = _phase_and_envelope(u, weights, df, noncentrality)
    return np.sin(phase - q * u) * envelope


def _tail_cos_coefficient(
    u: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> float:
    """Coefficient of ``cos(q u)`` in the oscillatory tail integrand."""
    phase, envelope = _phase_and_envelope(u, weights, df, noncentrality)
    return envelope * np.sin(phase)


def _tail_sin_coefficient(
    u: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> float:
    """Coefficient of ``sin(q u)`` in the oscillatory tail integrand."""
    phase, envelope = _phase_and_envelope(u, weights, df, noncentrality)
    return envelope * np.cos(phase)


def _cdf_single(
    q: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    split: float,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> tuple[float, float]:
    """``Pr(Q <= q)`` and its estimated absolute error for a scalar ``q``."""
    params: tuple[object, ...] = (weights, df, noncentrality)
    head, head_error = _quad(
        _head_integrand,
        0.0,
        split,
        (q, *params),
        None,
        None,
        epsabs,
        epsrel,
        limit,
    )
    tail_cos, tail_cos_error = _quad(
        _tail_cos_coefficient,
        split,
        np.inf,
        params,
        "cos",
        q,
        epsabs,
        epsrel,
        limit,
    )
    tail_sin, tail_sin_error = _quad(
        _tail_sin_coefficient,
        split,
        np.inf,
        params,
        "sin",
        q,
        epsabs,
        epsrel,
        limit,
    )
    cdf = 0.5 - (head + tail_cos - tail_sin) / np.pi
    quadrature_error = (head_error + tail_cos_error + tail_sin_error) / np.pi

    return cdf, quadrature_error


def _broadcast(values: ArrayLike, size: int, name: str) -> FloatArray:
    """Return ``values`` as a 1-D float array of length ``size``."""
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if arr.size == 1:
        arr = np.repeat(arr, size)
    if arr.size != size:
        raise ValueError(f"'{name}' must have length 1 or {size}, got {arr.size}")
    return arr


def _validate_inputs(
    q: FloatArray,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> None:
    r"""
    Check that the arguments describe a sum this module can evaluate.

    Out-of-domain values used to reach the integrand and come back as a QUADPACK
    message about roundoff, which names the integrator rather than the argument
    at fault.  Everything is therefore checked up front, and each message names
    the argument it is about.

    ``q`` may be infinite, which is a well-posed question answered exactly
    elsewhere.  Nothing else may be: an infinite weight or degrees of freedom
    describes no distribution.  NaN is never accepted.

    An all-zero term list leaves ``Q`` degenerate at 0, with no distribution to
    invert, so at least one weight must be non-zero.

    Parameters
    ----------
    q : numpy.ndarray
        Evaluation points, already coerced to a float array.
    weights, df, noncentrality : numpy.ndarray
        The per-term weights :math:`w_j`, degrees of freedom :math:`\nu_j`, and
        non-centrality parameters :math:`\delta_j^2`, already broadcast to a
        common length.

    Raises
    ------
    ValueError
        If any argument is outside the domain described above.
    """
    if np.isnan(q).any():
        raise ValueError("'q' must not be NaN")
    if not np.isfinite(weights).all():
        raise ValueError("'weights' must be finite")
    if not np.isfinite(df).all():
        raise ValueError("'df' must be finite")
    if np.any(df <= 0):
        raise ValueError("'df' must be positive")
    if not np.isfinite(noncentrality).all():
        raise ValueError("'noncentrality' must be finite")
    if np.any(noncentrality < 0):
        raise ValueError("'noncentrality' must be non-negative")
    if not np.any(weights != 0.0):
        raise ValueError("at least one weight must be non-zero")


def _collapse_terms(
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""
    Canonical term list: zero weights dropped, equal weights merged.

    Both steps are exact.  A zero weight contributes :math:`0 \cdot X_j`, and
    independent chi-squares sharing a weight add:

    .. math::

        w X(\nu_1, \delta_1^2) + w X(\nu_2, \delta_2^2)
            = w X(\nu_1 + \nu_2, \delta_1^2 + \delta_2^2).

    Weights merge on exact equality only.  A tolerance would silently replace a
    mixture of nearby weights by a different distribution.

    Parameters
    ----------
    weights, df, noncentrality : numpy.ndarray
        The per-term weights :math:`w_j`, degrees of freedom :math:`\nu_j`, and
        non-centrality parameters :math:`\delta_j^2`.

    Returns
    -------
    weights, df, noncentrality : numpy.ndarray
        The surviving terms, ordered by increasing weight.  Empty when every
        weight was zero; public input validation rejects that degenerate case.
    """
    keep = weights != 0.0
    weights, df, noncentrality = weights[keep], df[keep], noncentrality[keep]

    unique_weights, index = np.unique(weights, return_inverse=True)
    n_unique = unique_weights.size
    return (
        unique_weights,
        np.bincount(index, weights=df, minlength=n_unique),
        np.bincount(index, weights=noncentrality, minlength=n_unique),
    )


def _standard_deviation(
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> float:
    r"""
    Standard deviation of ``Q``, formed without squaring raw-scale inputs.

    The variance is

    .. math::

        \mathrm{sd}^2 =
            \sum_j w_j^2 \big(2 \nu_j + 4 \delta_j^2\big),

    but evaluating that directly squares the raw weights, so it overflows to
    infinity once the sum passes the largest representable double and underflows
    to zero for very small weights.  Both are silent: an infinite ``sd`` sends
    every normalised weight and the standardised frequency to zero, which returns
    a raw CDF of exactly ``0.5``.

    Factoring out the largest magnitude in the problem avoids both failure modes
    over the practically relevant float64 range.  Every ratio ``w_j / scale`` is
    then at most one, so no square can overflow, and the term attaining the
    maximum contributes at least ``2 nu_j``, so the scaled sum cannot underflow
    to zero either.

    Parameters
    ----------
    weights, df, noncentrality : numpy.ndarray
        The per-term weights :math:`w_j`, degrees of freedom :math:`\nu_j`, and
        non-centrality parameters :math:`\delta_j^2`.

    Returns
    -------
    float
        The standard deviation of ``Q``.  Strictly positive, since the caller has
        already checked that at least one weight is non-zero.
    """
    scale = float(np.max(np.abs(weights)))
    # The same variance measured in units of ``scale``, so of order sqrt(sum df).
    unit = np.sqrt(np.sum((weights / scale) ** 2 * (2.0 * df + 4.0 * noncentrality)))
    return scale * float(unit)


def _reduce(
    z: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> float | None:
    r"""
    ``Pr(Q <= q)`` in closed form where one exists, ``None`` otherwise.

    Three shapes are exact, and the last of them is the one the quadrature is
    worst on, so this runs before any numerical decision is made.

    **One term.**  :math:`w X` is a scaled (non-central) chi-square.  Dividing by
    a negative weight reverses the inequality, so the tails swap.

    **All weights positive, at a non-positive point.**  :math:`Q` is a positive
    combination of positive variables, hence positive almost surely, so the
    lower-tail probability is exactly zero.  Degrees of freedom are validated
    positive, and a non-centrality only shifts mass further right, so neither
    matters here.

    **Two terms of opposite sign, both central, at exactly zero.**  Writing
    :math:`Q = w_+ X_m + w_- Y_n` with :math:`w_+ > 0 > w_-`,

    .. math::

        \Pr(Q > 0) = \Pr(w_+ X_m > -w_- Y_n)
                   = \Pr\!\left(F_{m,n} > \frac{-w_-\, n}{w_+\, m}\right),

    which holds for any positive real :math:`m` and :math:`n`, not only integer
    degrees of freedom.  This is mgcv's estimated-dispersion smooth-term test
    after equal positive weights have been merged: with numerator rank ``r``,
    denominator degrees of freedom ``k0`` and test statistic ``d``, the weights
    are ``[1, -d/k0]`` and the p-value is ``f.sf(d/r, r, k0)``.

    Parameters
    ----------
    z : float
        Standardised evaluation point :math:`q / \mathrm{sd}`.
    weights, df, noncentrality : numpy.ndarray
        Standardised weights :math:`w_j / \mathrm{sd}`, degrees of freedom
        :math:`\nu_j` and non-centrality parameters :math:`\delta_j^2`, already
        collapsed to a canonical term list ordered by increasing weight.

    Returns
    -------
    float or None
        The lower-tail probability, or ``None`` when no closed form applies, in
        which case the caller consults :func:`_regime_gate`.
    """
    central = not np.any(noncentrality)

    if weights.size == 1:
        x = z / weights[0]
        if weights[0] > 0.0:
            if central:
                return float(chi2.cdf(x, df[0]))
            return float(ncx2.cdf(x, df[0], noncentrality[0]))
        if central:
            return float(chi2.sf(x, df[0]))
        return float(ncx2.sf(x, df[0], noncentrality[0]))

    if np.all(weights > 0.0) and z <= 0.0:
        return 0.0

    if z == 0.0 and central and weights.size == 2 and weights[0] < 0.0 < weights[1]:
        w_neg, w_pos = float(weights[0]), float(weights[1])
        n, m = float(df[0]), float(df[1])
        return 1.0 - float(f_dist.sf((-w_neg * n) / (w_pos * m), m, n))

    return None


def _regime_gate(
    z: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> None:
    r"""
    Refuse anything outside the regime this module is validated on.

    What reaches the quadrature is only what the GAM smooth-term tests produce:
    positive weights at any finite point (known dispersion, ``q = d``), and
    mixed-sign weights at exactly zero (estimated dispersion, the random
    denominator moved to the left-hand side).  Both central.

    Everything else is refused rather than integrated, because outside those two
    shapes there is no oracle for the answer, and the integrator's own error
    estimate is not one: it reports success with a tiny absolute error while
    being wrong in the second decimal place.  A single-term list never arrives
    here, since :func:`_reduce` answers all of those exactly.

    Parameters
    ----------
    z : float
        Standardised evaluation point :math:`q / \mathrm{sd}`.
    weights, df, noncentrality : numpy.ndarray
        Standardised weights, degrees of freedom and non-centrality parameters,
        as passed to :func:`_reduce`.

    Raises
    ------
    NotImplementedError
        If the standardised inputs fall outside the two supported shapes.  The
        message carries them, since the raw inputs do not determine what the
        method sees.
    """
    all_positive = bool(np.all(weights > 0.0))
    mixed_signs = bool(np.any(weights > 0.0) and np.any(weights < 0.0))
    central = not np.any(noncentrality)

    if all_positive and central:
        return
    elif mixed_signs and central and z == 0.0:
        return
    else:
        raise NotImplementedError(
            "psum_chisq: this input is outside the validated GAM regime, which "
            "is all-positive weights at any finite q, or mixed-sign weights at "
            "exactly q = 0, both with zero non-centrality. Got "
            f"z={z!r}, weights={weights.tolist()}, df={df.tolist()}, "
            f"noncentrality={noncentrality.tolist()}, where z and the weights "
            "are divided by the standard deviation of Q. Please report this "
            "case if you need it."
        )


def psum_chisq(
    q: ArrayLike,
    weights: ArrayLike,
    df: ArrayLike = 1.0,
    noncentrality: ArrayLike = 0.0,
    lower_tail: bool = False,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 200,
) -> float | FloatArray:
    r"""
    Distribution function of a weighted sum of chi-squared variables.

    Evaluates :math:`\Pr(Q \le q)` (or the upper tail) for
    :math:`Q = \sum_j w_j X_j`, where
    :math:`X_j \sim \chi^2_{\nu_j}(\delta_j^2)`, by numerically inverting the
    characteristic function of ``Q``.

    Parameters
    ----------
    q : array_like
        Point(s) at which to evaluate the distribution.  May be infinite, which
        is answered exactly.  Must not be NaN.
    weights : array_like
        The weights :math:`w_j`; may be positive or negative, and must be finite.
        At least one must be non-zero.
    df : array_like, optional
        Degrees of freedom :math:`\nu_j` (must be positive and finite).  A scalar
        is applied to every term.  Defaults to ``1``.
    noncentrality : array_like, optional
        Non-centrality parameters :math:`\delta_j^2` (must be non-negative and
        finite).  A scalar is applied to every term.  Defaults to ``0``
        (central).
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
        The (survival) probability.  A Python ``float`` is
        returned for scalar ``q``; otherwise an array matching ``q``.

    Raises
    ------
    ValueError
        If any argument lies outside the domain described above.
    NotImplementedError
        If the sum is outside the supported regime described in the notes.

    Notes
    -----
    The supported regime is the one the GAM smooth-term tests produce: weights
    that are all positive at any finite ``q``, and weights of mixed sign at
    exactly ``q = 0``, both with zero non-centrality.  Anything else raises,
    rather than returning a number no oracle has checked.

    Several shapes never reach the quadrature at all, because they have exact
    closed forms: a single term of either sign, positive weights at a
    non-positive ``q``, and two central terms of opposite sign at ``q = 0``,
    which is an F survival probability.  Non-centrality is therefore honoured
    only for a single term, where the answer is a non-central chi-square.

    Terms are canonicalised before anything is evaluated: zero-weight terms are
    dropped and equal weights are merged, both exactly.

    There is no additive normal term.  Davies' general form carries one,
    :math:`Q = \sum_j w_j X_j + \sigma Z`, and mgcv's ``psum.chisq`` exposes it
    as ``sigz``. A GAM test statistic is a pure quadratic form in the coefficients,
    so it never produces one, and no mgcv call site passes a non-zero ``sigz``.
    The argument was therefore removed rather than kept as a value that must always
    be zero.

    The problem is nondimensionalised before any numerical decision is made:
    ``q`` becomes ``z = q / sd`` and the weights are divided by ``sd``, the
    standard deviation of ``Q``.  The integrand is then split into a
    non-oscillatory head on ``[0, 1]`` in those standardised coordinates and a
    semi-infinite oscillatory tail integrated with SciPy's Fourier quadrature,
    which remains accurate for the slowly decaying tails produced by low degrees
    of freedom.
    """
    weight_arr = np.atleast_1d(np.asarray(weights, dtype=float))
    n_terms = int(weight_arr.size)
    df_arr = _broadcast(df, n_terms, "df")
    ncp_arr = _broadcast(noncentrality, n_terms, "noncentrality")
    q_arr = np.atleast_1d(np.asarray(q, dtype=float))

    _validate_inputs(q_arr, weight_arr, df_arr, ncp_arr)
    weight_arr, df_arr, ncp_arr = _collapse_terms(weight_arr, df_arr, ncp_arr)

    # Nondimensionalize: substituting t = sd * u maps the inversion integral
    # onto itself with weights w/sd, frequency z = q/sd and a split point of 1.
    # Every numerical decision below is then made on unit-free quantities, so
    # the result cannot depend on the units of q and the weights.
    sd = _standard_deviation(weight_arr, df_arr, ncp_arr)
    std_weights = weight_arr / sd

    out = np.empty(q_arr.shape, dtype=float)
    for i in range(q_arr.size):
        z = float(q_arr.flat[i]) / sd
        # An infinite z is q at infinitely many standard deviations, which is
        # the same statement as an infinite q.  Both are exact, and neither is
        # a question the quadrature can be asked.
        if np.isposinf(z):
            cdf = 1.0
        elif np.isneginf(z):
            cdf = 0.0
        elif np.isfinite(z):
            reduced = _reduce(z, std_weights, df_arr, ncp_arr)
            if reduced is None:
                _regime_gate(z, std_weights, df_arr, ncp_arr)
                cdf, _cdf_error = _cdf_single(
                    z,
                    std_weights,
                    df_arr,
                    ncp_arr,
                    1.0,
                    epsabs,
                    epsrel,
                    limit,
                )
            else:
                cdf = reduced
        else:
            raise RuntimeError(
                f"standardized evaluation point is not a number: "
                f"q={q_arr.flat[i]}, sd={sd}"
            )
        out.flat[i] = cdf if lower_tail else 1.0 - cdf

    if np.isnan(out).any():
        warnings.warn("psum_chisq: quadrature produced NaN", stacklevel=2)

    if np.any(out < 0.0) or np.any(out > 1.0):
        raise RuntimeError("Probabilities must be in [0, 1].")

    if np.ndim(q) == 0:
        return out.item()
    return out.reshape(np.shape(q))

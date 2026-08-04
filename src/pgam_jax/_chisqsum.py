r"""Distribution of a linear combination of chi-squared random variables.

Computes the (survival) probability of

.. math::

    Q = \sum_j w_j X_j,

where :math:`X_j \sim \chi^2_{\nu_j}(\delta_j^2)` are independent (possibly
non-central) chi-squared variables.  Weights may be of either sign.  This is the
null distribution used to obtain covariate-inclusion p-values for penalised GAM
smooth terms.

The general case is obtained by inverting the characteristic function of ``Q``
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
from scipy.integrate import quad, tanhsinh
from scipy.stats import chi2
from scipy.stats import f as f_dist
from scipy.stats import ncx2

__all__ = ["psum_chisq"]

FloatArray = NDArray[np.float64]

_Z_SWITCH = 5e-3
_TANHSINH_ATOL = 1e-13
_TANHSINH_RTOL = 1e-13
# Levels 6 through 8 can stop on an aliased node set for a positive three-term
# mixture near the handover even with ``success=True``.  Level 9 is the first
# forced refinement that meets the independent-oracle calibration.
_TANHSINH_MINLEVEL = 9

_TANHSINH_MAXLEVEL = 20

# QUADPACK's QAWF stops after this many oscillation cycles. SciPy's own default
# is 50 and is not part of its documented signature, so it is set here to keep
# the tail integrations reproducible across SciPy versions.
_LIMLST = 200

# The two quadrature routes, named so that the dispatcher, the cross-check and
# the failure messages all refer to them the same way.
_TANHSINH = "tanh-sinh"
_QAWF = "QAWF"

# Where each route is recomputed for its cross-check. Neither value changes the
# integral; both change the nodes used to find it.
_QAWF_CROSS_CHECK_SPLIT = 2.5
_TANHSINH_CROSS_CHECK_SPLIT = 1.0

# How far two independent computations of the same integral may sit apart before
# the disagreement is treated as evidence that one of them is wrong, as a
# multiple of their combined error estimates. Calibrated, not assumed: over 335
# healthy in-contract cases the worst ratio of disagreement to allowance was
# 0.018 for QAWF, and for tanh-sinh the ratio is meaningless because its
# estimates go to zero, which is what the floor below is for.
_ERROR_ESTIMATE_SAFETY_FACTOR = 8.0

# Both routes can report an error of exactly zero and still differ in the last
# bits, so the allowance never falls below this. It is the noise level of the
# comparison rather than an accuracy target: the worst disagreement measured
# over those same healthy cases was 2.5e-12, and this sits above it with room.
# The resulting trip point of 8e-11 is eight orders of magnitude below the
# smallest genuine failure this catches, which is 1.4e-2. A caller who requests
# an epsabs far below 1e-11 gets a cross-check no finer than this, because two
# routes that disagree by 2.5e-12 on healthy input cannot testify about less.
_CROSS_CHECK_FLOOR = 1e-11

# A probability this many times its own estimated error, or less, is reported as
# zero rather than as noise. One means a value is kept as soon as it exceeds
# its own error bar.
_FLOOR_SAFETY_FACTOR = 1.0

# The counterpart of _CROSS_CHECK_FLOOR: an error estimate of exactly zero is
# not a claim that the answer is exact to the last bit, and the probability is
# still assembled by two roundings. A rank-4 smooth at a near-zero statistic
# overshoots 1 by one ulp with an estimate of 0.0. Eight ulp leaves room and
# still sits fourteen orders below the smallest real violation, F1's 0.17.
_RANGE_CHECK_FLOOR = 8.0 * float(np.finfo(np.float64).eps)


class _QuadratureNotConverged(RuntimeError):
    """
    One quadrature route declined to answer.

    This is separate from a plain :class:`RuntimeError` because the two mean
    different things to the caller. A non-finite value or a negative error
    estimate is corruption, and nothing may be built on it. Non-convergence is
    a route reporting that it could not resolve this integrand to the requested
    accuracy, which leaves the other route free to try. Only the latter is
    caught by :func:`_cdf_approx`.

    It subclasses :class:`RuntimeError` so that callers who only care that the
    computation failed need not know about the distinction.
    """


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
    limlst: int | None = None,
) -> tuple[float, float]:
    """Thin typed wrapper around :func:`scipy.integrate.quad`.

    SciPy ships no type stubs, so its return value is untyped; isolating the call
    here keeps that "unknown" type from propagating through the module. When
    ``weight`` is ``None`` this is ordinary adaptive quadrature; when it is
    ``"cos"``/``"sin"`` SciPy uses its oscillatory Fourier integrator with angular
    frequency ``wvar``.

    ``limlst`` caps the number of oscillation cycles and applies only to the
    Fourier integrator, so it is passed only when one is requested. Leaving it
    unset would inherit SciPy's undocumented default.

    ``epsrel`` reaches QUADPACK only on the non-oscillatory branch. QAWF takes
    an absolute request alone, so on the tails the relative target is ignored.

    Returns the integral and its estimated absolute error. Raises
    :class:`_QuadratureNotConverged` if QUADPACK reports non-convergence, and
    :class:`RuntimeError` if it returns a non-finite value or an invalid error
    estimate.
    """
    extra: dict[str, int] = {} if limlst is None else {"limlst": limlst}
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
        **extra,
    )
    value, abs_error, _info, *diagnostics = result
    name = getattr(func, "__name__", "integrand")

    if diagnostics:
        raise _QuadratureNotConverged(f"{name} quadrature failed: {diagnostics[0]}")

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
    :math:`\sin\!\big(\phi(u) - q u\big)\, e^{-\psi(u)} / u`. This returns the
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


def _divide_with_fallback(
    numerator: FloatArray,
    denominator: FloatArray,
    fallback: float,
) -> FloatArray:
    """
    Divide elementwise, using ``fallback`` where the denominator is zero.

    ``numerator`` and ``denominator`` are arrays of tanh-sinh nodes, so this is
    the vectorized equivalent of an ``if denominator != 0`` at every node.
    NumPy does not evaluate the division where the mask is false. Those entries
    retain the fallback with which ``result`` was initialized.
    """
    result = np.full_like(numerator, fallback)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0.0,
    )
    return result


def _combined_integrand(
    u: FloatArray,
    q: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> FloatArray:
    r"""
    Combined Imhof integrand, vectorised for tanh-sinh quadrature.

    Unlike the QAWF path, this keeps :math:`\sin(\phi(u) - q u)` intact and
    integrates it over :math:`[0, \infty)` without a head/tail split.  SciPy's
    infinite-interval transform evaluates the function at both zero and values
    near the largest representable float, so both ends require explicit care.

    At zero, the apparent :math:`1/u` singularity is removable:

    .. math::

        \lim_{u \to 0}
        \frac{\sin(\phi(u) - q u)e^{-\psi(u)}}{u}
        = \sum_j w_j(\nu_j + \delta_j^2) - q.

    At the other end, the ratios involving :math:`x_j = 2w_ju` are expressed
    through :math:`h_j = \operatorname{hypot}(1, x_j) = \sqrt{1+x_j^2}`.
    Unlike forming :math:`x_j^2` directly, ``hypot`` remains finite whenever
    ``x_j`` does.
    """
    u_arr = np.asarray(u, dtype=float)

    # The final axis is the term axis. Every preceding axis indexes a batch of
    # tanh-sinh nodes. Some infinite-interval nodes are so large that this
    # product legitimately overflows to +/-inf.
    with np.errstate(over="ignore"):
        x = 2.0 * u_arr[..., np.newaxis] * weights

    # h = sqrt(1 + x**2) and s = x/h. np.hypot forms h without
    # overflowing when x is a large finite number.
    hypot_x = np.hypot(1.0, x)

    # This is an elementwise if: finite x uses x/h, while an x that overflowed
    # to +/-inf uses its analytic limit, +/-1
    with np.errstate(invalid="ignore"):
        unit_x = np.where(np.isfinite(x), x / hypot_x, np.sign(x))

    # The three potentially troublesome expressions now share the same stable
    # h and s:
    #   x / (1 + x**2) = s / h
    #   x**2 / (1 + x**2) = s**2
    #   log(1 + x**2) = 2 log(h)
    x_over_one_plus_x_sq = unit_x / hypot_x
    x_sq_over_one_plus_x_sq = unit_x**2
    log_one_plus_x_sq = 2.0 * np.log(hypot_x)

    phase = 0.5 * np.sum(
        df * np.arctan(x) + noncentrality * x_over_one_plus_x_sq,
        axis=-1,
    )
    log_modulus = -0.25 * np.sum(df * log_one_plus_x_sq, axis=-1)
    log_modulus -= 0.5 * np.sum(
        noncentrality * x_sq_over_one_plus_x_sq,
        axis=-1,
    )
    numerator = np.sin(phase - q * u_arr) * np.exp(log_modulus)

    # The apparent 0/0 at u=0 has this analytic limit. The helper performs the
    # division only at nonzero nodes and inserts the limit at zero.
    limit_at_zero = float(np.sum(weights * (df + noncentrality)) - q)
    return _divide_with_fallback(numerator, u_arr, limit_at_zero)


def _tanhsinh_piece(
    integrand: Callable[[FloatArray], FloatArray],
    a: float,
    b: float,
) -> tuple[float, float]:
    """One tanh-sinh integration, with its convergence report turned into a raise."""
    result = tanhsinh(
        integrand,
        a,
        b,
        atol=_TANHSINH_ATOL,
        rtol=_TANHSINH_RTOL,
        minlevel=_TANHSINH_MINLEVEL,
        maxlevel=_TANHSINH_MAXLEVEL,
    )

    integral = float(np.asarray(result.integral).item())
    abs_error = float(np.asarray(result.error).item())
    success = bool(np.asarray(result.success).item())
    if not success:
        status = int(np.asarray(result.status).item())
        nfev = int(np.asarray(result.nfev).item())
        maxlevel = int(np.asarray(result.maxlevel).item())
        raise _QuadratureNotConverged(
            "tanh-sinh quadrature failed: "
            f"status={status}, error={abs_error}, nfev={nfev}, "
            f"maxlevel={maxlevel}"
        )
    if not np.isfinite(integral):
        raise RuntimeError("tanh-sinh quadrature returned a non-finite value")
    if not np.isfinite(abs_error) or abs_error < 0.0:
        raise RuntimeError(
            f"tanh-sinh quadrature returned an invalid error estimate: {abs_error}"
        )

    return integral, abs_error


def _cdf_tanhsinh(
    q: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    domain_split: float | None = None,
) -> tuple[float, float]:
    """
    ``Pr(Q <= q)`` from the unsplit Imhof integrand via tanh-sinh.

    ``domain_split`` exists for the cross-check.  The integral over
    ``[0, inf)`` does not depend on where it is cut, but cutting it puts a
    different set of tanh-sinh nodes under the same integrand, so a run that
    resolved the mass badly moves.
    """

    def integrand(u: FloatArray) -> FloatArray:
        return _combined_integrand(u, q, weights, df, noncentrality)

    if domain_split is None:
        integral, abs_error = _tanhsinh_piece(integrand, 0.0, np.inf)
    else:
        head, head_error = _tanhsinh_piece(integrand, 0.0, domain_split)
        tail, tail_error = _tanhsinh_piece(integrand, domain_split, np.inf)
        integral, abs_error = head + tail, head_error + tail_error

    return 0.5 - integral / np.pi, abs_error / np.pi


def _cdf_qawf(
    q: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    split: float,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> tuple[float, float]:
    """
    ``Pr(Q <= q)`` and its estimated absolute error for a scalar ``q``.

    The value is assembled from three integrations, so ``epsabs`` is divided
    between them rather than handed to each in full.  Requesting the whole
    budget three times makes the public tolerance a per-piece figure that the
    end-to-end result can exceed, which is the accounting half of F6.  The
    tails also cancel against each other, so their errors are added, not
    combined in quadrature.

    ``epsrel`` reaches only the head.  QAWF accepts an absolute request alone.
    """
    params: tuple[object, ...] = (weights, df, noncentrality)
    piece_epsabs = epsabs / 3.0
    head, head_error = _quad(
        _head_integrand,
        0.0,
        split,
        (q, *params),
        None,
        None,
        piece_epsabs,
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
        piece_epsabs,
        epsrel,
        limit,
        _LIMLST,
    )
    tail_sin, tail_sin_error = _quad(
        _tail_sin_coefficient,
        split,
        np.inf,
        params,
        "sin",
        q,
        piece_epsabs,
        epsrel,
        limit,
        _LIMLST,
    )
    cdf = 0.5 - (head + tail_cos - tail_sin) / np.pi
    quadrature_error = (head_error + tail_cos_error + tail_sin_error) / np.pi

    return cdf, quadrature_error


def _route(
    name: str,
    q: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    split: float,
    epsabs: float,
    epsrel: float,
    limit: int,
    independent: bool,
) -> tuple[float, float]:
    """
    ``Pr(Q <= q)`` by the named quadrature route.

    With ``independent`` set, the same integral is computed a second way: the
    value is mathematically identical, but the nodes are not.  QAWF moves its
    split point, which is where its cycle grid starts, so a run that missed
    mass lands somewhere else. tanh-sinh cuts the domain in two, which
    replaces its node set entirely.
    """
    if name == _TANHSINH:
        return _cdf_tanhsinh(
            q,
            weights,
            df,
            noncentrality,
            _TANHSINH_CROSS_CHECK_SPLIT if independent else None,
        )
    elif name == _QAWF:
        return _cdf_qawf(
            q,
            weights,
            df,
            noncentrality,
            _QAWF_CROSS_CHECK_SPLIT if independent else split,
            epsabs,
            epsrel,
            limit,
        )
    else:
        raise NotImplementedError(f"unknown quadrature route: {name!r}")


def _cross_check(
    name: str,
    value: float,
    error: float,
    q: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    split: float,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> None:
    """
    Confirm a quadrature result against a numerically independent recomputation.

    This is the only guard here that can catch a failure mode nobody has
    catalogued.  It exists because F4 established that QUADPACK's own
    diagnostics are blind in this problem: it reports ``ier=0`` and an absolute
    error of ``4e-15`` on a value that is wrong in the second decimal place.
    An error estimate produced by the same nodes that missed the mass cannot
    detect that the mass was missed.  A different set of nodes can.

    The two outcomes are deliberately different exceptions. A recomputation
    that will not converge leaves this route unvalidated, which is the same
    thing to the caller as this route not converging, so
    :class:`_QuadratureNotConverged` is raised and the dispatcher may try the
    other route. A recomputation that converges to a *different* answer means
    one of the two is confidently wrong, which is not a reason to quietly
    change method, so that raises :class:`RuntimeError` and stops.

    Raises
    ------
    _QuadratureNotConverged
        If the independent recomputation does not converge.
    RuntimeError
        If the two routes converge to answers further apart than their combined
        error estimates allow.
    """
    other, other_error = _route(
        name,
        q,
        weights,
        df,
        noncentrality,
        split,
        epsabs,
        epsrel,
        limit,
        independent=True,
    )

    difference = abs(value - other)
    # The estimates are what is on trial here, so they cannot be the whole
    # allowance.  The floor covers the case where both routes report an error
    # of exactly zero and still differ in the last bits.
    allowance = max(error + other_error, _CROSS_CHECK_FLOOR)
    if difference > allowance * _ERROR_ESTIMATE_SAFETY_FACTOR:
        raise RuntimeError(
            f"psum_chisq: the {name} quadrature did not survive its "
            f"independent cross-check. It returned {value!r} with an estimated "
            f"error of {error!r}, while recomputing the same integral with a "
            f"different node set returned {other!r} with an estimated error of "
            f"{other_error!r}. The difference is {difference!r}, which exceeds "
            f"the combined estimate by more than the safety factor of "
            f"{_ERROR_ESTIMATE_SAFETY_FACTOR}. At least one of the two is "
            f"wrong. Standardized inputs: z={q!r}, "
            f"weights={weights.tolist()}, df={df.tolist()}, "
            f"noncentrality={noncentrality.tolist()}. Please report this case."
        )


def _cdf_approx(
    q: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    split: float,
    epsabs: float,
    epsrel: float,
    limit: int,
    check: bool = True,
) -> tuple[float, float]:
    """
    ``Pr(Q <= q)`` using whichever quadrature can actually resolve this input.

    ``abs(q)`` picks the route that should work: tanh-sinh handles the small
    frequencies where QAWF's first cycle steps over the whole integrand, and
    QAWF handles everything above, where tanh-sinh needs impractically many
    nodes.  That choice is a prediction, not a fact, and it is wrong for slowly
    decaying integrands.  The envelope falls off like
    ``u**-(1 + sum(df)/2)``, so a total of two degrees of freedom, which is what
    mgcv's fractional-rank test produces at known dispersion, decays only like
    ``u**-2`` and tanh-sinh will not converge on it anywhere in its own band.

    A route that declines is therefore not fatal: the other one is tried.  What
    makes that safe rather than a guess is the cross-check, which the fallback
    result must pass even when the caller asked to skip checking.  QAWF is
    silently wrong at small ``abs(q)`` on some structures, and falling back onto
    an unvalidated wrong answer would reintroduce exactly the failure this
    module exists to prevent.

    Raises
    ------
    RuntimeError
        If neither route produces a validated value.  The message names both.
    """
    if abs(q) <= _Z_SWITCH:
        preferred, fallback = _TANHSINH, _QAWF
    else:
        preferred, fallback = _QAWF, _TANHSINH

    arguments = (q, weights, df, noncentrality, split, epsabs, epsrel, limit)

    try:
        value, error = _route(preferred, *arguments, independent=False)
        if check:
            _cross_check(preferred, value, error, *arguments)
        return value, error
    except _QuadratureNotConverged as preferred_failure:
        try:
            value, error = _route(fallback, *arguments, independent=False)
            # Not conditional on ``check``.  The dispatcher predicted this route
            # would be the wrong one, so its result is only admissible with
            # independent evidence behind it.
            _cross_check(fallback, value, error, *arguments)
            return value, error
        except _QuadratureNotConverged as fallback_failure:
            raise RuntimeError(
                f"psum_chisq: neither quadrature route resolved this input. "
                f"The {preferred} route was chosen for z={q!r} and reported: "
                f"{preferred_failure}. Falling back to the {fallback} route "
                f"reported: {fallback_failure}. Standardized inputs: "
                f"weights={weights.tolist()}, df={df.tolist()}, "
                f"noncentrality={noncentrality.tolist()}, where z and the "
                f"weights are divided by the standard deviation of Q. Please "
                f"report this case."
            ) from fallback_failure


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


def _two_chi1_angular_integrand(
    theta: float,
    z: float,
    first_weight: float,
    second_weight: float,
    lower_tail: bool,
) -> float:
    """
    Angular CDF or survival integrand for two positive central chi-squares.

    If ``X1`` and ``X2`` are independent ``chi2_1`` variables, write their
    underlying normals in polar coordinates.  The squared radius is ``chi2_2``
    and the angle is uniform, which leaves a smooth integral over one quadrant.
    """
    cosine = np.cos(theta)
    cosine_squared = cosine * cosine
    scale = first_weight * cosine_squared + second_weight * (1.0 - cosine_squared)
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        exponent = -z / (2.0 * scale)
        if lower_tail:
            return float(-np.expm1(exponent))
        return float(np.exp(exponent))


def _two_chi1_probability(
    z: float,
    weights: FloatArray,
    lower_tail: bool,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> tuple[float, float]:
    """
    Evaluate one tail of two positive central ``chi2_1`` terms.

    The requested tail is integrated directly.  In particular, the lower-tail
    integrand uses ``-expm1`` rather than subtracting a survival probability
    close to one.  The quadrant is split where the two weighted angular
    contributions are equal and, when it lies inside the range of angular
    scales, where that scale equals ``z``.  These points expose the narrow
    endpoint layer that appears when one weight is almost zero.
    """
    first_weight, second_weight = sorted(map(float, weights))
    arguments: tuple[object, ...] = (
        z,
        first_weight,
        second_weight,
        lower_tail,
    )

    # Multiplication by 2 / pi turns the angular integral into a probability.
    # Dividing epsabs between every interval keeps their combined probability
    # error within the caller's end-to-end absolute budget.
    equal_contributions = np.arctan(np.sqrt(first_weight / second_weight))
    breakpoints = [0.0, float(equal_contributions), np.pi / 4.0, np.pi / 2.0]
    endpoint_layer = float(equal_contributions)
    if first_weight < z < second_weight:
        scale_fraction = (z - first_weight) / (second_weight - first_weight)
        matching_scale = np.arcsin(np.sqrt(scale_fraction))
        breakpoints.append(float(matching_scale))
        endpoint_layer = max(endpoint_layer, float(matching_scale))
    if endpoint_layer < np.pi / 4.0:
        breakpoints.append(float(np.sqrt(endpoint_layer * np.pi / 4.0)))
    breakpoints = sorted(set(breakpoints))

    n_intervals = len(breakpoints) - 1
    piece_epsabs = epsabs * np.pi / (2.0 * n_intervals)
    pieces = [
        _quad(
            _two_chi1_angular_integrand,
            left,
            right,
            arguments,
            None,
            None,
            piece_epsabs,
            epsrel,
            limit,
        )
        for left, right in zip(breakpoints[:-1], breakpoints[1:])
    ]
    factor = 2.0 / np.pi
    return (
        factor * sum(value for value, _ in pieces),
        factor * sum(error for _, error in pieces),
    )


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


def _cdf_single(
    q: float,
    sd: float,
    std_weights: FloatArray,
    df_arr: FloatArray,
    ncp_arr: FloatArray,
    epsabs: float,
    epsrel: float,
    limit: int,
    check: bool,
    lower_tail: bool,
) -> tuple[float, float]:
    """
    The requested tail and its absolute error at one point.

    Takes the raw ``q`` and the standard deviation of ``Q``, and standardises
    them here.  The weights arrive already standardised, which is what the
    ``std_`` prefix marks.  Closed-form answers return a zero error, since
    nothing was integrated to obtain them.  The specialized two-``chi2_1``
    reduction integrates the requested tail directly.
    """
    z = float(q) / sd
    # An infinite z is q at infinitely many standard deviations, which is
    # the same statement as an infinite q.  Both are exact, and neither is
    # a question the quadrature can be asked.
    if np.isposinf(z):
        return (1.0 if lower_tail else 0.0), 0.0
    if np.isneginf(z):
        return (0.0 if lower_tail else 1.0), 0.0
    if not np.isfinite(z):
        raise RuntimeError(
            f"standardized evaluation point is not a number: q={q}, sd={sd}"
        )

    reduced = _reduce(z, std_weights, df_arr, ncp_arr)
    if reduced is not None:
        return (reduced if lower_tail else 1.0 - reduced), 0.0

    central = not np.any(ncp_arr)
    two_positive_chi1 = (
        central
        and std_weights.size == 2
        and bool(np.all(std_weights > 0.0))
        and bool(np.all(df_arr == 1.0))
    )
    if two_positive_chi1:
        return _two_chi1_probability(
            z,
            std_weights,
            lower_tail,
            epsabs,
            epsrel,
            limit,
        )

    _regime_gate(z, std_weights, df_arr, ncp_arr)
    cdf, cdf_error = _cdf_approx(
        z,
        std_weights,
        df_arr,
        ncp_arr,
        1.0,
        epsabs,
        epsrel,
        limit,
        check,
    )
    return (cdf if lower_tail else 1.0 - cdf), cdf_error


def _finalize(
    probabilities: FloatArray,
    errors: FloatArray,
) -> FloatArray:
    """
    Bring computed probabilities into ``[0, 1]``, or refuse to.

    Two things happen here, and both are about the same fact: a quadrature
    result is a number plus an uncertainty, and reading it as though it were
    exact produces nonsense at the ends of the range.

    **The range check.**  A survival probability computed as ``0.5`` minus an
    integral can land just outside ``[0, 1]`` by less than its own error
    estimate.  That is arithmetic, not a bug, and clipping it is the correct
    reading.  Landing outside by more than the error allowance is a bug, and
    that still raises.  The old check had no allowance at all and rejected an
    overshoot of ``-1.3e-15`` on a value whose estimated error was ``3.8e-13``.

    The allowance never falls below :data:`_RANGE_CHECK_FLOOR`, or an estimate
    of exactly zero would reject a one-ulp overshoot.  A rank-4 smooth at a
    near-zero statistic does exactly that.

    **The floor.** Once the true probability drops below the resolution of the
    quadrature, what comes back is not a small number but noise, and noise is
    not monotone.  For ``[1, 0.6, 0.4]`` with ``df=[3, 1, 1]`` the survival
    function is genuine at ``q = 60`` (``1.2e-12``, against an estimated error
    of ``4.4e-13``) and pure noise by ``q = 80`` (``-1.3e-15``, against
    ``3.7e-13``).  Anything that does not exceed its own error estimate is
    reported as ``0.0`` with a warning, which is both honest and monotone.

    The floor is driven by the error estimate rather than by a fixed constant so
    that it cannot touch the exact closed forms.  Those return an error of
    exactly zero, having integrated nothing, and their deep-tail values are good
    to full relative precision.

    Parameters
    ----------
    probabilities : numpy.ndarray
        The probabilities in the tail the caller asked for.
    errors : numpy.ndarray
        Estimated absolute error of each, zero where the answer is exact.

    Returns
    -------
    numpy.ndarray
        The probabilities, clipped and floored.

    Raises
    ------
    RuntimeError
        If any probability lies outside ``[0, 1]`` by more than its error
        allowance.
    """
    allowance = np.maximum(errors * _ERROR_ESTIMATE_SAFETY_FACTOR, _RANGE_CHECK_FLOOR)
    overshoot = np.maximum(-probabilities, probabilities - 1.0)
    if np.any(overshoot > allowance):
        worst = int(np.argmax(overshoot - allowance))
        raise RuntimeError(
            "psum_chisq: computed a probability outside [0, 1] by more than "
            f"its own error allowance. Got {probabilities.flat[worst]!r} with "
            f"an estimated error of {errors.flat[worst]!r}."
        )

    probabilities = np.clip(probabilities, 0.0, 1.0)

    unresolved = (errors > 0.0) & (probabilities < errors * _FLOOR_SAFETY_FACTOR)
    if np.any(unresolved):
        warnings.warn(
            "psum_chisq: the probability is smaller than the quadrature can "
            "resolve at this tolerance and has been returned as 0.0. Tighten "
            "epsabs to resolve it, or read the result as an upper bound.",
            stacklevel=3,
        )
        probabilities = np.where(unresolved, 0.0, probabilities)

    return probabilities


def psum_chisq(
    q: ArrayLike,
    weights: ArrayLike,
    df: ArrayLike = 1.0,
    noncentrality: ArrayLike = 0.0,
    lower_tail: bool = False,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 200,
    check: bool = True,
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
        Absolute and relative accuracy targets.  ``epsabs`` is an end-to-end
        budget: the oscillatory route assembles its answer from three
        integrations and divides it between them.  For that route, ``epsrel``
        reaches only the non-oscillatory head, since QUADPACK's Fourier
        integrator accepts an absolute request alone.  The two-term angular
        reduction applies both tolerances to its two finite pieces.
        ``epsabs`` also sets the resolution floor described below.
    limit : int, optional
        Maximum number of quadrature subintervals.
    check : bool, optional
        Recompute every characteristic-function quadrature result by a
        numerically independent route and raise if the two disagree by more
        than their combined error estimates allow.  Defaults to ``True``, and
        roughly doubles the cost.  It is worth that: the integrator's own
        diagnostics report success with an absolute error of ``4e-15`` on
        values that are wrong in the second decimal place, so nothing else
        here can catch a failure mode that has not already been catalogued.  A
        result obtained from the route that ``q`` did *not* select is
        cross-checked even when this is ``False``.  The positive,
        two-``chi2_1`` angular reduction is a bounded non-oscillatory integral
        and uses its direct quadrature error instead.

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
    RuntimeError
        If no quadrature route resolves the input, or if a result fails its
        independent cross-check.  Both messages name what was tried.

    Warns
    -----
    UserWarning
        If the probability is smaller than the quadrature can resolve at the
        requested ``epsabs``, in which case ``0.0`` is returned.  Below that
        resolution what comes back is not a small number but noise, which is
        not even monotone, so it is reported as zero instead.  Tightening
        ``epsabs`` resolves more of the tail.  The exact closed forms below
        integrate nothing and are never subject to this.

    Notes
    -----
    The supported regime is the one the GAM smooth-term tests produce: weights
    that are all positive at any finite ``q``, and weights of mixed sign at
    exactly ``q = 0``, both with zero non-centrality.  Anything else raises,
    rather than returning a number no oracle has checked.

    Several shapes never reach characteristic-function quadrature.  A single
    term of either sign, positive weights at a non-positive ``q``, and two
    central terms of opposite sign at ``q = 0`` have closed forms.  Two
    positive central terms with one degree of freedom each use a bounded
    angular integral, with the requested tail evaluated directly.
    Non-centrality is therefore honoured only for a single term, where the
    answer is a non-central chi-square.

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
    standard deviation of ``Q``.  Two quadrature routes then share the range of
    ``z``: a tanh-sinh rule on the unsplit integrand where ``z`` is small enough
    that an oscillatory cycle would step over the whole integrand, and a
    non-oscillatory head on ``[0, 1]`` plus two semi-infinite Fourier tails
    everywhere else.

    Which of the two is tried first is a prediction from ``z``, and it is wrong
    for slowly decaying integrands: the envelope falls off like
    ``u**-(1 + sum(df)/2)``, so a total of two degrees of freedom decays only
    like ``u**-2`` and tanh-sinh will not converge on it.  A route that reports
    it cannot resolve the integrand is therefore not fatal; the other one is
    tried, and its answer is admitted only if it survives the cross-check.  A
    narrow band can remain for other slowly decaying mixtures where no two
    independent routes both converge.  That raises.
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
    errors = np.empty(q_arr.shape, dtype=float)
    for i in range(q_arr.size):
        probability, probability_error = _cdf_single(
            float(q_arr.flat[i]),
            sd,
            std_weights,
            df_arr,
            ncp_arr,
            epsabs,
            epsrel,
            limit,
            check,
            lower_tail,
        )
        out.flat[i] = probability
        errors.flat[i] = probability_error

    if np.isnan(out).any():
        warnings.warn("psum_chisq: quadrature produced NaN", stacklevel=2)

    out = _finalize(out, errors)

    if np.ndim(q) == 0:
        return out.item()
    return out.reshape(np.shape(q))

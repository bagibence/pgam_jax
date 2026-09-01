"""
Distribution of a weighted sum of chi-squared variables.

This module computes the probability that

    Q = sum_j w_j X_j

is above or below a given value q. The X_j are independent chi-squared
variables with nu_j degrees of freedom and non-centrality delta_j^2. The
weights w_j can be positive or negative. This is the null distribution behind
covariate-inclusion p-values for penalized GAM smooth terms.

The general case has no closed form. The probability comes from the
characteristic function of Q instead (Gil-Pelaez / Imhof, 1961):

    Pr(Q <= q) = 1/2 - (1/pi) * integral_0^inf sin(phi(u) - q u) exp(-psi(u))/u du

where, with x_j = 2 w_j u,

    phi(u) = (1/2) sum_j [ nu_j arctan(x_j) + delta_j^2 x_j / (1 + x_j^2) ]
    psi(u) = (1/4) sum_j nu_j log(1 + x_j^2)
             + (1/2) sum_j delta_j^2 x_j^2 / (1 + x_j^2)

phi is the phase of the integrand. exp(-psi(u))/u is its envelope, the positive
factor that makes the oscillation die out as u grows.

Before numerical integration, divide Q and q by s = sd(Q), so
Q/s = sum_j (w_j/s) X_j has variance one and z = q/s. With the t = s * u change
of variables, we have x_j = 2 (w_j/s) t, q u = z t, and du/u = dt/t. The integral
keeps the same form in dimensionless quantities, and P(Q <= q) = P(Q/s <= z).
A common positive rescaling of q and the weights leaves these quantities, and so
the result, unchanged. Without this normalization, a large common rescaling could
change the result returned by the numerical integration.

Two quadrature routes share the range of z:

- tanh-sinh integrates the whole normalized integrand over [0, inf) in one piece,
  with sin(phi(u) - z u) kept intact. It is used for small abs(z), where one
  oscillation cycle is longer than the region that carries the integrand.
- QAWF cuts the integral at u = 1. The head on [0, 1] does not oscillate and
  goes to ordinary adaptive quadrature. The tail on [1, inf) is written as a
  cosine part plus a sine part, and each part goes to QUADPACK's Fourier
  integrator with angular frequency z. It is used for every larger abs(z).

Which route runs first is a prediction from abs(z), and the prediction is
sometimes wrong. A route that cannot resolve the integrand says so, and the
other route is tried. See :func:`_cdf_approx`.

This is an alternative to Davies AS 155 algorithm for calculating the same
probability. Later versions may include that as an alternative.

References
----------
Imhof, J.P. (1961) "Computing the distribution of quadratic forms in normal
variables." *Biometrika* 48, 419-426.

Davies, R.B. (1980) "The distribution of a linear combination of chi-squared
random variables." *J. R. Statist. Soc. C* 29, 323-333.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad, tanhsinh
from scipy.stats import chi2
from scipy.stats import f as f_dist
from scipy.stats import ncx2

from ._numpy_utils import FloatArray, _broadcast, _divide_with_fallback

# The standardized point z = q / sd(Q) where the module changes route. At or below
# it tanh-sinh runs first, above it QAWF runs first. Near z = 0 one QAWF
# oscillation cycle is longer than the whole integrand, so QAWF steps over the
# integrand and reports success on a wrong answer.
_Z_SWITCH = 5e-3

# Calibrated ceilings for the absolute and relative tolerances passed to
# tanh-sinh. A tighter caller request takes precedence.
_TANHSINH_ATOL = 1e-13
_TANHSINH_RTOL = 1e-13

# Refinement levels tanh-sinh must and can use. Each level halves the node
# spacing, so the node count roughly doubles.
# Levels 6 through 8 can stop on an aliased node set for a positive three-term
# mixture near the handover point, even while reporting ``success=True``. Level
# 9 is the first forced refinement that meets the independent-oracle calibration.
_TANHSINH_MINLEVEL = 9
# On the 144 hardest measured cases, a ceiling of 14 leaves 23 of them raising,
# 18 leaves 2, 20 leaves none, and 22 buys nothing more.
_TANHSINH_MAXLEVEL = 20

# QUADPACK's QAWF stops after this many oscillation cycles. SciPy's own default
# is 50 and is not part of its documented signature, so it is set here to keep
# the tail integrations reproducible across SciPy versions.
_LIMLST = 200

# The two quadrature routes, named so that the dispatcher, the cross-check and
# the failure messages all refer to them the same way.
_TANHSINH = "tanh-sinh"
_QAWF = "QAWF"

# Where each route is cut when the cross-check recomputes it. Neither value
# changes the integral itself. Both change the nodes used to find it.
_QAWF_CROSS_CHECK_SPLIT = 2.5
_TANHSINH_CROSS_CHECK_SPLIT = 1.0

# The cross-check computes one integral twice with different nodes, then
# compares. The two answers are allowed to differ by the sum of their two error
# estimates, times this factor. A larger difference counts as evidence that one
# of the two runs is wrong.
# Calibrated: over 335 healthy in-contract cases the worst ratio of disagreement
# to allowance was 0.018 for QAWF. For tanh-sinh that ratio says nothing, because
# its error estimates go to zero. The floor below covers that.
_ERROR_ESTIMATE_SAFETY_FACTOR = 8.0

# The smallest allowance the cross-check ever uses. Both routes can report an
# error of exactly zero and still differ in the last bits, so an allowance built
# from the error estimates alone would fire on healthy input.
# This is the noise level of the comparison, not an accuracy target. The worst
# disagreement over those same healthy cases was 2.5e-12, and this sits above it
# with room. The resulting trip point of 8e-11 is eight orders of magnitude
# below the smallest genuine failure it catches, which is 1.4e-2. A caller who
# requests an epsabs far below 1e-11 still gets this floor, because two routes
# that disagree by 2.5e-12 on healthy input cannot testify about less.
_CROSS_CHECK_FLOOR = 1e-11

# A probability that does not exceed its own estimated error times this factor
# is reported as 0.0. Below that size the quadrature returns noise rather than a
# small number. One means a value is kept as soon as it exceeds its error bar.
_FLOOR_SAFETY_FACTOR = 1.0

# The smallest allowance used when checking that a probability lies in [0, 1].
# It is the counterpart of _CROSS_CHECK_FLOOR. An error estimate of exactly zero
# is not a claim that the answer is exact to the last bit, and the probability is
# still assembled by two roundings. A rank-4 smooth at a near-zero statistic
# overshoots 1 by one ulp with an estimate of 0.0. Eight ulp leaves room and
# still sits fourteen orders of magnitude below the smallest real violation (0.17).
_RANGE_CHECK_FLOOR = 8.0 * float(np.finfo(np.float64).eps)


class _QuadratureNotConverged(RuntimeError):
    """
    One quadrature route declined to answer.

    This is a separate exception from a plain :class:`RuntimeError` because the
    two mean different things. A non-finite value or a negative error estimate
    is corruption, and nothing can be built on it. Non-convergence only means
    that this route could not resolve this integrand to the requested accuracy,
    which leaves the other route free to try. :func:`_cdf_approx` catches this
    exception and not :class:`RuntimeError`.

    It is a subclass of :class:`RuntimeError`, so a caller that only cares that
    the computation failed does not need to know about the difference.
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
    """
    Thin typed wrapper around :func:`scipy.integrate.quad`.

    SciPy ships no type stubs, so the return value of ``quad`` is untyped. The
    call is isolated here to keep that unknown type out of the rest of the
    module.

    With ``weight=None`` this is ordinary adaptive quadrature. With
    ``weight="cos"`` or ``weight="sin"`` SciPy uses QAWF, its Fourier
    integrator, which handles an oscillation of angular frequency ``wvar`` over
    a semi-infinite interval.

    ``limlst`` caps the number of oscillation cycles, and only the Fourier
    integrator reads it. It is passed only when a Fourier integrator is
    requested, because leaving it unset would take SciPy's undocumented
    default.

    ``epsrel`` reaches QUADPACK only on the non-oscillatory branch. QAWF takes
    an absolute target alone, so the relative target is ignored on the tails.

    Returns
    -------
    The integral and its estimated absolute error.

    Raises
    ------
    _QuadratureNotConverged
        If QUADPACK reports that it did not converge.
    RuntimeError
        If QUADPACK returns a non-finite value or an invalid error estimate.
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
    """
    Phase and envelope of the inversion integrand at frequency ``u``.

    The normalized integrand is
    ``sin(phi(u) - z u) * exp(-psi(u)) / u``. This function returns the pair
    ``(phi(u), exp(-psi(u)) / u)``, where, with ``x_j = 2 w_j u``,

        phi(u) = (1/2) sum_j [ nu_j arctan(x_j) + delta_j^2 x_j / (1 + x_j^2) ]
        psi(u) = (1/4) sum_j nu_j log(1 + x_j^2)
                 + (1/2) sum_j delta_j^2 x_j^2 / (1 + x_j^2)

    The sum over the terms ``j`` is a NumPy reduction. This is the only place
    where the two formulas appear. Every other function assembles the same pair
    differently.

    Parameters
    ----------
    u : float
        Frequency at which the characteristic function is evaluated.
    weights, df, noncentrality : numpy.ndarray
        The per-term weights ``w_j``, degrees of freedom ``nu_j``, and
        non-centrality parameters ``delta_j^2``.

    Returns
    -------
    phase : float
        The phase ``phi(u)``.
    envelope : float
        The envelope ``exp(-psi(u)) / u``, which is positive.
    """
    x = 2.0 * weights * u
    x_sq = x**2
    ncp = noncentrality * x / (1.0 + x_sq)
    phase = 0.5 * np.sum(df * np.arctan(x) + ncp)
    log_modulus = -0.25 * np.sum(df * np.log1p(x_sq)) - 0.5 * np.sum(x * ncp)
    return phase, np.exp(log_modulus) / u


def _head_integrand(
    u: float,
    z: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> float:
    """
    The whole integrand on the head ``[0, a]``, where it does not oscillate.
    """
    phase, envelope = _phase_and_envelope(u, weights, df, noncentrality)
    return np.sin(phase - z * u) * envelope


def _tail_cos_coefficient(
    u: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> float:
    """
    Coefficient of ``cos(z u)`` in the oscillatory tail integrand.

    The angle addition formula splits the tail into two Fourier integrals:

        sin(phi - z u) = sin(phi) cos(z u) - cos(phi) sin(z u)

    QAWF integrates one of them against ``cos(z u)`` and the other against
    ``sin(z u)``, so it needs the two coefficients separately.
    """
    phase, envelope = _phase_and_envelope(u, weights, df, noncentrality)
    return envelope * np.sin(phase)


def _tail_sin_coefficient(
    u: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> float:
    """
    Coefficient of ``sin(z u)`` in the oscillatory tail integrand.

    See :func:`_tail_cos_coefficient` for the split. The minus sign in front of
    this piece is applied by :func:`_cdf_qawf`, not here.
    """
    phase, envelope = _phase_and_envelope(u, weights, df, noncentrality)
    return envelope * np.cos(phase)


def _combined_integrand(
    u: FloatArray,
    z: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> FloatArray:
    """
    The whole Imhof integrand in one piece, vectorized for tanh-sinh.

    The QAWF route splits ``sin(phi(u) - z u)`` into a cosine part and a sine
    part. This route keeps it intact and integrates over ``[0, inf)`` with no
    split. SciPy's transform for an infinite interval puts nodes at zero and at
    values close to the largest float, so both ends need care.

    **The node at u = 0.** The integrand carries a ``1 / u`` factor, and the
    numerator also goes to zero there, so the formula reads 0 / 0 and NumPy
    returns NaN. The ratio still has a finite limit. That is what a removable
    singularity is: the formula is undefined at one point, and filling in the
    limit makes the function smooth there. Use ``arctan(x) -> x`` and
    ``x / (1 + x^2) -> x`` for small ``x``, which give

        phi(u) -> u * sum_j w_j (nu_j + delta_j^2)   and   psi(u) -> 0.

    So the whole integrand tends to ``sum_j w_j (nu_j + delta_j^2) - z``. The
    code computes that number and puts it at every node that sits at zero. One
    NaN node would otherwise make the whole integral NaN.

    **The largest nodes.** There ``x_j = 2 w_j u`` overflows to ``+/-inf``, and
    ``x_j^2 / (1 + x_j^2)`` then reads inf / inf, which is NaN again. Every
    ratio is therefore written through ``h_j = hypot(1, x_j)``, which is
    ``sqrt(1 + x_j^2)`` computed without overflow, and through
    ``s_j = x_j / h_j``, which tends to ``+/-1``:

        x / (1 + x^2)   = s / h
        x^2 / (1 + x^2) = s^2
        log(1 + x^2)    = 2 log(h)

    An infinite ``x_j`` then gives ``s / h = 0``, ``s^2 = 1`` and an infinite
    ``log(h)``, so the envelope is zero there, which is the correct limit.
    """
    u_arr = np.asarray(u, dtype=float)

    # The final axis is the term axis. Every preceding axis indexes a batch of
    # tanh-sinh nodes. Some infinite-interval nodes are so large that this
    # product legitimately overflows to +/-inf.
    with np.errstate(over="ignore"):
        x = 2.0 * u_arr[..., np.newaxis] * weights

    # h = sqrt(1 + x**2), formed by np.hypot so that it does not overflow while
    # x is still finite.
    hypot_x = np.hypot(1.0, x)

    # s = x / h. This is an elementwise if: finite x uses x / h, and an x that
    # overflowed to +/-inf uses the limit, +/-1.
    with np.errstate(invalid="ignore"):
        unit_x = np.where(np.isfinite(x), x / hypot_x, np.sign(x))

    # The three ratios that would otherwise overflow, all written through the
    # stable h and s. See the docstring.
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
    numerator = np.sin(phase - z * u_arr) * np.exp(log_modulus)

    # The 0/0 at u = 0 has this limit. The helper divides at the nonzero nodes
    # only and puts the limit at the zero ones.
    limit_at_zero = float(np.sum(weights * (df + noncentrality)) - z)
    return _divide_with_fallback(numerator, u_arr, limit_at_zero)


def _tanhsinh_piece(
    integrand: Callable[[FloatArray], FloatArray],
    a: float,
    b: float,
    epsabs: float,
    epsrel: float,
) -> tuple[float, float]:
    """
    One tanh-sinh integration, with its convergence report turned into a raise.

    SciPy reports a tanh-sinh failure in a field of the result object instead of
    raising. This turns that field into the same exception QUADPACK failures
    produce, so the dispatcher can treat the two routes alike.
    """
    result = tanhsinh(
        integrand,
        a,
        b,
        atol=min(epsabs, _TANHSINH_ATOL),
        rtol=min(epsrel, _TANHSINH_RTOL),
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
    z: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    epsabs: float,
    epsrel: float,
    domain_split: float | None = None,
) -> tuple[float, float]:
    """
    The CDF at standardized point ``z``, by tanh-sinh quadrature.

    ``domain_split`` exists for the cross-check. The integral over ``[0, inf)``
    has the same value however the interval is cut, but tanh-sinh places its
    nodes to suit the interval it is given. A run over ``[0, s]`` plus a run
    over ``[s, inf)`` therefore samples the integrand in different places than
    one run over ``[0, inf)``. A first run that missed part of the mass then
    gives a visibly different answer.

    Parameters
    ----------
    z : float
        Standardized evaluation point.
    weights, df, noncentrality : numpy.ndarray
        Standardized weights, degrees of freedom, and non-centrality
        parameters.
    epsabs, epsrel : float
        Absolute and relative accuracy targets. Each is capped at its calibrated
        tanh-sinh tolerance, so tighter caller requests pass through unchanged.
    domain_split : float or None, optional
        ``None`` integrates ``[0, inf)`` in one run. A float cuts the domain
        there and adds the two runs. Only the cross-check passes a value.

    Returns
    -------
    The CDF at ``z`` and its estimated absolute error.
    """

    def integrand(u: FloatArray) -> FloatArray:
        return _combined_integrand(u, z, weights, df, noncentrality)

    if domain_split is None:
        integral, abs_error = _tanhsinh_piece(integrand, 0.0, np.inf, epsabs, epsrel)
    else:
        head, head_error = _tanhsinh_piece(integrand, 0.0, domain_split, epsabs, epsrel)
        tail, tail_error = _tanhsinh_piece(
            integrand, domain_split, np.inf, epsabs, epsrel
        )
        integral, abs_error = head + tail, head_error + tail_error

    return 0.5 - integral / np.pi, abs_error / np.pi


def _cdf_qawf(
    z: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    split: float,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> tuple[float, float]:
    """
    The CDF at standardized point ``z``, by the head plus QAWF route.

    The answer is assembled from three integrations: the head on
    ``[0, split]``, which does not oscillate, plus the cosine part and the sine
    part of the tail on ``[split, inf)``. ``epsabs`` is an end-to-end budget,
    so each piece gets a third of it rather than the whole.

    ``epsrel`` reaches only the head. QAWF accepts an absolute target alone.

    Returns
    -------
    The CDF at ``z`` and its estimated absolute error.
    """
    params = (weights, df, noncentrality)
    piece_epsabs = epsabs / 3.0
    head, head_error = _quad(
        _head_integrand,
        0.0,
        split,
        (z, *params),
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
        z,
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
        z,
        piece_epsabs,
        epsrel,
        limit,
        _LIMLST,
    )
    cdf = 0.5 - (head + tail_cos - tail_sin) / np.pi
    quadrature_error = (head_error + tail_cos_error + tail_sin_error) / np.pi

    return cdf, quadrature_error


def _cdf_by_route(
    route: str,
    z: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    split: float,
    epsabs: float,
    epsrel: float,
    limit: int,
    use_other_nodes: bool,
) -> tuple[float, float]:
    """
    The CDF at standardized point ``z``, by the named quadrature route.

    With ``use_other_nodes`` set, the route computes the same integral in a
    second way. The value is the same mathematically, but the nodes are not.
    QAWF moves its split point, which is where its cycle grid starts, so a run
    that missed mass lands somewhere else. tanh-sinh cuts the domain in two,
    which replaces its node set.
    :func:`_cross_check` sets this to True.

    Parameters
    ----------
    route : str
        Either :data:`_TANHSINH` or :data:`_QAWF`.
    z : float
        Standardized evaluation point.
    weights, df, noncentrality : numpy.ndarray
        Standardized weights, degrees of freedom, and non-centrality
        parameters.
    split, epsabs, epsrel, limit
        Quadrature settings. Both routes read the accuracy targets. Only QAWF
        reads ``split`` and ``limit``.
    use_other_nodes : bool
        Recompute with a different node placement, for the cross-check.

    Returns
    -------
    The CDF at ``z`` and its estimated absolute error.

    Raises
    ------
    NotImplementedError
        If ``route`` names no known route.
    """
    if route == _TANHSINH:
        return _cdf_tanhsinh(
            z,
            weights,
            df,
            noncentrality,
            epsabs,
            epsrel,
            _TANHSINH_CROSS_CHECK_SPLIT if use_other_nodes else None,
        )
    elif route == _QAWF:
        return _cdf_qawf(
            z,
            weights,
            df,
            noncentrality,
            _QAWF_CROSS_CHECK_SPLIT if use_other_nodes else split,
            epsabs,
            epsrel,
            limit,
        )
    else:
        raise NotImplementedError(f"unknown quadrature route: {route!r}")


def _cross_check(
    route: str,
    value: float,
    error: float,
    z: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
    split: float,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> None:
    """
    Check a quadrature result against a second, independent computation.

    This is the only guard here that can catch a failure nobody has catalogued.
    It exists because QUADPACK's own diagnostics are blind in the way that
    matters. QUADPACK has reported ``ier=0`` and an absolute error of ``4e-15``
    on a value that was wrong in the second decimal place. An error estimate
    built from the nodes that missed the mass cannot report that the mass was
    missed. A different set of nodes can.

    The two ways this can end are different exceptions on purpose. A second
    computation that does not converge leaves this route unchecked, which means
    the same thing to the caller as this route not converging, so
    :class:`_QuadratureNotConverged` is raised and the dispatcher can try the
    other route. A second computation that converges to a *different* answer
    means that one of the two is confidently wrong. That is not a reason to
    change method quietly, so it raises :class:`RuntimeError` and stops.

    Raises
    ------
    _QuadratureNotConverged
        If the second computation does not converge.
    RuntimeError
        If the two answers are further apart than their combined error
        estimates allow.
    """
    other, other_error = _cdf_by_route(
        route,
        z,
        weights,
        df,
        noncentrality,
        split,
        epsabs,
        epsrel,
        limit,
        use_other_nodes=True,
    )

    difference = abs(value - other)
    # _CROSS_CHECK_FLOOR covers the case where both runs report an error of
    # exactly zero and still differ in the last bits.
    allowance = max(error + other_error, _CROSS_CHECK_FLOOR)
    if difference > allowance * _ERROR_ESTIMATE_SAFETY_FACTOR:
        raise RuntimeError(
            f"psum_chisq: the {route} quadrature did not survive its "
            f"independent cross-check. It returned {value!r} with an estimated "
            f"error of {error!r}, while recomputing the same integral with a "
            f"different node set returned {other!r} with an estimated error of "
            f"{other_error!r}. The difference is {difference!r}, which exceeds "
            f"the combined estimate by more than the safety factor of "
            f"{_ERROR_ESTIMATE_SAFETY_FACTOR}. At least one of the two is "
            f"wrong. Standardized inputs: z={z!r}, "
            f"weights={weights.tolist()}, df={df.tolist()}, "
            f"noncentrality={noncentrality.tolist()}.\n\n"
            "PLEASE REPORT THIS CASE ON GITHUB:\n"
            "https://github.com/bagibence/pgam_jax/issues/new"
        )


def _cdf_approx(
    z: float,
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
    The CDF at standardized point ``z``, by whichever quadrature can resolve this input.

    ``abs(z)`` picks the route that is expected to work. tanh-sinh takes the
    small frequencies, where QAWF's first oscillation cycle steps over the whole
    integrand. QAWF takes everything above, where tanh-sinh needs impractically
    many nodes.

    That choice is a prediction, not a fact, and it is wrong when the integrand
    decays slowly. The envelope falls off like ``u ** -(1 + sum(df) / 2)``, so
    the smaller the total degrees of freedom, the longer the tail. At a total of
    2, it decays only like ``u ** -2``, and tanh-sinh does not converge on that
    anywhere in its own band.

    A route that declines is not fatal. The other one is tried as a fallback.
    When the fallback is tried, `check` is ignored, the cross-check always runs.
    At small ``abs(z)``, QAWF can miss the region carrying the integral while
    reporting success and a tiny error estimate. A fallback result is
    therefore accepted only if the independent cross-check agrees.

    Raises
    ------
    RuntimeError
        If neither route produces a checked value.
    """
    if abs(z) <= _Z_SWITCH:
        preferred, fallback = _TANHSINH, _QAWF
    else:
        preferred, fallback = _QAWF, _TANHSINH

    arguments = (z, weights, df, noncentrality, split, epsabs, epsrel, limit)

    try:
        value, error = _cdf_by_route(preferred, *arguments, use_other_nodes=False)
        if check:
            _cross_check(preferred, value, error, *arguments)
        return value, error
    except _QuadratureNotConverged as preferred_failure:
        try:
            value, error = _cdf_by_route(fallback, *arguments, use_other_nodes=False)
            # Always cross-checking. This is the route the dispatcher
            # predicted would be the wrong one, so its answer is allowed
            # only with cross-check evidence behind it.
            _cross_check(fallback, value, error, *arguments)
            return value, error
        except _QuadratureNotConverged as fallback_failure:
            raise RuntimeError(
                f"psum_chisq: neither quadrature route resolved this input. "
                f"The {preferred} route was chosen for z={z!r} and reported: "
                f"{preferred_failure}. Falling back to the {fallback} route "
                f"reported: {fallback_failure}. Standardized inputs: "
                f"weights={weights.tolist()}, df={df.tolist()}, "
                f"noncentrality={noncentrality.tolist()}, where z and the "
                f"weights are divided by the standard deviation of Q.\n\n"
                "PLEASE REPORT THIS CASE ON GITHUB:\n"
                "https://github.com/bagibence/pgam_jax/issues/new"
            ) from fallback_failure


def _validate_inputs(
    q: FloatArray,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> None:
    """
    Check that the arguments describe a sum this module can evaluate.

    - ``q`` must not be NaN.
    - Weights must be finite, with at least one non-zero value.
    - Degrees of freedom must be positive and finite.
    - Non-centrality parameters must be non-negative and finite.

    Parameters
    ----------
    q : numpy.ndarray
        Evaluation points, already turned into a float array.
    weights, df, noncentrality : numpy.ndarray
        The per-term weights ``w_j``, degrees of freedom ``nu_j``, and
        non-centrality parameters ``delta_j^2``, already broadcast to a common
        length.

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
    """
    Canonical term list: zero weights dropped, equal weights merged.

    Both steps are exact. A term with a zero weight contributes ``0 * X_j``,
    which is nothing. Independent chi-squares that share a weight add up:

        w X(nu_1, delta_1^2) + w X(nu_2, delta_2^2)
            = w X(nu_1 + nu_2, delta_1^2 + delta_2^2)

    Weights merge on exact equality only. A tolerance would quietly replace a
    mixture of nearby weights by a different distribution.

    Parameters
    ----------
    weights, df, noncentrality : numpy.ndarray
        The per-term weights ``w_j``, degrees of freedom ``nu_j``, and
        non-centrality parameters ``delta_j^2``.

    Returns
    -------
    weights, df, noncentrality : numpy.ndarray
        The surviving terms, ordered by increasing weight. The result is empty
        when every weight was zero, and :func:`_validate_inputs` has already
        rejected that case.
    """
    keep = weights != 0.0
    weights, df, noncentrality = weights[keep], df[keep], noncentrality[keep]

    unique_weights, index = np.unique(weights, return_inverse=True)

    combined_df = np.zeros(unique_weights.size, dtype=float)
    combined_noncentrality = np.zeros(unique_weights.size, dtype=float)

    np.add.at(combined_df, index, df)
    np.add.at(combined_noncentrality, index, noncentrality)

    return unique_weights, combined_df, combined_noncentrality


def _standard_deviation(
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> float:
    """
    Standard deviation of ``Q``, formed without squaring the raw weights.

    The variance is

        sd^2 = sum_j w_j^2 (2 nu_j + 4 delta_j^2)

    Evaluating this directly can make ``sd`` infinite for very large weights or zero
    for very small weights.
    An infinite ``sd`` sends every normalized weight and the standardized point to zero,
    and the module then returns a CDF of exactly ``0.5`` for every input.
    A zero ``sd`` make standardization infinite or undefined.

    Dividing by the largest absolute weight before squaring avoids both failures.

    Parameters
    ----------
    weights, df, noncentrality : numpy.ndarray
        The per-term weights ``w_j``, degrees of freedom ``nu_j``, and
        non-centrality parameters ``delta_j^2``.

    Returns
    -------
    float
        The standard deviation of ``Q``. It is strictly positive, because the
        caller has already checked that at least one weight is non-zero.
    """
    scale = float(np.max(np.abs(weights)))
    # The same variance, measured in units of ``scale``, so of order sqrt(sum df).
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
    Return the requested tail probability conditional on one polar angle.

    For ``Q = a X1 + b X2``, the polar reduction described in
    :func:`_two_chi1_probability` gives

        Q = R^2 g(theta)
        g(theta) = a cos^2(theta) + b sin^2(theta),

    where ``R^2`` is ``chi2_2``. Therefore, at a fixed angle,

        Pr(Q > z | theta) = exp(-z / [2 g(theta)])
        Pr(Q <= z | theta) = 1 - exp(-z / [2 g(theta)]).

    Here ``a`` and ``b`` are ``first_weight`` and ``second_weight``. The lower
    tail uses ``-expm1`` to remain accurate when its probability is close to
    zero. The caller averages this conditional probability over the angle.
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
    Compute one tail of two positive central ``chi2_1`` terms.

    Let

        Q = a X1 + b X2,

    where ``X1`` and ``X2`` are independent ``chi2_1`` variables. Write
    ``X1 = Z1^2`` and ``X2 = Z2^2`` for independent standard normals, then
    express the random point ``(Z1, Z2)`` in polar coordinates:

        Z1 = R cos(theta)
        Z2 = R sin(theta)
        Q = R^2 [a cos^2(theta) + b sin^2(theta)].

    The standard bivariate normal is rotationally symmetric. Its angle is
    uniform and independent of ``R``, while ``R^2`` is ``chi2_2``. Conditioning
    on the angle therefore gives

        Pr(Q > z | theta)
            = exp(-z / [2 (a cos^2(theta) + b sin^2(theta))]).

    All four quadrants are equivalent after squaring, so the upper-tail
    probability is

        Pr(Q > z)
            = (2 / pi) integral_0^(pi/2) Pr(Q > z | theta) dtheta.

    The lower tail replaces the conditional survival probability by its
    complement. The requested tail is integrated directly to avoid subtracting
    probabilities close to one.

    The weights are sorted so that ``a <= b``. The quadrant is split at angles
    that expose features adaptive quadrature could otherwise miss:

    - ``atan(sqrt(a / b))``, where the two weighted contributions are equal,
    - ``pi / 4``, a fixed interior landmark,
    - the angle where the angular scale equals ``z``, when ``a < z < b``,
    - an additional geometric-mean angle when these features lie near zero.

    These breakpoints do not change the integral. They isolate the narrow
    endpoint layer that appears when one weight is much smaller than the other.
    The absolute error budget is divided between the resulting intervals, and
    both the summed integral and its error are normalized by ``2 / pi``.

    Parameters
    ----------
    z : float
        Positive standardized evaluation point.
    weights : numpy.ndarray
        Two positive standardized weights.
    lower_tail : bool
        Whether to compute ``Pr(Q <= z)`` instead of ``Pr(Q > z)``.
    epsabs, epsrel : float
        Absolute and relative quadrature targets.
    limit : int
        Maximum number of quadrature subintervals for each angular piece.

    Returns
    -------
    probability : float
        Probability in the requested tail.
    error : float
        Estimated absolute error of that probability.
    """
    first_weight, second_weight = sorted(map(float, weights))
    arguments: tuple[object, ...] = (
        z,
        first_weight,
        second_weight,
        lower_tail,
    )

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

    # The angular integral becomes a probability after multiplication by 2 / pi,
    # so an angular error of epsabs * pi / 2 is a probability error of epsabs.
    # Splitting that between the pieces keeps their total inside the caller's
    # end-to-end absolute budget.
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
    lower_tail: bool,
) -> float | None:
    """
    The requested tail at standardized point ``z`` in closed form where one
    exists, ``None`` otherwise.

    The tail the caller asked for is read off the distribution directly. It is
    never formed by subtracting the other tail from one: once a tail falls
    below the spacing of floats near 1, the complement rounds to exactly 1.0
    and the subtraction returns 0.0. That loss is under one ulp in absolute
    terms, but these branches exist in order to be exact, and they report an
    error estimate of 0.0 to :func:`_cdf_single`, which would otherwise be a
    false claim of exactness.

    Three shapes are exact. The quadrature is at its worst on the last of them,
    so this function runs before any numerical decision is made.

    **One term.** ``w X`` is a scaled chi-square, central or non-central.
    Dividing by a negative weight reverses the inequality, so the tails swap.

    **All weights positive, at a non-positive point.** ``Q`` is then a positive
    combination of positive variables, so it is positive with probability one
    and the lower tail is exactly zero. The degrees of freedom are already
    checked positive, and a non-centrality only moves mass further right, so
    neither changes this.

    **Two terms of opposite sign, both central, at exactly zero.** Write
    ``Q = w_pos X_m + w_neg Y_n`` with ``w_pos > 0 > w_neg``. Then

        Pr(Q > 0) = Pr(w_pos X_m > -w_neg Y_n)
                  = Pr(F_{m,n} > -w_neg n / (w_pos m))

    because the ratio of two independent chi-squares, each divided by its
    degrees of freedom, is an F variable. This holds for any positive real ``m``
    and ``n``, not only integers.

    Parameters
    ----------
    z : float
        Standardized evaluation point ``q / sd``.
    weights, df, noncentrality : numpy.ndarray
        Standardized weights ``w_j / sd``, degrees of freedom ``nu_j``, and
        non-centrality parameters ``delta_j^2``, already collapsed to a
        canonical term list ordered by increasing weight.
    lower_tail : bool
        Whether ``Pr(Q <= z)`` is wanted rather than ``Pr(Q > z)``. Selects
        which of ``cdf`` and ``sf`` is evaluated, so that the requested tail is
        never obtained by subtraction.

    Returns
    -------
    float or None
        The probability in the requested tail, or ``None`` when no closed form
        applies. The caller then goes on to :func:`_regime_gate`.
    """
    if np.any(df <= 0.0):
        raise ValueError("`df` must be positive")

    central = not np.any(noncentrality)

    # one term: w X
    if weights.size == 1:
        weight = weights[0]
        x = z / weight

        if central:
            distribution = chi2(df[0])
        else:
            distribution = ncx2(df[0], noncentrality[0])

        # if w > 0: Pr(Q <= z) = Pr(wX <= z) = Pr(X <= z/w)
        # if w < 0: Pr(Q <= z) = Pr(wX <= z) = Pr(X >= z/w)
        #
        # Each tail comes from its own function. ``chi2(3).cdf(100)`` rounds to
        # exactly 1.0, so an upper tail formed as ``1 - cdf`` would be 0.0 where
        # the true value is 1.55e-21.
        if weight > 0.0:
            probability = distribution.cdf(x) if lower_tail else distribution.sf(x)
        elif weight < 0.0:
            probability = distribution.sf(x) if lower_tail else distribution.cdf(x)
        else:
            raise RuntimeError("Single chi2 weight must be non-zero.")

        return probability

    # all weights positive evaluated at a non-positive point. Q > 0 almost
    # surely, so both tails are exact here and neither is a rounding of the
    # other.
    if np.all(weights > 0.0) and z <= 0.0:
        return 0.0 if lower_tail else 1.0

    # rewrite as an F distribution
    if z == 0.0 and central and weights.size == 2 and weights[0] < 0.0 < weights[1]:
        w_neg, w_pos = float(weights[0]), float(weights[1])
        n, m = float(df[0]), float(df[1])
        threshold = (-w_neg * n) / (w_pos * m)
        # Pr(Q > 0) = Pr(F_{m,n} > threshold) is the survival function itself.
        # Routing it through 1 - (1 - sf) costs the far tail: at
        # weights = [-1e6, 1] with df = [50, 1] the upper tail is 1.1e-151 and
        # the round trip through 1.0 returns 0.0.
        if lower_tail:
            return float(f_dist.cdf(threshold, m, n))
        return float(f_dist.sf(threshold, m, n))

    return None


def _regime_gate(
    z: float,
    weights: FloatArray,
    df: FloatArray,
    noncentrality: FloatArray,
) -> None:
    """
    Refuse anything outside the regime this module is checked on.

    Only what the GAM smooth-term tests produce is allowed through to the
    quadrature. There are two such shapes, and both are central:

    - positive weights at any finite point, which is the test at known
      dispersion, where ``q`` is the test statistic and ``z = q / sd``
    - weights of mixed sign at exactly zero, which is the test at estimated
      dispersion, where the random denominator has been moved to the left-hand
      side of the comparison.

    Anything else is refused because it is outside the validated GAM contract.
    An integrator's own error estimate is not always trustworthy: QAWF has
    reported an error below 5e-15 for a result wrong by 1.5e-2.
    A single-term list never arrives here, because :func:`_reduce` answers
    all of those exactly.

    Parameters
    ----------
    z : float
        Standardized evaluation point ``q / sd``.
    weights, df, noncentrality : numpy.ndarray
        Standardized weights, degrees of freedom, and non-centrality
        parameters, as passed to :func:`_reduce`.

    Returns
    -------
    None if the parameter combination is allowed. Raises otherwise.

    Raises
    ------
    NotImplementedError
        If the standardized inputs fall outside the two supported shapes.
    """
    all_positive = bool(np.all(weights > 0.0))
    mixed_signs = bool(np.any(weights > 0.0) and np.any(weights < 0.0))
    central = not np.any(noncentrality)

    # known scale / dispersion
    if all_positive and central:
        return
    # estimated dispersion
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
    z: float,
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

    The evaluation point ``z = q / sd`` and the weights arrive standardized.
    The ``std_`` prefix distinguishes the weights from their unscaled values.

    The order of the branches is the whole control flow of the module. An
    infinite point, then the closed forms, then the two-``chi2_1`` angular
    reduction, then the regime gate and the characteristic-function quadrature.
    A closed form returns an error of exactly zero, because nothing was
    integrated to get it.
    """
    # An infinite z is an evaluation point infinitely many standard deviations
    # from zero. Its probability is exact.
    if np.isposinf(z):
        return (1.0 if lower_tail else 0.0), 0.0
    if np.isneginf(z):
        return (0.0 if lower_tail else 1.0), 0.0
    if not np.isfinite(z):
        raise RuntimeError(f"standardized evaluation point is not a number: z={z}")

    # cases we can analytically reduce and don't need numerical integration.
    # _reduce returns the tail the caller asked for, already correct in both
    # directions, so nothing is subtracted here.
    reduced = _reduce(z, std_weights, df_arr, ncp_arr, lower_tail)
    if reduced is not None:
        return reduced, 0.0

    # sum of two central chi2_1 variables with positive weights is better evaluated using a bounded angular integral
    two_positive_chi1 = (
        not np.any(ncp_arr)  # central
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

    # raise if we're trying to integrate something that's not supported
    _regime_gate(z, std_weights, df_arr, ncp_arr)

    # numerical integral
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
    Validate and clean up computed probabilities.

    A quadrature result is a number plus an uncertainty. Two safeguards are applied:

    1) For a probability with estimated absolute error ``e``, an excursion outside
    ``[0, 1]`` is allowed up to

        max(_ERROR_ESTIMATE_SAFETY_FACTOR * e, _RANGE_CHECK_FLOOR)

    Larger excursions raise because they indicate a numerical failure.
    Smaller ones are consistent with the reported numerical uncertainty and are
    clipped to the nearest endpoint.
    Without :data:`_RANGE_CHECK_FLOOR` an error estimate of exactly zero could
    reject an overshoot of one ULP.

    2) If ``e > 0`` and the probability ``p`` satisfies

         p < _FLOOR_SAFETY_FACTOR * e

    the value is replaced with ``0.0`` and a warning is emitted.
    With the current safety factor of one, this means the estimated error is
    larger than the probability itself, so the quadrature cannot distinguish it
    from zero.

    Parameters
    ----------
    probabilities : numpy.ndarray
        The probabilities in the tail the caller asked for.
    errors : numpy.ndarray
        Corresponding estimated absolute errors. Exact reductions report zero.

    Returns
    -------
    numpy.ndarray
        Probabilities clipped to ``[0, 1]``, with unresolved values replaced by zero.

    Warns
    -----
    UserWarning
        If any probability is replaced by zero because it is unresolved.

    Raises
    ------
    RuntimeError
        If any probability lies outside ``[0, 1]`` by more than its allowance.
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
            "epsabs to try to resolve it. Otherwise, treat 0.0 as an unresolved "
            "small value rather than an exact zero.",
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
    """
    Distribution function of a weighted sum of chi-squared variables.

    By default, computes ``Pr(Q > q)`` for ``Q = sum_j w_j X_j``, where
    ``X_j`` are independent chi-square variables with ``nu_j`` degrees of
    freedom and non-centrality ``delta_j^2``.
    Where no specialized reduction applies, the probability is computed by
    numerical inversion of the characteristic function of ``Q`` using the
    Gil-Pelaez/Imhof formula.

    Parameters
    ----------
    q : array_like
        The point or points at which to evaluate the distribution. An infinite
        value is allowed and is answered exactly. NaN is not allowed.
    weights : array_like
        The weights ``w_j``. They can be positive or negative, and must be
        finite. At least one must be non-zero.
    df : array_like, optional
        Degrees of freedom ``nu_j``, positive and finite. A scalar is applied to
        every term. Defaults to ``1``.
    noncentrality : array_like, optional
        Non-centrality parameters ``delta_j^2``, non-negative and finite. A
        scalar is applied to every term. Defaults to ``0``, which is central.
    lower_tail : bool, optional
        If ``True``, return ``Pr(Q <= q)``. Otherwise return the survival
        function ``Pr(Q > q)``. Defaults to ``False``, returning the upper tail
        used for smooth-term p-values in the GAMs supported here.
    epsabs, epsrel : float, optional
        Absolute and relative accuracy targets for all quadrature routes.
        ``epsabs`` is an end-to-end budget. The oscillatory route assembles its
        answer from three integrations and divides the budget between them. On
        that route ``epsrel`` reaches only the head, because QUADPACK's Fourier
        integrator accepts an absolute target alone. The two-term angular
        reduction applies both targets to each of its pieces. Tanh-sinh uses the
        tighter of each requested target and its calibrated ``1e-13`` ceiling.
        ``epsabs`` also sets the resolution floor described under Warns.
    limit : int, optional
        Maximum number of quadrature subintervals for the QAWF and two-term angular
        routes. The tanh-sinh route does not use this argument. It has a fixed maximum
        refinement level.
    check : bool, optional
        Recompute every characteristic-function result by a second, independent
        route, and raise if the two disagree by more than their combined error
        estimates allow. Defaults to ``True``, and roughly doubles the cost.
        That cost is worth paying, because the diagnostics of the integrator
        report success with an absolute error of ``4e-15`` on values that are
        wrong in the second decimal place. Nothing else here can catch a failure
        that is not already catalogued. A result from the route that ``z`` did
        *not* select is cross-checked even when this is ``False``.
        The angular reduction for two positive ``chi2_1`` terms does not use this
        cross-check. It integrates a bounded non-oscillatory function over a finite
        interval and uses the quadrature routine’s error estimate directly.

    Returns
    -------
    float or numpy.ndarray
        The probability, in the lower or the upper tail. A scalar ``q`` gives a
        Python ``float``. Anything else gives an array with the shape of ``q``.

    Raises
    ------
    ValueError
        If any argument lies outside the domain described above.
    NotImplementedError
        If the sum is outside the supported regime described in the notes.
    RuntimeError
        If no quadrature route resolves the input, or if a result fails its
        independent cross-check. Both messages name what was tried.

    Warns
    -----
    UserWarning
        If the probability is smaller than the quadrature can resolve at the
        requested ``epsabs``. ``0.0`` is returned in that case. Below that
        resolution what comes back is not a small number but noise, so it is
        reported as zero instead. Setting ``epsabs`` may resolve smaller
        probabilities, but require more computation.
        The exact closed forms integrate nothing and never trigger this warning.

    Notes
    -----
    The supported regime is the one the GAM smooth-term tests produce: weights
    that are all positive at any finite ``q``, and weights of mixed sign at
    exactly ``q = 0``, both with zero non-centrality. Anything else raises,
    instead of returning a number that no oracle has checked.

    Several shapes never reach the characteristic-function quadrature. A single
    term of either sign, positive weights at a non-positive ``q``, and two
    central terms of opposite sign at ``q = 0`` all have closed forms. Two
    positive central terms with one degree of freedom each use a bounded angular
    integral, which evaluates the requested tail directly. A non-centrality is
    therefore honored only for a single term, where the answer is a non-central
    chi-square.

    Terms are put in canonical form before anything is evaluated. Zero-weight
    terms are dropped and equal weights are merged, both exactly.

    There is no additive normal term. The general form of Davies AS 155 carries
    one, ``Q = sum_j w_j X_j + sigma Z``. A GAM test statistic is a pure quadratic
    form in the coefficients, so it never produces such a term.
    The argument was therefore removed rather than kept as a value that must
    always be zero.

    Before integration, q and the weights are divided by the standard deviation
    of Q. Tanh-sinh is tried near the resulting z = q / sd = 0, and QAWF elsewhere.
    """
    weight_arr = np.atleast_1d(np.asarray(weights, dtype=float))
    n_terms = int(weight_arr.size)
    df_arr = _broadcast(df, n_terms, "df")
    ncp_arr = _broadcast(noncentrality, n_terms, "noncentrality")
    q_arr = np.atleast_1d(np.asarray(q, dtype=float))

    _validate_inputs(q_arr, weight_arr, df_arr, ncp_arr)
    weight_arr, df_arr, ncp_arr = _collapse_terms(weight_arr, df_arr, ncp_arr)

    # Remove the units: substituting t = sd * u rewrites the inversion integral
    # in the same form using weights w/sd and frequency z = q/sd, with a split
    # point of 1. Every numerical decision below is then made on pure numbers, so
    # the result cannot depend on the units of q and the weights.
    sd = _standard_deviation(weight_arr, df_arr, ncp_arr)
    std_weights = weight_arr / sd

    out = np.empty(q_arr.shape, dtype=float)
    errors = np.empty(q_arr.shape, dtype=float)
    for i in range(q_arr.size):
        z = float(q_arr.flat[i]) / sd
        probability, probability_error = _cdf_single(
            z,
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

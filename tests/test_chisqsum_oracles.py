"""
Exact references for :mod:`pgam_jax._chisqsum`, and their calibration.

None of the oracles below involves a characteristic function, a Fourier
inversion, or any part of the module under test.  That is the whole point.  The
mgcv fixture in ``test_chisqsum.py`` and Davies' published cases in
``test_chisqsum_davies.py`` are transcriptions of other implementations of the
same inversion, so they cannot detect an error shared by the whole lineage.
These can.

Lifted from ``notes/chi-sq-sum/2026-07-23-simple-vs-davies.py``, section 0,
where they were written to measure the failures catalogued in
``notes/chi-sq-sum/2026-07-23-simple-method-report.md``.

Everything the F1 and F2 rows of ``test_chisqsum_failures.py`` assert rests on
these, so the second half of this file calibrates them against ``scipy.stats``
closed forms.  Each check is a case where the mixture degenerates to a single
named distribution: equal weights on two 1-df terms give ``chi2_2``, equal
weights throughout give a plain ``chi2``, and equal positive weights in the
estimated-scale statistic give an exact F.

The Gauss-Hermite reference from section 0 of that script is deliberately not
lifted.  It loses its spectral convergence once the kink at ``b = q`` sits
inside the bulk of the Gaussian weight, and is wrong by 1.7e-05 at ``q = 4``.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import i0e
from scipy.stats import chi2
from scipy.stats import f as f_dist

# --------------------------------------------------------------------------- #
# The oracles.
# --------------------------------------------------------------------------- #


def sf_two_chi1(t, a, b, n=40001):
    """
    Exact ``P(a X1 + b X2 > t)`` for independent ``X ~ chi2_1`` and ``a, b > 0``.

    With ``Z = (R cos T, R sin T)``, the sum is ``R^2 (a cos^2 T + b sin^2 T)``
    where ``R^2 ~ chi2_2`` and ``T`` is uniform, so the survival function is the
    angular mean of ``exp(-t / (2 g(T)))``.  The integrand is smooth and
    periodic, so the trapezoid rule converges geometrically.

    Parameters
    ----------
    t : array_like
        Evaluation point.  Non-positive values return 1, since the sum is
        non-negative.
    a, b : float
        Positive weights.
    n : int, optional
        Number of angular nodes.

    Returns
    -------
    numpy.ndarray
        The survival probability.
    """
    t = np.asarray(t, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    g = a * np.cos(theta) ** 2 + b * np.sin(theta) ** 2
    positive = np.maximum(t, 0.0)
    sf = np.mean(np.exp(-positive[..., None] / (2.0 * g)), axis=-1)
    return np.where(t > 0.0, sf, 1.0)


def sf_two_chi1_bessel(t, a, b):
    """
    Independent Bessel-density reference for two positive ``chi2_1`` terms.

    The density of ``a X1 + b X2`` is evaluated with the exponentially scaled
    modified Bessel function.  The substitution ``x = root_x**2`` smooths the
    near-zero density as one weight vanishes.  Integrating the smaller tail
    avoids subtracting two numbers close to one except when the survival
    probability itself is close to one, where subtracting the small CDF is
    well conditioned in absolute error.
    """
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    product = a * b
    difference_scale = abs(a - b) / (4.0 * product)
    normalization = 2.0 * np.sqrt(product)
    largest_weight = max(a, b)

    def density(x):
        argument = x * difference_scale
        return np.exp(-x / (2.0 * largest_weight)) * i0e(argument) / normalization

    def square_transformed_density(root_x):
        return 2.0 * root_x * density(root_x**2)

    for index in np.ndindex(t.shape):
        point = float(t[index])
        if point <= 0.0:
            out[index] = 1.0
        elif point <= a + b:
            cdf = quad(
                square_transformed_density,
                0.0,
                np.sqrt(point),
                epsabs=1e-15,
                epsrel=1e-13,
                limit=500,
            )[0]
            out[index] = 1.0 - cdf
        else:
            out[index] = quad(
                density,
                point,
                np.inf,
                epsabs=1e-15,
                epsrel=1e-13,
                limit=500,
            )[0]
    return out


def sf_teststat(d, weights_pos, k0):
    """
    Exact p-value of the estimated-scale smooth-term test.

    ``P(sum_j v_j X_j - (d / k0) S > 0)`` with ``X ~ chi2_1`` and
    ``S ~ chi2_k0``, which is the ``q = 0`` signed-weight call mgcv's
    ``testStat`` makes.  Obtained by conditioning on ``S``, using the exact F
    distribution for one positive weight and :func:`sf_two_chi1` for two.

    Parameters
    ----------
    d : float
        Test statistic.
    weights_pos : sequence of float
        One or two positive weights on 1-df terms.  Their number is the
        numerator rank, which need not be an integer once they are unequal.
    k0 : float
        Residual degrees of freedom.

    Returns
    -------
    float
        ``P(Q > 0)``.
    """
    if len(weights_pos) == 1:
        return float(f_dist.sf(d / weights_pos[0], 1, k0))
    a, b = weights_pos

    def inner(s):
        return chi2.pdf(s, k0) * float(sf_two_chi1(np.array(d * s / k0), a, b))

    lo, hi = chi2.ppf([1e-15, 1 - 1e-15], k0)
    return quad(inner, lo, hi, limit=500, epsabs=1e-15, epsrel=1e-13)[0]


def sf_teststat_at_q(q, d, weights_pos, k0):
    """
    ``P(sum_j v_j X_j - (d / k0) S > q)`` at an arbitrary ``q``.

    :func:`sf_teststat` generalised off ``q = 0``, which is what the F2 sweep
    needs: the failure being measured there is the behaviour of the quadrature
    as ``abs(q)/sd`` approaches zero, so the reference has to exist on both
    sides of it rather than only at the point itself.

    Conditioning on ``S`` turns the negative weight into a shift of the
    evaluation point, leaving a positive-weight problem inside, which
    :func:`sf_two_chi1` and :func:`scipy.stats.chi2` answer exactly.  At
    ``q = 0`` this agrees with :func:`sf_teststat` by construction, and
    :func:`test_generalisation_agrees_with_sf_teststat_at_zero` checks it.

    Parameters
    ----------
    q : float
        Evaluation point.
    d : float
        Test statistic, giving the negative weight ``-d / k0``.
    weights_pos : sequence of float
        One or two positive weights on 1-df terms.
    k0 : float
        Residual degrees of freedom.

    Returns
    -------
    float
        ``P(Q > q)``.
    """
    c = d / k0

    if len(weights_pos) == 1:
        (v,) = weights_pos

        def inner(s):
            return chi2.pdf(s, k0) * float(chi2.sf((q + c * s) / v, 1))

    else:
        a, b = weights_pos

        def inner(s):
            return chi2.pdf(s, k0) * float(sf_two_chi1(np.array(q + c * s), a, b))

    lo, hi = chi2.ppf([1e-15, 1 - 1e-15], k0)
    # The inner survival function has a kink where its argument crosses zero.
    kink = -q / c
    points = [kink] if lo < kink < hi else None
    return quad(inner, lo, hi, points=points, limit=500, epsabs=1e-15, epsrel=1e-13)[0]


def sf_chi2_2_plus_weighted_chi1(t, weight):
    """
    Exact ``P(R + weight * X > t)`` for ``R ~ chi2_2`` and ``X ~ chi2_1``.

    Conditioning on ``X = x``, the survival probability of ``R`` is
    ``exp(-(t - weight * x) / 2)`` until ``x = t / weight`` and one above it.
    The first part is a truncated moment-generating function of ``chi2_1``.
    Rescaling it by ``1 - weight`` gives the closed form below, valid for
    positive ``t`` and ``0 < weight < 1``.

    Parameters
    ----------
    t : float
        Positive evaluation point.
    weight : float
        Weight on the 1-df term, strictly between zero and one.

    Returns
    -------
    float
        The survival probability.
    """
    cutoff = t / weight
    return float(
        chi2.sf(cutoff, 1)
        + np.exp(-t / 2.0)
        / np.sqrt(1.0 - weight)
        * chi2.cdf((1.0 - weight) * cutoff, 1)
    )


def sf_two_chi1_plus_chi2k(t, a, b, c, k):
    """
    Exact ``P(a X1 + b X2 + c S > t)`` with ``X ~ chi2_1`` and ``S ~ chi2_k``.

    Conditions on ``S`` and uses the exact two-term formula inside, so this is a
    single smooth 1-D quadrature with an integrable kink at ``s = t / c``, which
    is handed to QUADPACK as a break point.

    The conditioning runs on ``s = y**2``, the same substitution
    :func:`sf_two_chi1_bessel` uses. Below ``k = 2`` the chi-square density has
    an integrable singularity at the origin, and for ``k = 1`` the substitution
    cancels it exactly: ``2 y f(y**2) = 2 exp(-y**2 / 2) / sqrt(2 pi)``.
    Integrating in ``s`` instead, QUADPACK cannot resolve that end of the range
    once ``t`` is small, and returns a survival probability above 1: measured
    1.00107 at ``k = 1`` and ``t`` around ``1e-6`` times the standard deviation,
    which is 1.1e-3 too large.

    Parameters
    ----------
    t : float
        Evaluation point.
    a, b, c : float
        Positive weights, ``a`` and ``b`` on the 1-df terms and ``c`` on the
        ``k``-df term.
    k : float
        Degrees of freedom of the third term.

    Returns
    -------
    float
        The survival probability.
    """

    def inner(y):
        s = y * y
        return 2.0 * y * chi2.pdf(s, k) * float(sf_two_chi1(np.array(t - c * s), a, b))

    lo, hi = np.sqrt(chi2.ppf([1e-15, 1 - 1e-15], k))
    root_kink = np.sqrt(t / c)
    brk = [root_kink] if lo < root_kink < hi else []
    return quad(
        inner, lo, hi, points=brk or None, limit=500, epsabs=1e-15, epsrel=1e-13
    )[0]


# --------------------------------------------------------------------------- #
# The oracles are themselves under test.
# --------------------------------------------------------------------------- #

# The trapezoid rule on a smooth periodic integrand converges geometrically, so
# the two-term formula is exact to rounding.  The conditioning oracles carry a
# QUADPACK call on top of that, and are measured at 7.6e-14 worst.
_TRAPEZOID = 1e-15
_CONDITIONED = 1e-13


@pytest.mark.parametrize(
    "t, dof",
    [(0.1, 2), (5.0, 2), (30.0, 2)],
)
def test_two_equal_unit_weights_are_a_chi_square(t, dof):
    """``X1 + X2 ~ chi2_2``, the degenerate case of the polar-angle formula."""
    assert float(sf_two_chi1(np.array(t), 1.0, 1.0)) == pytest.approx(
        chi2.sf(t, dof), rel=_TRAPEZOID
    )


@pytest.mark.parametrize("scale", [0.5, 2.0, 100.0])
def test_two_equal_scaled_weights_rescale_the_chi_square(scale):
    """``a (X1 + X2) > t`` is ``chi2_2`` evaluated at ``t / a``."""
    assert float(sf_two_chi1(np.array(6.0 * scale), scale, scale)) == pytest.approx(
        chi2.sf(6.0, 2), rel=_TRAPEZOID
    )


def test_one_vanishing_weight_degenerates_to_one_term():
    """
    As ``b`` goes to zero the sum becomes ``chi2_1``.

    A limit rather than an identity, so the agreement is the size of ``b``
    rather than machine precision.  It is here because it is the only check
    that the formula behaves when the two scales are far apart, which is the
    regime a mixture with one tiny weight puts it in.
    """
    assert float(sf_two_chi1(np.array(4.0), 1.0, 1e-9)) == pytest.approx(
        chi2.sf(4.0, 1), rel=1e-8
    )
    assert float(sf_two_chi1_bessel(np.array(4.0), 1.0, 1e-9)) == pytest.approx(
        chi2.sf(4.0, 1), rel=1e-8
    )


@pytest.mark.parametrize(
    "t, a, b",
    [
        (1e-10, 0.125, 0.25),
        (1e-5, 0.125, 0.625),
        (0.5, 0.2, 1.3),
        (4.0, 0.6, 1.2),
        (30.0, 0.4, 2.0),
    ],
)
def test_bessel_reference_agrees_with_angular_reference(t, a, b):
    """The two independent two-term representations agree."""
    assert float(sf_two_chi1_bessel(np.array(t), a, b)) == pytest.approx(
        sf_two_chi1(np.array(t), a, b), rel=2e-12, abs=2e-14
    )


@pytest.mark.parametrize("d, k0", [(3.84, 50), (1.0, 5), (12.0, 500)])
def test_one_positive_weight_is_an_f_probability(d, k0):
    """
    Rank 1: ``P(X - (d/k0) S > 0) = f.sf(d, 1, k0)``.

    Tautological, since this branch of :func:`sf_teststat` returns exactly that
    call.  Kept as a statement of the convention, so a later edit that changes
    the meaning of ``d`` cannot pass silently.
    """
    assert sf_teststat(d, [1.0], k0) == pytest.approx(f_dist.sf(d, 1, k0), rel=1e-15)


@pytest.mark.parametrize("d, k0", [(3.0, 50), (1.0, 5), (10.0, 500), (0.25, 12)])
def test_two_equal_positive_weights_are_an_f_probability(d, k0):
    """
    Rank 2: equal weights give ``f.sf(d, 2, k0)``.

    Unlike the rank-1 row this exercises the whole construction, conditioning
    on the denominator chi-square with :func:`sf_two_chi1` inside.
    """
    assert sf_teststat(2 * d, [1.0, 1.0], k0) == pytest.approx(
        f_dist.sf(d, 2, k0), rel=_CONDITIONED
    )


@pytest.mark.parametrize("t, k", [(2.0, 1), (6.0, 3), (25.0, 10)])
def test_three_equal_weights_are_a_chi_square(t, k):
    """``X1 + X2 + S ~ chi2_(k+2)`` when every weight is 1."""
    assert sf_two_chi1_plus_chi2k(t, 1.0, 1.0, 1.0, k) == pytest.approx(
        chi2.sf(t, k + 2), rel=_CONDITIONED
    )


@pytest.mark.parametrize("t", [1e-9, 1e-6, 1e-3])
def test_three_equal_weights_stay_a_chi_square_near_zero(t):
    """
    The same identity where the conditioning integral used to fail.

    At ``k = 1`` the conditioning density has an integrable singularity at the
    origin.  Integrating in ``s`` rather than in ``sqrt(s)``, QUADPACK could not
    resolve it once ``t`` was small and returned survival probabilities above 1,
    measured at 1.00107. A reference that reports a probability of 1.001 cannot
    referee anything, so the range is asserted alongside the value.
    """
    got = sf_two_chi1_plus_chi2k(t, 1.0, 1.0, 1.0, 1)

    assert 0.0 <= got <= 1.0
    assert got == pytest.approx(chi2.sf(t, 3), rel=_CONDITIONED)


@pytest.mark.parametrize("weight", [1e-9, 1e-6])
@pytest.mark.parametrize("t", [1e-5, 1.0, 6.0])
def test_conditioning_oracle_resolves_a_tiny_weight(t, weight):
    """
    Compare the imbalanced conditioning integral with an exact closed form.

    The first and third unit-weight 1-df terms sum to ``chi2_2``, leaving the
    second term as ``weight * chi2_1``. The reference retains that term exactly,
    and the tolerance is small enough that dropping it would fail. The band
    this oracle referees in ``test_chisqsum_failures.py`` has this shape: two
    weights many orders of magnitude apart.
    """
    expected = sf_chi2_2_plus_weighted_chi1(t, weight)
    without_tiny_term = chi2.sf(t, 2)

    assert expected - without_tiny_term > max(_CONDITIONED * expected, 2e-15)
    assert sf_two_chi1_plus_chi2k(t, 1.0, weight, 1.0, 1) == pytest.approx(
        expected, rel=_CONDITIONED, abs=2e-15
    )


@pytest.mark.parametrize(
    "weights_pos, k0, d",
    [
        ([1.0], 50, 3.84),
        ([1.1830127, 0.3169873], 50, 3.84),
        ([1.9486833, 0.0513167], 5, 2.0),
    ],
)
def test_generalisation_agrees_with_sf_teststat_at_zero(weights_pos, k0, d):
    """
    :func:`sf_teststat_at_q` is :func:`sf_teststat` extended off ``q = 0``.

    The two are written differently, the second conditioning through
    :func:`scipy.stats.chi2` for rank 1 where the first uses an exact F, so
    agreement at the shared point is a real check on both.
    """
    assert sf_teststat_at_q(0.0, d, weights_pos, k0) == pytest.approx(
        sf_teststat(d, weights_pos, k0), rel=_CONDITIONED
    )


def test_generalisation_is_monotone_in_q():
    """A survival function decreases, which the added shift must not break."""
    values = [
        sf_teststat_at_q(q, 3.84, [1.1830127, 0.3169873], 50)
        for q in (-2.0, -0.5, 0.0, 0.5, 2.0)
    ]
    assert np.all(np.diff(values) < 0.0)

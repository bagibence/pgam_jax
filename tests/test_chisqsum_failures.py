"""
Failure-mode regression tests for :mod:`pgam_jax._chisqsum`.
"""

import numpy as np
import pytest
from scipy.stats import chi2, ncx2

import pgam_jax._chisqsum as chisqsum
from pgam_jax._chisqsum import psum_chisq


def _unreachable(*_args, **_kwargs):
    """Stand-in for :func:`_cdf_single` on paths that must never integrate."""
    raise AssertionError("quadrature must not be reached")


# --------------------------------------------------------------------------- #
# F3.  Nothing in the method was scale-free.
#
# Multiplying ``q`` and every weight by a common positive constant is a change
# of units.  It cannot change the probability.  The implementation used to pass
# the raw ``q`` as the oscillatory quadrature's frequency while placing the split
# point at ``1/sd``, so the answer moved: correct at ``c = 1e6``, raising at
# ``c = 1e8``, and silently returning the complement at ``c = 1e10``.
# --------------------------------------------------------------------------- #

# Spans a large range. 1e7 to 1e12 is where the old code broke.
_RESCALINGS = [1e-6, 1e-3, 1.0, 1e3, 1e6, 1e7, 1e8, 1e9, 1e10, 1e12, 1e15]


@pytest.mark.parametrize("c", _RESCALINGS)
def test_single_term_scale_invariance(c):
    """``c * chi2_1`` evaluated at ``c`` is ``chi2_1`` evaluated at 1, for any ``c``."""
    assert psum_chisq(c, [c], df=1) == pytest.approx(chi2.sf(1.0, 1), rel=1e-12)


@pytest.mark.parametrize("c", _RESCALINGS)
def test_multi_term_scale_invariance(c):
    """A three-term mixture is invariant under a common rescale of ``q`` and weights."""
    base = psum_chisq(5.0, [1.0, 0.6, 0.4], df=[3, 1, 1])
    got = psum_chisq(5.0 * c, [1.0 * c, 0.6 * c, 0.4 * c], df=[3, 1, 1])
    assert got == pytest.approx(base, rel=1e-12)


@pytest.mark.parametrize("c", _RESCALINGS)
def test_noncentrality_is_dimensionless(c):
    """``noncentrality`` is unit-free, so it stays fixed while ``q`` and weights rescale."""
    base = psum_chisq(10.0, [1.0, 0.5], df=[3, 1], noncentrality=[2.0, 0.5])
    got = psum_chisq(10.0 * c, [1.0 * c, 0.5 * c], df=[3, 1], noncentrality=[2.0, 0.5])
    assert got == pytest.approx(base, rel=1e-12)


# The naive variance ``sum(w**2 * (2 df + 4 ncp))`` squares the raw weights, so it
# overflows to inf above roughly ``1.34e154 / sqrt(sum(2 nu + 4 delta**2))`` and
# underflows to zero below about 2.2e-162.  On overflow every normalized weight
# and the frequency collapse to 0 and the raw CDF is exactly 0.5, silently.
# These rows probe extreme magnitudes beyond realistic GAM inputs.


@pytest.mark.parametrize("c", [1e-300, 1e-200, 1e160, 1e300])
def test_scale_invariance_beyond_naive_variance_range(c):
    """The rescale still holds where the naive variance over- or underflows."""
    assert psum_chisq(c, [c], df=1) == pytest.approx(chi2.sf(1.0, 1), rel=1e-12)


def test_many_terms_at_the_overflow_edge():
    """More terms lower the overflow threshold: 20 terms at ``df=5`` break by 1e153."""
    df = np.full(20, 5.0)
    base = psum_chisq(50.0, np.ones(20), df=df)
    got = psum_chisq(50.0 * 1e153, np.full(20, 1e153), df=df)
    assert got == pytest.approx(base, rel=1e-12)


# --------------------------------------------------------------------------- #
# F8.  The input domain was never stated, so out-of-domain values reached the
# integrand.
#
# A NaN weight, degree of freedom or evaluation point used to travel all the way
# into QUADPACK and come back as ``RuntimeError: _head_integrand quadrature
# failed: The occurrence of roundoff error is detected``, which names the
# integrator rather than the offending argument.  An infinite ``q`` is a
# well-posed question with an exact answer, and got the same treatment.
# --------------------------------------------------------------------------- #

_TERM_SHAPES = [
    pytest.param({"weights": [1.0], "df": 1}, id="single-positive"),
    pytest.param({"weights": [1.0, 0.6, 0.4], "df": [3, 1, 1]}, id="positive-mixture"),
    pytest.param({"weights": [1.0, -0.5], "df": [2, 4]}, id="signed"),
    pytest.param({"weights": [-1.0, -2.0], "df": [1, 1]}, id="all-negative"),
]


@pytest.mark.parametrize("case", _TERM_SHAPES)
def test_positive_infinite_q_is_exact(case):
    """``Pr(Q <= +inf)`` is 1 whatever the weights are, and is not a quadrature."""
    assert psum_chisq(np.inf, lower_tail=True, **case) == 1.0
    assert psum_chisq(np.inf, lower_tail=False, **case) == 0.0


@pytest.mark.parametrize("case", _TERM_SHAPES)
def test_negative_infinite_q_is_exact(case):
    """``Pr(Q <= -inf)`` is 0 whatever the weights are."""
    assert psum_chisq(-np.inf, lower_tail=True, **case) == 0.0
    assert psum_chisq(-np.inf, lower_tail=False, **case) == 1.0


def test_infinite_q_does_not_reach_quadrature(monkeypatch):
    """The answer at an infinite ``q`` is exact, so no integrand is evaluated."""
    monkeypatch.setattr(chisqsum, "_cdf_single", _unreachable)

    assert psum_chisq(np.inf, [1.0, 0.6], df=[3, 1]) == 0.0
    assert psum_chisq(-np.inf, [1.0, 0.6], df=[3, 1]) == 1.0


def test_infinite_q_inside_an_array():
    """A mixed array is answered elementwise, the finite entries by quadrature."""
    got = psum_chisq([-np.inf, 2.0, np.inf], [1.0], df=1)
    assert got[0] == 1.0
    assert got[1] == pytest.approx(chi2.sf(2.0, 1), abs=1e-9)
    assert got[2] == 0.0


def test_finite_q_beyond_the_standardized_range_saturates():
    """
    ``z = q / sd`` overflows here, which says ``q`` is infinitely many standard
    deviations out.  That is the same statement as an infinite ``q``, so it gets
    the same exact answer rather than a quadrature failure.
    """
    assert psum_chisq(1e300, [1e-300], df=1) == 0.0
    assert psum_chisq(1e300, [1e-300], df=1, lower_tail=True) == 1.0
    assert psum_chisq(-1e300, [1e-300], df=1) == 1.0
    assert psum_chisq(-1e300, [1e-300], df=1, lower_tail=True) == 0.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"q": np.nan, "weights": [1.0]}, "'q'"),
        ({"q": [1.0, np.nan], "weights": [1.0]}, "'q'"),
        ({"q": 1.0, "weights": [np.nan]}, "'weights'"),
        ({"q": 1.0, "weights": [1.0, np.nan]}, "'weights'"),
        ({"q": 1.0, "weights": [1.0], "df": [np.nan]}, "'df'"),
        ({"q": 1.0, "weights": [1.0, 1.0], "df": [1.0, np.nan]}, "'df'"),
        ({"q": 1.0, "weights": [1.0], "noncentrality": [np.nan]}, "'noncentrality'"),
    ],
)
def test_nan_inputs_raise(kwargs, match):
    """A NaN anywhere is a caller error, reported against the argument it is in."""
    with pytest.raises(ValueError, match=match):
        psum_chisq(**kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"weights": [np.inf]}, "'weights'"),
        ({"weights": [1.0, -np.inf]}, "'weights'"),
        ({"weights": [1.0], "df": [np.inf]}, "'df'"),
        ({"weights": [1.0], "noncentrality": [np.inf]}, "'noncentrality'"),
    ],
)
def test_infinite_parameters_raise(kwargs, match):
    """Only ``q`` may be infinite.  An infinite weight or dof describes no sum."""
    with pytest.raises(ValueError, match=match):
        psum_chisq(1.0, **kwargs)


def test_validation_precedes_quadrature(monkeypatch):
    """
    The old symptom of every one of these was a QUADPACK message about roundoff.
    Nothing may reach the integrator before the inputs have been checked.
    """
    monkeypatch.setattr(chisqsum, "_cdf_single", _unreachable)

    with pytest.raises(ValueError, match="'q'"):
        psum_chisq(np.nan, [1.0])
    with pytest.raises(ValueError, match="'weights'"):
        psum_chisq(1.0, [1.0, np.nan])
    with pytest.raises(ValueError, match="'df'"):
        psum_chisq(1.0, [1.0], df=[np.inf])


# --------------------------------------------------------------------------- #
# F9.  Degenerate term lists were rejected or left uncanonicalised.
#
# Zero weights and repeated weights are exactly removable.
# --------------------------------------------------------------------------- #


def test_all_zero_weights_raise():
    """``Q`` is then degenerate at 0, which is not a distribution to invert."""
    with pytest.raises(ValueError, match="non-zero"):
        psum_chisq(1.0, [0.0, 0.0])


def test_additive_normal_term_is_not_an_argument():
    """
    Davies' ``sigma`` and mgcv's ``sigz`` have no GAM call site, so there is no
    argument to pass and the mistake is loud.
    """
    with pytest.raises(TypeError, match="sigma"):
        psum_chisq(1.0, [1.0], sigma=2.0)  # type: ignore[call-arg]


@pytest.mark.parametrize("q", [0.5, 2.0, 9.0])
def test_zero_weight_terms_are_dropped(q):
    """A zero weight contributes nothing, and takes its degrees of freedom with it."""
    assert psum_chisq(q, [1.0, 0.0], df=[1, 99]) == pytest.approx(
        chi2.sf(q, 1), abs=1e-9
    )


@pytest.mark.parametrize("q", [0.5, 2.0, 9.0])
def test_equal_weights_merge(q):
    """Three unit-weight 1-dof terms are one unit-weight 3-dof term."""
    assert psum_chisq(q, [1.0, 1.0, 1.0], df=1) == pytest.approx(
        chi2.sf(q, 3), abs=1e-9
    )


@pytest.mark.parametrize("q", [1.0, 5.0, 12.0])
def test_merging_adds_noncentralities(q):
    """Merging equal weights adds the degrees of freedom and the non-centralities."""
    got = psum_chisq(q, [1.0, 1.0], df=[1, 1], noncentrality=[0.5, 1.5])
    assert got == pytest.approx(ncx2.sf(q, 2, 2.0), abs=1e-9)


def test_collapse_terms_drops_zero_weights():
    w, df, ncp = chisqsum._collapse_terms(
        np.array([1.0, 0.0, 2.0]),
        np.array([1.0, 99.0, 3.0]),
        np.array([0.0, 7.0, 0.5]),
    )
    assert w.tolist() == [1.0, 2.0]
    assert df.tolist() == [1.0, 3.0]
    assert ncp.tolist() == [0.0, 0.5]


def test_collapse_terms_merges_equal_weights():
    w, df, ncp = chisqsum._collapse_terms(
        np.array([1.0, 0.5, 1.0]),
        np.array([2.0, 1.0, 3.0]),
        np.array([0.25, 0.0, 0.75]),
    )
    assert w.tolist() == [0.5, 1.0]
    assert df.tolist() == [1.0, 5.0]
    assert ncp.tolist() == [0.0, 1.0]


def test_collapse_terms_is_order_independent():
    """The collapsed list is canonical, so the input order cannot survive it."""
    args = (
        np.array([1.0, 0.5, 1.0, 0.0]),
        np.array([2.0, 1.0, 3.0, 9.0]),
        np.array([0.25, 0.0, 0.75, 4.0]),
    )
    shuffled = [arg[[3, 1, 0, 2]] for arg in args]

    for left, right in zip(
        chisqsum._collapse_terms(*args), chisqsum._collapse_terms(*shuffled)
    ):
        assert left.tolist() == right.tolist()


def test_collapse_terms_returns_empty_when_every_weight_is_zero():
    w, df, ncp = chisqsum._collapse_terms(np.zeros(3), np.ones(3), np.zeros(3))
    assert w.size == 0 and df.size == 0 and ncp.size == 0


def test_collapse_terms_leaves_unequal_weights_alone():
    """Merging is on exact equality.  Nearby weights are a different mixture."""
    weights = np.array([1.0, 1.0 + 1e-12])
    w, _df, _ncp = chisqsum._collapse_terms(weights, np.ones(2), np.zeros(2))
    assert w.tolist() == sorted(weights.tolist())


# ``q = 0`` is left out below: with signed weights that is F1, which still raises
# "Probabilities must be in [0, 1]" and is fixed in a later pass.
@pytest.mark.parametrize("q", [-4.0, -1.0, 1.0, 4.0])
def test_reflection_identity(q):
    """``Pr(Q > q) = Pr(-Q <= -q)``, so negating both sides must agree."""
    weights = [1.0, 0.6, -0.4]
    df = [3, 1, 2]
    upper = psum_chisq(q, weights, df=df, lower_tail=False)
    lower = psum_chisq(-q, [-w for w in weights], df=df, lower_tail=True)
    assert upper == pytest.approx(lower, abs=1e-10)

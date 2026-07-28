"""
Failure-mode regression tests for :mod:`pgam_jax._chisqsum`.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi2
from scipy.stats import f as f_dist
from scipy.stats import ncx2
from test_chisqsum_oracles import (
    sf_teststat,
    sf_teststat_at_q,
    sf_two_chi1,
    sf_two_chi1_plus_chi2k,
)

import pgam_jax._chisqsum as chisqsum
from pgam_jax._chisqsum import psum_chisq


def _unreachable(*_args, **_kwargs):
    """Stand-in for :func:`_cdf_approx` on paths that must never integrate."""
    raise AssertionError("quadrature must not be reached")


def _sf_by_quadrature(q, weights, df=1.0, noncentrality=0.0):
    """
    ``Pr(Q > q)`` by nondimensionalisation and quadrature alone.

    Exact reductions answer several of the rows below before the integrator is
    reached, which would leave those rows testing SciPy rather than this module.
    This routes the same call through the standardisation and integration
    ``psum_chisq`` performs, with no reduction and no regime gate, so a test that
    is about the quadrature stays about the quadrature.

    :func:`test_quadrature_helper_matches_public_path` pins this to the wiring in
    ``psum_chisq``, which is what keeps it from drifting into testing a
    convention the module no longer uses.
    """
    w = np.atleast_1d(np.asarray(weights, dtype=float))
    d = chisqsum._broadcast(df, w.size, "df")
    ncp = chisqsum._broadcast(noncentrality, w.size, "noncentrality")
    w, d, ncp = chisqsum._collapse_terms(w, d, ncp)

    sd = chisqsum._standard_deviation(w, d, ncp)
    cdf, _error = chisqsum._cdf_approx(q / sd, w / sd, d, ncp, 1.0, 1e-10, 1e-10, 200)
    return 1.0 - cdf


def test_quadrature_helper_matches_public_path():
    """No reduction applies to this mixture, so both routes are the same code."""
    q, weights, df = 5.0, [1.0, 0.6, 0.4], [3, 1, 1]
    assert _sf_by_quadrature(q, weights, df) == pytest.approx(
        psum_chisq(q, weights, df=df), rel=1e-14
    )


def _standard_deviation(weights, df):
    """
    ``sd(Q)`` for a central mixture, computed here rather than imported.

    The sweeps below place ``q`` at a chosen multiple of the standard
    deviation, so taking it from the module under test would make the grid
    depend on the thing being measured.
    """
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(w**2 * 2 * np.asarray(df, dtype=float))))


def test_divide_with_fallback_skips_zero_denominators():
    """The fallback branch must not evaluate the masked division by zero."""
    numerator = np.array([6.0, 1.0, 8.0])
    denominator = np.array([2.0, 0.0, 4.0])

    with np.errstate(divide="raise", invalid="raise"):
        got = chisqsum._divide_with_fallback(numerator, denominator, fallback=99.0)

    np.testing.assert_array_equal(got, [3.0, 99.0, 2.0])


def test_combined_integrand_handles_zero_and_an_overflowed_x():
    """The removable zero limit and the large-node analytic limit stay finite."""
    got = chisqsum._combined_integrand(
        np.array([0.0, np.finfo(float).max]),
        0.0,
        np.array([1.0]),
        np.array([1.0]),
        np.array([0.0]),
    )

    np.testing.assert_array_equal(got, [1.0, 0.0])


# The five estimated-dispersion structures of the report, as
# ``(positive weights on 1-df terms, k0, d)``.  The negative weight is
# ``-d / k0`` on a ``k0``-df term.  Rank is the number of positive weights only
# when they are equal, so the unequal rows are the fractional-rank tests that
# ``_reduce`` cannot answer: merging leaves three terms, not two.
_FRACTIONAL_RANK_STRUCTURES = [
    pytest.param([1.0], 50, 3.84, id="rank1.0-k50-d3.84"),
    pytest.param([1.1830127, 0.3169873], 50, 3.84, id="rank1.5-k50-d3.84"),
    pytest.param([1.1830127, 0.3169873], 50, 1e-4, id="rank1.5-k50-d1e-4"),
    pytest.param([1.1830127, 0.3169873], 500, 1.0, id="rank1.5-k500-d1.0"),
    pytest.param([1.9486833, 0.0513167], 5, 2.0, id="rank1.9-k5-d2.0"),
]


def _teststat_call(weights_pos, k0, d):
    """The weights and degrees of freedom mgcv's ``testStat`` passes."""
    return list(weights_pos) + [-d / k0], [1.0] * len(weights_pos) + [float(k0)]


# --------------------------------------------------------------------------- #
# F1.  QAWF's ``omega == 0`` branch restarts the cycle grid at 0, so the head of
# the integrand is counted twice at exactly ``q = 0``.
#
# That is the estimated-dispersion mainline.  Step 4 closed the integer-rank
# half of it with an exact F reduction, which never reaches the integrator.  The
# fractional-rank half is what remains: two unequal positive weights leave three
# terms after merging, no closed form applies, and the quadrature answers.  Two
# of the rows below currently return a confidently wrong number, and two raise.
#
# The raw-CDF rows are the direct statement of the bug, with no reduction in the
# way: a single positive term at ``q = 0`` has survival probability exactly 1,
# and the integrator returns 1.17 and 1.30.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("weights_pos, k0, d", _FRACTIONAL_RANK_STRUCTURES)
def test_teststat_at_zero_matches_the_exact_p_value(weights_pos, k0, d):
    """
    The mgcv ``testStat`` p-value, against conditioning on the denominator.

    The rank-1.0 row is answered by the step-4 reduction and passes already.
    The other four are the outstanding failure, and two of them are silent.
    """
    weights, df = _teststat_call(weights_pos, k0, d)
    assert psum_chisq(0.0, weights, df=df) == pytest.approx(
        sf_teststat(d, weights_pos, k0), rel=1e-12
    )


@pytest.mark.parametrize("dof", [1, 3, 10])
def test_raw_survival_at_zero_reaches_one_by_computation(dof):
    """
    ``P(chi2_k > 0) == 1``, obtained rather than clipped.

    A single term is answered exactly by :func:`_reduce` on the public path, so
    this goes through the quadrature helper.  The raw survival probabilities
    are 1.17 and 1.30 today, which is the doubled head showing through
    undisguised.
    """
    assert _sf_by_quadrature(0.0, [1.0], df=dof) == pytest.approx(1.0, abs=1e-10)


def test_rank_one_row_is_correct_through_the_quadrature():
    """
    The exact reduction must not be the only thing standing between ``q = 0``
    and a wrong answer.

    Same row as ``test_teststat_headline_row``, routed past the reduction.  It
    comes back as -0.3736 today, which is what the public path raised on before
    step 4 gave it a closed form.
    """
    assert _sf_by_quadrature(0.0, [1.0, -3.84 / 50], df=[1, 50]) == pytest.approx(
        f_dist.sf(3.84, 1, 50), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# F2.  The first QAWF cycle steps over the whole integrand once ``abs(z)`` is
# small, so every node lands in underflow and the answer is built from nothing.
#
# Measured: the signed structure is wrong by 74% at ``z <= 1e-4`` and exact to
# 7e-15 from ``z = 1e-3`` up.  The positive structure is wrong by 13% at
# ``z = 1e-9``, raises at ``z = 1e-6``, and is exact from ``z = 1e-4`` up.  So
# the handover to the tanh-sinh branch has to sit above 1e-3, and both methods
# are exact there, which is what makes ``_Z_SWITCH`` calibratable rather than
# guessed.
#
# The signed sweep goes through the quadrature helper because signed weights
# away from ``q = 0`` are outside the public contract.  That is deliberate: the
# branch on ``abs(z)`` belongs inside ``_cdf_approx``, so that every route into
# the integrator gets it, including this helper and the F3 rows below.  A branch
# added above ``_cdf_approx`` instead would leave these rows red.
# --------------------------------------------------------------------------- #

_Z_SWEEP = [1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]


@pytest.mark.parametrize("z", _Z_SWEEP)
def test_signed_structure_is_accurate_at_small_z(z):
    """A fractional-rank ``testStat`` mixture, swept towards ``q = 0``."""
    weights_pos, k0, d = [1.1830127, 0.3169873], 50, 3.84
    weights, df = _teststat_call(weights_pos, k0, d)
    q = z * _standard_deviation(weights, df)

    assert _sf_by_quadrature(q, weights, df=df) == pytest.approx(
        sf_teststat_at_q(q, d, weights_pos, k0), rel=1e-10
    )


@pytest.mark.parametrize("z", _Z_SWEEP)
def test_positive_structure_is_accurate_at_small_z(z):
    """
    The same sweep on an in-contract mixture, through the public API.

    Positive weights at a positive ``q`` are supported, so this is a failure a
    caller can reach today without leaving the documented regime.
    """
    weights, df = [1.0, 0.6, 0.4], [3, 1, 1]
    q = z * _standard_deviation(weights, df)

    assert psum_chisq(q, weights, df=df) == pytest.approx(
        sf_two_chi1_plus_chi2k(q, 0.6, 0.4, 1.0, 3), rel=1e-10
    )


# The eight ``testStat`` structures used to calibrate the original prototype,
# plus the positive mixture from the sweep above.  These are raw weights and
# degrees of freedom; the test standardizes them before calling either
# quadrature directly.
_HANDOVER_STRUCTURES = [
    pytest.param([1.0, -3.84 / 50], [1, 50], id="rank1-k50-d3.84"),
    pytest.param([1.0, -0.5 / 5], [1, 5], id="rank1-k5-d0.5"),
    pytest.param([1.0, -12.0 / 500], [1, 500], id="rank1-k500-d12"),
    pytest.param(
        [1.1830127, 0.3169873, -3.84 / 50],
        [1, 1, 50],
        id="rank1.5-k50-d3.84",
    ),
    pytest.param(
        [1.1830127, 0.3169873, -1e-4 / 50],
        [1, 1, 50],
        id="rank1.5-k50-d1e-4",
    ),
    pytest.param(
        [1.1830127, 0.3169873, -1.0 / 500],
        [1, 1, 500],
        id="rank1.5-k500-d1",
    ),
    pytest.param(
        [1.9486833, 0.0513167, -2.0 / 5],
        [1, 1, 5],
        id="rank1.9-k5-d2",
    ),
    pytest.param([1.0, 1.0, -6.0 / 50], [1, 1, 50], id="rank2-k50-d6"),
    pytest.param([1.0, 0.6, 0.4], [3, 1, 1], id="positive-mixture"),
]
_HANDOVER_Z = np.geomspace(1e-3, 1e-2, 9)


@pytest.mark.parametrize("weights, df", _HANDOVER_STRUCTURES)
@pytest.mark.parametrize("z", _HANDOVER_Z)
def test_quadrature_methods_agree_across_the_handover_band(z, weights, df):
    """
    The method switch lies inside an overlap where both quadratures are accurate.

    QAWF is requested at ``1e-11`` for this calibration: that is the largest
    per-piece absolute tolerance measured to give 1e-12 relative agreement over
    the whole set.  The public default remains 1e-10, and the dispatcher keeps
    QAWF away from the small-z rows where that looser request misses the
    integrand.
    """
    w = np.asarray(weights, dtype=float)
    d = np.asarray(df, dtype=float)
    ncp = np.zeros_like(w)
    w, d, ncp = chisqsum._collapse_terms(w, d, ncp)
    sd = chisqsum._standard_deviation(w, d, ncp)
    w = w / sd

    tanhsinh_cdf, _ = chisqsum._cdf_tanhsinh(z, w, d, ncp)
    qawf_cdf, _ = chisqsum._cdf_qawf(
        z,
        w,
        d,
        ncp,
        1.0,
        1e-11,
        1e-11,
        200,
    )

    assert 1.0 - qawf_cdf == pytest.approx(1.0 - tanhsinh_cdf, rel=1e-12)


def test_quadrature_dispatches_at_the_calibrated_switch(monkeypatch):
    """
    The boundary itself belongs to tanh-sinh; the next float belongs to QAWF.

    ``check=False`` keeps this about the dispatch decision.  With the
    cross-check on, each route is also recomputed by its independent variant,
    which doubles the recorded calls without saying anything about which branch
    the switch chose.
    """
    calls = []

    def tanhsinh_branch(*_args):
        calls.append("tanhsinh")
        return 0.25, 0.0

    def qawf_branch(*_args):
        calls.append("qawf")
        return 0.75, 0.0

    monkeypatch.setattr(chisqsum, "_cdf_tanhsinh", tanhsinh_branch)
    monkeypatch.setattr(chisqsum, "_cdf_qawf", qawf_branch)

    args = (
        np.array([1.0]),
        np.array([1.0]),
        np.array([0.0]),
        1.0,
        1e-10,
        1e-10,
        200,
        False,
    )
    assert chisqsum._cdf_approx(chisqsum._Z_SWITCH, *args) == (0.25, 0.0)
    assert chisqsum._cdf_approx(np.nextafter(chisqsum._Z_SWITCH, np.inf), *args) == (
        0.75,
        0.0,
    )
    assert calls == ["tanhsinh", "qawf"]


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

# The single-term rows go through ``_sf_by_quadrature``.  A single term now has
# an exact reduction, so calling ``psum_chisq`` would answer them from
# ``chi2.sf`` and stop testing the standardisation these rows exist to check.


@pytest.mark.parametrize("c", _RESCALINGS)
def test_single_term_scale_invariance(c):
    """``c * chi2_1`` evaluated at ``c`` is ``chi2_1`` evaluated at 1, for any ``c``."""
    assert _sf_by_quadrature(c, [c], df=1) == pytest.approx(chi2.sf(1.0, 1), rel=1e-12)


@pytest.mark.parametrize("c", _RESCALINGS)
def test_multi_term_scale_invariance(c):
    """A three-term mixture is invariant under a common rescale of ``q`` and weights."""
    base = psum_chisq(5.0, [1.0, 0.6, 0.4], df=[3, 1, 1])
    got = psum_chisq(5.0 * c, [1.0 * c, 0.6 * c, 0.4 * c], df=[3, 1, 1])
    assert got == pytest.approx(base, rel=1e-12)


# A multi-term non-central sum is outside the supported regime, so this row also
# goes through the quadrature helper.  With the public path gated, that helper is
# the only thing keeping the integrand's non-centrality terms under test.


@pytest.mark.parametrize("c", _RESCALINGS)
def test_noncentrality_is_dimensionless(c):
    """``noncentrality`` is unit-free, so it stays fixed while ``q`` and weights rescale."""
    base = _sf_by_quadrature(10.0, [1.0, 0.5], df=[3, 1], noncentrality=[2.0, 0.5])
    got = _sf_by_quadrature(
        10.0 * c, [1.0 * c, 0.5 * c], df=[3, 1], noncentrality=[2.0, 0.5]
    )
    assert got == pytest.approx(base, rel=1e-12)


# The naive variance ``sum(w**2 * (2 df + 4 ncp))`` squares the raw weights, so it
# overflows to inf above roughly ``1.34e154 / sqrt(sum(2 nu + 4 delta**2))`` and
# underflows to zero below about 2.2e-162.  On overflow every normalized weight
# and the frequency collapse to 0 and the raw CDF is exactly 0.5, silently.
# These rows probe extreme magnitudes beyond realistic GAM inputs.


@pytest.mark.parametrize("c", [1e-300, 1e-200, 1e160, 1e300])
def test_scale_invariance_beyond_naive_variance_range(c):
    """The rescale still holds where the naive variance over- or underflows."""
    assert _sf_by_quadrature(c, [c], df=1) == pytest.approx(chi2.sf(1.0, 1), rel=1e-12)


def test_many_terms_at_the_overflow_edge():
    """More terms lower the overflow threshold: 20 terms at ``df=5`` break by 1e153."""
    df = np.full(20, 5.0)
    base = _sf_by_quadrature(50.0, np.ones(20), df=df)
    got = _sf_by_quadrature(50.0 * 1e153, np.full(20, 1e153), df=df)
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
    monkeypatch.setattr(chisqsum, "_cdf_approx", _unreachable)

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
    monkeypatch.setattr(chisqsum, "_cdf_approx", _unreachable)

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


# The reflection identity ``Pr(Q > q) = Pr(-Q <= -q)`` used to be checked here on
# ``[1, 0.6, -0.4]`` at ``q = +/-1, +/-4``.  Both sides of it are now outside the
# supported regime, the left as signed weights away from zero and the right as an
# all-negative list, so it lives in the regime-gate section below as two rejected
# shapes rather than as a numerical identity.


# --------------------------------------------------------------------------- #
# Exact reductions.  Some shapes have closed forms, and the quadrature is worst
# on the one that matters most.
#
# mgcv's estimated-dispersion smooth-term test is ``P(Q > 0)`` for
# ``Q = sum_j v_j X_j - (d/k0) S``.  Once equal positive weights are merged that
# is two terms of opposite sign at exactly ``q = 0``, which is an F survival
# probability and needs no inversion at all.  The quadrature put the rank-1 case
# at a raw survival probability of -0.3736, so it raised "Probabilities must be
# in [0, 1]" where the answer is 0.05563.  F1 is the double-counted head at
# ``q = 0``, and it is only half closed here: three or more signed terms still
# reach the integrator.
# --------------------------------------------------------------------------- #


def _sf_two_term_by_conditioning(w_pos, m, w_neg, n):
    """
    ``P(w_pos X_m + w_neg Y_n > 0)`` for independent central chi-squares.

    Conditions on ``Y_n`` and integrates the exact conditional probability, so
    this is an oracle for the F identity itself, independent of
    :func:`scipy.stats.f.sf` and of any characteristic-function inversion.
    """

    def integrand(y):
        return chi2.pdf(y, n) * chi2.sf(-w_neg * y / w_pos, m)

    return quad(integrand, 0.0, np.inf, limit=400, epsabs=1e-14, epsrel=1e-13)[0]


# (rank, k0, d): ``rank`` unit weights against ``-d/k0``.
_TESTSTAT_ROWS = [
    pytest.param(1, 5, 3.84, id="rank1-k5"),
    pytest.param(1, 50, 3.84, id="rank1-k50"),
    pytest.param(1, 50, 1.0, id="rank1-k50-weak"),
    pytest.param(1, 500, 2.0, id="rank1-k500"),
    pytest.param(2, 50, 1.0, id="rank2-k50"),
    pytest.param(3, 5, 12.0, id="rank3-k5"),
]


@pytest.mark.parametrize("rank, k0, d", _TESTSTAT_ROWS)
def test_teststat_at_zero_is_an_f_probability(rank, k0, d):
    """The estimated-dispersion p-value is ``f.sf(d/r, r, k0)``, exactly."""
    weights = [1.0] * rank + [-d / k0]
    df = [1.0] * rank + [float(k0)]
    assert psum_chisq(0.0, weights, df=df) == pytest.approx(
        f_dist.sf(d / rank, rank, k0), rel=1e-12
    )


def test_teststat_headline_row():
    """Rank 1, ``k0 = 50``, ``d = 3.84``: the row the report opens with."""
    assert psum_chisq(0.0, [1.0, -3.84 / 50], df=[1, 50]) == pytest.approx(
        0.05563, abs=1e-5
    )


@pytest.mark.parametrize(
    "w_pos, m, w_neg, n",
    [
        (1.0, 1.0, -3.84 / 50, 50.0),
        (2.5, 3.0, -0.4, 7.0),
        (1.0, 2.5, -0.07, 12.5),  # non-integer degrees of freedom on both sides
        (0.3, 1.0, -1.7, 2.0),  # the negative weight dominates
    ],
)
def test_two_term_reduction_matches_an_independent_integral(w_pos, m, w_neg, n):
    """The F identity is checked against conditioning on the negative term."""
    assert psum_chisq(0.0, [w_pos, w_neg], df=[m, n]) == pytest.approx(
        _sf_two_term_by_conditioning(w_pos, m, w_neg, n), rel=1e-10
    )


def test_two_term_reduction_does_not_integrate(monkeypatch):
    """The closed form is exact, so no integrand is evaluated."""
    monkeypatch.setattr(chisqsum, "_cdf_approx", _unreachable)

    assert psum_chisq(0.0, [1.0, -3.84 / 50], df=[1, 50]) == pytest.approx(
        f_dist.sf(3.84, 1, 50), rel=1e-12
    )


@pytest.mark.parametrize(
    "z, weights, df, noncentrality",
    [
        pytest.param(1e-12, [-0.0768, 1.0], [50.0, 1.0], [0.0, 0.0], id="q-not-zero"),
        pytest.param(
            0.0, [-0.0768, 0.5, 1.0], [50.0, 1.0, 1.0], [0.0] * 3, id="three-terms"
        ),
        pytest.param(0.0, [-0.0768, 1.0], [50.0, 1.0], [0.0, 2.0], id="non-central"),
    ],
)
def test_two_term_reduction_declines_outside_its_domain(z, weights, df, noncentrality):
    """
    Each boundary condition of the identity, perturbed one at a time.

    The reducer must decline rather than apply the formula where it does not
    hold.  Declining sends the input to the regime gate, which is where the
    decision to refuse or integrate belongs.
    """
    reduced = chisqsum._reduce(
        z, np.array(weights), np.array(df), np.array(noncentrality)
    )
    assert reduced is None


@pytest.mark.parametrize(
    "q, weight, dof, ncp",
    [
        (2.0, 1.0, 1, 0.0),
        (5.0, 2.0, 3, 0.0),
        (10.0, 1.0, 3, 2.0),
        (6.0, 0.5, 2, 1.5),
    ],
)
def test_single_term_is_exact_and_does_not_integrate(monkeypatch, q, weight, dof, ncp):
    """One term is a scaled (non-central) chi-square, whatever its weight."""
    monkeypatch.setattr(chisqsum, "_cdf_approx", _unreachable)

    expected = chi2.sf(q / weight, dof) if ncp == 0.0 else ncx2.sf(q / weight, dof, ncp)
    got = psum_chisq(q, [weight], df=[dof], noncentrality=[ncp])
    assert got == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("q", [-4.0, -1.0, 0.0, 2.0])
def test_single_negative_term_is_exact(q):
    """``Pr(w X > q)`` with ``w < 0`` is ``Pr(X < q/w)``, so the tails swap."""
    assert psum_chisq(q, [-2.0], df=[3]) == pytest.approx(
        chi2.cdf(q / -2.0, 3), rel=1e-12
    )


@pytest.mark.parametrize("q", [-1e300, -5.0, -1e-300, 0.0])
def test_positive_weights_are_certain_at_non_positive_q(monkeypatch, q):
    """``Q`` is positive almost surely, so ``Pr(Q > q) = 1`` for ``q <= 0``."""
    monkeypatch.setattr(chisqsum, "_cdf_approx", _unreachable)

    case = {"weights": [1.0, 0.6, 0.4], "df": [3, 1, 1]}
    assert psum_chisq(q, lower_tail=False, **case) == 1.0
    assert psum_chisq(q, lower_tail=True, **case) == 0.0


# --------------------------------------------------------------------------- #
# The regime gate.
#
# The quadrature has no independent oracle outside the GAM regime, and F4 shows
# QUADPACK's own diagnostics do not notice when it is wrong.  Everything the
# reductions do not answer and the GAM path does not need is therefore refused,
# loudly, rather than integrated and returned.
# --------------------------------------------------------------------------- #

_UNSUPPORTED = [
    pytest.param({"q": 2.0, "weights": [1.0, 0.6, -0.4], "df": [3, 1, 2]}, id="signed"),
    pytest.param(
        {"q": -2.0, "weights": [1.0, -0.4], "df": [1, 50]}, id="signed-two-term"
    ),
    pytest.param(
        {"q": 2.0, "weights": [-1.0, -0.6, 0.4], "df": [3, 1, 2]},
        id="signed-mirror",
    ),
    pytest.param({"q": 2.0, "weights": [-1.0, -2.0], "df": [1, 1]}, id="all-negative"),
    pytest.param(
        {
            "q": 10.0,
            "weights": [1.0, 0.5],
            "df": [3, 1],
            "noncentrality": [2.0, 0.5],
        },
        id="multi-term-non-central",
    ),
    pytest.param(
        {
            "q": 0.0,
            "weights": [1.0, -0.4],
            "df": [1, 50],
            "noncentrality": [2.0, 0.0],
        },
        id="signed-non-central-at-zero",
    ),
]


@pytest.mark.parametrize("case", _UNSUPPORTED)
def test_unsupported_regimes_raise(case):
    """Outside the validated regime the answer is an error, not a number."""
    with pytest.raises(NotImplementedError, match="validated GAM regime"):
        psum_chisq(**case)


@pytest.mark.parametrize("case", _UNSUPPORTED)
def test_unsupported_regimes_do_not_integrate(monkeypatch, case):
    """The refusal comes before the integrator, not from it."""
    monkeypatch.setattr(chisqsum, "_cdf_approx", _unreachable)

    with pytest.raises(NotImplementedError):
        psum_chisq(**case)


def test_gate_message_names_the_standardized_inputs():
    """
    The message has to carry enough to decide whether to widen the contract.

    That means the standardised inputs, since the raw ones do not determine what
    the method sees.
    """
    with pytest.raises(NotImplementedError) as excinfo:
        psum_chisq(2.0, [1.0, 0.6, -0.4], df=[3, 1, 2])

    message = str(excinfo.value)
    assert "z=" in message
    assert "weights=" in message
    assert "df=" in message
    assert "noncentrality=" in message
    assert "report" in message


@pytest.mark.parametrize("q", [0.5, 5.0, 40.0])
def test_positive_central_mixtures_stay_supported(q):
    """The gate must not narrow the mainline: positive central at any finite q."""
    assert psum_chisq(q, [1.0, 0.6, 0.4], df=[3, 1, 1]) == pytest.approx(
        _sf_by_quadrature(q, [1.0, 0.6, 0.4], df=[3, 1, 1]), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# F4.  QUADPACK reports ``ier=0`` and an absolute error of 4e-15 on a value that
# is wrong in the second decimal place.  An error estimate produced by the same
# nodes that missed the mass cannot report that the mass was missed, so the only
# guard that can see it is a recomputation using different nodes.
#
# This is also the guard that makes the fallback below admissible: without it,
# falling back onto a route the dispatcher had already predicted was wrong would
# be a guess.
# --------------------------------------------------------------------------- #


def _standardize(weights, df):
    """Weights and sd for a central structure, as the cross-check sees them."""
    w = np.asarray(weights, dtype=float)
    d = np.asarray(df, dtype=float)
    ncp = np.zeros_like(w)
    w, d, ncp = chisqsum._collapse_terms(w, d, ncp)
    sd = chisqsum._standard_deviation(w, d, ncp)
    return w / sd, d, ncp


@pytest.mark.parametrize(
    "weights, df",
    [
        pytest.param([1.0, -3.84 / 50], [1, 50], id="rank1-k50-d3.84"),
        pytest.param(
            [1.1830127, 0.3169873, -3.84 / 50], [1, 1, 50], id="rank1.5-k50-d3.84"
        ),
        pytest.param([1.0, 1.0, -6.0 / 50], [1, 1, 50], id="rank2-k50-d6"),
    ],
)
def test_cross_check_sees_what_quadpack_reports_as_success(weights, df):
    """
    The measured F4 rows: QAWF converges confidently on a wrong answer.

    At ``z = 1e-4`` these structures make QAWF's first cycle step over the
    integrand.  It returns without complaint, and its own error estimate is
    around 1e-11 while the answer is wrong in the second decimal place.  Moving
    the split point, which is where the cycle grid starts, moves the answer by
    more than a hundred million times the claimed uncertainty.
    """
    w, d, ncp = _standardize(weights, df)
    z = 1e-4

    value, error = chisqsum._cdf_qawf(z, w, d, ncp, 1.0, 1e-10, 1e-10, 200)
    other, other_error = chisqsum._cdf_qawf(
        z, w, d, ncp, chisqsum._QAWF_CROSS_CHECK_SPLIT, 1e-10, 1e-10, 200
    )

    # The failure is real and large, and neither run suspects it.
    assert abs(value - other) > 1e-2
    assert error + other_error < 1e-9

    with pytest.raises(RuntimeError, match="did not survive its independent"):
        chisqsum._cross_check(
            chisqsum._QAWF, value, error, z, w, d, ncp, 1.0, 1e-10, 1e-10, 200
        )


def _corrupt_the_first_split_only(offset=1e-3):
    """
    A ``_quad`` stand-in that spoils the cosine tail on the primary split only.

    Perturbing both runs by the same amount would move them together and the
    cross-check would rightly stay quiet, so the corruption is keyed to the
    split point the primary computation uses.  This imitates the real failure:
    one cycle grid misses mass that the other one finds.
    """
    truthful = chisqsum._quad

    def corrupt(func, a, b, *args, **kwargs):
        value, error = truthful(func, a, b, *args, **kwargs)
        if func is chisqsum._tail_cos_coefficient and a == 1.0:
            return value + offset, error
        return value, error

    return corrupt


def test_cross_check_raises_when_a_piece_is_corrupted(monkeypatch):
    """A wrong value from one of the three QAWF pieces must not get through."""
    monkeypatch.setattr(chisqsum, "_quad", _corrupt_the_first_split_only())

    with pytest.raises(RuntimeError, match="did not survive its independent"):
        psum_chisq(5.0, [1.0, 0.6, 0.4], df=[3, 1, 1])


def test_check_false_bypasses_the_cross_check(monkeypatch):
    """The guard is skippable, and skipping it is what ``check=False`` means."""
    monkeypatch.setattr(chisqsum, "_quad", _corrupt_the_first_split_only())

    # The same call that raises above returns a (wrong) number here, which is
    # the point: the caller asked for the check to be skipped.
    assert 0.0 <= psum_chisq(5.0, [1.0, 0.6, 0.4], df=[3, 1, 1], check=False) <= 1.0


def test_cross_check_counts_a_second_computation(monkeypatch):
    """The guard must actually recompute, not re-read the first result."""
    calls = []
    truthful = chisqsum._cdf_qawf

    def counting(q, weights, df, ncp, split, *args):
        calls.append(split)
        return truthful(q, weights, df, ncp, split, *args)

    monkeypatch.setattr(chisqsum, "_cdf_qawf", counting)
    psum_chisq(5.0, [1.0, 0.6, 0.4], df=[3, 1, 1])

    assert calls == [1.0, chisqsum._QAWF_CROSS_CHECK_SPLIT]


def test_no_spurious_cross_check_failures_on_healthy_inputs():
    """
    The guard must be quiet on inputs where nothing is wrong.

    A guard that fires on healthy input is worse than no guard, because it
    trains the caller to disable it.  The safety factor and the comparison
    floor were both calibrated against this sweep.
    """
    rng = np.random.default_rng(20260728)
    checked = 0
    for _ in range(150):
        size = int(rng.integers(2, 6))
        weights = np.unique(rng.uniform(0.05, 3.0, size=size))
        if weights.size < 2:
            continue
        df = rng.integers(1, 8, size=weights.size).astype(float)
        mean = float(np.sum(weights * df))
        q = float(rng.uniform(0.0, 2.0)) * mean

        # No assertion on the value: this is only about the guard staying quiet.
        psum_chisq(q, weights, df=df)
        checked += 1

    assert checked > 100


# --------------------------------------------------------------------------- #
# The fallback between quadrature routes.
#
# ``abs(z)`` predicts which route will work, and the prediction is wrong for
# slowly decaying integrands.  The envelope falls off like ``u**-(1+sum(df)/2)``,
# so a total of two degrees of freedom decays only like ``u**-2`` and tanh-sinh
# will not converge on it anywhere in its own band.
#
# This is not a corner case.  ``notes/mgcv.r:3837`` is the known-dispersion
# branch of mgcv's ``testStat``, ``psum.chisq(d, val)`` with ``df`` defaulting to
# all ones, and for a rank between 1 and 2 it builds ``val`` of length exactly
# two.  Poisson GAMs have ``scale.estimated = FALSE``, so that is the branch this
# package takes.  Every one of these rows raised before the fallback existed.
# --------------------------------------------------------------------------- #


def _mgcv_fractional_rank_weights(rank):
    """``val`` as ``mgcv:::testStat`` builds it for a rank between 1 and 2."""
    rp = rank  # mgcv's rp = nu + 1, with nu = rank - 1
    first = (rp + np.sqrt(rp * (2.0 - rp))) / 2.0
    return [first, rp - first]


@pytest.mark.parametrize("rank", [1.2, 1.5, 1.8])
@pytest.mark.parametrize("stat", [1e-4, 1e-3, 1e-2, 1e-1])
def test_mgcv_fractional_rank_known_dispersion_is_answered(rank, stat):
    """The rank-1.x rows of mgcv's known-dispersion test, which used to raise."""
    first, second = _mgcv_fractional_rank_weights(rank)

    assert psum_chisq(stat, [first, second], df=[1, 1]) == pytest.approx(
        sf_two_chi1(np.array(stat), first, second), rel=1e-10
    )


def test_fallback_is_reached_by_the_low_degree_of_freedom_rows():
    """
    These rows are answered by the route ``abs(z)`` did not pick.

    Without this the test above would keep passing if the switch were merely
    retuned, and the fallback it is supposed to exercise could rot.
    """
    first, second = _mgcv_fractional_rank_weights(1.5)
    w, d, ncp = _standardize([first, second], [1, 1])
    z = 1e-3 / chisqsum._standard_deviation(
        *(np.asarray(x, dtype=float) for x in ([first, second], [1.0, 1.0])),
        np.zeros(2),
    )

    # The dispatcher prefers tanh-sinh here, and tanh-sinh cannot do it.
    assert abs(z) <= chisqsum._Z_SWITCH
    with pytest.raises(chisqsum._QuadratureNotConverged):
        chisqsum._cdf_tanhsinh(z, w, d, ncp)

    # The fallback route can, and the dispatcher returns its answer.
    value, _error = chisqsum._cdf_approx(z, w, d, ncp, 1.0, 1e-10, 1e-10, 200)
    qawf_value, _ = chisqsum._cdf_qawf(z, w, d, ncp, 1.0, 1e-10, 1e-10, 200)
    assert value == pytest.approx(qawf_value, rel=1e-12)


def test_fallback_is_cross_checked_even_when_checking_is_off():
    """
    ``check=False`` does not extend to a fallback result.

    The dispatcher predicted this route was the wrong one for this input, so its
    answer is only admissible with independent evidence behind it.  QAWF is
    silently wrong at small ``abs(z)`` on other structures, and an unvalidated
    fallback onto it would reintroduce F4.
    """
    first, second = _mgcv_fractional_rank_weights(1.5)
    w, d, ncp = _standardize([first, second], [1, 1])
    z = 1e-4

    calls = []
    truthful = chisqsum._cdf_qawf

    def counting(q, weights, dof, ncp_, split, *args):
        calls.append(split)
        return truthful(q, weights, dof, ncp_, split, *args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chisqsum, "_cdf_qawf", counting)
        chisqsum._cdf_approx(z, w, d, ncp, 1.0, 1e-10, 1e-10, 200, check=False)

    assert calls == [1.0, chisqsum._QAWF_CROSS_CHECK_SPLIT]


def test_both_routes_failing_names_both():
    """
    A band remains where no two independent routes both converge.

    It must raise, and the message must say what was tried.  Returning the one
    route that happened to converge would be returning an unvalidated number,
    which is the thing this module exists not to do.
    """
    first, second = _mgcv_fractional_rank_weights(1.5)

    with pytest.raises(RuntimeError) as excinfo:
        psum_chisq(1e-5, [first, second], df=[1, 1])

    message = str(excinfo.value)
    assert "neither quadrature route" in message
    assert chisqsum._TANHSINH in message
    assert chisqsum._QAWF in message
    assert "weights=" in message
    assert "report" in message


def test_a_corrupt_result_is_not_laundered_through_the_fallback():
    """
    Disagreement stops; only non-convergence falls back.

    A route that converges to a different answer means one of the two is
    confidently wrong, which is a reason to raise, not a reason to quietly
    change method and report the other one.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chisqsum, "_quad", _corrupt_the_first_split_only())
        with pytest.raises(RuntimeError) as excinfo:
            psum_chisq(5.0, [1.0, 0.6, 0.4], df=[3, 1, 1])

    assert "did not survive its independent" in str(excinfo.value)
    assert "neither quadrature route" not in str(excinfo.value)


def test_tanhsinh_accuracy_limit_at_two_degrees_of_freedom():
    """
    A known limitation, pinned so that it cannot quietly get worse.

    Just below where tanh-sinh stops converging on a two-degrees-of-freedom
    integrand there is a narrow band where it converges to an answer about twice
    the requested absolute tolerance away, and its domain-split variant agrees
    with it, so the cross-check does not see it.  The measured worst case is
    2.2e-10 against a request of 1e-10.  QAWF is accurate to 1e-13 there but the
    dispatcher does not prefer it, because ``abs(z)`` is well inside tanh-sinh's
    band.

    This is a tolerance miss of about a factor of two, not an F4-class failure,
    and it is confined to the lowest degrees of freedom the contract allows.
    The bound is asserted one-sided: the limitation must not get worse, and a
    future SciPy that resolves it should not turn this red.

    Which ``z`` converges is not stable enough to pin, so this sweeps a band
    and looks only at the points where tanh-sinh reported success.
    """
    worst = 0.0
    converged = 0
    for a, b in [(0.125, 0.25), (0.125, 1.0), (0.5, 1.0), (1.0, 2.0)]:
        w, d, ncp = _standardize([a, b], [1, 1])
        sd = chisqsum._standard_deviation(
            np.array([a, b]), np.array([1.0, 1.0]), np.zeros(2)
        )
        for z in np.geomspace(1e-5, 1e-4, 12):
            exact = 1.0 - float(sf_two_chi1(np.array(z * sd), a, b))
            try:
                value, _error = chisqsum._cdf_tanhsinh(z, w, d, ncp)
            except chisqsum._QuadratureNotConverged:
                continue
            converged += 1
            worst = max(worst, abs(value - exact))

    assert converged > 0
    assert worst < 1e-8


# --------------------------------------------------------------------------- #
# F5.  Below the resolution of the quadrature the survival function is not a
# small number, it is noise, and noise is not monotone.  The old code returned
# 7.8e-16 for a true 1.6e-21, and raised outright on a -1.3e-15 whose own error
# estimate was 3.8e-13.
# --------------------------------------------------------------------------- #

_DEEP_TAIL_STRUCTURE = ([1.0, 0.6, 0.4], [3.0, 1.0, 1.0])


def test_deep_tail_is_monotone_and_never_leaves_the_unit_interval():
    """The whole point of the floor: an ordered, in-range survival function."""
    weights, df = _DEEP_TAIL_STRUCTURE
    q = np.linspace(1.0, 200.0, 60)

    with pytest.warns(UserWarning, match="smaller than the quadrature can resolve"):
        p = psum_chisq(q, weights, df=df)

    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(np.diff(p) <= 0.0)


def test_below_the_floor_is_exactly_zero():
    """Not a small number, and not a negative one."""
    weights, df = _DEEP_TAIL_STRUCTURE

    with pytest.warns(UserWarning, match="smaller than the quadrature can resolve"):
        p = psum_chisq([80.0, 129.0, 200.0], weights, df=df)

    assert np.all(p == 0.0)


def test_q_129_no_longer_raises():
    """
    The measured regression: a value inside its own error allowance was rejected.

    ``-1.3e-15`` against an estimated error of ``3.8e-13`` is arithmetic, not a
    bug, and the old range check had no allowance at all.
    """
    weights, df = _DEEP_TAIL_STRUCTURE

    with pytest.warns(UserWarning, match="smaller than the quadrature can resolve"):
        assert psum_chisq(129.0, weights, df=df) == 0.0


@pytest.mark.parametrize("nu, k0", [(0.2, 15), (0.2, 60), (0.5, 15)])
def test_rank_four_at_a_near_zero_statistic_does_not_raise(nu, k0):
    """
    The same guard one step further: an error estimate of exactly zero.

    An allowance proportional to the estimate covers a small estimate, not a
    zero one.  This is the estimated-dispersion call for a rank-4 smooth at a
    near-zero statistic, where the survival probability is exactly ``1``, both
    quadrature pieces report ``0.0``, and the assembled value lands one ulp
    above ``1``.  Measured before the floor: ``RuntimeError`` on
    ``sf - 1 == 2.220e-16``.
    """
    val = np.ones(5)
    rp = 1.0 + nu
    val[3] = (rp + np.sqrt(rp * (2.0 - rp))) / 2.0
    val[4] = rp - val[3]
    d = 1e-7 * np.sqrt(2.0 * np.sum(val**2))

    weights = np.concatenate([val, [-d / k0]])
    df = np.concatenate([np.ones(val.size), [float(k0)]])

    assert psum_chisq(0.0, weights, df=df) == 1.0


def test_range_check_still_rejects_a_real_violation():
    """
    The floor is round-off sized, so it cannot mask a genuine failure.

    It sits fourteen orders of magnitude below the smallest real miss on
    record, the ``1.17`` of F1.
    """
    assert chisqsum._RANGE_CHECK_FLOOR < 1e-14

    with pytest.raises(RuntimeError, match=r"outside \[0, 1\]"):
        chisqsum._finalize(np.array([1.17]), np.array([0.0]))


def test_tightening_epsabs_resolves_more_of_the_tail():
    """
    The floor is the requested tolerance, not a fixed property of the module.

    This is what makes the warning's advice actionable.
    """
    weights, df = _DEEP_TAIL_STRUCTURE

    with pytest.warns(UserWarning):
        loose = psum_chisq(60.0, weights, df=df)
    tight = psum_chisq(60.0, weights, df=df, epsabs=1e-13, epsrel=1e-13)

    assert loose == 0.0
    assert 1e-13 < tight < 1e-11


def test_values_kept_above_the_floor_are_accurate():
    """A kept value must be worth keeping, not merely above the threshold."""
    weights, df = _DEEP_TAIL_STRUCTURE
    q = [30.0, 35.0, 40.0, 45.0, 50.0]

    loose = psum_chisq(q, weights, df=df)
    tight = psum_chisq(q, weights, df=df, epsabs=1e-13, epsrel=1e-13)

    assert np.all(loose > 0.0)
    np.testing.assert_allclose(loose, tight, rtol=1e-6, atol=1e-14)


def test_exact_reductions_are_never_floored():
    """
    The floor is driven by the error estimate so that it cannot touch these.

    A closed form integrated nothing, reports an error of exactly zero, and is
    good to full relative precision far below any quadrature floor.
    """
    assert psum_chisq(400.0, [1.0], df=[1]) == pytest.approx(
        chi2.sf(400.0, 1), rel=1e-12
    )
    assert psum_chisq(400.0, [1.0], df=[1]) < 1e-88


def test_a_probability_outside_its_allowance_still_raises():
    """The range check is relaxed by the error estimate, not removed."""
    probabilities = np.array([-1e-6])
    errors = np.array([1e-13])

    with pytest.raises(RuntimeError, match="outside \\[0, 1\\]"):
        chisqsum._finalize(probabilities, errors)


def test_a_probability_inside_its_allowance_is_clipped():
    """
    An overshoot smaller than the error estimate is arithmetic, so it clips.

    The clipped zero is then also below the floor, which is consistent: a value
    the quadrature placed below zero is certainly one it could not resolve.
    """
    probabilities = np.array([-1.3e-15, 1.0 + 1e-15])
    errors = np.array([3.8e-13, 3.8e-13])

    with pytest.warns(UserWarning, match="smaller than the quadrature can resolve"):
        finalized = chisqsum._finalize(probabilities, errors)

    assert np.all(finalized == np.array([0.0, 1.0]))


# --------------------------------------------------------------------------- #
# F6 and F7.  The public ``epsabs`` was handed in full to each of three
# integrations, so it was a per-piece figure that the assembled result could
# exceed.  ``limlst`` was inherited from a SciPy default that is not part of its
# documented signature.
# --------------------------------------------------------------------------- #


def test_each_quadrature_piece_gets_a_third_of_the_budget():
    """``epsabs`` is an end-to-end budget for the three pieces together."""
    requested = []

    def recording(func, a, b, args, weight, wvar, epsabs, epsrel, limit, limlst=None):
        recording.calls.append((func.__name__, epsabs, limlst))
        return 0.0, 0.0

    recording.calls = requested

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chisqsum, "_quad", recording)
        chisqsum._cdf_qawf(
            2.0,
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.0]),
            1.0,
            3e-10,
            1e-10,
            200,
        )

    assert [call[1] for call in requested] == pytest.approx([1e-10, 1e-10, 1e-10])


def test_limlst_is_set_explicitly_on_both_fourier_tails():
    """
    The oscillatory cycle cap must not be inherited from SciPy's hidden default.

    It has no measurable effect on any case probed here, so there is no outcome
    to assert against.  What can be asserted is that the module states it.
    """
    calls = []

    def recording(func, a, b, args, weight, wvar, epsabs, epsrel, limit, limlst=None):
        calls.append((weight, limlst))
        return 0.0, 0.0

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chisqsum, "_quad", recording)
        chisqsum._cdf_qawf(
            2.0,
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.0]),
            1.0,
            1e-10,
            1e-10,
            200,
        )

    assert calls == [
        (None, None),
        ("cos", chisqsum._LIMLST),
        ("sin", chisqsum._LIMLST),
    ]


def test_limlst_reaches_scipy():
    """The wrapper passes it on rather than accepting and dropping it."""
    seen = {}

    def fake_quad(*args, **kwargs):
        seen.update(kwargs)
        return 1.25, 2.5e-9, {"neval": 21}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chisqsum, "quad", fake_quad)
        chisqsum._quad(lambda x: x, 0.0, np.inf, (), "cos", 1.0, 1e-10, 1e-10, 200, 321)

    assert seen["limlst"] == 321


def test_limlst_is_not_passed_to_the_non_oscillatory_head():
    """SciPy only uses it for the Fourier integrator, so it is not sent there."""
    seen = {}

    def fake_quad(*args, **kwargs):
        seen.update(kwargs)
        return 1.25, 2.5e-9, {"neval": 21}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chisqsum, "quad", fake_quad)
        chisqsum._quad(lambda x: x, 0.0, 1.0, (), None, None, 1e-10, 1e-10, 200)

    assert "limlst" not in seen


def test_non_convergence_is_distinguishable_from_corruption():
    """
    The fallback keys off this distinction, so it has to be real.

    A route reporting that it could not resolve the integrand leaves the other
    route free to try.  A route returning a non-finite number does not.
    """
    assert issubclass(chisqsum._QuadratureNotConverged, RuntimeError)

    def diverging(*_args, **_kwargs):
        return 1.25, 0.1, {}, "maximum number of cycles reached", {1: "failed"}

    def corrupting(*_args, **_kwargs):
        return np.inf, 0.1, {"neval": 21}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chisqsum, "quad", diverging)
        with pytest.raises(chisqsum._QuadratureNotConverged):
            chisqsum._quad(lambda x: x, 0.0, np.inf, (), "cos", 1.0, 1e-10, 1e-10, 200)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chisqsum, "quad", corrupting)
        with pytest.raises(RuntimeError) as excinfo:
            chisqsum._quad(lambda x: x, 0.0, 1.0, (), None, None, 1e-10, 1e-10, 200)

    assert not isinstance(excinfo.value, chisqsum._QuadratureNotConverged)

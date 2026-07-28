"""
Failure-mode regression tests for :mod:`pgam_jax._chisqsum`.
"""

import numpy as np
import pytest
from scipy.stats import chi2

from pgam_jax._chisqsum import psum_chisq

# --------------------------------------------------------------------------- #
# F3.  Nothing in the method was scale-free.
#
# Multiplying ``q``, every weight and ``sigma`` by a common positive constant is
# a change of units.  It cannot change the probability.  The implementation used
# to pass the raw ``q`` as the oscillatory quadrature's frequency while placing
# the split point at ``1/sd``, so the answer moved: correct at ``c = 1e6``,
# raising at ``c = 1e8``, and silently returning the complement at ``c = 1e10``.
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
@pytest.mark.parametrize("lower_tail", [False, True])
def test_sigma_scale_invariance(c, lower_tail):
    """``sigma`` carries the same units as ``q``, so it rescales with everything else."""
    base = psum_chisq(3.0, [1.0, 0.5], df=[2, 1], sigma=0.7, lower_tail=lower_tail)
    got = psum_chisq(
        3.0 * c,
        [1.0 * c, 0.5 * c],
        df=[2, 1],
        sigma=0.7 * c,
        lower_tail=lower_tail,
    )
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

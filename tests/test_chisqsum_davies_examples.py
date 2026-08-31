"""
Davies' 39 published test cases for :mod:`pgam_jax._chisqsum`.

These are the cases Robert Davies published with the method itself, in
``qf.dat`` and ``qf.txt`` inside ``http://www.robertnz.net/ftp/qf.tar.gz``.
``tests/data/davies_published_cases.tsv`` is Hail's verbatim transcription of
that set, one row per case, taken from
``hail/hail/test/resources/davies-genchisq-tests.tsv`` in ``hail-is/hail``.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import pgam_jax._chisqsum as chisqsum
from pgam_jax._chisqsum import psum_chisq

_FIXTURE = Path(__file__).parent / "data" / "davies_published_cases.tsv"

# Davies computed the published values at his own ``acc = 1e-4``, which QF.htm
# calls the method's comfort zone, and printed them to six decimals.  So the set
# supports a tolerance of about 1e-4 and no more.  Measured agreement over the
# 24 supported rows is 1.788e-05.
_PUBLISHED_ACCURACY = 1e-4


@dataclass(frozen=True)
class _Case:
    """
    One published case, as parsed from the fixture.

    Attributes
    ----------
    number : int
        Position in the published set, 1 to 39.  The fixture has a header, so
        this is the line number minus one.
    q : float
        Evaluation point, Davies' ``c``.
    weights, df, noncentrality : numpy.ndarray
        The mixture, Davies' ``weights``, ``k`` and ``lam``.
    expected : float
        The published probability.  It is the **lower** tail, ``P(Q <= q)``.
    """

    number: int
    q: float
    weights: np.ndarray
    df: np.ndarray
    noncentrality: np.ndarray
    expected: float

    @property
    def is_central(self) -> bool:
        return not np.any(self.noncentrality)

    @property
    def in_contract(self) -> bool:
        """
        Whether the supported regime covers this case.

        The contract is all-positive weights at any finite ``q``, or mixed-sign
        weights at exactly ``q = 0``, both central.  No published case sits at
        ``q = 0``, so the second shape never applies and this reduces to
        all-positive and central.
        """
        return bool(np.all(self.weights > 0.0)) and self.is_central


def _load_cases() -> list[_Case]:
    """
    Parse the fixture.

    The ``sigma``, ``lim``, ``acc`` and ``expected_n_iterations`` columns are
    kept in the file for provenance but are not used here.  ``sigma`` has no
    argument to be passed to, and the other three are parameters of Davies'
    algorithm rather than of this one.
    """
    with open(_FIXTURE, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [
        _Case(
            number=i,
            q=float(row["c"]),
            weights=np.array(json.loads(row["weights"]), dtype=float),
            df=np.array(json.loads(row["k"]), dtype=float),
            noncentrality=np.array(json.loads(row["lam"]), dtype=float),
            expected=float(row["expected"]),
        )
        for i, row in enumerate(rows, start=1)
    ]


_CASES = _load_cases()
_IN_CONTRACT = [c for c in _CASES if c.in_contract]
_OUT_OF_CONTRACT = [c for c in _CASES if not c.in_contract]


def _params(cases: list[_Case]):
    return [pytest.param(case, id=f"case{case.number:02d}") for case in cases]


# --------------------------------------------------------------------------- #
# The fixture itself.  These guard the shape of the set, so that an edit to the
# file cannot quietly change what the rows below are testing.
# --------------------------------------------------------------------------- #


def test_fixture_is_the_published_set():
    """39 cases, splitting 24 supported and 15 refused against the contract."""
    assert len(_CASES) == 39
    assert len(_IN_CONTRACT) == 24
    assert len(_OUT_OF_CONTRACT) == 15

    noncentral = [c for c in _OUT_OF_CONTRACT if np.all(c.weights > 0.0)]
    signed = [c for c in _OUT_OF_CONTRACT if np.any(c.weights < 0.0)]
    assert len(noncentral) == 12
    assert len(signed) == 3


def test_no_case_carries_an_additive_normal_term():
    """
    ``sigma`` is 0.000000 in all 39 rows.

    Davies' general form is ``Q = sum_j w_j X_j + sigma Z``, and this module
    drops the second piece.  The method's own author published no case
    exercising it, which is part of why it was dropped rather than kept as an
    argument with no oracle.  Asserted on the raw file, since the parser has no
    field to put it in.
    """
    with open(_FIXTURE, newline="") as fh:
        sigmas = {row["sigma"] for row in csv.DictReader(fh, delimiter="\t")}
    assert sigmas == {"0.000000"}


def test_published_values_were_computed_at_the_tolerance_used_here():
    """The 1e-4 bound below is Davies' own accuracy request, not a guess."""
    with open(_FIXTURE, newline="") as fh:
        accuracies = {float(row["acc"]) for row in csv.DictReader(fh, delimiter="\t")}
    assert accuracies == {_PUBLISHED_ACCURACY}


def test_the_published_set_avoids_every_known_failure_regime():
    """
    The published cases all sit in the bulk, so they validate none of the error cases.

    This is the reason the set is a coarse regression net rather than evidence
    about accuracy. The implementation that returned a p-value of 0 where the answer
    was 0.0556 passed all 39 of them.
    """
    z = np.array(
        [
            abs(case.q)
            / np.sqrt(np.sum(case.weights**2 * (2 * case.df + 4 * case.noncentrality)))
            for case in _CASES
        ]
    )
    probabilities = np.array([case.expected for case in _CASES])

    # F1 is exactly ``q = 0`` and F2 is ``abs(z)`` below about 1e-3.
    assert all(case.q != 0.0 for case in _CASES)
    assert z.min() > 0.1
    assert z.max() < 7.0

    # F5 is the deep tail, below an absolute floor of about 1e-13.
    assert probabilities.min() > 1e-3
    assert probabilities.max() < 0.999

    # F3 is the units bug, which needs a scale many orders away from 1.
    assert max(abs(case.q) for case in _CASES) < 1e3
    assert max(np.abs(case.weights).max() for case in _CASES) < 1e2


# --------------------------------------------------------------------------- #
# The 24 supported rows, as a regression net.
#
# All of them are multi-term positive central mixtures at a positive ``q``, so
# no exact reduction applies and every one exercises the quadrature.  That is
# what makes them useful ahead of a rewrite of the integrator.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _params(_IN_CONTRACT))
def test_supported_case_matches_the_published_value(case):
    """``expected`` is the lower tail, so this is the ``lower_tail=True`` call."""
    got = psum_chisq(case.q, case.weights, df=case.df, lower_tail=True)
    assert got == pytest.approx(case.expected, abs=_PUBLISHED_ACCURACY)


def test_every_supported_case_reaches_the_quadrature(monkeypatch):
    """
    No supported row is answered by a closed form.

    Without this, a later reduction could take over some of the rows above and
    they would go on passing while testing SciPy instead of the integrator.
    """
    calls = []
    integrate = chisqsum._cdf_approx

    def counting(*args, **kwargs):
        calls.append(args[0])
        return integrate(*args, **kwargs)

    monkeypatch.setattr(chisqsum, "_cdf_approx", counting)

    for case in _IN_CONTRACT:
        del calls[:]
        psum_chisq(case.q, case.weights, df=case.df, lower_tail=True)
        assert len(calls) == 1, f"case {case.number} did not integrate"


# --------------------------------------------------------------------------- #
# The 15 refused rows.
#
# Twelve are multi-term non-central and three are signed weights away from
# ``q = 0``.  Both are outside the GAM regime, so the module must refuse them
# rather than return a number.  Their published values stay in the fixture, so
# that widening the contract later starts with an oracle already in place.  The
# current code does in fact reproduce them, to 8.9e-6, which is the evidence
# that widening would be feasible.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _params(_OUT_OF_CONTRACT))
def test_unsupported_case_is_refused(case):
    """Outside the validated regime the answer is a raise, not a number."""
    with pytest.raises(NotImplementedError, match="outside the validated GAM regime"):
        psum_chisq(
            case.q,
            case.weights,
            df=case.df,
            noncentrality=case.noncentrality,
            lower_tail=True,
        )


def test_no_unsupported_case_reaches_the_quadrature(monkeypatch):
    """The refusal is a gate decision, so no integrand is evaluated."""

    def unreachable(*_args, **_kwargs):
        raise AssertionError("quadrature must not be reached")

    monkeypatch.setattr(chisqsum, "_cdf_approx", unreachable)

    for case in _OUT_OF_CONTRACT:
        with pytest.raises(NotImplementedError):
            psum_chisq(
                case.q,
                case.weights,
                df=case.df,
                noncentrality=case.noncentrality,
                lower_tail=True,
            )

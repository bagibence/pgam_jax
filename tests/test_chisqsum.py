"""Tests for :mod:`pgam_jax._chisqsum` (weighted sum-of-chi-squares CDF)."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2, ncx2

from pgam_jax._chisqsum import psum_chisq

_FIXTURE = Path(__file__).parent / "data" / "chisqsum_cases.json"


def _load_cases():
    with open(_FIXTURE) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Regression against the committed mgcv-generated fixture (no R needed).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _load_cases())
def test_matches_mgcv_fixture(case):
    """Every fixture case reproduces mgcv's psum.chisq to quadrature accuracy."""
    got = psum_chisq(
        case["q"], case["weights"], df=case["df"], lower_tail=case["lower_tail"]
    )
    assert got == pytest.approx(case["expected"], abs=1e-6)


# --------------------------------------------------------------------------- #
# Exact ground truth via SciPy, independent of the fixture / R.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "q, dof",
    [
        (0.5, 1),
        (1.0, 1),
        (3.84, 1),
        (25.0, 8),
        (15.0, 10),
        # large dof (many predictors / rich bases): bulk and deep tail
        (90.0, 50),
        (130.0, 100),
        (250.0, 200),
        (320.0, 200),  # ~ mean + 6 sd, p ~ 1e-7
    ],
)
def test_single_central_matches_chi2(q, dof):
    """A single unit-weight term is an ordinary chi-square survival function."""
    assert psum_chisq(q, [1.0], df=[dof]) == pytest.approx(chi2.sf(q, dof), abs=1e-9)


@pytest.mark.parametrize("q, weight, dof", [(5.0, 2.0, 3), (4.0, 0.5, 2)])
def test_single_scaled_matches_chi2(q, weight, dof):
    """A single scaled term ``w * chi2_k`` matches ``chi2.sf(q / w, k)``."""
    assert psum_chisq(q, [weight], df=[dof]) == pytest.approx(
        chi2.sf(q / weight, dof), abs=1e-9
    )


@pytest.mark.parametrize("q, dof, ncp", [(10.0, 3, 2.0), (6.0, 1, 1.5)])
def test_single_noncentral_matches_ncx2(q, dof, ncp):
    """A single non-central term matches the non-central chi-square."""
    assert psum_chisq(q, [1.0], df=[dof], noncentrality=[ncp]) == pytest.approx(
        ncx2.sf(q, dof, ncp), abs=1e-9
    )


# --------------------------------------------------------------------------- #
# Distributional / API properties.
# --------------------------------------------------------------------------- #


def test_tails_sum_to_one():
    upper = psum_chisq(5.0, [1, 0.6, 0.4], df=[3, 1, 1], lower_tail=False)
    lower = psum_chisq(5.0, [1, 0.6, 0.4], df=[3, 1, 1], lower_tail=True)
    assert upper + lower == pytest.approx(1.0, abs=1e-10)


def test_survival_is_monotone_decreasing():
    q = np.array([0.5, 1.0, 3.0, 7.0, 15.0, 30.0])
    p = psum_chisq(q, [1, 0.6, 0.4], df=[3, 1, 1], lower_tail=False)
    assert np.all(np.diff(p) < 0)
    assert np.all((p >= 0) & (p <= 1))


def test_scalar_returns_float_array_returns_array():
    scalar = psum_chisq(5.0, [1, 1], df=[1, 1])
    assert isinstance(scalar, float)
    arr = psum_chisq([2.0, 5.0], [1, 1], df=[1, 1])
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)


def test_large_dof_deep_tail_is_accurate():
    """Large total dof with a deep-tail point: default tol agrees with a tight one."""
    rng = np.random.default_rng(0)
    w = rng.uniform(0.2, 2.0, size=30)
    df = rng.integers(1, 6, size=30)  # total dof ~ 100
    mean = float(np.sum(w * df))
    sd = float(np.sqrt(np.sum(w**2 * 2 * df)))
    q = mean + 6 * sd  # p ~ 1e-6
    default = psum_chisq(q, w, df=df)
    tight = psum_chisq(q, w, df=df, epsabs=1e-13, epsrel=1e-13, limit=400)
    assert default == pytest.approx(tight, rel=1e-9)


def test_scalar_df_broadcasts():
    """A scalar ``df`` applies to every weight."""
    a = psum_chisq(5.0, [1, 1, 1], df=1)
    b = psum_chisq(5.0, [1, 1, 1], df=[1, 1, 1])
    assert a == pytest.approx(b, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"weights": [0.0, 0.0]}, "non-zero"),
        ({"weights": [1.0], "df": [0]}, "positive"),
        ({"weights": [1.0], "noncentrality": [-1.0]}, "non-negative"),
        ({"weights": [1.0, 1.0], "df": [1, 1, 1]}, "length"),
    ],
)
def test_invalid_inputs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        psum_chisq(5.0, **kwargs)


# --------------------------------------------------------------------------- #
# Optional live cross-check against mgcv (skipped when R/mgcv is unavailable).
# --------------------------------------------------------------------------- #

_R_SCRIPT = r"""
suppressMessages(ok <- require(mgcv, quietly = TRUE))
if (!ok) quit(status = 2)
args <- commandArgs(trailingOnly = TRUE)
q <- as.numeric(args[1]); lb <- as.numeric(strsplit(args[2], ",")[[1]])
df <- as.numeric(strsplit(args[3], ",")[[1]])
cat(sprintf("%.12g", psum.chisq(q, lb, df = df, lower.tail = FALSE, tol = 1e-7)))
"""


def _have_r_mgcv():
    if shutil.which("Rscript") is None:
        return False
    check = subprocess.run(
        [
            "Rscript",
            "-e",
            "quit(status = if (requireNamespace('mgcv', quietly=TRUE)) 0 else 2)",
        ],
        capture_output=True,
    )
    return check.returncode == 0


@pytest.mark.skipif(not _have_r_mgcv(), reason="Rscript with mgcv not available")
@pytest.mark.parametrize(
    "q, lb, df",
    [
        (6.0, [1.0, 2.0], [1, 3]),
        (12.0, [1, 0.8, 0.6, 0.4, 0.2], [2, 2, 1, 1, 1]),
        (3.0, [1.0, 0.3], [4, 1]),
    ],
)
def test_live_matches_mgcv(q, lb, df):
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(_R_SCRIPT)
        script = fh.name
    proc = subprocess.run(
        [
            "Rscript",
            script,
            repr(q),
            ",".join(map(repr, lb)),
            ",".join(map(str, df)),
        ],
        capture_output=True,
        text=True,
    )
    expected = float(proc.stdout.strip())
    got = psum_chisq(q, lb, df=df, lower_tail=False)
    assert got == pytest.approx(expected, abs=1e-6)

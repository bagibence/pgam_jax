"""Tests for `pgam_jax.concurvity` and `GAM.concurvity`.

Three layers:

- Pure linear-algebra unit tests on `concurvity()` with hand-built design
  matrices whose answer is known from the geometry (orthogonal blocks,
  span containment). These check the QR machinery directly, fast, no fit.
- Scenario tests that fit the three worked GAMs from
  `docs/concurvity_examples.md` and assert (a) the math-guaranteed
  invariants tightly and (b) the reported magnitudes as loose ranges.
  These are self-consistency tests: they lock in current behaviour and
  catch pipeline breakage.
- A golden cross-check (`test_matches_mgcv_golden`) that runs
  `concurvity()` on design matrices and coefficients saved from mgcv fits
  of the same three scenarios and requires agreement with
  `mgcv::concurvity` to 1e-14. Regenerate the golden files with
  `scripts/generate_concurvity_test_data.py` (needs R with mgcv and
  jsonlite).
"""

import json
from pathlib import Path

import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest

from pgam_jax import GAM
from pgam_jax.concurvity import TermBlock, concurvity, term_blocks_for_gam

# Seed shared by the scenario data generators. Each scenario gets its own
# fresh Generator so the fits are independent of test execution order
# (matters under pytest-xdist).
SEED = 123
N = 600


# ---------------------------------------------------------------------------
# Pure linear-algebra unit tests (no GAM, no fit).
# ---------------------------------------------------------------------------


def _blocks_3_2():
    """A 3-column block "A" followed by a 2-column block "B"."""
    return [TermBlock("A", 0, 2), TermBlock("B", 3, 4)]


def test_orthonormal_blocks_have_zero_worst():
    rng = np.random.default_rng(0)
    # Columns of Q are orthonormal, so block A and block B are orthogonal:
    # no direction of one lives in the span of the other.
    Q = np.linalg.qr(rng.standard_normal((200, 5)))[0]
    out = concurvity(jnp.asarray(Q), _blocks_3_2(), full=True)
    assert np.all(np.asarray(out["worst"]) < 1e-10)
    assert np.all(np.asarray(out["estimate"]) < 1e-10)


def test_block_in_span_has_worst_one():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((200, 3))
    B = A @ rng.standard_normal((3, 2))  # span(B) is contained in span(A)
    X = jnp.asarray(np.hstack([A, B]))
    out = concurvity(X, _blocks_3_2(), full=True)
    worst = np.asarray(out["worst"])
    # B lies entirely in span(A). A also has a 2-D direction coinciding with
    # span(B), so both worst-case ratios are 1.
    np.testing.assert_allclose(worst, [1.0, 1.0], atol=1e-8)
    # estimate (Frobenius, averaged over coefficient space) is 1 only for the
    # fully-contained block B, and strictly below 1 for A.
    estimate = np.asarray(out["estimate"])
    np.testing.assert_allclose(estimate[1], 1.0, atol=1e-8)
    assert estimate[0] < 1.0


def test_precondition_flag_does_not_change_result():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((200, 3))
    B = A @ rng.standard_normal((3, 2))
    X = jnp.asarray(np.hstack([A, B]))
    beta = jnp.asarray(rng.standard_normal(5))
    on = concurvity(X, _blocks_3_2(), beta=beta, precondition=True)
    off = concurvity(X, _blocks_3_2(), beta=beta, precondition=False)
    assert set(on) == set(off) == {"worst", "observed", "estimate"}
    for key in on:
        np.testing.assert_allclose(
            np.asarray(on[key]), np.asarray(off[key]), atol=1e-12
        )


@pytest.mark.parametrize("full", [True, False])
def test_observed_requires_beta(full):
    rng = np.random.default_rng(3)
    X = jnp.asarray(rng.standard_normal((200, 5)))
    without = concurvity(X, _blocks_3_2(), full=full)
    assert "observed" not in without
    with_beta = concurvity(
        X, _blocks_3_2(), beta=jnp.asarray(rng.standard_normal(5)), full=full
    )
    assert "observed" in with_beta


@pytest.mark.parametrize("full", [True, False])
def test_all_measures_within_unit_interval(full):
    rng = np.random.default_rng(4)
    X = jnp.asarray(rng.standard_normal((200, 5)))
    out = concurvity(
        X, _blocks_3_2(), beta=jnp.asarray(rng.standard_normal(5)), full=full
    )
    for key, vals in out.items():
        vals = np.asarray(vals)
        assert np.all(vals >= -1e-9), key
        assert np.all(vals <= 1 + 1e-9), key


# ---------------------------------------------------------------------------
# Scenario fixtures: the three worked examples from the docs.
# Each returns a fitted GAM plus the inputs it was fit on.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenario1():
    """Independent covariates -> low concurvity."""
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    eta = 1.0 + 0.6 * np.sin(4 * np.pi * x1) + 0.8 * (x2 - 0.5) ** 2
    y = rng.poisson(np.exp(eta))
    basis = nmo.basis.BSplineEval(
        12, bounds=(0.0, 1.0), label="s(x1)"
    ) + nmo.basis.BSplineEval(12, bounds=(0.0, 1.0), label="s(x2)")
    gam = GAM(basis, use_scipy=True, maxiter=15).fit((x1, x2), y)
    return gam, (x1, x2)


@pytest.fixture(scope="module")
def scenario2():
    """
    One covariate is a noisy smooth function of another

    Gives high, symmetric concurvity (the example from `?mgcv::concurvity`).
    """
    rng = np.random.default_rng(SEED)

    def f2(z):
        return 0.2 * z**11 * (10 * (1 - z)) ** 6 + 10 * (10 * z) ** 3 * (1 - z) ** 10

    t = np.sort(rng.uniform(0, 1, N))
    x = f2(t) + rng.normal(0, 3, N)
    eta = 0.5 + 0.8 * np.sin(4 * np.pi * t) + 0.03 * x
    y = rng.poisson(np.exp(eta))
    basis = nmo.basis.BSplineEval(
        15, bounds=(float(t.min()), float(t.max())), label="s(t)"
    ) + nmo.basis.BSplineEval(15, bounds=(float(x.min()), float(x.max())), label="s(x)")
    gam = GAM(basis, use_scipy=True, maxiter=20).fit((t, x), y)
    return gam, (t, x)


@pytest.fixture(scope="module")
def scenario3():
    """Three smooths, one jointly determined by the other two -> asymmetric pairwise structure."""
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(-1, 1, N)
    x2 = rng.uniform(-1, 1, N)
    x3 = 0.8 * x1 + 0.5 * np.sin(2 * x2) + rng.normal(0, 0.3, N)
    eta = x1 + x2**2 + np.sin(2 * x3)
    y = rng.poisson(np.exp(eta))
    basis = (
        nmo.basis.BSplineEval(10, bounds=(-1.0, 1.0), label="s(x1)")
        + nmo.basis.BSplineEval(10, bounds=(-1.0, 1.0), label="s(x2)")
        + nmo.basis.BSplineEval(
            10, bounds=(float(x3.min()), float(x3.max())), label="s(x3)"
        )
    )
    gam = GAM(basis, use_scipy=True, maxiter=20).fit((x1, x2, x3), y)
    return gam, (x1, x2, x3)


# ---------------------------------------------------------------------------
# Math-guaranteed invariants (asserted tightly).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("full", [True, False])
@pytest.mark.parametrize("fixture", ["scenario1", "scenario2", "scenario3"])
def test_gam_measures_in_unit_interval(fixture, full, request):
    gam, xi = request.getfixturevalue(fixture)
    out = gam.concurvity(xi, full=full)
    assert set(out) == {"worst", "observed", "estimate"}
    for key, vals in out.items():
        vals = np.asarray(vals)
        assert np.all(vals >= -1e-9), key
        assert np.all(vals <= 1 + 1e-9), key


@pytest.mark.parametrize("fixture", ["scenario1", "scenario2", "scenario3"])
def test_parametric_block_is_negligible(fixture, request):
    # Centered smooth bases are orthogonal to the intercept column, so the
    # 'para' entry sits at machine epsilon.
    gam, xi = request.getfixturevalue(fixture)
    out = gam.concurvity(xi)
    for key in out:
        assert float(np.asarray(out[key])[0]) < 1e-12, key


@pytest.mark.parametrize("fixture", ["scenario1", "scenario2", "scenario3"])
def test_pairwise_worst_is_symmetric(fixture, request):
    gam, xi = request.getfixturevalue(fixture)
    worst = np.asarray(gam.concurvity(xi, full=False)["worst"])
    np.testing.assert_allclose(worst, worst.T, atol=1e-9)


@pytest.mark.parametrize("fixture", ["scenario1", "scenario2", "scenario3"])
def test_pairwise_diagonal_is_one(fixture, request):
    gam, xi = request.getfixturevalue(fixture)
    out = gam.concurvity(xi, full=False)
    for mat in out.values():
        mat = np.asarray(mat)
        np.testing.assert_array_equal(np.diag(mat), np.ones(mat.shape[0]))


def test_prefit_matches_postfit_for_coefficient_free_measures(scenario2):
    # worst and estimate are properties of the design matrix, so an unfitted
    # model must report the same values as the fitted one on the same inputs.
    fitted, xi = scenario2
    post = fitted.concurvity(xi)

    basis = nmo.basis.BSplineEval(
        15, bounds=fitted.basis.basis1.bounds, label="s(t)"
    ) + nmo.basis.BSplineEval(15, bounds=fitted.basis.basis2.bounds, label="s(x)")
    unfit = GAM(basis, use_scipy=True, maxiter=20)
    with pytest.warns(UserWarning, match="not fitted"):
        pre = unfit.concurvity(xi)

    assert "observed" not in pre
    np.testing.assert_allclose(
        np.asarray(pre["worst"]), np.asarray(post["worst"]), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(pre["estimate"]), np.asarray(post["estimate"]), atol=1e-12
    )


def test_dataframe_labels(scenario3):
    gam, xi = scenario3
    df = gam.concurvity(xi, as_dataframe=True)
    assert list(df.index) == ["para", "s(x1)", "s(x2)", "s(x3)"]
    assert list(df.columns) == ["worst", "observed", "estimate"]
    assert df.index.name == "term"

    pair = gam.concurvity(xi, full=False, as_dataframe=True)
    assert list(pair["estimate"].index) == ["para", "s(x1)", "s(x2)", "s(x3)"]
    assert list(pair["estimate"].columns) == ["para", "s(x1)", "s(x2)", "s(x3)"]
    assert pair["estimate"].index.name == "explainer"
    assert pair["estimate"].columns.name == "focal"


def test_two_dimensional_smooth_term_blocks():
    """
    A multiplicative (tensor) component must translate into a single term
    block that tiles the design matrix together with the 1-D smooth.
    Uses the pre-fit path, so this also checks the DataFrame layout when
    the 'observed' column is unavailable.
    """
    tensor = nmo.basis.BSplineEval(4, bounds=(0.0, 1.0)) * nmo.basis.BSplineEval(
        4, bounds=(0.0, 1.0)
    )
    tensor.label = "te(b,c)"
    basis = nmo.basis.BSplineEval(6, bounds=(0.0, 1.0), label="s(a)") + tensor
    gam = GAM(basis, use_scipy=True, maxiter=20)

    rng = np.random.default_rng(5)
    xi = tuple(rng.uniform(0, 1, 150) for _ in range(3))
    with pytest.warns(UserWarning, match="not fitted"):
        df = gam.concurvity(xi, as_dataframe=True)

    # Column layout: intercept, then 6 - 1 columns for s(a), then
    # 4 * 4 - 1 for the tensor term (one column dropped per component
    # for identifiability).
    blocks = term_blocks_for_gam(gam)
    assert blocks == [
        TermBlock("para", 0, 0),
        TermBlock("s(a)", 1, 5),
        TermBlock("te(b,c)", 6, 20),
    ]
    # The blocks must cover the design matrix exactly.
    X_smooths = gam._compute_uncentered_design_matrix(xi, setup_basis=False)
    assert X_smooths.shape[1] + 1 == sum(b.ncol for b in blocks)

    assert list(df.index) == ["para", "s(a)", "te(b,c)"]
    assert list(df.columns) == ["worst", "estimate"]
    assert (df.values > -1e-9).all() and (df.values < 1 + 1e-9).all()


# ---------------------------------------------------------------------------
# Reported magnitudes (asserted as loose ranges, grounded in measured output).
# ---------------------------------------------------------------------------


def test_scenario1_low_concurvity(scenario1):
    gam, xi = scenario1
    df = gam.concurvity(xi, as_dataframe=True)
    # Independent covariates: everything stays small.
    assert (df.loc[["s(x1)", "s(x2)"], "worst"] < 0.2).all()
    assert (df.loc[["s(x1)", "s(x2)"], "estimate"] < 0.15).all()
    assert (df.loc[["s(x1)", "s(x2)"], "observed"] < 0.15).all()


def test_scenario2_symmetric_high_worst(scenario2):
    gam, xi = scenario2
    df = gam.concurvity(xi, as_dataframe=True)
    w_t, w_x = df.loc["s(t)", "worst"], df.loc["s(x)", "worst"]
    # Two-term case: worst is a property of the pair of subspaces, so equal.
    np.testing.assert_allclose(w_t, w_x, atol=1e-9)
    assert 0.35 < w_t < 0.7
    # The fit puts most signal in s(t), so s(x) is more re-explainable.
    assert df.loc["s(x)", "observed"] > df.loc["s(t)", "observed"]


def test_scenario3_ordering_and_asymmetry(scenario3):
    gam, xi = scenario3
    df = gam.concurvity(xi, as_dataframe=True)
    w1, w2, w3 = (df.loc[f"s(x{i})", "worst"] for i in (1, 2, 3))
    # x3 is jointly recoverable from (x1, x2). x1 is partly re-explainable
    # because x3 carries a copy of it. x2 sits lowest.
    assert w3 > w1 > w2
    assert 0.6 < w3 < 0.95
    assert 0.4 < w2 < 0.85
    # x2's fitted direction is hard to mimic -> small observed.
    assert df.loc["s(x2)", "observed"] < 0.3

    pair = gam.concurvity(xi, full=False, as_dataframe=True)
    pair_est = pair["estimate"]
    # rows = explainer, cols = focal. x1 explains more of x3 than vice versa.
    assert pair_est.loc["s(x1)", "s(x3)"] > pair_est.loc["s(x3)", "s(x1)"]
    # Independent generators: x1 and x2 barely explain each other.
    assert pair_est.loc["s(x1)", "s(x2)"] < 0.1
    assert pair_est.loc["s(x2)", "s(x1)"] < 0.1

    pair_obs = pair["observed"]
    # Observed uses fitted coefficients, so ordered-pair values need not mirror
    # estimate. Here x3 explains more of fitted x1 than the reverse.
    assert pair_obs.loc["s(x3)", "s(x1)"] > pair_obs.loc["s(x1)", "s(x3)"]


# ---------------------------------------------------------------------------
# The real mgcv cross-check, against R-generated golden files.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["scenario1", "scenario2", "scenario3"])
def test_matches_mgcv_golden(scenario):
    """
    Feed the design matrix and coefficients saved from an `mgcv::gam` fit
    through `concurvity()` and assert the three measures match the saved
    `mgcv::concurvity` output (full=TRUE and full=FALSE) to 1e-14.

    The golden files (tests/data/concurvity_mgcv_*.json) come from
    `scripts/generate_concurvity_test_data.py`, which refits the three
    scenario datasets above with mgcv. Running on mgcv's own model matrix
    rather than ours isolates the concurvity linear algebra from
    basis-construction differences between nemos and mgcv.
    """
    path = Path(__file__).parent / "data" / f"concurvity_mgcv_{scenario}.json"
    data = json.loads(path.read_text())
    X = jnp.asarray(data["X"])
    beta = jnp.asarray(data["beta"])
    blocks = [TermBlock(b["label"], b["start"], b["stop"]) for b in data["blocks"]]
    assert [b.label for b in blocks] == data["labels"]

    full = concurvity(X, blocks, beta=beta, full=True)
    for measure in ("worst", "observed", "estimate"):
        np.testing.assert_allclose(
            np.asarray(full[measure]),
            data["full"][measure],
            rtol=0,
            atol=1e-14,
            err_msg=f"full=True, measure={measure}",
        )

    pairwise = concurvity(X, blocks, beta=beta, full=False)
    for measure in ("worst", "observed", "estimate"):
        np.testing.assert_allclose(
            np.asarray(pairwise[measure]),
            data["pairwise"][measure],
            rtol=0,
            atol=1e-14,
            err_msg=f"full=False, measure={measure}",
        )

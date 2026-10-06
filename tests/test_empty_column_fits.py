"""End-to-end tests for fitting with empty columns dropped."""

import warnings

import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest
from conftest import any_columns_dropped, n_columns_dropped

from pgam_jax import GAM
from pgam_jax._penalty_handler import _KroneckerWithNullPenalty
from pgam_jax.concurvity import TermBlock, concurvity, term_blocks_from_infos

jax.config.update("jax_enable_x64", True)


def _bspline(k):
    return nmo.basis.BSplineEval(k, bounds=(0.0, 1.0))


def _fit(basis, xi, y, drop_empty_columns, method="pql_gcv", **kwargs):
    gam = GAM(basis, drop_empty_columns=drop_empty_columns, method=method, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gam.fit(xi, y)
    return gam


def _spread_1d(n=400, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.02, 0.98, n)
    y = jnp.asarray(rng.poisson(np.exp(0.3 + np.sin(5 * x))).astype(float))
    return (x,), y


def _partial_1d(n=500, seed=6, hi=0.45):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.02, hi, n)
    y = jnp.asarray(rng.poisson(np.exp(0.3 + np.sin(5 * x))).astype(float))
    return (x,), y


def _island_2d(n=700, seed=7, radius=0.24, center=(0.32, 0.36)):
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    log_rate = -0.5 + 1.8 * np.exp(
        -0.5 * (((x - center[0]) / 0.11) ** 2 + ((y - center[1]) / 0.11) ** 2)
    )
    counts = jnp.asarray(rng.poisson(np.exp(log_rate)).astype(float))
    return (x, y), counts, log_rate


class TestAFullyActiveDesignIsUnchanged:
    """With nothing to drop, the flag must change nothing at all."""

    @pytest.mark.parametrize("method", ["pql_gcv", "pql_reml"])
    def test_one_dimensional_results_are_identical(self, method):
        xi, y = _spread_1d()
        on = _fit(_bspline(8), xi, y, True, method=method)
        off = _fit(_bspline(8), xi, y, False, method=method)
        assert not any_columns_dropped(on)
        np.testing.assert_array_equal(on.coef_, off.coef_)
        np.testing.assert_array_equal(on.intercept_, off.intercept_)
        assert float(on.edf_) == float(off.edf_)

    def test_two_dimensional_results_are_identical(self):
        rng = np.random.default_rng(8)
        n = 700
        x = rng.uniform(0.02, 0.98, n)
        y = rng.uniform(0.02, 0.98, n)
        counts = jnp.asarray(rng.poisson(1.5, n).astype(float))
        on = _fit(_bspline(6) * _bspline(6), (x, y), counts, True)
        off = _fit(_bspline(6) * _bspline(6), (x, y), counts, False)
        assert not any_columns_dropped(on)
        np.testing.assert_array_equal(on.coef_, off.coef_)

    def test_the_kronecker_fast_path_is_still_taken(self):
        rng = np.random.default_rng(9)
        n = 700
        x = rng.uniform(0.02, 0.98, n)
        y = rng.uniform(0.02, 0.98, n)
        counts = jnp.asarray(rng.poisson(1.5, n).astype(float))
        gam = _fit(_bspline(6) * _bspline(6), (x, y), counts, True)
        ph = gam._build_penalty_handler(
            gam._get_penalty_tree(gam.component_infos_), gam.component_infos_
        )
        assert isinstance(ph._penalties[0], _KroneckerWithNullPenalty)


class TestMaskingShrinksTheModel:
    def test_term_blocks_follow_the_fitted_additive_design(self):
        xi, y = _partial_1d()
        second = np.random.default_rng(17).uniform(0.02, 0.98, len(y))
        basis = _bspline(12) + _bspline(6)
        gam = _fit(basis, (*xi, second), y, True)

        blocks = term_blocks_from_infos(gam.component_infos_)
        assert blocks[0] == TermBlock("para", 0, 0)
        assert blocks[1].ncol < 11
        assert blocks[2].ncol == 5
        assert blocks[2].start == blocks[1].stop + 1
        assert sum(block.ncol for block in blocks) == gam.coef_.size + 1

        for index, values in enumerate((xi[0], second)):
            smooth, lower, upper = gam.smooth_compute((values,), index)
            raw = tuple(basis)[index]._compute_features(values)
            reduced = gam.component_infos_[index].reduce_features(raw)
            coefficient_slice = slice(
                blocks[index + 1].start - 1, blocks[index + 1].stop
            )
            expected = (reduced - reduced.mean(axis=0)) @ gam.coef_[coefficient_slice]
            np.testing.assert_allclose(smooth, expected, atol=1e-12)
            assert np.all(np.isfinite(lower))
            assert np.all(np.isfinite(upper))

    def test_the_coefficient_vector_gets_shorter(self):
        xi, y = _partial_1d()
        on = _fit(_bspline(12), xi, y, True)
        off = _fit(_bspline(12), xi, y, False)
        assert any_columns_dropped(on)
        assert on.coef_.shape[0] < off.coef_.shape[0]
        n_kept = on.component_infos_[0].n_kept
        assert on.coef_.shape[0] == n_kept - 1

    def test_the_covariance_matches_the_coefficients(self):
        xi, y = _partial_1d()
        on = _fit(_bspline(12), xi, y, True)
        assert on.cov_beta_.shape == (on.coef_.shape[0] + 1,) * 2

    def test_predict_returns_one_value_per_input_row(self):
        xi, y = _partial_1d()
        on = _fit(_bspline(12), xi, y, True)
        grid = np.linspace(0.0, 1.0, 77)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert on.predict((grid,)).shape == (77,)

    def test_a_larger_min_obs_drops_more(self):
        # Per-column counts here are 79, 176, 274, 380, 421, 324, 226, 120 and
        # then six zeros, so a threshold of 150 removes two more columns.
        xi, y = _partial_1d()
        few = _fit(_bspline(14), xi, y, 1)
        many = _fit(_bspline(14), xi, y, 150)
        assert n_columns_dropped(few) == 6
        assert n_columns_dropped(many) == 8


class TestMaskingKeepsTheFitOnTheObservedRegion:
    """
    Masking forces the empty coefficients to zero rather than leaving them free.

    That is a slightly more constrained model, so the fits agree closely but
    not exactly. What must hold is that the model still describes the data.
    """

    @pytest.mark.parametrize("method", ["pql_gcv", "pql_reml"])
    def test_predicted_means_agree_closely(self, method):
        xi, y = _partial_1d()
        on = _fit(_bspline(12), xi, y, True, method=method)
        off = _fit(_bspline(12), xi, y, False, method=method)
        mu_on = np.asarray(on.predict(xi))
        mu_off = np.asarray(off.predict(xi))
        np.testing.assert_allclose(mu_on, mu_off, rtol=0.10)

    @pytest.mark.parametrize("method", ["pql_gcv", "pql_reml"])
    def test_the_log_likelihood_does_not_get_worse(self, method):
        xi, y = _partial_1d()
        on = _fit(_bspline(12), xi, y, True, method=method)
        off = _fit(_bspline(12), xi, y, False, method=method)
        assert float(on.score(xi, y)) > float(off.score(xi, y)) - 0.01

    def test_the_effective_degrees_of_freedom_stay_close(self):
        xi, y = _partial_1d()
        on = _fit(_bspline(12), xi, y, True)
        off = _fit(_bspline(12), xi, y, False)
        assert abs(float(on.edf_) - float(off.edf_)) < 1.0


class TestDegenerateInputs:
    def test_an_all_empty_component_is_named_in_the_error(self):
        """
        The guard names the component that failed, not just the fact of failure.

        A covariate that is entirely NaN or entirely out of bounds cannot reach
        this guard, because nemos rejects it earlier with "Invalid input data".
        A threshold that only the wider component fails does reach it.
        """
        rng = np.random.default_rng(10)
        n = 600
        x1 = rng.uniform(0.02, 0.98, n)
        x2 = rng.uniform(0.02, 0.98, n)
        counts = jnp.asarray(rng.poisson(1.0, n).astype(float))
        gam = GAM(_bspline(4) + _bspline(40), drop_empty_columns=100)
        with pytest.raises(ValueError, match="component 1"):
            gam.fit((x1, x2), counts)

    def test_an_impossible_min_obs_raises(self):
        xi, y = _partial_1d()
        gam = GAM(_bspline(8), drop_empty_columns=10**6)
        with pytest.raises(ValueError, match="drops all columns"):
            gam.fit(xi, y)

    @pytest.mark.parametrize("threshold, expected_width", [(400, 1), (375, 2), (26, 6)])
    @pytest.mark.parametrize("method", ["pql_gcv", "pql_reml"])
    def test_nonconstant_survivors_are_retained(
        self, threshold, expected_width, method
    ):
        """
        Threshold masking leaves independent columns alongside the intercept.
        The resulting model must retain those directions and fit successfully.

        Without the guard this reached the penalty eigendecomposition and died
        with "zero-size array to reduction operation max", which says nothing
        about the cause. Counts here are 100, 231, 375, 400, 300, 169, 25 and
        then three zeros, so min_obs=400 leaves exactly one column.
        """
        rng = np.random.default_rng(0)
        x = rng.uniform(0.02, 0.45, 400)
        y = jnp.asarray(rng.poisson(1.0, 400).astype(float))
        gam = _fit(_bspline(10), (x,), y, threshold, method=method)
        info = gam.component_infos_[0]
        raw = info.basis._compute_features(x)[:, info.nonempty_mask]
        assert (
            np.linalg.matrix_rank(np.column_stack((np.ones(len(x)), raw)))
            == expected_width + 1
        )
        assert gam.coef_.shape == (expected_width,)
        assert gam.cov_beta_.shape == (expected_width + 1,) * 2
        assert np.all(np.isfinite(gam.predict((x,))))
        assert all(np.all(np.isfinite(v)) for v in gam.smooth_compute((x,), 0))

    def test_refitting_recomputes_the_mask(self):
        narrow, y_narrow = _partial_1d(hi=0.35)
        wide, y_wide = _spread_1d()
        gam = GAM(_bspline(12), drop_empty_columns=True, method="pql_gcv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gam.fit(narrow, y_narrow)
            dropped_first = n_columns_dropped(gam)
            gam.fit(wide, y_wide)
        assert dropped_first > 0
        assert n_columns_dropped(gam) == 0


_FITTED_ATTRIBUTES = (
    "component_infos_",
    "feature_mean_",
    "coef_",
    "intercept_",
    "regularizer_strength_",
    "cov_beta_",
    "scale_",
    "n_iter_",
    "dof_resid_",
)


def _fail(*_args, **_kwargs):
    raise RuntimeError("injected failure")


def _break_fit(monkeypatch, failing_step):
    """Make one step of ``GAM.fit`` raise."""
    if failing_step == "solver":
        monkeypatch.setattr("pgam_jax.gam.pql_outer_iteration", _fail)
    elif failing_step == "covariance":
        monkeypatch.setattr(GAM, "_compute_cov_beta_from_fit_state", _fail)
    else:
        raise NotImplementedError(f"Unknown failing step: {failing_step!r}")


@pytest.mark.parametrize("failing_step", ["solver", "covariance"])
class TestAFailedFitStoresNothing:
    """
    ``fit`` stores fitted state only after every step succeeded.

    The column layout depends on the data. A layout stored before a later step
    raises would disagree in width with the coefficients of the previous fit.
    """

    def test_a_failed_first_fit_leaves_the_model_unfitted(
        self, monkeypatch, failing_step
    ):
        xi, y = _partial_1d()
        gam = GAM(_bspline(12), drop_empty_columns=True, method="pql_gcv")
        _break_fit(monkeypatch, failing_step)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match="injected failure"):
                gam.fit(xi, y)
        stored = [name for name in _FITTED_ATTRIBUTES if hasattr(gam, name)]
        assert stored == []

    def test_a_failed_refit_keeps_the_previous_fit(self, monkeypatch, failing_step):
        narrow, y_narrow = _partial_1d(hi=0.35)
        wide, y_wide = _spread_1d()
        gam = _fit(_bspline(12), narrow, y_narrow, True)
        assert n_columns_dropped(gam) > 0
        before = {name: getattr(gam, name) for name in _FITTED_ATTRIBUTES}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = np.asarray(gam.predict(narrow))
            smooth = [np.asarray(v) for v in gam.smooth_compute(narrow, 0)]
            concurvity_before = gam.concurvity(narrow)

        # The wide inputs drop no column, so their layout has another width.
        _break_fit(monkeypatch, failing_step)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match="injected failure"):
                gam.fit(wide, y_wide)

        for name, value in before.items():
            assert getattr(gam, name) is value, name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.testing.assert_array_equal(np.asarray(gam.predict(narrow)), prediction)
            for actual, expected in zip(gam.smooth_compute(narrow, 0), smooth):
                np.testing.assert_array_equal(np.asarray(actual), expected)
            concurvity_after = gam.concurvity(narrow)
        for measure, expected in concurvity_before.items():
            np.testing.assert_array_equal(
                np.asarray(concurvity_after[measure]), np.asarray(expected)
            )


@pytest.mark.slow
class TestIslandRecovery:
    """The 2-D case that motivates the feature: data on a disc in a square."""

    def test_the_bump_is_recovered_inside_the_disc(self):
        xi, counts, log_rate = _island_2d()
        gam = _fit(_bspline(9) * _bspline(9), xi, counts, True, method="pql_reml")
        assert n_columns_dropped(gam) > 20

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = np.log(np.asarray(gam.predict(xi)))
        centered_fit = fitted - fitted.mean()
        centered_true = log_rate - log_rate.mean()
        corr = np.corrcoef(centered_fit, centered_true)[0, 1]
        assert corr > 0.9

    def test_masking_beats_no_masking_on_parameter_count(self):
        xi, counts, _ = _island_2d()
        on = _fit(_bspline(9) * _bspline(9), xi, counts, True, method="pql_reml")
        off = _fit(_bspline(9) * _bspline(9), xi, counts, False, method="pql_reml")
        assert on.coef_.shape[0] < 0.7 * off.coef_.shape[0]
        mu_on = np.asarray(on.predict(xi))
        mu_off = np.asarray(off.predict(xi))
        np.testing.assert_allclose(mu_on, mu_off, rtol=0.15)


@pytest.mark.parametrize("drop_empty_columns", [False, True, 26])
@pytest.mark.parametrize("nan_handling", ["zero", "drop"])
@pytest.mark.parametrize("full", [False, True])
def test_prefit_concurvity_respects_column_selection(
    drop_empty_columns, nan_handling, full
):
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0.02, 0.45, 400)
    x2 = rng.uniform(0.02, 0.98, 400)
    x2[:80] = np.nan
    gam = GAM(
        _bspline(10) + _bspline(8),
        drop_empty_columns=drop_empty_columns,
        nan_handling=nan_handling,
    )
    gam.basis.setup_basis(x1, x2)
    raw_blocks = [b._compute_features(x) for b, x in zip(gam.basis, (x1, x2))]
    if nan_handling == "drop":
        valid = ~np.isnan(np.hstack(raw_blocks)).any(axis=1)
        raw_blocks = [b[valid] for b in raw_blocks]
    reduced = []
    terms = [TermBlock("para", 0, 0)]
    start = 1
    for raw in raw_blocks:
        counts = np.sum(np.abs(raw) > 0, axis=0)
        keep = (
            counts >= int(drop_empty_columns)
            if drop_empty_columns
            else np.ones(raw.shape[1], bool)
        )
        block = raw[:, keep]
        if not np.any((counts > 0) & ~keep):
            block = block[:, :-1]
        block = np.nan_to_num(block)
        reduced.append(block)
        terms.append(TermBlock(str(start), start, start + block.shape[1] - 1))
        start += block.shape[1]
    smooths = jnp.asarray(np.hstack(reduced))
    smooths = smooths - smooths.mean(axis=0)
    design = jnp.column_stack([jnp.ones(smooths.shape[0]), smooths])
    expected = concurvity(jnp.asarray(design), terms, full=full)
    with pytest.warns(UserWarning, match="GAM is not fitted"):
        actual = gam.concurvity((x1, x2), full=full)
    for measure in expected:
        np.testing.assert_allclose(actual[measure], expected[measure], atol=1e-12)
    assert not hasattr(gam, "component_infos_")
    assert not hasattr(gam, "feature_mean_")

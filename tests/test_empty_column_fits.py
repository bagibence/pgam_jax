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
from pgam_jax.concurvity import TermBlock, term_blocks_for_gam

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
        ph = gam._build_penalty_handler(gam._get_penalty_tree())
        assert isinstance(ph._penalties[0], _KroneckerWithNullPenalty)


class TestMaskingShrinksTheModel:
    def test_term_blocks_follow_the_fitted_additive_design(self):
        xi, y = _partial_1d()
        second = np.random.default_rng(17).uniform(0.02, 0.98, len(y))
        basis = _bspline(12) + _bspline(6)
        gam = _fit(basis, (*xi, second), y, True)

        blocks = term_blocks_for_gam(gam)
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
        with pytest.raises(ValueError, match="keeps 0 column"):
            gam.fit(xi, y)

    def test_a_single_surviving_column_is_rejected_not_crashed(self):
        """
        One survivor plus the identifiability drop leaves a zero-width block.

        Without the guard this reached the penalty eigendecomposition and died
        with "zero-size array to reduction operation max", which says nothing
        about the cause. Counts here are 100, 231, 375, 400, 300, 169, 25 and
        then three zeros, so min_obs=400 leaves exactly one column.
        """
        rng = np.random.default_rng(0)
        x = rng.uniform(0.02, 0.45, 400)
        y = jnp.asarray(rng.poisson(1.0, 400).astype(float))
        gam = GAM(_bspline(10), drop_empty_columns=400)
        with pytest.raises(ValueError, match="keeps 1 column"):
            gam.fit((x,), y)

    def test_two_surviving_columns_still_fit(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.02, 0.45, 400)
        y = jnp.asarray(rng.poisson(1.0, 400).astype(float))
        gam = _fit(_bspline(10), (x,), y, 375)
        assert gam.component_infos_[0].n_kept == 2
        assert gam.coef_.shape[0] == 1

    def test_a_stale_mask_does_not_leak_into_prefit_concurvity(self):
        """
        Concurvity before a fit must use the full basis, mask or no mask.

        A fit that raises after ``_fit_design_matrix`` leaves
        ``component_infos_`` set while ``coef_`` is absent. The pre-fit branch
        must ignore it, or the design and the term blocks disagree in width.
        """
        from pgam_jax._identifiable_features import _get_basis_component_infos

        xi, _ = _spread_1d()
        gam = GAM(_bspline(10), drop_empty_columns=True)
        stale_block = np.ones((8, 10))
        stale_block[:, :4] = 0.0
        gam.component_infos_ = _get_basis_component_infos(
            gam.basis, drop_conv_basis_col=False, blocks=[stale_block], min_obs=1
        )
        assert not hasattr(gam, "coef_")
        assert sum(block.ncol for block in term_blocks_for_gam(gam)) == 10

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = gam.concurvity(xi)
        assert set(out) == {"worst", "estimate"}

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

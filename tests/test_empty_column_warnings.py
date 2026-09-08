"""Tests for the warnings raised when a fit dropped empty columns."""

import warnings

import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest

from pgam_jax import GAM, EmptyColumnWarning

jax.config.update("jax_enable_x64", True)


def _bspline(k):
    return nmo.basis.BSplineEval(k, bounds=(0.0, 1.0))


@pytest.fixture(scope="module")
def masked_fit():
    """A 1-D fit whose data cover only the lower part of the basis range."""
    rng = np.random.default_rng(11)
    x = rng.uniform(0.02, 0.4, 500)
    counts = rng.poisson(np.exp(0.5 + np.sin(8 * x))).astype(float)
    gam = GAM(_bspline(12), drop_empty_columns=True, method="pql_gcv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gam.fit((x,), jnp.asarray(counts))
    return gam, x, jnp.asarray(counts)


@pytest.fixture(scope="module")
def unmasked_fit():
    rng = np.random.default_rng(12)
    x = rng.uniform(0.02, 0.98, 500)
    counts = rng.poisson(np.exp(0.5 + np.sin(4 * x))).astype(float)
    gam = GAM(_bspline(8), drop_empty_columns=True, method="pql_gcv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gam.fit((x,), jnp.asarray(counts))
    return gam, x, jnp.asarray(counts)


class TestTheFixturesAreWhatTheyClaim:
    def test_the_masked_fit_dropped_columns(self, masked_fit):
        gam, _, _ = masked_fit
        assert gam.nonempty_columns_.any_dropped

    def test_the_unmasked_fit_dropped_nothing(self, unmasked_fit):
        gam, _, _ = unmasked_fit
        assert not gam.nonempty_columns_.any_dropped


class TestPredictWarns:
    def test_predicting_outside_the_observed_region_warns(self, masked_fit):
        gam, _, _ = masked_fit
        grid = np.linspace(0.0, 1.0, 200)
        with pytest.warns(EmptyColumnWarning, match="extrapolates"):
            gam.predict((grid,))

    def test_predicting_inside_the_observed_region_is_quiet(self, masked_fit):
        gam, x, _ = masked_fit
        with warnings.catch_warnings():
            warnings.simplefilter("error", EmptyColumnWarning)
            gam.predict((x,))

    def test_an_unmasked_fit_never_warns(self, unmasked_fit):
        gam, _, _ = unmasked_fit
        grid = np.linspace(0.0, 1.0, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("error", EmptyColumnWarning)
            gam.predict((grid,))

    def test_the_message_counts_the_affected_rows(self, masked_fit):
        gam, _, _ = masked_fit
        grid = np.linspace(0.9, 1.0, 30)
        with pytest.warns(EmptyColumnWarning) as record:
            gam.predict((grid,))
        assert "30 row(s)" in str(record[0].message)


class TestScoreWarns:
    def test_scoring_outside_the_observed_region_warns(self, masked_fit):
        gam, _, _ = masked_fit
        grid = np.linspace(0.0, 1.0, 60)
        y = jnp.asarray(np.ones(60))
        with pytest.warns(EmptyColumnWarning):
            gam.score((grid,), y)


class TestSmoothComputeWarns:
    def test_a_grid_over_the_full_range_warns(self, masked_fit):
        gam, _, _ = masked_fit
        grid = np.linspace(0.0, 1.0, 100)
        with pytest.warns(EmptyColumnWarning, match="smooth_compute"):
            gam.smooth_compute((grid,), 0)

    def test_a_grid_inside_the_observed_region_is_quiet(self, masked_fit):
        gam, _, _ = masked_fit
        grid = np.linspace(0.05, 0.35, 100)
        with warnings.catch_warnings():
            warnings.simplefilter("error", EmptyColumnWarning)
            gam.smooth_compute((grid,), 0)


class TestRestrictedMeaningWarnings:
    def test_significance_test_warns_about_the_observed_region(self, masked_fit):
        gam, _, _ = masked_fit
        with pytest.warns(EmptyColumnWarning, match="observed region"):
            gam.test_smooth_significance(0)

    def test_concurvity_warns_about_the_observed_region(self, masked_fit):
        gam, x, _ = masked_fit
        with pytest.warns(EmptyColumnWarning, match="observed region"):
            gam.concurvity((x,))

    def test_concurvity_is_quiet_without_a_mask(self, unmasked_fit):
        gam, x, _ = unmasked_fit
        with warnings.catch_warnings():
            warnings.simplefilter("error", EmptyColumnWarning)
            gam.concurvity((x,))


class TestTheWarningNamesTheRealCaller:
    def test_score_says_score(self, masked_fit):
        gam, _, _ = masked_fit
        grid = np.linspace(0.0, 1.0, 40)
        with pytest.warns(EmptyColumnWarning) as record:
            gam.score((grid,), jnp.asarray(np.ones(40)))
        assert str(record[0].message).startswith("score:")

    def test_predict_says_predict(self, masked_fit):
        gam, _, _ = masked_fit
        grid = np.linspace(0.0, 1.0, 40)
        with pytest.warns(EmptyColumnWarning) as record:
            gam.predict((grid,))
        assert str(record[0].message).startswith("predict:")

    @pytest.mark.parametrize("method", ["predict", "score", "concurvity"])
    def test_the_warning_points_at_the_caller(self, masked_fit, method):
        """stacklevel must name this test file, not a frame inside GAM."""
        gam, _, _ = masked_fit
        grid = np.linspace(0.0, 1.0, 40)
        calls = {
            "predict": lambda: gam.predict((grid,)),
            "score": lambda: gam.score((grid,), jnp.asarray(np.ones(40))),
            "concurvity": lambda: gam.concurvity((grid,)),
        }
        with pytest.warns(EmptyColumnWarning) as record:
            calls[method]()
        active = [r for r in record if "extrapolates" in str(r.message)]
        assert active, "no extrapolation warning was raised"
        assert active[0].filename == __file__


class TestMinObsFollowsThePublicFlag:
    def test_changing_the_flag_after_construction_takes_effect(self):
        gam = GAM(_bspline(8))
        assert gam.min_obs is None
        gam.drop_empty_columns = True
        assert gam.min_obs == 1
        gam.drop_empty_columns = 7
        assert gam.min_obs == 7

    def test_a_bad_value_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="at least 1"):
            GAM(_bspline(8), drop_empty_columns=0)

    def test_a_flag_set_after_construction_actually_masks(self):
        rng = np.random.default_rng(21)
        x = rng.uniform(0.02, 0.4, 400)
        y = jnp.asarray(rng.poisson(2.0, 400).astype(float))
        gam = GAM(_bspline(12), method="pql_gcv")
        gam.drop_empty_columns = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gam.fit((x,), y)
        assert gam.nonempty_columns_.any_dropped


class TestTheCategoryIsFilterable:
    def test_it_is_a_user_warning_subclass(self):
        assert issubclass(EmptyColumnWarning, UserWarning)

    def test_it_can_be_silenced_on_its_own(self, masked_fit):
        gam, _, _ = masked_fit
        grid = np.linspace(0.0, 1.0, 50)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", category=EmptyColumnWarning)
            gam.predict((grid,))

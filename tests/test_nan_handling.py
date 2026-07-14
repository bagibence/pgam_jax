import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest

from pgam_jax import GAM
from pgam_jax._nan_policy import (
    _rows_with_nan,
    apply_nan_policy_for_fit,
    get_valid_y_rows,
)
from pgam_jax._utils import prepend_ones_for_intercept
from pgam_jax.concurvity import concurvity as low_level_concurvity
from pgam_jax.concurvity import term_blocks_for_gam


def _eval_basis(label=None):
    return nmo.basis.BSplineEval(
        n_basis_funcs=6,
        order=4,
        bounds=(0.0, 1.0),
        label=label,
    )


def _prepare_fitted_state(nan_handling):
    gam = GAM(_eval_basis(), nan_handling=nan_handling)
    x_train = np.linspace(0.0, 1.0, 40)
    y_train = jnp.arange(x_train.size, dtype=float)
    X, _ = gam._fit_design_matrix((x_train,), y_train)
    gam.coef_ = jnp.linspace(0.05, 0.15, X.shape[1])
    gam.intercept_ = jnp.array([0.1])
    gam.scale_ = jnp.array(1.0)
    return gam


def test_constructor_rejects_unknown_nan_handling():
    with pytest.raises(ValueError, match="nan_handling.*unknown"):
        GAM(_eval_basis(), nan_handling="unknown")


def test_default_zero_matches_legacy_zero_fill_then_center_behavior():
    gam = GAM(_eval_basis())
    x = np.linspace(0.0, 1.0, 20)
    x[4] = np.nan
    y = jnp.arange(x.size, dtype=float)

    X, y_aligned = gam._fit_design_matrix((x,), y)
    X_raw = gam._compute_raw_design_matrix((x,), setup_basis=False)
    X_zero = jnp.where(jnp.isnan(X_raw), 0.0, X_raw)

    assert gam.nan_handling == "zero"
    np.testing.assert_allclose(gam.feature_mean_, X_zero.mean(axis=0))
    np.testing.assert_allclose(X, X_zero - X_zero.mean(axis=0))
    np.testing.assert_allclose(y_aligned, y)


@pytest.mark.parametrize("nan_handling", ["zero", "drop"])
def test_gam_fit_design_omits_nan_response_before_centering(nan_handling):
    gam = GAM(_eval_basis(), nan_handling=nan_handling)
    x = np.linspace(0.0, 1.0, 20)
    y = jnp.arange(x.size, dtype=float).at[5].set(jnp.nan)

    X, y_aligned = gam._fit_design_matrix((x,), y)
    X_raw = gam._compute_raw_design_matrix((x,), setup_basis=False)
    expected_X, expected_y, expected_feature_mean = apply_nan_policy_for_fit(
        X_raw, y, nan_handling
    )

    np.testing.assert_allclose(X, expected_X)
    np.testing.assert_allclose(y_aligned, expected_y)
    np.testing.assert_allclose(gam.feature_mean_, expected_feature_mean)
    assert X.shape[0] == x.size - 1


def test_drop_conv_filters_the_built_design_without_changing_history():
    gam = GAM(
        nmo.basis.BSplineConv(n_basis_funcs=5, window_size=4),
        nan_handling="drop",
    )
    x = np.arange(20.0)
    y = jnp.arange(x.size, dtype=float).at[10].set(jnp.nan)

    X, y_aligned = gam._fit_design_matrix((x,), y)
    X_raw = gam._compute_raw_design_matrix((x,), setup_basis=False)
    valid_y_rows = get_valid_y_rows(y, n_rows=X_raw.shape[0])
    kept_rows = valid_y_rows & ~jnp.any(jnp.isnan(X_raw), axis=1)
    expected_uncentered = X_raw[kept_rows]

    np.testing.assert_allclose(
        gam.feature_mean_,
        expected_uncentered.mean(axis=0),
    )
    np.testing.assert_allclose(
        X,
        expected_uncentered - expected_uncentered.mean(axis=0),
    )
    np.testing.assert_allclose(y_aligned, y[kept_rows])


def test_drop_full_fit_uses_effective_sample_for_residual_dof():
    rng = np.random.default_rng(12)
    x = np.linspace(0.0, 1.0, 80)
    y = rng.poisson(np.exp(0.2 + 0.3 * x)).astype(float)
    x[3] = np.nan
    y[9] = np.nan
    gam = GAM(
        nmo.basis.BSplineEval(
            n_basis_funcs=10,
            order=4,
            bounds=(0.0, 1.0),
        ),
        nan_handling="drop",
        maxiter=2,
        use_scipy=True,
    )

    gam.fit((x,), y)

    np.testing.assert_allclose(gam.dof_resid_, 78 - gam.edf_)
    prediction = np.asarray(gam.predict((x,)))
    assert prediction.shape == x.shape
    assert np.isnan(prediction[3])


@pytest.mark.parametrize("nan_handling", ["zero", "drop"])
def test_predict_preserves_rows_and_applies_policy_mask(nan_handling):
    gam = _prepare_fitted_state(nan_handling)
    x = np.linspace(0.1, 0.9, 7)
    x[3] = np.nan

    prediction = np.asarray(gam.predict((x,)))

    assert prediction.shape == x.shape
    if nan_handling == "zero":
        assert np.isfinite(prediction).all()
    else:
        assert np.isnan(prediction[3])
        assert np.isfinite(np.delete(prediction, 3)).all()


@pytest.mark.parametrize("nan_handling", ["zero", "drop"])
def test_score_matches_manual_effective_row_likelihood(nan_handling):
    gam = _prepare_fitted_state(nan_handling)
    x = np.linspace(0.1, 0.9, 9)
    x[2] = np.nan
    y = jnp.arange(1.0, 10.0).at[6].set(jnp.nan)

    X_transformed = gam._transform_design_matrix_with_policy((x,))
    valid_X_rows = ~_rows_with_nan(X_transformed)
    valid_y_rows = get_valid_y_rows(y, n_rows=x.size)
    score_rows = valid_X_rows & valid_y_rows
    mu = gam.observation_model.default_inverse_link_function(
        X_transformed[score_rows] @ gam.coef_ + gam.intercept_
    )
    expected = gam.observation_model.log_likelihood(
        y[score_rows],
        mu,
        scale=gam.scale_,
        aggregate_sample_scores=jnp.sum,
    )

    actual = gam.score((x,), y, aggregate_sample_scores=jnp.sum)
    np.testing.assert_allclose(actual, expected)


def test_drop_postfit_concurvity_matches_manually_filtered_design():
    basis = _eval_basis("s(x1)") + _eval_basis("s(x2)")
    gam = GAM(basis, nan_handling="drop")
    x1_train = np.linspace(0.0, 1.0, 80)
    x2_train = np.linspace(1.0, 0.0, 80) ** 2
    X_train, _ = gam._fit_design_matrix(
        (x1_train, x2_train),
        jnp.ones(x1_train.size),
    )
    gam.coef_ = jnp.linspace(0.05, 0.2, X_train.shape[1])
    gam.intercept_ = jnp.array([0.1])

    x1 = x1_train.copy()
    x2 = x2_train.copy()
    x1[10] = np.nan
    X_transformed = gam._transform_design_matrix_with_policy((x1, x2))
    valid_X_rows = ~_rows_with_nan(X_transformed)
    X = prepend_ones_for_intercept(X_transformed[valid_X_rows])
    beta = jnp.concatenate([gam.intercept_, gam.coef_])
    expected = low_level_concurvity(
        X,
        term_blocks_for_gam(gam),
        beta=beta,
    )

    actual = gam.concurvity((x1, x2))
    assert actual.keys() == expected.keys()
    for measure in actual:
        np.testing.assert_allclose(actual[measure], expected[measure])


def test_all_invalid_drop_behaviors():
    # A convolution window longer than the input yields an entirely NaN raw
    # design without requiring NaN inputs during basis setup.
    unfitted = GAM(
        nmo.basis.BSplineConv(n_basis_funcs=5, window_size=8),
        nan_handling="drop",
    )
    with pytest.raises(ValueError, match="No rows remain"):
        unfitted._fit_design_matrix((np.arange(6.0),), jnp.ones(6))

    gam = GAM(
        nmo.basis.BSplineConv(n_basis_funcs=5, window_size=8),
        nan_handling="drop",
    )
    X_train, _ = gam._fit_design_matrix((np.arange(30.0),), jnp.ones(30))
    gam.coef_ = jnp.linspace(0.05, 0.15, X_train.shape[1])
    gam.intercept_ = jnp.array([0.1])
    gam.scale_ = jnp.array(1.0)

    all_invalid = np.arange(6.0)
    prediction = np.asarray(gam.predict((all_invalid,)))
    assert prediction.shape == all_invalid.shape
    assert np.isnan(prediction).all()

    with pytest.raises(ValueError, match="No valid rows remain for scoring"):
        gam.score((all_invalid,), jnp.ones(all_invalid.size))

    with pytest.raises(ValueError, match="No valid design rows remain"):
        gam.concurvity((all_invalid,))

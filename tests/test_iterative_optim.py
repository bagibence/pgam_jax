import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nemos.observation_models import PoissonObservations

from pgam_jax.iterative_optim import (
    _tree_max_leaf_l2_delta,
    check_pql_convergence,
    pql_outer_iteration,
)
from pgam_jax.penalty_utils import tree_compute_sqrt_penalty

jax.config.update("jax_enable_x64", True)


def test_tree_max_leaf_l2_delta_uses_max_over_leaves():
    # max(L2([3, 4]), L2([12])) == max(5, 12) == 12.
    # Whole-tree L2 would be sqrt(25 + 144) == 13 — guarding against that regression.
    left = (jnp.array([3.0, 4.0]), {"x": jnp.array([12.0])})
    right = (jnp.zeros(2), {"x": jnp.zeros(1)})

    np.testing.assert_allclose(_tree_max_leaf_l2_delta(left, right), 12.0)


def _params(value):
    """Mock (coef, intercept) tuple with both leaves equal to ``value``."""
    return (jnp.asarray(value, dtype=float), jnp.asarray(value, dtype=float))


def test_check_pql_convergence_coef_triggers_only_on_delta_coef():
    # 1e-4 relative change in params (below tol). reg_strength changes by a huge
    # absolute amount but is ignored.
    assert check_pql_convergence(
        "coef",
        iteration=1,
        tol=1e-3,
        old_params=_params(1.0),
        new_params=_params(1.0 + 1e-4),
        old_reg_strength=_params(1.0),
        new_reg_strength=_params(1e6),
    )
    # 1e-2 relative change in params (above tol). Should not converge.
    assert not check_pql_convergence(
        "coef",
        iteration=1,
        tol=1e-3,
        old_params=_params(1.0),
        new_params=_params(1.0 + 1e-2),
        old_reg_strength=_params(1.0),
        new_reg_strength=_params(1.0),
    )


def test_check_pql_convergence_coef_warmup_skips_iteration_zero():
    # Identical params would otherwise trigger convergence (delta = 0); warmup
    # blocks at iteration 0.
    assert not check_pql_convergence(
        "coef",
        iteration=0,
        tol=1e-3,
        old_params=_params(1.0),
        new_params=_params(1.0),
        old_reg_strength=_params(1.0),
        new_reg_strength=_params(1.0),
    )


def test_check_pql_convergence_coef_and_reg_requires_both_below_tol():
    # Both relative deltas below tol: converged.
    assert check_pql_convergence(
        "coef_and_reg",
        iteration=1,
        tol=1e-3,
        old_params=_params(1.0),
        new_params=_params(1.0 + 1e-4),
        old_reg_strength=_params(1.0),
        new_reg_strength=_params(1.0 + 1e-4),
    )
    # delta_params relative ratio too big.
    assert not check_pql_convergence(
        "coef_and_reg",
        iteration=1,
        tol=1e-3,
        old_params=_params(1.0),
        new_params=_params(1.0 + 1e-2),
        old_reg_strength=_params(1.0),
        new_reg_strength=_params(1.0 + 1e-4),
    )
    # delta_reg_strength relative ratio too big.
    assert not check_pql_convergence(
        "coef_and_reg",
        iteration=1,
        tol=1e-3,
        old_params=_params(1.0),
        new_params=_params(1.0 + 1e-4),
        old_reg_strength=_params(1.0),
        new_reg_strength=_params(1.0 + 1e-2),
    )


def test_check_pql_convergence_coef_and_reg_warmup_skips_iteration_zero():
    # Identical params would otherwise trigger convergence; warmup blocks iter 0.
    assert not check_pql_convergence(
        "coef_and_reg",
        iteration=0,
        tol=1e-3,
        old_params=_params(1.0),
        new_params=_params(1.0),
        old_reg_strength=_params(1.0),
        new_reg_strength=_params(1.0),
    )


def test_check_pql_convergence_gcv_matches_legacy_min_iteration():
    # gcv ignores params/reg deltas; it only looks at scores. Pass huge deltas
    # to demonstrate that.
    assert not check_pql_convergence(
        "gcv",
        iteration=3,
        tol=1e-5,
        old_params=_params(0.0),
        new_params=_params(1e6),
        old_reg_strength=_params(0.0),
        new_reg_strength=_params(1e6),
        old_score=1.0,
        new_score=1.0,
    )
    assert check_pql_convergence(
        "gcv",
        iteration=4,
        tol=1e-5,
        old_params=_params(0.0),
        new_params=_params(1e6),
        old_reg_strength=_params(0.0),
        new_reg_strength=_params(1e6),
        old_score=1.0,
        new_score=1.0,
    )


def test_check_pql_convergence_gcv_handles_negative_scores():
    # The relative threshold uses |new_score|, so negative scores converge the
    # same as positive ones. Without the abs the threshold flips sign and
    # convergence never triggers.
    assert check_pql_convergence(
        "gcv",
        iteration=4,
        tol=1e-5,
        old_params=_params(0.0),
        new_params=_params(0.0),
        old_reg_strength=_params(0.0),
        new_reg_strength=_params(0.0),
        old_score=-1.0,
        new_score=-1.0,
    )


def test_check_pql_convergence_gcv_returns_false_when_old_score_missing():
    # First post-warmup iteration: old_inner_score is still None.
    assert not check_pql_convergence(
        "gcv",
        iteration=4,
        tol=1e-5,
        old_params=_params(0.0),
        new_params=_params(0.0),
        old_reg_strength=_params(0.0),
        new_reg_strength=_params(0.0),
        old_score=None,
        new_score=1.0,
    )


def test_check_pql_convergence_invalid_criterion_raises():
    with pytest.raises(ValueError, match="convergence_criterion must be one of"):
        check_pql_convergence(
            "bad_value",
            iteration=1,
            tol=1e-3,
            old_params=_params(0.0),
            new_params=_params(0.0),
            old_reg_strength=_params(0.0),
            new_reg_strength=_params(0.0),
        )

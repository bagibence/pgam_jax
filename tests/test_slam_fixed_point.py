"""
The Appendix B from Wood 2011 transform stops at its fixed point instead of always
running one step per column. This checks that the shortcut changes nothing.

The reference below is the fixed-length loop that ``_run_to_fixed_point``
replaced. Every case must agree with it exactly, not approximately.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from pgam_jax._slam_compute import (
    _make_scan_body,
    transform_slam,
    transform_slam_with_Q,
)


def reference_transform(S_tensor, lams):
    """
    Run the Appendix B step exactly once per column, as the code used to.
    """
    n_pen, n_col = S_tensor.shape[0], S_tensor.shape[1]
    body = _make_scan_body(lams, n_col)
    init = (
        S_tensor,
        S_tensor,
        jnp.eye(n_col),
        jnp.ones(n_pen, dtype=bool),
        jnp.zeros((), dtype=jnp.int32),
    )
    (_, S_i_out, Q_s, _, _), _ = lax.scan(body, init, None, length=n_col)
    return S_i_out, Q_s


def random_psd(rng, n_col, rank):
    factor = rng.normal(size=(n_col, rank))
    return factor @ factor.T


def make_lams(rng, n_pen, kind):
    """
    Build one smoothing-parameter vector of the requested kind.
    """
    if kind == "ordinary":
        return jnp.asarray(np.exp(rng.normal(size=n_pen)))
    elif kind == "extreme_scales":
        return jnp.asarray(np.exp(rng.normal(size=n_pen) * 20))
    elif kind == "tied":
        return jnp.asarray(np.full(n_pen, 3.0))
    elif kind == "some_zero":
        return jnp.asarray(np.exp(rng.normal(size=n_pen)) * (rng.random(n_pen) > 0.5))
    elif kind == "all_zero":
        return jnp.zeros(n_pen)
    else:
        raise NotImplementedError(f"Unknown smoothing-parameter kind {kind!r}.")


KINDS = ("ordinary", "extreme_scales", "tied", "some_zero", "all_zero")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n_pen", (1, 2, 3, 5))
def test_fixed_point_matches_one_step_per_column(kind, n_pen):
    """
    The early stop must reproduce the full-budget loop bit for bit.
    """
    rng = np.random.default_rng(100 * KINDS.index(kind) + n_pen)
    for _ in range(5):
        n_col = int(rng.integers(n_pen + 1, 26))
        blocks = [
            random_psd(rng, n_col, int(rng.integers(1, n_col))) for _ in range(n_pen)
        ]
        S_tensor = jnp.asarray(np.stack(blocks))
        lams = make_lams(rng, n_pen, kind)

        expected_S, expected_Q = reference_transform(S_tensor, lams)
        got_S, got_Q = transform_slam_with_Q(S_tensor, lams)

        np.testing.assert_array_equal(np.asarray(got_S), np.asarray(expected_S))
        np.testing.assert_array_equal(np.asarray(got_Q), np.asarray(expected_Q))


@pytest.mark.parametrize("n_pen, n_col", ((2, 60), (3, 40)))
def test_fixed_point_matches_at_wide_shapes(n_pen, n_col):
    """
    Wide penalties are where the early stop saves the most, so check them too.
    """
    rng = np.random.default_rng(n_pen * 1000 + n_col)
    blocks = [random_psd(rng, n_col, int(rng.integers(1, n_col))) for _ in range(n_pen)]
    S_tensor = jnp.asarray(np.stack(blocks))
    lams = jnp.asarray(np.exp(rng.normal(size=n_pen) * 3))

    expected_S, expected_Q = reference_transform(S_tensor, lams)
    got_S, got_Q = transform_slam_with_Q(S_tensor, lams)

    np.testing.assert_array_equal(np.asarray(got_S), np.asarray(expected_S))
    np.testing.assert_array_equal(np.asarray(got_Q), np.asarray(expected_Q))


def test_transform_slam_takes_log_scale_parameters():
    """
    transform_slam takes rho and must agree with the reference on exp(rho).
    """
    rng = np.random.default_rng(11)
    n_pen, n_col = 3, 20
    blocks = [random_psd(rng, n_col, int(rng.integers(1, n_col))) for _ in range(n_pen)]
    S_tensor = jnp.asarray(np.stack(blocks))
    rho = jnp.asarray(rng.normal(size=n_pen))

    expected_S, _ = reference_transform(S_tensor, jnp.exp(rho))
    got_S = transform_slam(S_tensor, rho)

    np.testing.assert_array_equal(np.asarray(got_S), np.asarray(expected_S))

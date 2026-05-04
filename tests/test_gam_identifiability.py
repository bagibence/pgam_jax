"""Tests for per-leaf identifiability handling in GAM.

The design matrix is built per leaf (``compute_features_identifiable``):
``BSplineConv`` leaves follow the ``GAM`` constructor flag, other leaves drop the last column.
The penalty path must drop columns in the same per-leaf pattern, otherwise
the WLS step inside the PQL outer loop fails with a contracting-dim mismatch.
"""

import nemos as nmo
import numpy as np
import pytest

from pgam_jax import GAM
from pgam_jax._identifiable_features import compute_features_identifiable


@pytest.fixture
def small_inputs():
    rng = np.random.default_rng(0)
    T = 800
    x1 = rng.standard_normal(T)
    x2 = rng.standard_normal(T)
    events = np.zeros(T)
    events[::5] = 1
    y = rng.poisson(0.05, size=T)
    return x1, x2, events, y


def _expected_n_cols(basis, drop_conv_basis_col):
    """Number of columns the design matrix should have, per the per-leaf rule."""
    n = 0
    for b in basis:
        if isinstance(b, nmo.basis._basis_mixin.ConvBasisMixin):
            n += b.n_basis_funcs - int(drop_conv_basis_col)
        else:
            n += b.n_basis_funcs - 1
    return n


@pytest.mark.parametrize(
    "drop_conv_basis_col",
    [True, False],
    ids=["drop-conv", "keep-conv"],
)
@pytest.mark.parametrize(
    "make_basis",
    # build a fresh basis for each case
    [
        lambda: nmo.basis.BSplineEval(n_basis_funcs=10, order=4),
        lambda: nmo.basis.BSplineEval(n_basis_funcs=10, order=4)
        + nmo.basis.BSplineEval(n_basis_funcs=10, order=4),
        lambda: nmo.basis.BSplineConv(n_basis_funcs=10, window_size=51),
        lambda: nmo.basis.BSplineEval(n_basis_funcs=10, order=4)
        + nmo.basis.BSplineEval(n_basis_funcs=10, order=4)
        + nmo.basis.BSplineConv(n_basis_funcs=10, window_size=51),
        lambda: nmo.basis.BSplineConv(n_basis_funcs=10, window_size=51)
        + nmo.basis.BSplineEval(n_basis_funcs=10, order=4),
    ],
    ids=["eval", "eval+eval", "conv", "eval+eval+conv", "conv+eval"],
)
def test_design_matrix_and_fit_shapes_match(
    make_basis,
    small_inputs,
    drop_conv_basis_col,
):
    """
    For each supported basis composition, the design matrix width and the
    fitted coefficient vector must agree, and the outer PQL loop must run.
    """
    x1, x2, events, y = small_inputs
    basis = make_basis()

    inputs = []
    eval_count = 0
    for b in basis:
        if isinstance(b, nmo.basis._basis_mixin.ConvBasisMixin):
            inputs.append(events)
        else:
            inputs.append(x1 if eval_count == 0 else x2)
            eval_count += 1
    inputs = tuple(inputs)

    expected_cols = _expected_n_cols(basis, drop_conv_basis_col)
    X = compute_features_identifiable(
        basis,
        *inputs,
        drop_conv_basis_col=drop_conv_basis_col,
    )
    assert X.shape == (len(y), expected_cols)

    gam = GAM(
        basis,
        use_scipy=True,
        maxiter=3,
        drop_conv_basis_col=drop_conv_basis_col,
    )
    gam.fit(inputs, y)

    assert gam.coef_.shape == (expected_cols,)
    pred = gam.predict(inputs)
    assert pred.shape == (len(y),)


@pytest.mark.parametrize(
    ("drop_conv_basis_col", "expected_conv_cols"),
    [(True, 9), (False, 10)],
    ids=["drop-conv", "keep-conv"],
)
def test_per_leaf_identifiability_is_a_tuple_of_callables(
    drop_conv_basis_col,
    expected_conv_cols,
):
    """
    The per-leaf identifiability containers must be tuples (hashable, so
    they can be used as ``static_argnames`` of jit), with one entry per basis
    component. Conv components follow the constructor flag.
    """
    spatial = nmo.basis.BSplineEval(n_basis_funcs=10, order=4)
    temporal = nmo.basis.BSplineConv(n_basis_funcs=10, window_size=51)
    basis = spatial + temporal
    gam = GAM(
        basis,
        drop_conv_basis_col=drop_conv_basis_col,
    )

    assert isinstance(gam._apply_identifiability_column, tuple)
    assert isinstance(gam._apply_identifiability_square, tuple)
    assert len(gam._apply_identifiability_column) == 2
    assert len(gam._apply_identifiability_square) == 2

    # tuples must be hashable for jit static-arg caching
    hash(gam._apply_identifiability_column)
    hash(gam._apply_identifiability_square)

    arr = np.ones((4, 10))
    sq = np.ones((4, 10, 10))
    # eval leaf drops the last column / row+col
    assert gam._apply_identifiability_column[0](arr).shape == (4, 9)
    assert gam._apply_identifiability_square[0](sq).shape == (4, 9, 9)
    # conv leaf follows the constructor flag
    assert gam._apply_identifiability_column[1](arr).shape == (4, expected_conv_cols)
    assert gam._apply_identifiability_square[1](sq).shape == (
        4,
        expected_conv_cols,
        expected_conv_cols,
    )


def test_predict_reuses_fitted_basis_and_training_centering():
    """Prediction must transform with fitted basis state and training column means."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, size=200)
    y = rng.poisson(0.1, size=x.shape[0])
    basis = nmo.basis.BSplineEval(n_basis_funcs=8, order=4, bounds=(-1.0, 1.0))
    gam = GAM(basis, use_scipy=True, maxiter=2)

    gam.fit((x,), y)
    feature_mean = np.asarray(gam.feature_mean_)
    assert feature_mean.shape == (gam.coef_.shape[0],)

    def fail_setup_basis(*_args, **_kwargs):
        raise AssertionError("predict should not call setup_basis")

    gam.basis.setup_basis = fail_setup_basis
    x_pred = np.linspace(-0.5, 0.5, 50)
    pred = gam.predict((x_pred,))

    assert pred.shape == x_pred.shape
    transformed = gam._transform_design_matrix((x_pred,))
    uncentered = gam._compute_uncentered_design_matrix((x_pred,), setup_basis=False)
    np.testing.assert_allclose(
        np.asarray(transformed),
        np.asarray(uncentered) - feature_mean,
    )


def test_tree_compute_sqrt_penalty_accepts_per_leaf_callables():
    """
    ``tree_compute_sqrt_penalty`` must accept a list/tuple of per-leaf
    identifiability functions and apply each to its corresponding leaf,
    instead of the same function to every leaf.
    """
    import jax.numpy as jnp

    from pgam_jax.penalty_utils import tree_compute_sqrt_penalty

    # two penalty leaves, both 10x10, both with one penalty matrix (m=1)
    eye = jnp.eye(10)[None]
    tree_penalty = [eye, eye]
    reg = [jnp.zeros(1), jnp.zeros(1)]

    # leaf 0: drop last column; leaf 1: keep all columns
    apply_id = (lambda x: x[..., :-1], lambda x: x)
    out = tree_compute_sqrt_penalty(
        tree_penalty, reg, shift_by=0, apply_identifiability=apply_id
    )
    # block-diag of (10 x 9) and (10 x 10) → (20, 19)
    assert out.shape == (20, 19)


def test_compute_penalty_blocks_accepts_per_leaf_callables():
    """
    ``compute_penalty_blocks`` must likewise accept a tuple of per-leaf
    identifiability functions.
    """
    import jax.numpy as jnp

    from pgam_jax.penalty_utils import compute_penalty_blocks

    eye = jnp.eye(10)[None]
    tree_penalty = [eye, eye]

    apply_id = (lambda x: x[..., :-1, :-1], lambda x: x)
    out = compute_penalty_blocks(
        tree_penalty, shift_by=1, apply_identifiability=apply_id
    )
    assert len(out) == 2
    for leaf in out:
        assert leaf.shape == (1, 20, 20)

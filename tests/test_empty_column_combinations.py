"""
Column masking combined with NaN policies, convolution, and post-fit methods.

The expected values come from small reference helpers that work on raw basis
output. They do not use the library layout helpers, ``component_infos_``, or
``feature_mean_``. Those are the outputs under test.
"""

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest
from scipy import stats as sts

from pgam_jax import GAM
from pgam_jax.concurvity import TermBlock, concurvity

jax.config.update("jax_enable_x64", True)

RTOL = 1e-10
ATOL = 1e-12


def _assert_close(actual, expected, context=""):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=context)


N_ROWS = 600
WINDOW = 40
NAN_POLICIES = ("zero", "drop")

# NaN sources sit on disjoint rows, so each exclusion can be told apart.
CASE_INPUT_NAN_ROWS = np.arange(100, 140)
SECOND_INPUT_NAN_ROWS = np.arange(200, 210)
CONV_INPUT_NAN_ROW = 300
RESPONSE_NAN_ROWS = np.arange(450, 455)
# Design rows a convolution invalidates: its padding, then the window after
# the input NaN.
CONV_DESIGN_NAN_ROWS = np.concatenate(
    [np.arange(WINDOW), CONV_INPUT_NAN_ROW + 1 + np.arange(WINDOW)]
)

DISC_CENTER = (0.32, 0.36)
DISC_RADIUS = 0.24

ZERO_ONLY_EVENTS = tuple(range(588, 600))
THRESHOLD_EVENTS = (50, 90, 130, 210, 250, 350, 390, 500)


# ─── Cases ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    """
    One model of the grid: a case smooth plus a second smooth.

    ``n_removed`` holds the literal number of masked columns of the case
    component and of the second smooth. It guards against data drift. The
    counts are the same under both NaN policies for this data.
    """

    name: str
    kind: str
    drop_empty_columns: bool | int
    drop_conv_basis_col: bool
    n_removed: tuple[int, int]
    conv_events: tuple[int, ...] = ()
    conv_signal_column: int | None = None


MASKED_CASES = (
    Case("eval-zero_only", "eval", True, False, (4, 4)),
    Case("eval-threshold", "eval", 30, False, (5, 5)),
    Case("conv-zero_only", "conv", True, False, (3, 4), ZERO_ONLY_EVENTS, 1),
    Case("conv-threshold", "conv", 80, False, (2, 5), THRESHOLD_EVENTS, 3),
    Case("conv-threshold-drop_col", "conv", 80, True, (2, 5), THRESHOLD_EVENTS, 3),
    Case("tensor-zero_only", "tensor", True, False, (13, 4)),
    Case("tensor-threshold", "tensor", 20, False, (16, 5)),
)

UNMASKED_CONTROLS = (
    Case("eval-unmasked", "eval", False, False, (0, 0)),
    Case("conv-unmasked", "conv", False, False, (0, 0), THRESHOLD_EVENTS, 3),
    Case("tensor-unmasked", "tensor", False, False, (0, 0)),
)


def _grid(cases):
    return [
        pytest.param(case, nan_handling, id=f"{case.name}-nan_{nan_handling}")
        for case in cases
        for nan_handling in NAN_POLICIES
    ]


# ─── Deterministic builders ───────────────────────────────────────────────────


def _bspline(k):
    return nmo.basis.BSplineEval(k, bounds=(0.0, 1.0))


def _case_basis(kind):
    if kind == "eval":
        return _bspline(12)
    elif kind == "conv":
        return nmo.basis.BSplineConv(8, window_size=WINDOW)
    elif kind == "tensor":
        return _bspline(7) * _bspline(7)
    else:
        raise NotImplementedError(f"Unknown basis kind: {kind!r}")


def _n_case_inputs(kind):
    if kind == "eval" or kind == "conv":
        return 1
    elif kind == "tensor":
        return 2
    else:
        raise NotImplementedError(f"Unknown basis kind: {kind!r}")


def _make_basis(kind):
    """Return ``case smooth + second smooth``."""
    return _case_basis(kind) + _bspline(12)


def _make_gam(case, nan_handling, **kwargs):
    return GAM(
        _make_basis(case.kind),
        drop_empty_columns=case.drop_empty_columns,
        drop_conv_basis_col=case.drop_conv_basis_col,
        nan_handling=nan_handling,
        **kwargs,
    )


def _conv_signal(events, column):
    """Return one scaled raw convolution column, built on a zero-filled input."""
    filled = np.nan_to_num(events)
    basis = _case_basis("conv")
    basis.setup_basis(filled)
    # The raw output still has NaN padding rows after zero-filling the input.
    values = np.nan_to_num(np.asarray(basis._compute_features(filled))[:, column])
    assert np.all(np.isfinite(values))
    assert values.max() > 0
    return 2 * values / values.max()


@dataclass(frozen=True)
class Data:
    """
    Inputs of one case.

    ``inputs`` and ``y`` are the training data. ``extra_inputs`` is a second
    prediction dataset. ``grids`` holds one smooth grid per component.
    """

    inputs: tuple
    y: np.ndarray
    extra_inputs: tuple
    grids: tuple


def _training_data(case):
    rng = np.random.default_rng(0)
    x = rng.uniform(0.02, 0.45, N_ROWS)
    second = rng.uniform(0.55, 0.98, N_ROWS)
    radius = DISC_RADIUS * np.sqrt(rng.uniform(0, 1, N_ROWS))
    angle = rng.uniform(0, 2 * np.pi, N_ROWS)

    second[SECOND_INPUT_NAN_ROWS] = np.nan
    if case.kind == "eval":
        x[CASE_INPUT_NAN_ROWS] = np.nan
        case_inputs = (x,)
        case_signal = np.sin(12 * np.nan_to_num(x))
    elif case.kind == "conv":
        events = np.zeros(N_ROWS)
        events[list(case.conv_events)] = 1.0
        events[CONV_INPUT_NAN_ROW] = np.nan
        case_inputs = (events,)
        case_signal = _conv_signal(events, case.conv_signal_column)
    elif case.kind == "tensor":
        first = DISC_CENTER[0] + radius * np.cos(angle)
        other = DISC_CENTER[1] + radius * np.sin(angle)
        first[CASE_INPUT_NAN_ROWS] = np.nan
        case_inputs = (first, other)
        distance_sq = (np.nan_to_num(first) - DISC_CENTER[0]) ** 2 + (
            other - DISC_CENTER[1]
        ) ** 2
        case_signal = 1.8 * np.exp(-0.5 * distance_sq / 0.11**2)
    else:
        raise NotImplementedError(f"Unknown basis kind: {case.kind!r}")

    rate = np.exp(-0.2 + case_signal + np.sin(8 * np.nan_to_num(second)))
    assert np.all(np.isfinite(rate))
    y = np.random.default_rng(3).poisson(rate).astype(float)
    y[RESPONSE_NAN_ROWS] = np.nan
    return (*case_inputs, second), y


def _extra_inputs(kind):
    """Return prediction inputs that also cover the range unseen in training."""
    if kind == "eval":
        x = np.linspace(0.0, 1.0, 77)
        x[3] = np.nan
        return (x, np.linspace(0.0, 1.0, 77))
    elif kind == "conv":
        # The impulse is far from both ends, so the full lag response shows.
        events = np.zeros(200)
        events[60] = 1.0
        events[150] = np.nan
        return (events, np.linspace(0.0, 1.0, 200))
    elif kind == "tensor":
        # The whole square, which includes points outside the training disc.
        first, other = (v.ravel() for v in np.meshgrid(*[np.linspace(0, 1, 9)] * 2))
        first = first.copy()
        first[3] = np.nan
        return (first, other, np.linspace(0.0, 1.0, first.size))
    else:
        raise NotImplementedError(f"Unknown basis kind: {kind!r}")


def _smooth_grids(kind):
    second = np.linspace(0.0, 1.0, 50)
    second[20] = np.nan
    if kind == "eval":
        x = np.linspace(0.0, 1.0, 50)
        x[7] = np.nan
        case_grid = (x,)
    elif kind == "conv":
        events = np.zeros(120)
        events[60] = 1.0
        case_grid = (events,)
    elif kind == "tensor":
        axis = np.linspace(0.05, 0.95, 6)
        first, other = (v.ravel() for v in np.meshgrid(axis, axis))
        first = first.copy()
        first[4] = np.nan
        case_grid = (first, other)
    else:
        raise NotImplementedError(f"Unknown basis kind: {kind!r}")
    return (case_grid, (second,))


def _build_data(case):
    inputs, y = _training_data(case)
    return Data(inputs, y, _extra_inputs(case.kind), _smooth_grids(case.kind))


def _literal_design_nan_rows(kind):
    """Rows of the training design that hold a NaN, from the literal data layout."""
    if kind == "eval" or kind == "tensor":
        case_rows = CASE_INPUT_NAN_ROWS
    elif kind == "conv":
        case_rows = CONV_DESIGN_NAN_ROWS
    else:
        raise NotImplementedError(f"Unknown basis kind: {kind!r}")
    return np.concatenate([case_rows, SECOND_INPUT_NAN_ROWS])


def _literal_rows(kind, nan_handling, with_response):
    """Row indices that survive, built from the literal exclusions."""
    excluded = RESPONSE_NAN_ROWS if with_response else np.array([], dtype=int)
    if nan_handling == "zero":
        pass
    elif nan_handling == "drop":
        excluded = np.concatenate([excluded, _literal_design_nan_rows(kind)])
    else:
        raise NotImplementedError(f"Unknown NaN policy: {nan_handling!r}")
    return np.setdiff1d(np.arange(N_ROWS), excluded)


def _expected_fit_row_count(kind, nan_handling):
    if nan_handling == "zero":
        return 595
    elif nan_handling == "drop":
        if kind == "eval" or kind == "tensor":
            return 545
        elif kind == "conv":
            return 505
        else:
            raise NotImplementedError(f"Unknown basis kind: {kind!r}")
    else:
        raise NotImplementedError(f"Unknown NaN policy: {nan_handling!r}")


# ─── Independent references ───────────────────────────────────────────────────


@dataclass(frozen=True)
class RefComponent:
    """Reference layout of one component."""

    mask: np.ndarray
    removed_nonempty: bool
    drops_column: bool
    coef_slice: slice


def _ref_min_obs(drop_empty_columns):
    if drop_empty_columns is False:
        return None
    elif drop_empty_columns is True:
        return 1
    elif isinstance(drop_empty_columns, int):
        return drop_empty_columns
    else:
        raise NotImplementedError(f"Unknown flag: {drop_empty_columns!r}")


def _raw_blocks(basis, kind, inputs):
    """Evaluate the full-width block of each component. The basis must be set up."""
    case_component, second_component = tuple(basis)
    n_case = _n_case_inputs(kind)
    return [
        np.asarray(case_component._compute_features(*inputs[:n_case])),
        np.asarray(second_component._compute_features(*inputs[n_case:])),
    ]


def _ref_rows(raw_blocks, y, nan_handling):
    """Rows used for mask detection and fitting. Pass ``y=None`` for no response."""
    n_rows = raw_blocks[0].shape[0]
    keep = np.ones(n_rows, dtype=bool) if y is None else ~np.isnan(np.asarray(y))
    if nan_handling == "zero":
        return keep
    elif nan_handling == "drop":
        return keep & ~np.isnan(np.hstack(raw_blocks)).any(axis=1)
    else:
        raise NotImplementedError(f"Unknown NaN policy: {nan_handling!r}")


def _ref_layout(raw_blocks, kinds, keep, min_obs, drop_conv_basis_col):
    """Masks, identifiability flags, and cumulative coefficient slices."""
    layout = []
    start = 0
    for raw, kind in zip(raw_blocks, kinds, strict=True):
        if min_obs is None:
            mask = np.ones(raw.shape[1], dtype=bool)
            removed_nonempty = False
        else:
            # A NaN compares as not greater than zero, so it counts as zero.
            counts = np.sum(np.abs(raw[keep]) > 0, axis=0)
            mask = counts >= min_obs
            removed_nonempty = bool(np.any((counts > 0) & ~mask))
        if kind == "conv":
            drops_column = drop_conv_basis_col
        elif kind == "eval" or kind == "tensor":
            drops_column = not removed_nonempty
        else:
            raise NotImplementedError(f"Unknown basis kind: {kind!r}")
        width = int(mask.sum()) - int(drops_column)
        assert width > 0
        layout.append(
            RefComponent(
                mask,
                removed_nonempty,
                drops_column,
                slice(start, start + width),
            )
        )
        start += width
    return layout


def _ref_reduce(raw, component):
    """Apply the mask, then the identifiability drop."""
    block = raw[:, component.mask]
    if component.drops_column:
        block = block[:, :-1]
    return block


def _ref_reduced_design(raw_blocks, layout):
    return np.hstack(
        [_ref_reduce(raw, comp) for raw, comp in zip(raw_blocks, layout, strict=True)]
    )


def _ref_fit_design(raw_blocks, layout, rows, nan_handling):
    """Centered fitting design and its column means on the retained rows."""
    reduced = _ref_reduced_design(raw_blocks, layout)[rows]
    if nan_handling == "zero":
        reduced = np.nan_to_num(reduced)
    elif nan_handling == "drop":
        assert not np.isnan(reduced).any()
    else:
        raise NotImplementedError(f"Unknown NaN policy: {nan_handling!r}")
    mean = reduced.mean(axis=0)
    return reduced - mean, mean


def _ref_transform(raw_blocks, layout, mean, nan_handling):
    """Row-preserving design for prediction, scoring, and post-fit concurvity."""
    reduced = _ref_reduced_design(raw_blocks, layout)
    if nan_handling == "zero":
        return np.nan_to_num(reduced) - mean
    elif nan_handling == "drop":
        return reduced - mean
    else:
        raise NotImplementedError(f"Unknown NaN policy: {nan_handling!r}")


def _ref_term_blocks(layout):
    """Concurvity term blocks. The intercept shifts every coefficient by one."""
    blocks = [TermBlock("para", 0, 0)]
    for index, component in enumerate(layout):
        blocks.append(
            TermBlock(
                str(index),
                component.coef_slice.start + 1,
                component.coef_slice.stop,
            )
        )
    return blocks


def _with_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def _reference(basis, case, data, nan_handling, with_response=True):
    """Return ``(raw_blocks, rows, layout, X, mean)`` for the training data."""
    raw_blocks = _raw_blocks(basis, case.kind, data.inputs)
    rows = _ref_rows(raw_blocks, data.y if with_response else None, nan_handling)
    layout = _ref_layout(
        raw_blocks,
        (case.kind, "eval"),
        rows,
        _ref_min_obs(case.drop_empty_columns),
        case.drop_conv_basis_col,
    )
    X, mean = _ref_fit_design(raw_blocks, layout, rows, nan_handling)
    return raw_blocks, rows, layout, X, mean


def _assert_layout_matches(infos, layout, context):
    assert len(infos) == len(layout), context
    for index, (info, expected) in enumerate(zip(infos, layout)):
        where = f"{context}, component {index}"
        np.testing.assert_array_equal(info.nonempty_mask, expected.mask, err_msg=where)
        assert info.drops_identifiability_column is expected.drops_column, where
        assert info.identifiable_feature_slice == expected.coef_slice, where


_RECORDER_RESULT = object()


@contextmanager
def _recorded_concurvity_calls():
    """
    Replace the low-level concurvity call of ``GAM.concurvity`` with a recorder.

    Masking changes only the arguments of that call, so the tests compare the
    arguments. The measures are not a stable target. Several designs here are
    almost rank deficient, and a last-bit change of the column means then
    moves the measures by far more than the tolerance. The recorder accepts
    only the calls that the real function accepts.
    """
    calls = []
    signature = inspect.signature(concurvity)

    def record(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(bound.arguments)
        return _RECORDER_RESULT

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("pgam_jax.gam._concurvity", record)
        yield calls


def _assert_concurvity_call(call, design, layout, beta, full, context):
    """Compare one recorded call with the reference design, blocks, and beta."""
    expected_X = _with_intercept(design)
    assert call["X"].shape == expected_X.shape, context
    _assert_close(call["X"], expected_X, context)
    # The labels come from the basis and are not under test.
    assert [(block.start, block.stop) for block in call["term_blocks"]] == [
        (block.start, block.stop) for block in _ref_term_blocks(layout)
    ], context
    if beta is None:
        assert call["beta"] is None, context
    else:
        assert call["beta"].shape == beta.shape, context
        _assert_close(call["beta"], beta, context)
    assert call["full"] is full, context


# ─── Fast tests without solver fits ───────────────────────────────────────────


@pytest.mark.parametrize("case, nan_handling", _grid(MASKED_CASES))
def test_fit_preparation_matches_the_reference(case, nan_handling):
    data = _build_data(case)
    gam = _make_gam(case, nan_handling)
    X, y_aligned, infos, feature_mean = gam._fit_design_matrix(
        data.inputs, jnp.asarray(data.y)
    )

    reference_basis = _make_basis(case.kind)
    reference_basis.setup_basis(*data.inputs)
    raw_blocks, rows, layout, _, _ = _reference(
        reference_basis, case, data, nan_handling
    )

    # Rows: the reference selects exactly the literal survivors.
    literal_rows = _literal_rows(case.kind, nan_handling, with_response=True)
    assert literal_rows.size == _expected_fit_row_count(case.kind, nan_handling)
    np.testing.assert_array_equal(np.flatnonzero(rows), literal_rows)

    # Layout.
    _assert_layout_matches(infos, layout, case.name)
    removed = tuple(int((~component.mask).sum()) for component in layout)
    assert removed == case.n_removed
    assert removed[1] > 0
    if case.drop_empty_columns is True:
        assert removed[0] >= 1
        assert not any(component.removed_nonempty for component in layout)
    else:
        # The data must remove an observed column of the case component.
        assert layout[0].removed_nonempty

    # Design, response, and means, taken from the raw data on the literal rows.
    expected_X, expected_mean = _ref_fit_design(
        raw_blocks, layout, literal_rows, nan_handling
    )
    assert X.shape == expected_X.shape
    _assert_close(X, expected_X)
    _assert_close(feature_mean, expected_mean)
    np.testing.assert_array_equal(y_aligned, data.y[literal_rows])

    assert not hasattr(gam, "component_infos_")
    assert not hasattr(gam, "feature_mean_")


# Knots of ``_bspline(8)`` sit at multiples of 0.2. Column 6 is nonzero on
# (0.6, 1) and column 7 on (0.8, 1).
_BULK = np.linspace(0.03, 0.57, 60)
_DECISIVE = np.array([0.63, 0.67])
_SUPPORT = np.array([0.71, 0.75])
_TAIL = np.array([0.85, 0.93])
_CROSSING_X = np.concatenate([_BULK, _DECISIVE, _SUPPORT, _TAIL])
_DECISIVE_ROWS = np.arange(_BULK.size, _BULK.size + _DECISIVE.size)
_CROSSING_MIN_OBS = 5


def _crossing_counts(raw, rows):
    return np.sum(np.abs(raw[rows]) > 0, axis=0)


@pytest.mark.parametrize("nan_handling", NAN_POLICIES)
def test_response_nan_rows_do_not_count_toward_min_obs(nan_handling):
    """
    A column falls below the threshold only because of response-NaN rows.

    Column 6 has six observations in total and four on rows with a response.
    Column 7 has two, so an observed column is removed either way and no
    identifiability column is dropped. Counting the response-NaN rows would
    keep column 6 and widen the design.
    """
    x = _CROSSING_X
    y = np.random.default_rng(1).poisson(1.0, x.size).astype(float)
    y[_DECISIVE_ROWS] = np.nan
    rows = ~np.isnan(y)

    reference_basis = _bspline(8)
    reference_basis.setup_basis(x)
    raw = np.asarray(reference_basis._compute_features(x))
    assert _crossing_counts(raw, rows)[6] == _CROSSING_MIN_OBS - 1
    assert _crossing_counts(raw, slice(None))[6] >= _CROSSING_MIN_OBS
    assert 0 < _crossing_counts(raw, rows)[7] < _CROSSING_MIN_OBS

    gam = GAM(
        _bspline(8), drop_empty_columns=_CROSSING_MIN_OBS, nan_handling=nan_handling
    )
    X, y_aligned, infos, feature_mean = gam._fit_design_matrix((x,), jnp.asarray(y))

    expected_mask = np.array([True] * 6 + [False] * 2)
    np.testing.assert_array_equal(infos[0].nonempty_mask, expected_mask)
    assert infos[0].drops_identifiability_column is False
    assert infos[0].identifiable_feature_slice == slice(0, 6)
    expected = raw[rows][:, :6]
    _assert_close(feature_mean, expected.mean(axis=0))
    _assert_close(X, expected - expected.mean(axis=0))
    np.testing.assert_array_equal(y_aligned, y[rows])


@pytest.mark.parametrize("nan_handling", NAN_POLICIES)
def test_nan_rows_of_another_component_count_only_under_zero(nan_handling):
    """
    NaNs in the second input decide a column of the first component.

    The rows that are NaN in the second input give column 6 of the first
    component its decisive observations. ``"drop"`` excludes those rows, so
    the column falls below the threshold. ``"zero"`` keeps the rows and the
    column. Every response is valid.
    """
    x = _CROSSING_X
    second = np.linspace(0.02, 0.98, x.size)
    second[_DECISIVE_ROWS] = np.nan
    y = np.random.default_rng(1).poisson(1.0, x.size).astype(float)

    reference_basis = _bspline(8) + _bspline(6)
    reference_basis.setup_basis(x, second)
    raw, raw_second = (
        np.asarray(component._compute_features(values))
        for component, values in zip(reference_basis, (x, second))
    )
    valid = ~np.isnan(second)
    assert _crossing_counts(raw, valid)[6] == _CROSSING_MIN_OBS - 1
    assert _crossing_counts(raw, slice(None))[6] >= _CROSSING_MIN_OBS
    assert 0 < _crossing_counts(raw, slice(None))[7] < _CROSSING_MIN_OBS
    # The second component stays unmasked under both policies.
    assert _crossing_counts(raw_second, valid).min() >= _CROSSING_MIN_OBS

    if nan_handling == "zero":
        rows = np.ones(x.size, dtype=bool)
        n_kept = 7
    elif nan_handling == "drop":
        rows = valid
        n_kept = 6
    else:
        raise NotImplementedError(f"Unknown NaN policy: {nan_handling!r}")

    gam = GAM(
        _bspline(8) + _bspline(6),
        drop_empty_columns=_CROSSING_MIN_OBS,
        nan_handling=nan_handling,
    )
    X, y_aligned, infos, feature_mean = gam._fit_design_matrix(
        (x, second), jnp.asarray(y)
    )

    expected_mask = np.arange(8) < n_kept
    np.testing.assert_array_equal(infos[0].nonempty_mask, expected_mask)
    assert infos[0].drops_identifiability_column is False
    assert infos[0].identifiable_feature_slice == slice(0, n_kept)
    assert infos[1].nonempty_mask.all()
    assert infos[1].drops_identifiability_column is True
    assert infos[1].identifiable_feature_slice == slice(n_kept, n_kept + 5)
    expected = np.nan_to_num(np.hstack([raw[:, :n_kept], raw_second[:, :5]])[rows])
    _assert_close(feature_mean, expected.mean(axis=0))
    _assert_close(X, expected - expected.mean(axis=0))
    np.testing.assert_array_equal(y_aligned, y[rows])


@pytest.mark.parametrize("case, nan_handling", _grid(MASKED_CASES + UNMASKED_CONTROLS))
def test_prefit_concurvity_respects_column_selection(case, nan_handling):
    data = _build_data(case)

    reference_basis = _make_basis(case.kind)
    reference_basis.setup_basis(*data.inputs)
    _, rows, layout, X, _ = _reference(
        reference_basis, case, data, nan_handling, with_response=False
    )
    # Concurvity takes no response, so only the design rule removes rows.
    np.testing.assert_array_equal(
        np.flatnonzero(rows),
        _literal_rows(case.kind, nan_handling, with_response=False),
    )

    gam = _make_gam(case, nan_handling)
    with _recorded_concurvity_calls() as calls:
        with pytest.warns(UserWarning, match="GAM is not fitted"):
            result = gam.concurvity(data.inputs)

    assert result is _RECORDER_RESULT
    (call,) = calls
    _assert_concurvity_call(call, X, layout, beta=None, full=True, context=case.name)
    assert not hasattr(gam, "component_infos_")
    assert not hasattr(gam, "feature_mean_")


@pytest.mark.parametrize("full", [False, True])
def test_prefit_concurvity_measures_match_the_reference(full):
    """
    Run the real concurvity math through ``GAM.concurvity`` for one case.

    The other concurvity checks in this file record the call and skip the
    math. This is also the only check of ``full=False`` before a fit.

    This design is well conditioned. Its smallest singular value was 1.6e-2
    of the largest, and a last-bit change of the column means moved the
    measures by less than 1e-4 of the tolerance.
    """
    case = next(case for case in MASKED_CASES if case.name == "eval-threshold")
    nan_handling = "zero"
    data = _build_data(case)

    reference_basis = _make_basis(case.kind)
    reference_basis.setup_basis(*data.inputs)
    _, _, layout, X, _ = _reference(
        reference_basis, case, data, nan_handling, with_response=False
    )
    expected = concurvity(
        jnp.asarray(_with_intercept(X)), _ref_term_blocks(layout), full=full
    )

    gam = _make_gam(case, nan_handling)
    with pytest.warns(UserWarning, match="GAM is not fitted"):
        actual = gam.concurvity(data.inputs, full=full)

    assert set(actual) == set(expected) == {"worst", "estimate"}
    for measure in ("worst", "estimate"):
        assert np.all(np.isfinite(expected[measure])), measure
        assert actual[measure].shape == expected[measure].shape, measure
        _assert_close(actual[measure], expected[measure], measure)


# ─── Fitted integration grid ──────────────────────────────────────────────────


def _assert_fitted_layout(gam, layout, mean, context):
    _assert_layout_matches(gam.component_infos_, layout, context)
    _assert_close(gam.feature_mean_, mean, context)
    width = layout[-1].coef_slice.stop
    assert gam.coef_.shape == (width,), context
    assert gam.cov_beta_.shape == (width + 1, width + 1), context
    for index, component in enumerate(layout):
        # A near-zero block would hide a wrong coefficient slice downstream.
        largest = float(np.max(np.abs(np.asarray(gam.coef_)[component.coef_slice])))
        assert largest > 1e-2, f"{context}, component {index}"

    # The per-coefficient values are shared with the library. Only the slice
    # is under test. Entry 0 belongs to the intercept.
    edf_by_coef = np.asarray(gam._edf_by_coef)
    assert edf_by_coef.shape == (width + 1,), context
    expected_edf = [edf_by_coef[1:][comp.coef_slice].sum() for comp in layout]
    for index, expected in enumerate(expected_edf):
        _assert_close(
            gam.edf_of_component(index), expected, f"{context}, component {index}"
        )
    _assert_close(gam.edf_by_component_, expected_edf, context)


def _assert_prediction(gam, case, layout, mean, inputs, context):
    raw_blocks = _raw_blocks(gam.basis, case.kind, inputs)
    design = _ref_transform(raw_blocks, layout, mean, gam.nan_handling)
    inverse_link = gam.observation_model.default_inverse_link_function
    expected = np.asarray(
        inverse_link(design @ np.asarray(gam.coef_) + np.asarray(gam.intercept_))
    )
    actual = np.asarray(gam.predict(inputs))

    assert actual.shape == (len(inputs[0]),), context
    raw_nan_rows = np.isnan(np.hstack(raw_blocks)).any(axis=1)
    assert raw_nan_rows.any(), context
    if gam.nan_handling == "zero":
        assert np.all(np.isfinite(actual)), context
    elif gam.nan_handling == "drop":
        np.testing.assert_array_equal(np.isnan(actual), raw_nan_rows, err_msg=context)
        assert np.all(np.isfinite(actual[~raw_nan_rows])), context
    else:
        raise NotImplementedError(f"Unknown NaN policy: {gam.nan_handling!r}")
    _assert_close(actual, expected, context)


def _assert_score(gam, case, data, layout, mean, context):
    raw_blocks = _raw_blocks(gam.basis, case.kind, data.inputs)
    design = _ref_transform(raw_blocks, layout, mean, gam.nan_handling)
    rows = _literal_rows(case.kind, gam.nan_handling, with_response=True)
    # Under "zero" the design-NaN rows stay. Under "drop" they do not.
    assert np.isnan(design).any() == (gam.nan_handling == "drop"), context
    assert not np.isnan(design[rows]).any(), context
    rate = np.exp(design[rows] @ np.asarray(gam.coef_) + np.asarray(gam.intercept_))
    expected = sts.poisson.logpmf(data.y[rows], rate).mean()
    actual = gam.score(data.inputs, jnp.asarray(data.y))
    _assert_close(actual, expected, context)


def _assert_smooth_compute(gam, data, layout, context):
    z = sts.norm().ppf(0.975)
    coef = np.asarray(gam.coef_)
    cov_beta = np.asarray(gam.cov_beta_)
    for index, (component, grid) in enumerate(zip(layout, data.grids, strict=True)):
        raw = np.asarray(tuple(gam.basis)[index]._compute_features(*grid))
        reduced = _ref_reduce(raw, component)
        invalid = np.isnan(np.asarray(grid)).any(axis=0) | np.isnan(reduced).any(axis=1)
        assert invalid.any() and not invalid.all(), context
        smooth_design = np.nan_to_num(reduced) - reduced[~invalid].mean(axis=0)
        smooth_design[invalid] = 0.0
        expected_smooth = smooth_design @ coef[component.coef_slice]
        coef_index = (
            np.arange(component.coef_slice.start, component.coef_slice.stop) + 1
        )

        for se_with_mean in (True, False):
            where = f"{context}, component {index}, se_with_mean={se_with_mean}"
            if se_with_mean:
                se_design = _with_intercept(smooth_design)
                cov_index = np.concatenate([[0], coef_index])
                invalid_half_width = z * np.sqrt(cov_beta[0, 0])
            else:
                se_design = smooth_design
                cov_index = coef_index
                invalid_half_width = 0.0
            cov = cov_beta[np.ix_(cov_index, cov_index)]
            variance = np.diag(se_design @ cov @ se_design.T)
            half_width = z * np.sqrt(np.clip(variance, 0.0, None))

            smooth, lower, upper = (
                np.asarray(values)
                for values in gam.smooth_compute(grid, index, se_with_mean=se_with_mean)
            )
            _assert_close(smooth, expected_smooth, where)
            _assert_close(lower, expected_smooth - half_width, where)
            _assert_close(upper, expected_smooth + half_width, where)
            # Invalid rows: zero smooth, and a band from the intercept alone.
            np.testing.assert_array_equal(smooth[invalid], 0.0, err_msg=where)
            _assert_close(upper[invalid], invalid_half_width, where)
            _assert_close(lower[invalid], -invalid_half_width, where)


def _assert_postfit_concurvity(gam, case, data, layout, mean, context):
    raw_blocks = _raw_blocks(gam.basis, case.kind, data.inputs)
    design = _ref_transform(raw_blocks, layout, mean, gam.nan_handling)
    valid = ~np.isnan(design).any(axis=1)
    # No response is involved, so training response-NaN rows stay.
    np.testing.assert_array_equal(
        np.flatnonzero(valid),
        _literal_rows(case.kind, gam.nan_handling, with_response=False),
        err_msg=context,
    )
    beta = np.concatenate([np.atleast_1d(gam.intercept_), gam.coef_])
    for full in (True, False):
        with _recorded_concurvity_calls() as calls:
            result = gam.concurvity(data.inputs, full=full)
        assert result is _RECORDER_RESULT, context
        (call,) = calls
        _assert_concurvity_call(
            call, design[valid], layout, beta, full, f"{context}, full={full}"
        )
    # The real call must still accept the fitted layout.
    measures = gam.concurvity(data.inputs)
    assert set(measures) == {"worst", "estimate", "observed"}, context


def _assert_significance_smoke(gam, layout, context):
    """
    Check only that a p-value comes out. This does not validate indexing.

    ``test_significance_separates_null_and_signal_smooths`` reacts to a wrong
    coefficient slice.
    """
    for index in range(len(layout)):
        with pytest.warns(UserWarning, match="approximate"):
            p_value = float(gam.test_smooth_significance(index))
        assert np.isfinite(p_value), f"{context}, component {index}"
        assert 0.0 <= p_value <= 1.0, f"{context}, component {index}"


@pytest.mark.slow
@pytest.mark.parametrize("case, nan_handling", _grid(MASKED_CASES))
def test_masking_combinations(case, nan_handling):
    """Fit once per case and policy, then check every method on that fit."""
    context = f"{case.name}, nan_handling={nan_handling}"
    data = _build_data(case)
    gam = _make_gam(case, nan_handling, method="pql_reml")
    gam.fit(data.inputs, jnp.asarray(data.y))

    # The basis state comes from the fit. It is not set up again below.
    _, _, layout, _, mean = _reference(gam.basis, case, data, nan_handling)

    _assert_fitted_layout(gam, layout, mean, context)

    _assert_prediction(gam, case, layout, mean, data.inputs, f"{context}, training")
    extra_raw = _raw_blocks(gam.basis, case.kind, data.extra_inputs)[0]
    # The extra inputs must reach a column that training removed.
    assert np.any(np.abs(np.nan_to_num(extra_raw[:, ~layout[0].mask])) > 0), context
    _assert_prediction(gam, case, layout, mean, data.extra_inputs, f"{context}, extra")

    _assert_score(gam, case, data, layout, mean, context)
    _assert_smooth_compute(gam, data, layout, context)
    _assert_postfit_concurvity(gam, case, data, layout, mean, context)
    _assert_significance_smoke(gam, layout, context)


def _null_and_signal_data(null_index):
    """Return two partly covered inputs. Only the one that is not null drives ``y``."""
    rng = np.random.default_rng(0)
    inputs = (rng.uniform(0.02, 0.45, N_ROWS), rng.uniform(0.55, 0.98, N_ROWS))
    # The baseline log rate of 1 keeps the intercept far from zero.
    log_rate = 1.0 + np.sin(12 * inputs[1 - null_index])
    y = np.random.default_rng(3).poisson(np.exp(log_rate)).astype(float)
    return inputs, y


@pytest.mark.slow
@pytest.mark.parametrize("null_index", [0, 1])
def test_significance_separates_null_and_signal_smooths(null_index):
    """
    One smooth has no effect on the response and the other has a strong one.

    A wrong coefficient slice gives the null smooth a significant coefficient.
    Without the intercept offset, component 0 takes the intercept and
    component 1 takes the last coefficient of component 0.
    """
    inputs, y = _null_and_signal_data(null_index)
    gam = GAM(_bspline(12) + _bspline(12), drop_empty_columns=True, method="pql_reml")
    gam.fit(inputs, jnp.asarray(y))
    # Each component loses four empty columns and its identifiability column.
    assert gam.coef_.shape == (14,)
    assert abs(float(gam.intercept_)) > 0.5

    p_values = []
    for index in range(2):
        with pytest.warns(UserWarning, match="approximate"):
            p_values.append(float(gam.test_smooth_significance(index)))
    assert p_values[null_index] > 0.01
    assert p_values[1 - null_index] < 1e-6

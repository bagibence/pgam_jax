"""Tests for empty-column detection on a single basis component."""

import nemos as nmo
import numpy as np
import pytest

from pgam_jax._identifiable_features import (
    BasisComponentInfo,
    _get_basis_component_infos,
    resolve_min_obs,
)


def _eval_basis(k):
    """An evaluation basis. It always drops a column for identifiability."""
    return nmo.basis.BSplineEval(k, bounds=(0.0, 1.0))


def _conv_basis(k):
    """A convolutional basis. It drops a column only when asked to."""
    return nmo.basis.BSplineConv(k, window_size=2 * k)


def _from_block(basis, block, min_obs=1, *, index=0, drop_conv_basis_col=False):
    """Build one component record, with the layout defaults the tests do not vary."""
    return BasisComponentInfo.from_block(
        basis,
        index=index,
        input_start=0,
        feature_start=0,
        drop_conv_basis_col=drop_conv_basis_col,
        block=block,
        min_obs=min_obs,
    )


class TestResolveMinObs:
    def test_false_disables(self):
        assert resolve_min_obs(False) is None

    def test_true_means_one(self):
        assert resolve_min_obs(True) == 1

    def test_int_sets_the_threshold(self):
        assert resolve_min_obs(5) == 5

    def test_zero_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            resolve_min_obs(0)

    def test_negative_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            resolve_min_obs(-3)

    def test_non_integer_is_rejected(self):
        with pytest.raises(TypeError, match="bool or int"):
            resolve_min_obs(1.5)


class TestDetection:
    def test_all_columns_nonempty(self):
        info = _from_block(_eval_basis(4), np.ones((3, 4)))
        np.testing.assert_array_equal(info.nonempty_mask, [True] * 4)
        assert info.is_masked is False
        assert info.n_kept == 4
        assert info.n_dropped == 0

    def test_an_all_zero_column_is_empty(self):
        block = np.ones((3, 4))
        block[:, 1] = 0.0
        info = _from_block(_eval_basis(4), block)
        np.testing.assert_array_equal(info.nonempty_mask, [True, False, True, True])
        assert info.is_masked is True
        assert info.n_dropped == 1

    def test_min_obs_above_one_drops_sparse_columns(self):
        block = np.ones((3, 4))
        block[1:, 0] = 0.0  # one observation in the first column, three elsewhere
        info = _from_block(_eval_basis(4), block, min_obs=2)
        np.testing.assert_array_equal(info.nonempty_mask, [False, True, True, True])

    def test_nan_counts_as_no_observation(self):
        block = np.ones((3, 4))
        block[:, 2] = np.nan
        info = _from_block(_eval_basis(4), block)
        np.testing.assert_array_equal(info.nonempty_mask, [True, True, False, True])

    def test_negative_values_count_as_observations(self):
        block = np.full((3, 4), -2.0)
        info = _from_block(_eval_basis(4), block)
        np.testing.assert_array_equal(info.nonempty_mask, [True] * 4)

    def test_the_mask_is_read_only(self):
        info = _from_block(_eval_basis(4), np.ones((3, 4)))
        with pytest.raises(ValueError, match="read-only"):
            info.nonempty_mask[0] = False

    def test_the_block_is_not_kept_by_reference(self):
        block = np.ones((3, 4))
        block[:, 1] = 0.0
        info = _from_block(_eval_basis(4), block)
        block[:] = 1.0
        np.testing.assert_array_equal(info.nonempty_mask, [True, False, True, True])


class TestDetectionIsOff:
    def test_no_min_obs_keeps_every_column(self):
        info = _from_block(_eval_basis(5), None, min_obs=None)
        np.testing.assert_array_equal(info.nonempty_mask, [True] * 5)
        assert info.n_dropped == 0

    def test_a_missing_block_is_rejected(self):
        with pytest.raises(ValueError, match="requires a training feature block"):
            _from_block(_eval_basis(4), None, min_obs=1)

    def test_a_one_dimensional_block_is_rejected(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            _from_block(_eval_basis(4), np.ones(4))

    def test_a_block_of_the_wrong_width_is_rejected(self):
        with pytest.raises(ValueError, match="with 4 columns"):
            _from_block(_eval_basis(4), np.ones((3, 5)))


class TestTheSurvivorGuard:
    def test_a_fully_empty_component_raises_and_names_its_index(self):
        with pytest.raises(ValueError, match="component 1 keeps 0 column"):
            _from_block(_eval_basis(4), np.zeros((3, 4)), index=1)

    def test_one_survivor_is_rejected_when_a_column_is_dropped_after(self):
        """A single survivor plus the identifiability drop leaves no column."""
        block = np.zeros((3, 4))
        block[:, 0] = 1.0
        with pytest.raises(ValueError, match="keeps 1 column"):
            _from_block(_eval_basis(4), block)

    def test_the_message_explains_the_identifiability_drop(self):
        block = np.zeros((3, 4))
        block[:, 0] = 1.0
        with pytest.raises(ValueError, match="identifiability"):
            _from_block(_eval_basis(4), block)

    def test_one_survivor_is_fine_when_no_column_is_dropped_after(self):
        block = np.zeros((3, 4))
        block[:, 0] = 1.0
        info = _from_block(_conv_basis(4), block, drop_conv_basis_col=False)
        np.testing.assert_array_equal(info.nonempty_mask, [True, False, False, False])
        assert info.identifiable_feature_slice == slice(0, 1)

    def test_two_survivors_are_enough_when_a_column_is_dropped_after(self):
        block = np.zeros((3, 4))
        block[:, :2] = 1.0
        info = _from_block(_eval_basis(4), block)
        np.testing.assert_array_equal(info.nonempty_mask, [True, True, False, False])
        assert info.identifiable_feature_slice == slice(0, 1)


class TestDetectionOverComponents:
    def test_one_mask_per_component(self):
        basis = _eval_basis(4) + _eval_basis(5)
        blocks = [np.ones((4, 4)), np.ones((4, 5))]
        blocks[1][:, 1] = 0.0
        infos = _get_basis_component_infos(
            basis, drop_conv_basis_col=False, blocks=blocks, min_obs=1
        )
        assert len(infos) == 2
        np.testing.assert_array_equal(infos[0].nonempty_mask, [True] * 4)
        np.testing.assert_array_equal(
            infos[1].nonempty_mask, [True, False, True, True, True]
        )

    def test_the_failing_component_is_named(self):
        basis = _eval_basis(4) + _eval_basis(5)
        blocks = [np.ones((4, 4)), np.zeros((4, 5))]
        with pytest.raises(ValueError, match="component 1"):
            _get_basis_component_infos(
                basis, drop_conv_basis_col=False, blocks=blocks, min_obs=1
            )

    def test_a_wrong_number_of_blocks_is_rejected(self):
        basis = _eval_basis(4) + _eval_basis(5)
        with pytest.raises(ValueError, match="one feature block per basis component"):
            _get_basis_component_infos(
                basis,
                drop_conv_basis_col=False,
                blocks=[np.ones((4, 4))],
                min_obs=1,
            )

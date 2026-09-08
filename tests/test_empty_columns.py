"""Tests for empty-column detection and the NonemptyColumns record."""

import numpy as np
import pytest

from pgam_jax._empty_columns import (
    NonemptyColumns,
    resolve_min_obs,
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


class TestFromComponentBlocks:
    def test_all_columns_nonempty(self):
        block = np.array([[1.0, 2.0], [3.0, 4.0]])
        nec = NonemptyColumns.from_component_blocks([block], 1, [False])
        np.testing.assert_array_equal(nec.masks[0], [True, True])
        assert nec.any_dropped is False
        assert nec.n_dropped == 0

    def test_an_all_zero_column_is_empty(self):
        block = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]])
        nec = NonemptyColumns.from_component_blocks([block], 1, [False])
        np.testing.assert_array_equal(nec.masks[0], [True, False, True])
        assert nec.any_dropped is True
        assert nec.n_dropped == 1

    def test_min_obs_above_one_drops_sparse_columns(self):
        block = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        nec = NonemptyColumns.from_component_blocks([block], 2, [False])
        np.testing.assert_array_equal(nec.masks[0], [False, True])

    def test_one_mask_per_component(self):
        blocks = [np.ones((4, 3)), np.zeros((4, 2))]
        blocks[1][0, 1] = 7.0
        nec = NonemptyColumns.from_component_blocks(blocks, 1, [False, False])
        assert len(nec.masks) == 2
        np.testing.assert_array_equal(nec.masks[0], [True, True, True])
        np.testing.assert_array_equal(nec.masks[1], [False, True])

    def test_nan_counts_as_no_observation(self):
        block = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        nec = NonemptyColumns.from_component_blocks([block], 1, [False])
        np.testing.assert_array_equal(nec.masks[0], [False, True])

    def test_a_fully_empty_component_raises(self):
        blocks = [np.ones((4, 2)), np.zeros((4, 3))]
        with pytest.raises(ValueError, match="component 1"):
            NonemptyColumns.from_component_blocks(blocks, 1, [False, False])

    def test_one_survivor_is_rejected_when_a_column_is_dropped_after(self):
        """A single survivor plus the identifiability drop leaves no column."""
        block = np.array([[1.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="keeps 1 column"):
            NonemptyColumns.from_component_blocks([block], 1, [True])

    def test_one_survivor_is_fine_when_no_column_is_dropped_after(self):
        block = np.array([[1.0, 0.0], [1.0, 0.0]])
        nec = NonemptyColumns.from_component_blocks([block], 1, [False])
        np.testing.assert_array_equal(nec.masks[0], [True, False])

    def test_two_survivors_are_enough_when_a_column_is_dropped_after(self):
        block = np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
        nec = NonemptyColumns.from_component_blocks([block], 1, [True])
        np.testing.assert_array_equal(nec.masks[0], [True, True, False])


class TestAllKept:
    def test_builds_all_true_masks(self):
        nec = NonemptyColumns.all_kept([3, 5])
        assert len(nec.masks) == 2
        assert nec.masks[0].tolist() == [True] * 3
        assert nec.masks[1].tolist() == [True] * 5
        assert nec.any_dropped is False

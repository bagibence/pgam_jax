"""Tests for masked feature construction and component bookkeeping."""

import nemos as nmo
import numpy as np
import pytest

from pgam_jax._empty_columns import NonemptyColumns
from pgam_jax._identifiable_features import (
    _component_feature_blocks,
    _compute_features_identifiable,
    _get_basis_component_infos,
    compute_features_identifiable,
)


def _bspline(k):
    return nmo.basis.BSplineEval(k, bounds=(0.0, 1.0))


@pytest.fixture
def additive_basis():
    return _bspline(6) + _bspline(7)


@pytest.fixture
def mixed_basis():
    return (_bspline(6) + _bspline(7)) + (_bspline(5) * _bspline(4))


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    return tuple(rng.uniform(0.05, 0.95, 50) for _ in range(4))


class TestComponentFeatureBlocks:
    def test_one_full_width_block_per_component(self, mixed_basis, inputs):
        mixed_basis.setup_basis(*inputs)
        blocks = _component_feature_blocks(mixed_basis, *inputs)
        assert [b.shape[1] for b in blocks] == [6, 7, 20]
        assert all(b.shape[0] == 50 for b in blocks)

    def test_blocks_concatenate_to_the_full_design(self, additive_basis, inputs):
        xi = inputs[:2]
        additive_basis.setup_basis(*xi)
        blocks = _component_feature_blocks(additive_basis, *xi)
        expected = additive_basis.compute_features(*xi)
        np.testing.assert_allclose(np.hstack(blocks), expected)

    def test_wrong_number_of_inputs_raises(self, additive_basis, inputs):
        additive_basis.setup_basis(*inputs[:2])
        with pytest.raises(ValueError, match="expects 2 input array"):
            _component_feature_blocks(additive_basis, *inputs[:3])


class TestMaskedFeatures:
    def test_no_mask_matches_the_unmasked_result(self, mixed_basis, inputs):
        mixed_basis.setup_basis(*inputs)
        without = _compute_features_identifiable(
            mixed_basis, *inputs, drop_conv_basis_col=False
        )
        kept = NonemptyColumns.all_kept([6, 7, 20])
        with_all_true = _compute_features_identifiable(
            mixed_basis, *inputs, drop_conv_basis_col=False, nonempty=kept
        )
        np.testing.assert_array_equal(without, with_all_true)

    def test_mask_is_applied_before_the_identifiability_drop(
        self, additive_basis, inputs
    ):
        xi = inputs[:2]
        additive_basis.setup_basis(*xi)
        blocks = _component_feature_blocks(additive_basis, *xi)

        m0 = np.ones(6, dtype=bool)
        m0[2] = False
        m1 = np.ones(7, dtype=bool)
        m1[6] = False  # the column the identifiability rule would have dropped
        nonempty = NonemptyColumns((m0, m1))

        got = _compute_features_identifiable(
            additive_basis, *xi, drop_conv_basis_col=False, nonempty=nonempty
        )
        expected = np.hstack([blocks[0][:, m0][:, :-1], blocks[1][:, m1][:, :-1]])
        np.testing.assert_array_equal(got, expected)
        assert got.shape[1] == (5 - 1) + (6 - 1)

    def test_setup_wrapper_forwards_the_mask(self, additive_basis, inputs):
        xi = inputs[:2]
        m0 = np.ones(6, dtype=bool)
        m0[0] = False
        nonempty = NonemptyColumns((m0, np.ones(7, dtype=bool)))
        got = compute_features_identifiable(
            additive_basis, *xi, drop_conv_basis_col=False, nonempty=nonempty
        )
        assert got.shape[1] == (5 - 1) + (7 - 1)


class TestComponentInfos:
    def test_slices_shrink_with_the_mask(self, mixed_basis):
        m0 = np.ones(6, dtype=bool)
        m0[1] = False
        m2 = np.ones(20, dtype=bool)
        m2[:5] = False
        nonempty = NonemptyColumns((m0, np.ones(7, dtype=bool), m2))

        infos = _get_basis_component_infos(
            mixed_basis, drop_conv_basis_col=False, nonempty=nonempty
        )
        widths = [
            i.identifiable_feature_slice.stop - i.identifiable_feature_slice.start
            for i in infos
        ]
        assert widths == [5 - 1, 7 - 1, 15 - 1]
        starts = [i.identifiable_feature_slice.start for i in infos]
        assert starts == [0, 4, 10]

    def test_input_slices_are_unchanged_by_the_mask(self, mixed_basis):
        nonempty = NonemptyColumns.all_kept([6, 7, 20])
        infos = _get_basis_component_infos(
            mixed_basis, drop_conv_basis_col=False, nonempty=nonempty
        )
        assert [(i.input_slice.start, i.input_slice.stop) for i in infos] == [
            (0, 1),
            (1, 2),
            (2, 4),
        ]


class TestMaskingIsARestriction:
    """
    The masked design must be a column subset of the unmasked design.

    Masking removes basis functions. It never rescales or recombines them. If
    this fails, the masked model is not the unmasked model restricted to the
    kept coefficients, and every comparison between the two is meaningless.
    """

    def _kept_indices(self, mask):
        """Indices the reduced design keeps, in the full-width numbering."""
        kept = np.flatnonzero(mask)
        return kept[:-1]  # the identifiability column comes off the survivors

    def test_columns_are_a_subset_of_the_unmasked_columns(self, additive_basis, inputs):
        xi = inputs[:2]
        additive_basis.setup_basis(*xi)
        blocks = _component_feature_blocks(additive_basis, *xi)

        m0 = np.ones(6, dtype=bool)
        m0[[1, 4]] = False
        m1 = np.ones(7, dtype=bool)
        m1[6] = False  # the column the unmasked rule would have dropped
        nonempty = NonemptyColumns((m0, m1))

        masked = _compute_features_identifiable(
            additive_basis, *xi, drop_conv_basis_col=False, nonempty=nonempty
        )
        expected = np.hstack(
            [
                blocks[0][:, self._kept_indices(m0)],
                blocks[1][:, self._kept_indices(m1)],
            ]
        )
        np.testing.assert_array_equal(masked, expected)

    def test_every_masked_column_appears_in_the_unmasked_design(
        self, additive_basis, inputs
    ):
        xi = inputs[:2]
        additive_basis.setup_basis(*xi)
        unmasked = _compute_features_identifiable(
            additive_basis, *xi, drop_conv_basis_col=False
        )
        m0 = np.ones(6, dtype=bool)
        m0[2] = False
        m1 = np.ones(7, dtype=bool)
        m1[3] = False
        masked = _compute_features_identifiable(
            additive_basis,
            *xi,
            drop_conv_basis_col=False,
            nonempty=NonemptyColumns((m0, m1)),
        )
        for j in range(masked.shape[1]):
            hits = np.all(np.isclose(unmasked, masked[:, [j]]), axis=0)
            assert (
                hits.any()
            ), f"masked column {j} is not a column of the unmasked design"

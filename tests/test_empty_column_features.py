"""Tests for masked feature construction and component bookkeeping."""

import nemos as nmo
import numpy as np
import pytest

from pgam_jax._identifiable_features import (
    _compute_features_identifiable,
    _compute_full_width_blocks,
    _get_basis_component_infos,
    compute_features_identifiable,
    reduce_component_blocks,
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


def _detection_blocks(widths, empty=None):
    """
    Build training blocks whose named columns hold no observations.

    ``empty`` maps a component index to the full-basis columns to empty out.
    Detection over these blocks produces the mask the test asks for.
    """
    empty = {} if empty is None else empty
    blocks = []
    for index, width in enumerate(widths):
        block = np.ones((4, width))
        block[:, list(empty.get(index, ()))] = 0.0
        blocks.append(block)
    return blocks


def _detected_infos(basis, widths, empty=None, drop_conv_basis_col=False):
    """Component records whose masks come from synthetic detection blocks."""
    return _get_basis_component_infos(
        basis,
        drop_conv_basis_col=drop_conv_basis_col,
        blocks=_detection_blocks(widths, empty),
        min_obs=1,
    )


def _masked_features(infos, *inputs):
    """Reduce the real feature blocks with an already detected layout."""
    return reduce_component_blocks(_compute_full_width_blocks(infos, *inputs), infos)


class TestComponentFeatureBlocks:
    def test_one_full_width_block_per_component(self, mixed_basis, inputs):
        mixed_basis.setup_basis(*inputs)
        blocks = _compute_full_width_blocks(
            _get_basis_component_infos(mixed_basis, drop_conv_basis_col=False), *inputs
        )
        assert [b.shape[1] for b in blocks] == [6, 7, 20]
        assert all(b.shape[0] == 50 for b in blocks)

    def test_blocks_concatenate_to_the_full_design(self, additive_basis, inputs):
        xi = inputs[:2]
        additive_basis.setup_basis(*xi)
        blocks = _compute_full_width_blocks(
            _get_basis_component_infos(additive_basis, drop_conv_basis_col=False), *xi
        )
        expected = additive_basis.compute_features(*xi)
        np.testing.assert_allclose(np.hstack(blocks), expected)

    def test_wrong_number_of_inputs_raises(self, additive_basis, inputs):
        additive_basis.setup_basis(*inputs[:2])
        with pytest.raises(ValueError, match="expects 2 input array"):
            _compute_full_width_blocks(
                _get_basis_component_infos(additive_basis, drop_conv_basis_col=False),
                *inputs[:3],
            )


class TestMaskedFeatures:
    def test_an_empty_mask_matches_the_unmasked_result(self, mixed_basis, inputs):
        mixed_basis.setup_basis(*inputs)
        without = _compute_features_identifiable(
            mixed_basis, *inputs, drop_conv_basis_col=False
        )
        infos = _detected_infos(mixed_basis, [6, 7, 20])
        with_all_true = _masked_features(infos, *inputs)
        np.testing.assert_array_equal(without, with_all_true)

    def test_mask_is_applied_before_the_identifiability_drop(
        self, additive_basis, inputs
    ):
        xi = inputs[:2]
        additive_basis.setup_basis(*xi)
        blocks = _compute_full_width_blocks(
            _get_basis_component_infos(additive_basis, drop_conv_basis_col=False), *xi
        )

        # Component 1 loses the column the identifiability rule would have taken.
        infos = _detected_infos(additive_basis, [6, 7], {0: [2], 1: [6]})
        m0, m1 = (info.nonempty_mask for info in infos)

        got = _masked_features(infos, *xi)
        expected = np.hstack([blocks[0][:, m0][:, :-1], blocks[1][:, m1][:, :-1]])
        np.testing.assert_array_equal(got, expected)
        assert got.shape[1] == (5 - 1) + (6 - 1)

    def test_the_setup_wrapper_evaluates_without_a_prior_setup(
        self, additive_basis, inputs
    ):
        xi = inputs[:2]
        got = compute_features_identifiable(
            additive_basis, *xi, drop_conv_basis_col=False
        )
        assert got.shape == (50, (6 - 1) + (7 - 1))


class TestComponentInfos:
    @pytest.mark.parametrize("convolution", [False, True])
    @pytest.mark.parametrize("drop_conv", [False, True])
    def test_reference_and_features_use_full_basis_coordinates(
        self, convolution, drop_conv
    ):
        basis = nmo.basis.BSplineConv(6, window_size=12) if convolution else _bspline(6)
        (info,) = _detected_infos(
            basis, [6], {0: [1, 4, 5]}, drop_conv_basis_col=drop_conv
        )
        np.testing.assert_array_equal(
            info.nonempty_mask, [True, False, True, True, False, False]
        )
        drops = not convolution or drop_conv
        assert info.identifiability_column == (3 if drops else None)
        block = np.arange(18).reshape(3, 6)
        expected = block[:, [0, 2] if drops else [0, 2, 3]]
        np.testing.assert_array_equal(info.reduce_features(block), expected)
        assert info.identifiable_feature_slice == slice(0, expected.shape[1])

    def test_counts_follow_the_mask(self):
        (info,) = _detected_infos(_bspline(6), [6], {0: [1, 4, 5]})
        assert info.n_kept == 3
        assert info.n_dropped == 3
        assert info.is_masked is True

    def test_slices_shrink_with_the_mask(self, mixed_basis):
        infos = _detected_infos(mixed_basis, [6, 7, 20], {0: [1], 2: list(range(5))})
        widths = [
            i.identifiable_feature_slice.stop - i.identifiable_feature_slice.start
            for i in infos
        ]
        assert widths == [5 - 1, 7 - 1, 15 - 1]
        starts = [i.identifiable_feature_slice.start for i in infos]
        assert starts == [0, 4, 10]

    def test_input_slices_are_unchanged_by_the_mask(self, mixed_basis):
        infos = _detected_infos(mixed_basis, [6, 7, 20], {0: [1], 2: list(range(5))})
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
        blocks = _compute_full_width_blocks(
            _get_basis_component_infos(additive_basis, drop_conv_basis_col=False), *xi
        )

        # Component 1 loses the column the unmasked rule would have taken.
        infos = _detected_infos(additive_basis, [6, 7], {0: [1, 4], 1: [6]})
        m0, m1 = (info.nonempty_mask for info in infos)

        masked = _masked_features(infos, *xi)
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
        infos = _detected_infos(additive_basis, [6, 7], {0: [2], 1: [3]})
        masked = _masked_features(infos, *xi)
        for j in range(masked.shape[1]):
            hits = np.all(np.isclose(unmasked, masked[:, [j]]), axis=0)
            assert (
                hits.any()
            ), f"masked column {j} is not a column of the unmasked design"

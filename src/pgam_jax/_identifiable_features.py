from dataclasses import dataclass

import nemos as nmo
import numpy as np

from ._empty_columns import NonemptyColumns
from ._nemos_compat import get_n_inputs


@dataclass(frozen=True)
class BasisComponentInfo:
    """Slices for one component after identifiability column dropping."""

    index: int
    basis: object
    input_slice: slice
    identifiable_feature_slice: slice


def _should_drop_basis_col(
    basis,
    drop_conv_basis_col: bool,
) -> bool:
    """
    Return whether this basis component should drop its last column.

    Evaluation bases always drop, convolutional bases drop if ``drop_conv_basis_col`` is True.
    Convolution doesn't create linearly dependent columns, so in theory there is no need to drop,
    but the option is added for matching the original implementation if required.
    """
    if isinstance(basis, nmo.basis._basis_mixin.ConvBasisMixin):
        return drop_conv_basis_col
    return True


def _iter_components_with_inputs(basis, inputs):
    """
    Pair each basis component with the inputs it consumes.

    Iterating a nemos basis flattens additive composition, so a tensor product
    counts as one component. The input arrays are split in the same order.
    """
    n_expected = sum(get_n_inputs(component) for component in basis)
    if len(inputs) != n_expected:
        raise ValueError(
            f"This basis expects {n_expected} input array(s), got {len(inputs)}."
        )
    start = 0
    for component in basis:
        n_inputs = get_n_inputs(component)
        yield component, inputs[start : start + n_inputs]
        start += n_inputs


def _component_feature_blocks(basis, *inputs) -> list[np.ndarray]:
    """
    Evaluate one full-width feature block per basis component.

    The blocks carry every basis function, so no identifiability column and no
    empty column is removed yet. Empty-column detection needs this full width,
    because a nonempty mask indexes basis functions rather than fitted
    coefficients.
    """
    return [
        component._compute_features(*component_inputs)
        for component, component_inputs in _iter_components_with_inputs(basis, inputs)
    ]


def _mask_for_component(nonempty: NonemptyColumns | None, index: int):
    """
    Return the nonempty mask for one component, or None when unmasked.

    An all-true mask returns None so that the caller skips the copy that
    boolean indexing would make.
    """
    if nonempty is None:
        return None
    mask = nonempty.masks[index]
    return None if mask.all() else mask


def _get_basis_component_infos(
    basis,
    *,
    drop_conv_basis_col: bool,
    nonempty: NonemptyColumns | None = None,
) -> list[BasisComponentInfo]:
    """Return component slices matching the identifiable feature matrix columns."""
    infos = []
    input_start = 0
    out_start = 0
    for index, component in enumerate(basis):
        n_inputs = get_n_inputs(component)

        mask = _mask_for_component(nonempty, index)
        n_outputs = component.n_basis_funcs if mask is None else int(mask.sum())
        if _should_drop_basis_col(component, drop_conv_basis_col):
            n_outputs -= 1

        infos.append(
            BasisComponentInfo(
                index=index,
                basis=component,
                input_slice=slice(input_start, input_start + n_inputs),
                identifiable_feature_slice=slice(out_start, out_start + n_outputs),
            )
        )
        input_start += n_inputs
        out_start += n_outputs
    return infos


def compute_features_identifiable(
    basis,
    *inputs,
    drop_conv_basis_col: bool,
    nonempty: NonemptyColumns | None = None,
):
    """Build the identifiability-constrained design matrix, **uncentered**.

    The returned matrix has one column dropped per eval-basis component to
    remove collinearity with the intercept, but is NOT mean-centered.  Callers
    that want a usable design must subtract the per-column means of the
    training matrix.  ``GAM._fit_design_matrix`` does this and stores the
    means as ``feature_mean_`` for reuse at prediction time
    (``GAM._transform_design_matrix``).

    Without that centering, smooth columns remain correlated with the
    intercept, which both leaves the model only weakly identifiable in
    finite samples and inflates the conditioning of ``H + S_λ/φ``.

    When ``nonempty`` is given, each component drops its empty columns first,
    and the identifiability column is then taken from the survivors.
    """
    basis.setup_basis(*inputs)
    return _compute_features_identifiable(
        basis,
        *inputs,
        drop_conv_basis_col=drop_conv_basis_col,
        nonempty=nonempty,
    )


# TODO: Should this be a method in nemos basis classes?
def reduce_component_blocks(
    blocks,
    basis,
    *,
    drop_conv_basis_col: bool,
    nonempty: NonemptyColumns | None = None,
):
    """
    Turn full-width component blocks into the reduced design matrix.

    Each block drops its empty columns first. The identifiability column is
    then taken from the survivors.
    """
    out = []
    for index, (component, block) in enumerate(zip(basis, blocks)):
        mask = _mask_for_component(nonempty, index)
        if mask is not None:
            block = block[:, mask]
        if _should_drop_basis_col(component, drop_conv_basis_col):
            block = block[:, :-1]
        out.append(block)
    return np.hstack(out)


def _compute_features_identifiable(
    basis,
    *inputs,
    drop_conv_basis_col: bool,
    nonempty: NonemptyColumns | None = None,
):
    return reduce_component_blocks(
        _component_feature_blocks(basis, *inputs),
        basis,
        drop_conv_basis_col=drop_conv_basis_col,
        nonempty=nonempty,
    )

from dataclasses import dataclass

import nemos as nmo
import numpy as np

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


def _get_basis_component_infos(
    basis,
    *,
    drop_conv_basis_col: bool,
) -> list[BasisComponentInfo]:
    """Return component slices matching the identifiable feature matrix columns."""
    infos = []
    input_start = 0
    out_start = 0
    for index, component in enumerate(basis):
        n_inputs = get_n_inputs(component)

        n_outputs = component.n_basis_funcs
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
):
    basis.setup_basis(*inputs)
    return _compute_features_identifiable(
        basis,
        *inputs,
        drop_conv_basis_col=drop_conv_basis_col,
    )


# TODO: Should this be a method in nemos basis classes?
def _compute_features_identifiable(
    basis,
    *inputs,
    drop_conv_basis_col: bool,
):
    if isinstance(basis, nmo.basis.AdditiveBasis):
        n1 = get_n_inputs(basis.basis1)
        x1 = _compute_features_identifiable(
            basis.basis1,
            *inputs[:n1],
            drop_conv_basis_col=drop_conv_basis_col,
        )
        x2 = _compute_features_identifiable(
            basis.basis2,
            *inputs[n1:],
            drop_conv_basis_col=drop_conv_basis_col,
        )
        return np.hstack([x1, x2])
    else:
        X = basis._compute_features(*inputs)
        if _should_drop_basis_col(basis, drop_conv_basis_col):
            return X[:, :-1]
        return X

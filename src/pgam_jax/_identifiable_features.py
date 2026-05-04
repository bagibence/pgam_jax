import nemos as nmo
import numpy as np


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
        n1 = basis.basis1._n_inputs
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

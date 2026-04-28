import nemos as nmo
import numpy as np


def compute_features_identifiable(basis, *inputs):
    basis.setup_basis(*inputs)
    return _compute_features_identifiable(basis, *inputs)


# TODO: Should this be a method in nemos basis classes?
def _compute_features_identifiable(basis, *inputs):
    if isinstance(basis, nmo.basis.AdditiveBasis):
        n1 = basis.basis1._n_input_dimensionality
        x1 = _compute_features_identifiable(basis.basis1, *inputs[:n1])
        x2 = _compute_features_identifiable(basis.basis2, *inputs[n1:])
        return np.hstack([x1, x2])
    else:
        # convolution doesn't create linearly dependent columns, so no need to drop
        if isinstance(basis, nmo.basis._basis_mixin.ConvBasisMixin):
            return basis._compute_features(*inputs)

        # BSplineEval and MultiplicativeBasis just drop a column
        return basis._compute_features(*inputs)[:, :-1]

from nemos.identifiability_constraints import apply_identifiability_constraints
from nemos.basis import AdditiveBasis, BSplineEval, MultiplicativeBasis
from ._basis import GAMAdditiveBasis, GAMMultiplicativeBasis
from ._bspline import GAMBSplineEval


def to_gam_basis(basis, identifiability):
    _type_dict = {
        BSplineEval: GAMBSplineEval,
        AdditiveBasis: GAMAdditiveBasis,
        MultiplicativeBasis: GAMMultiplicativeBasis,
    }

    # don't convert if already converted
    if isinstance(basis, tuple(_type_dict.values())):
        return basis

    if isinstance(basis, (AdditiveBasis, MultiplicativeBasis)):
        gam_composite_type = _type_dict[type(basis)]

        basis1 = to_gam_basis(basis.basis1, identifiability)
        basis2 = to_gam_basis(basis.basis2, identifiability)

        return gam_composite_type(basis1, basis2)

    if isinstance(basis, BSplineEval):
        # replace the default label
        label = basis.label
        if label.startswith("BSplineEval"):
            label = label.replace("BSplineEval", "GAMBSplineEval")

        return GAMBSplineEval(
            n_basis_funcs=basis.n_basis_funcs,
            order=basis.order,
            bounds=basis.bounds,
            label=label,
            identifiability=identifiability,
        )

    raise TypeError(
        "basis has to be one of BSplineEval, AdditiveBasis, MultiplicativeBasis"
    )


# TODO: Use this somewhere?
def apply_constraints(X, add_intercept: bool = True):
    """
    Drop linearly depednent columns.

    Parameters
    ----------
    X:
        A design matrix.
    add_intercept:
        True, if one must add a constant column: [1, X].

    Returns
    -------
        The full rank matrix.

    """
    X_identifiable, kept_index = apply_identifiability_constraints(
        X[..., ::-1], add_intercept=add_intercept
    )
    kept_index = X.shape[-1] - kept_index[::-1] - 1
    return X_identifiable[:, ::-1], kept_index

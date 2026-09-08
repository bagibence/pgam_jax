from dataclasses import dataclass

import nemos as nmo
import numpy as np

from ._empty_columns import NonemptyColumns
from ._nemos_compat import get_n_inputs


@dataclass(frozen=True, eq=False)
class BasisComponentInfo:
    """
    Layout of one smooth in the full basis and fitted coefficient vector.

    ``nonempty_mask`` indexes the full basis, before identifiability. The
    last surviving column is removed when ``drops_identifiability_column``
    is True. ``identifiable_feature_slice`` indexes the resulting design
    and coefficients, without the intercept.

    Penalties use the nonempty mask before rebuilding their null-space term.
    Their identifiability transform remains a separate, later operation.
    """

    index: int
    basis: object
    input_slice: slice
    identifiable_feature_slice: slice
    nonempty_mask: np.ndarray
    drops_identifiability_column: bool

    @property
    def n_inputs(self) -> int:
        """Number of input arrays consumed by this component."""
        return self.input_slice.stop - self.input_slice.start

    @property
    def is_masked(self) -> bool:
        """Whether any full-basis columns were removed as empty."""
        return not bool(self.nonempty_mask.all())

    @property
    def identifiability_column(self) -> int | None:
        """Full-basis index of the last survivor, or None without a constraint."""
        if not self.drops_identifiability_column:
            return None
        return int(np.flatnonzero(self.nonempty_mask)[-1])

    def reduce_features(self, block: np.ndarray) -> np.ndarray:
        """Remove empty columns, then the identifiability column."""
        if self.is_masked:
            block = block[:, self.nonempty_mask]
        if self.drops_identifiability_column:
            block = block[:, :-1]
        return block

    def compute_features(self, *inputs) -> np.ndarray:
        """Evaluate reduced features using the already configured basis."""
        return self.reduce_features(self.basis._compute_features(*inputs))


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


def _component_feature_blocks(infos, *inputs) -> list[np.ndarray]:
    """Evaluate full-width blocks using the component records' input slices."""
    n_expected = sum(info.n_inputs for info in infos)
    if len(inputs) != n_expected:
        raise ValueError(
            f"This basis expects {n_expected} input array(s), got {len(inputs)}."
        )
    return [info.basis._compute_features(*inputs[info.input_slice]) for info in infos]


def _get_basis_component_infos(
    basis,
    *,
    drop_conv_basis_col: bool,
    nonempty: NonemptyColumns | None = None,
) -> tuple[BasisComponentInfo, ...]:
    """Build component records with full-width masks and reduced slices."""
    infos = []
    input_start = 0
    out_start = 0
    components = tuple(basis)
    if nonempty is not None and len(nonempty) != len(components):
        raise ValueError("Expected one nonempty mask per basis component.")
    for index, component in enumerate(components):
        n_inputs = get_n_inputs(component)

        mask = (
            np.ones(component.n_basis_funcs, dtype=bool)
            if nonempty is None
            else np.array(nonempty.masks[index], dtype=bool, copy=True)
        )
        if mask.shape != (component.n_basis_funcs,):
            raise ValueError(
                f"Mask for component {index} must have shape "
                f"{(component.n_basis_funcs,)}, got {mask.shape}."
            )
        drops_column = _should_drop_basis_col(component, drop_conv_basis_col)
        n_outputs = int(mask.sum()) - int(drops_column)
        if n_outputs < 1:
            raise ValueError(f"Basis component {index} has no fitted columns.")
        mask.setflags(write=False)

        infos.append(
            BasisComponentInfo(
                index=index,
                basis=component,
                input_slice=slice(input_start, input_start + n_inputs),
                identifiable_feature_slice=slice(out_start, out_start + n_outputs),
                nonempty_mask=mask,
                drops_identifiability_column=drops_column,
            )
        )
        input_start += n_inputs
        out_start += n_outputs
    return tuple(infos)


def compute_features_identifiable(
    basis,
    *inputs,
    drop_conv_basis_col: bool,
    nonempty: NonemptyColumns | None = None,
):
    """
    Build the identifiability-constrained design matrix, **uncentered**.

    The returned matrix has one column dropped per eval-basis component to
    remove collinearity with the intercept, but is NOT mean-centered. Callers
    that want a usable design must subtract the per-column means of the
    training matrix. ``GAM._fit_design_matrix`` does this and stores the
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


def reduce_component_blocks(blocks, infos):
    """Reduce full-width blocks using their matching component records."""
    return np.hstack(
        [info.reduce_features(block) for info, block in zip(infos, blocks, strict=True)]
    )


def _compute_features_identifiable(
    basis,
    *inputs,
    drop_conv_basis_col: bool,
    nonempty: NonemptyColumns | None = None,
):
    infos = _get_basis_component_infos(
        basis, drop_conv_basis_col=drop_conv_basis_col, nonempty=nonempty
    )
    return reduce_component_blocks(_component_feature_blocks(infos, *inputs), infos)

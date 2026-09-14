from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import nemos as nmo
import numpy as np

from ._nemos_compat import get_n_inputs


def resolve_min_obs(drop_empty_columns: bool | int) -> int | None:
    """
    Turn the user-facing flag into a ``min_obs`` threshold.

    ``False`` disables dropping and returns None. ``True`` means a threshold of
    1. An integer sets the threshold directly.
    """
    if isinstance(drop_empty_columns, bool):
        return 1 if drop_empty_columns else None
    if isinstance(drop_empty_columns, (int, np.integer)):
        min_obs = int(drop_empty_columns)
        if min_obs < 1:
            raise ValueError(
                f"drop_empty_columns must be at least 1 when given as an "
                f"integer, got {min_obs}."
            )
        return min_obs
    raise TypeError(
        f"drop_empty_columns must be a bool or int, got "
        f"{type(drop_empty_columns).__name__}."
    )


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

    @classmethod
    def from_block(
        cls,
        basis,
        *,
        index: int,
        input_start: int,
        feature_start: int,
        drop_conv_basis_col: bool,
        block: np.ndarray | None = None,
        min_obs: int | None = None,
    ) -> BasisComponentInfo:
        """
        Build a component record from an uncentered training feature block.

        The caller selects the fitting rows before supplying ``block``.
        ``min_obs=None`` keeps every column and does not require a block.
        Otherwise, NaNs count as zero, matching the zero-fill NaN policy.
        Removing a nonempty evaluation column disables the additional
        identifiability drop. Zero-only removal preserves the existing rule.
        """
        removed_nonempty = False
        if min_obs is None:
            mask = np.ones(basis.n_basis_funcs, dtype=bool)
        else:
            if block is None:
                raise ValueError("Column detection requires a training feature block.")
            block = np.asarray(block)
            if block.ndim != 2 or block.shape[1] != basis.n_basis_funcs:
                raise ValueError(
                    f"Feature block for component {index} must be two-dimensional "
                    f"with {basis.n_basis_funcs} columns, got {block.shape}."
                )
            counts = np.sum(np.abs(block) > 0, axis=0)
            mask = counts >= min_obs
            removed_nonempty = bool(np.any((counts > 0) & ~mask))

        drops_column = _should_drop_basis_col(
            basis,
            drop_conv_basis_col=drop_conv_basis_col,
            removed_nonempty=removed_nonempty,
        )
        needed = 1 + int(drops_column)
        n_kept = int(mask.sum())
        if n_kept < needed:
            raise ValueError(
                f"Basis component {index} keeps {n_kept} column(s) at "
                f"min_obs={min_obs}, but it needs {needed}. "
                + (
                    "One more column comes off that component for "
                    "identifiability, so a single survivor leaves it empty. "
                    if drops_column
                    else ""
                )
                + "Lower min_obs, use a smaller basis for that covariate, "
                "or remove the component."
            )
        mask.setflags(write=False)
        return cls(
            index=index,
            basis=basis,
            input_slice=slice(input_start, input_start + get_n_inputs(basis)),
            identifiable_feature_slice=slice(
                feature_start, feature_start + n_kept - int(drops_column)
            ),
            nonempty_mask=mask,
            drops_identifiability_column=drops_column,
        )

    @property
    def n_kept(self) -> int:
        """Number of nonempty columns, before the identifiability constraint."""
        return int(self.nonempty_mask.sum())

    @property
    def n_dropped(self) -> int:
        """Number of full-basis columns removed as empty."""
        return self.nonempty_mask.size - self.n_kept

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

    def compute_reduced_features(self, *inputs) -> np.ndarray:
        """Evaluate reduced features using the already configured basis."""
        return self.reduce_features(self.basis._compute_features(*inputs))


def _should_drop_basis_col(
    basis,
    *,
    drop_conv_basis_col: bool,
    removed_nonempty: bool,
) -> bool:
    """
    Return whether this basis component should drop its last column.

    Without masking or masking only all-zero columns, evaluation bases drop
    and convolutional bases drop if ``drop_conv_basis_col`` is True.
    When threshold masking removes nonempty columns, there is no need to drop
    for evaluation bases either, so it gets disabled.

    Convolution doesn't create linearly dependent columns, so in theory there is no need to drop,
    but the option is added for matching the original implementation if required.
    """
    if isinstance(basis, nmo.basis._basis_mixin.ConvBasisMixin):
        return drop_conv_basis_col
    return not removed_nonempty


def _compute_full_width_blocks(infos, *inputs) -> list[np.ndarray]:
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
    blocks: Sequence[np.ndarray] | None = None,
    min_obs: int | None = None,
) -> tuple[BasisComponentInfo, ...]:
    """
    Build component records and assign consecutive coefficient slices.

    With ``min_obs`` set, detect empty columns directly from full-width
    training blocks whose rows have already been selected for fitting.
    Otherwise, build an unmasked layout.
    """
    components = tuple(basis)
    if blocks is not None and len(blocks) != len(components):
        raise ValueError("Expected one feature block per basis component.")
    infos = []
    input_start = 0
    feature_start = 0
    for index, component in enumerate(components):
        info = BasisComponentInfo.from_block(
            component,
            index=index,
            input_start=input_start,
            feature_start=feature_start,
            drop_conv_basis_col=drop_conv_basis_col,
            block=None if blocks is None else blocks[index],
            min_obs=min_obs,
        )
        infos.append(info)
        input_start = info.input_slice.stop
        feature_start = info.identifiable_feature_slice.stop
    return tuple(infos)


def compute_features_identifiable(
    basis,
    *inputs,
    drop_conv_basis_col: bool,
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
    """
    basis.setup_basis(*inputs)
    return _compute_features_identifiable(
        basis,
        *inputs,
        drop_conv_basis_col=drop_conv_basis_col,
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
):
    infos = _get_basis_component_infos(basis, drop_conv_basis_col=drop_conv_basis_col)
    return reduce_component_blocks(_compute_full_width_blocks(infos, *inputs), infos)

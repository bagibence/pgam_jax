"""
Detection and bookkeeping for empty design columns.

An *empty column* is a design column whose count of non-zero entries, over the
rows that reach the solver, is below ``min_obs``. The data carry no information
about its coefficient. A basis built over a full covariate range produces such
columns whenever the data cover only part of that range.

A *nonempty mask* is the per-component boolean record of which columns are not
empty. Its length is the full basis width of the component, before the
identifiability column is removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


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


def _nonzero_counts(block: np.ndarray) -> np.ndarray:
    """
    Count entries per column that are neither zero nor NaN.

    NaN never compares greater than zero, so a NaN entry counts as no
    observation. That matches ``nan_handling="zero"``, which turns design NaNs
    into zeros before the solver sees them.
    """
    return np.sum(np.abs(np.asarray(block)) > 0, axis=0)


@dataclass(frozen=True)
class NonemptyColumns:
    """
    Per-component record of which design columns are not empty.

    ``masks[i]`` has the full basis width of component ``i``, so it indexes
    basis functions, not fitted coefficients. The identifiability column is
    removed after this mask is applied.
    """

    masks: tuple[np.ndarray, ...]

    @classmethod
    def all_kept(cls, widths: Sequence[int]) -> NonemptyColumns:
        """Build a record that keeps every column of every component."""
        return cls(tuple(np.ones(int(w), dtype=bool) for w in widths))

    @classmethod
    def from_component_blocks(
        cls,
        blocks: Sequence[np.ndarray],
        min_obs: int,
        drops_identifiability_column: Sequence[bool],
    ) -> NonemptyColumns:
        """
        Detect empty columns in one full-width feature block per component.

        Each block must be the uncentered, unmasked feature matrix of one basis
        component, restricted to the rows that reach the solver.

        ``drops_identifiability_column`` says, per component, whether one more
        column comes off after this mask. Such a component needs two survivors,
        not one, or the design matrix ends up with no column for it at all.
        """
        masks = []
        for index, block in enumerate(blocks):
            mask = _nonzero_counts(block) >= min_obs
            needed = 2 if drops_identifiability_column[index] else 1
            n_kept = int(mask.sum())
            if n_kept < needed:
                raise ValueError(
                    f"Basis component {index} keeps {n_kept} column(s) at "
                    f"min_obs={min_obs}, but it needs {needed}. "
                    + (
                        "One more column comes off that component for "
                        "identifiability, so a single survivor leaves it empty. "
                        if needed == 2
                        else ""
                    )
                    + "Lower min_obs, use a smaller basis for that covariate, "
                    "or remove the component."
                )
            masks.append(mask)
        return cls(tuple(masks))

    @property
    def any_dropped(self) -> bool:
        """Whether any component lost at least one column."""
        return any(not mask.all() for mask in self.masks)

    @property
    def n_dropped(self) -> int:
        """Total number of empty columns across all components."""
        return int(sum(mask.size - mask.sum() for mask in self.masks))

    @property
    def n_kept(self) -> tuple[int, ...]:
        """Number of surviving columns per component."""
        return tuple(int(mask.sum()) for mask in self.masks)

    def __len__(self) -> int:
        return len(self.masks)

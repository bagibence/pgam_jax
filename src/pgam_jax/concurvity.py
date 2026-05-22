"""JAX implementation of mgcv's `concurvity()`.

The user-facing entry point is :meth:`pgam_jax.GAM.concurvity`. This
module exposes the low-level building blocks it delegates to:

- :func:`concurvity` : pure linear-algebra implementation that operates on
  a raw design matrix and explicit term column slices. Mirrors mgcv exactly
  and is what the validation harness in ``_scripts/`` checks against
  ``mgcv::concurvity`` outputs to ~1e-14.
- :class:`TermBlock` : column range + label for one term.
- :func:`design_with_intercept`, :func:`term_blocks_for_gam` : adapters
  that translate a fitted :class:`pgam_jax.GAM` into the inputs of
  :func:`concurvity`.

Mathematical derivation: ``docs/concurvity_mgcv.md``.

Numerical note: float64 is required to reproduce mgcv's LAPACK output;
the diagnostic is sensitive to the near-rank-deficient end of the QR and
float32 diverges by ~1e-4. Callers should set
``jax.config.update("jax_enable_x64", True)`` before invoking this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import jax.scipy.linalg as jsl
from ._identifiable_features import _get_basis_component_infos


@dataclass(frozen=True)
class TermBlock:
    """Column range and label for one term in the design matrix."""
    label: str
    start: int  # 0-indexed, inclusive
    stop: int   # 0-indexed, inclusive (i.e. `X[:, start:stop+1]` is the block)

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop + 1)

    @property
    def ncol(self) -> int:
        return self.stop - self.start + 1


def _precondition(X):
    """Replace X with the R factor of its QR; preserves X.T @ X exactly."""
    return jnp.linalg.qr(X, mode="r")


def _block_qr_pieces(X, term: TermBlock):
    """Build R_12 and R-tilde for one term (the rest-of-model is everything
    else). Returns (R12, Rt, R_right) following the doc's notation."""
    p = X.shape[1]
    cols_i = jnp.arange(term.start, term.stop + 1)
    # Boolean mask for "other" columns.
    mask_other = jnp.ones(p, dtype=bool).at[cols_i].set(False)
    other_idx = jnp.nonzero(mask_other, size=p - term.ncol)[0]

    X_other = X[:, other_idx]
    X_i = X[:, cols_i]
    r = X_other.shape[1]

    # Non-pivoted QR of [X_other | X_i]; keep the right block of R.
    R = jnp.linalg.qr(jnp.concatenate([X_other, X_i], axis=1), mode="r")
    R_right = R[:, r:]            # (r+k) x k = [[R12], [R22]]
    R12 = R_right[:r, :]          # r x k

    # Second QR to get R-tilde (k x k upper triangular).
    Rt = jnp.linalg.qr(R_right, mode="r")
    return R12, Rt, R_right


def _measures_for_term(X, term: TermBlock, beta_block):
    R12, Rt, R_right = _block_qr_pieces(X, term)

    # worst: top singular value squared of R12 @ inv(Rt).
    # M = inv(Rt.T) @ R12.T = (R12 @ inv(Rt)).T   -> svd same singular values.
    # `solve_triangular(Rt.T, R12.T, lower=True)` handles the forward solve.
    M = jsl.solve_triangular(Rt.T, R12.T, lower=True)
    s = jnp.linalg.svd(M, compute_uv=False)
    worst = s[0] ** 2

    # estimate: Frobenius ratio.
    estimate = jnp.sum(R12 ** 2) / jnp.sum(R_right ** 2)

    # observed: only if a coefficient block was supplied.
    if beta_block is None:
        observed = None
    else:
        num = jnp.sum((R12 @ beta_block) ** 2)
        den = jnp.sum((Rt  @ beta_block) ** 2)
        observed = num / den

    return worst, observed, estimate


def concurvity(
    X,
    term_blocks: Sequence[TermBlock],
    beta=None,
    full: bool = True,
    precondition: bool = True,
    as_dataframe: bool = False,
):
    """Compute mgcv-style concurvity for the given design matrix and terms.

    Parameters
    ----------
    X : array, shape (n, p)
        Design matrix; columns must be grouped by term according to
        `term_blocks`. Rows with NaNs should already be removed by the
        caller (mgcv does this internally; we leave it explicit).
    term_blocks : sequence of TermBlock
        One per term, plus a leading TermBlock for the parametric block
        if the model has one (label conventionally 'para'). Slices must
        be disjoint and together cover the columns referenced.
    beta : array, shape (p,), optional
        Fitted coefficients. Required for the 'observed' measure.
    full : bool, default True
        If True, each term is compared against everything else. If False,
        compare each ordered pair (i, j).
    precondition : bool, default True
        Apply the QR-R preconditioning of X before the main loop. Pure
        optimization; gives identical results up to floating point.
    as_dataframe : bool, default False
        If True, return pandas DataFrames instead of raw arrays:
        - full=True: a single DataFrame indexed by term label with one
          column per measure.
        - full=False: dict[measure -> DataFrame] of m x m matrices with
          term labels on both axes (row = explainer, col = focal).

    Returns
    -------
    dict[str, jnp.ndarray] | pandas.DataFrame | dict[str, pandas.DataFrame]
        Shape depends on `full` and `as_dataframe`. Measure keys are
        'worst', 'estimate', and 'observed' (only if beta was supplied).
    """
    if precondition:
        X = _precondition(X)

    m = len(term_blocks)

    if full:
        worst = jnp.zeros(m)
        estim = jnp.zeros(m)
        obs   = jnp.zeros(m) if beta is not None else None
        for i, t in enumerate(term_blocks):
            bi = beta[t.slice] if beta is not None else None
            w, o, e = _measures_for_term(X, t, bi)
            worst = worst.at[i].set(w)
            estim = estim.at[i].set(e)
            if obs is not None:
                obs = obs.at[i].set(o)
        out = {"worst": worst, "estimate": estim}
        if obs is not None:
            out["observed"] = obs
        return _to_dataframe(out, term_blocks, full=True) if as_dataframe else out

    # Pairwise. Convention follows mgcv's `concurvity(., full=FALSE)`:
    # entry [i, j] is "how much of term j is explained by term i", i.e.
    # term i plays the role of 'rest of model' and term j is the focal
    # term whose curve gets decomposed. `worst` is symmetric so this
    # convention is invisible to it; `observed` and `estimate` are not.
    worst = jnp.ones((m, m))   # diagonal is 1 by convention in mgcv
    estim = jnp.ones((m, m))
    obs   = jnp.ones((m, m)) if beta is not None else None
    for i, t_rest in enumerate(term_blocks):
        for j, t_focal in enumerate(term_blocks):
            if i == j:
                continue
            # Build [X_rest | X_focal]; synthetic block points at X_focal.
            cols = jnp.concatenate([
                jnp.arange(t_rest.start,  t_rest.stop  + 1),
                jnp.arange(t_focal.start, t_focal.stop + 1),
            ])
            Xpair = X[:, cols]
            r = t_rest.ncol
            synth = TermBlock(label=t_focal.label,
                              start=r, stop=r + t_focal.ncol - 1)
            b_focal = beta[t_focal.slice] if beta is not None else None
            w, o, e = _measures_for_term(Xpair, synth, b_focal)
            worst = worst.at[i, j].set(w)
            estim = estim.at[i, j].set(e)
            if obs is not None:
                obs = obs.at[i, j].set(o)
    out = {"worst": worst, "estimate": estim}
    if obs is not None:
        out["observed"] = obs
    return _to_dataframe(out, term_blocks, full=False) if as_dataframe else out


_MEASURE_ORDER = ("worst", "observed", "estimate")


def _to_dataframe(result, term_blocks, full):
    """Wrap the raw-array result dict in pandas DataFrames.

    full=True  -> single DataFrame, rows=terms, cols=measures.
    full=False -> dict[measure -> DataFrame] of m x m matrices labeled on
                  both axes (row = explainer, col = focal).
    """
    import pandas as pd

    labels = [t.label for t in term_blocks]
    measures = [m for m in _MEASURE_ORDER if m in result]

    if full:
        return pd.DataFrame({m: result[m] for m in measures},
                            index=pd.Index(labels, name="term"))

    return {
        m: pd.DataFrame(result[m],
                        index=pd.Index(labels, name="explainer"),
                        columns=pd.Index(labels, name="focal"))
        for m in measures
    }


# ---------------------------------------------------------------------------
# GAM-aware helpers. Used by `GAM.concurvity` (see `gam.py`); pulled out
# here so the design-matrix → term-block translation lives next to the
# math that consumes it.
# ---------------------------------------------------------------------------

def design_with_intercept(gam, inputs):
    """Centered design matrix for a fitted GAM, with the intercept column
    prepended. Used by `GAM.concurvity` to assemble the full `X`."""
    X_smooths = gam._transform_design_matrix(inputs)
    intercept_col = jnp.ones((X_smooths.shape[0], 1))
    return jnp.concatenate([intercept_col, X_smooths], axis=1)


def term_blocks_for_gam(gam):
    """Build the TermBlock list for a fitted GAM (parametric block first,
    then one block per smooth component), using the same slice convention
    as `_get_basis_component_infos`. The +1 shifts account for the
    prepended intercept column in `design_with_intercept`."""

    infos = _get_basis_component_infos(
        gam.basis, drop_conv_basis_col=gam.drop_conv_basis_col
    )
    blocks = [TermBlock(label="para", start=0, stop=0)]
    for info in infos:
        s = info.identifiable_feature_slice
        blocks.append(TermBlock(
            label=getattr(info.basis, "label", f"s_{len(blocks)}"),
            start=s.start + 1,
            stop=s.stop,  # slice.stop is exclusive -> +1 -1 cancels
        ))
    return blocks

Written by Claude Code documenting the motivation and decisions behind dropping the all-zero columns.

# Apply the empty-column mask upstream of the penalty handler

A basis built over a full covariate range can produce design columns that no
data activate, for example a 2-D spline over a square arena that the animal
only partly visits. Those columns enlarge the solves and can affect the fit
and smoothing-parameter selection through penalty coupling to active columns.
REML includes a penalty log-determinant. GCV has no such term and instead uses
residual error and effective degrees of freedom. Empty columns do not by
themselves establish bias in either criterion. The opt-in `drop_empty_columns`
flag on `GAM` removes them to fit a reduced model.

The mask is computed inside `fit` and applied to the design matrix and to the
penalty tensors before `PenaltyHandler` is built. No low-level API changes.

## Considered options

Two rejected alternatives put the mask inside the penalty layer. One carries
it in a per-component callable that replaces `DROP_LAST_COL`. The other adds a
dynamic index array field to the penalty leaves. Both fail on the same fact:
`apply_identifiability` is a `jax.jit` static argument in `_pql_gcv.py` and
`_pql_reml.py`, and it is part of `_group_key` for `vmap` batching. A
data-dependent mask in either place puts fit data into a compilation key.

A `MaskedBasis` wrapper around the nemos basis was also rejected.
`compute_energy_penalty_tensor`, `compute_energy_penalty_factors`, and
`_should_drop_basis_col` all dispatch on nemos types with `isinstance`, which
a wrapper breaks.

## Consequences

At `min_obs = 1`, the mask removes only columns that are zero on every fitting
row, with NaNs counted as zero. The existing identifiability rule is preserved.
For evaluation B-splines on valid rows, the columns form a partition of unity,
and zero-only removal preserves that dependency with the intercept. The last
surviving column is still removed by `DROP_LAST_COL`. Fits with no removed
columns follow the existing path.

At a higher `min_obs`, a dropped column can be nonzero on up to `min_obs - 1`
fitting rows. Removing it can break the partition-of-unity dependency. The
count threshold does not bound the size of the removed values or guarantee
near-collinearity with the intercept. If any removed column has nonzero
observations, evaluation components therefore keep every surviving column
and use `IDENTITY` instead of `DROP_LAST_COL`. If only zero columns are
removed, the existing rule remains. Convolutional bases retain the explicit
`drop_conv_basis_col` setting in either case. The integer form discards
columns supported by few observations, but does not guarantee better
conditioning.

Legacy `PGAM` instead masks first and force-keeps the final column as a
placeholder, so that a later `X[:, :-1]` slice removes it. pgam_jax does not
reproduce the placeholder, and the resulting column counts can differ. For
zero-only masking of an evaluation component, if the original final column is
empty and k other columns survive, legacy keeps k columns after removing the
placeholder. pgam_jax removes the last survivor and keeps k - 1. If the
original final column survives, both procedures keep k - 1 of k survivors.

A tensor-product component with any removed column currently leaves the
Kronecker fast path. Arbitrary masks generally break Kronecker-sum structure,
but masks whose retained indices form a Cartesian product preserve it through
restrictions of the individual factors. The implementation does not detect
that special case. A masked tensor-product component falls back to
`PenaltyHandler.add`, which
routes to `_GeneralPenalty`. Components with no empty columns keep the fast
path. Two optimizations are deliberately deferred: a rank-k Schur correction
that would keep the Kronecker structure under masking, and keeping the
Kronecker path under `pql_gcv`, which needs no log-determinant.

The null-space term of a masked component is rebuilt from the reduced
penalty, not carried over from the full one. Restricting a positive-semidefinite
penalty to retained coordinates leaves a null direction only if a full-space
null vector is zero on every removed coordinate. Masking can therefore make
the reduced penalty full rank, but does not always do so. For example,
removing one coordinate from a penalty with a two-dimensional null space
leaves at least one null direction.

A sliced full-space null-space projector generally remains nonzero and still
penalizes retained coefficients. It need not target the reduced energy
penalty's null space, so carrying it over defines a different penalty. In a
2-D island experiment that carried it over, its log smoothing parameter went
from 9.9 to 54.9, the effective degrees of freedom rose from 8.2 to 13.0, and
the log-likelihood got worse. These observations do not establish that the
parameter has nothing to penalize or must tend to infinity.

`compute_energy_penalty_tensor` takes the mask and slices before measuring the
null space. When null-space penalization is enabled, it adds a term only if
the reduced penalty still has a null space. A masked component has one fewer
smoothing parameter when masking eliminates a null-space term present in the
unmasked component. Otherwise, the number need not change.

Because `coef_` and `cov_beta_` are reduced, `test_smooth_significance` and
`concurvity` answer a question restricted to the observed region. They warn
when a mask is active.

Dropping empty columns is not only a computation saving. It also changes the
model. In an unmasked fit the coefficients of empty columns are free, and the
smoothness penalty couples them to their neighbors, so they take the values
that make the penalty smallest. Masking forces them to zero instead. The two
fits need not agree closely. On the 2-D island fit the
predicted means differ by about 3 percent at most, and the effective degrees of
freedom fall slightly, from 8.19 to 8.06 under `pql_reml`.

Speed moves in both directions. Measured on that fit, `laplace_reml` went from
25.0 s to 5.4 s, while `pql_gcv` went from 3.2 s to 21.8 s, because the general
path eigendecomposes the penalty at every smoothing parameter while the
Kronecker path does not. Treat masking as a choice of reduced model, with
performance gains depending on the data and fitting method.

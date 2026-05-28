---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Stable computation of `sqrt_penalty` for GCV and REML scores

## Background

Both the GCV score (PQL) and the REML score (PQL and Laplace) depend on the
singular value decomposition of the stacked matrix

$$
A = \begin{bmatrix} R \\ B \end{bmatrix}
$$

where

- $R$ is the upper-triangular QR factor of $\sqrt{W}\, X$, so $R^\top R = X^\top W X = \phi H$
- $B$ is the **square root of the penalty** satisfying $B^\top B = S_\lambda = \sum_k \lambda_k S_k$ — **no $1/\phi$**

From $A = U \Sigma V^\top$:

$$
A^\top A = V \Sigma^2 V^\top = X^\top W X + S_\lambda = \phi\,H + S_\lambda
$$

The $1/\phi$ is **not** folded into $B$; it is recovered explicitly in each derived quantity:

| Quantity | Formula |
|---|---|
| $\log \lvert H + S_\lambda \rvert$ | $2 \sum_k \log \sigma_k - r \log \phi$ |
| $(H + S_\lambda)^{-1}$ | $\phi \, V \,\mathrm{diag}(\sigma^{-2})\, V^\top$ |
| $H + S_\lambda$ | $V \,\mathrm{diag}(\sigma^2)\, V^\top / \phi$ |
| Influence matrix $A_\gamma$ | $U_1 U_1^\top$ ($U_1$ = first $n$ rows of $U$) |
| $\operatorname{tr}(A_\gamma)$ | $\| U_1 \|_F^2$ |

**The correctness and numerical stability of every score that appears in
fitting a GAM depends entirely on having an accurate $B$.**

---

## The problem

The current production path is:

```python
S_sum = compute_weighted_penalty(S_tensor, rho)   # naive float64 sum
B     = symmetric_sqrt(S_sum)                      # eigh, then mask small eigs
```

Two compounding issues:

1. **Masking threshold uses `jnp.finfo(float).eps`** (which resolves to float32 eps
   ≈ 1.2 × 10⁻⁷ unless the user has enabled float64).
   When $\lambda_1 \gg \lambda_2$, the threshold $\varepsilon \cdot \sigma_{\max}$
   exceeds the valid small eigenvalues of $S_\lambda$, masking them to zero and
   producing a wrong log-determinant.  This bug appears in both `_symmetric_sqrt_jax`
   and `FLOAT_EPS` in `_pql_gcv.py`.

2. **Naive float64 sum loses small contributions** when
   $\lambda_1 / \lambda_2 \gtrsim 1/\varepsilon_{\mathrm{f64}} \approx 4.5 \times 10^{15}$
   ($\rho_1 - \rho_2 \gtrsim 35$).  At that point the $\lambda_2 S_2$ elements are
   simply lost in the float64 addition.

---

## Four stable approaches, by penalty structure

The appropriate method depends on the algebraic structure of the penalties
within a given smooth block ($S_\lambda = \sum_{k=1}^M \lambda_k S_k$,
all matrices of size $q \times q$).

### Case 1 — Single penalty ($M = 1$)

No sum is needed. $S_\lambda = \lambda_1 S_1$.

**sqrt:** $B = \sqrt{\lambda_1}\, B_1$ where $B_1^\top B_1 = S_1$.

**log-det:** $\log\lvert S_\lambda \rvert_+ = r_1 \log\lambda_1 + \log\lvert S_1 \rvert_+$

*Arises for:* any smooth with a single penalty matrix (e.g. `penalty_type="EqSpaced"`, ridge).

### Case 2 — Orthogonal penalties ($M = 2$, complementary projectors)

Applies when $S_1 S_2 = 0$ and $\operatorname{range}(S_1) \perp \operatorname{range}(S_2)$.
The canonical instance is the **derivative + null-space** pair for 1D B-spline smooths:
$S_\mathrm{der}$ penalizes curvature, $S_\mathrm{null}$ regularizes the null space, and the
two ranges are complementary.

**sqrt:** Stack per-component roots — no sum needed:

$$
B = \begin{bmatrix} \sqrt{\lambda_1}\, B_1 \\ \sqrt{\lambda_2}\, B_2 \end{bmatrix},
\qquad B^\top B = \lambda_1 S_1 + \lambda_2 S_2 \quad \checkmark
$$

This is exact at any $\lambda_1, \lambda_2$ because the ranges are orthogonal.

**log-det:** $\log\lvert S_\lambda \rvert_+ = r_1 \log\lambda_1 + \log\lvert S_1 \rvert_+ + r_2 \log\lambda_2 + \log\lvert S_2 \rvert_+$

### Case 3 — Kronecker-sum structure ($M = 2$, tensor-product smooth)

Applies to 2D (or higher) tensor-product smooths:
$S_1 = S_x \otimes I_y$, $S_2 = I_x \otimes S_y$.

Eigenvalues of $S_\lambda$ factor as:

$$
\mu_{ij} = \lambda_1\, d^x_i + \lambda_2\, d^y_j
$$

where $d^x_i, d^y_j$ are the eigenvalues of $S_x, S_y$ individually.
These are computable at any $\lambda$ ratio without forming the large sum.

**sqrt:**

$$
B = \operatorname{diag}\!\left(\sqrt{\mu_{ij}}\right)_{\mu_{ij}>0} (U_x \otimes U_y)^\top
$$

**log-det:** $\log\lvert S_\lambda \rvert_+ = \sum_{(i,j):\, \mu_{ij}>0} \log(\mu_{ij})$

### Case 4 — General overlapping penalties (App.B rotation)

No special structure; the ranges of $S_k$ overlap and neither orthogonality
nor Kronecker factorization applies.

The Wood (2011) Appendix B algorithm (`transform_slam`) constructs a
cumulative orthogonal rotation $Q$ such that in the rotated basis

$$
S_\lambda^{\mathrm{rot}} = Q^\top S_\lambda\, Q
  = \sum_k \lambda_k S_k^{\mathrm{out}}
$$

is block-diagonal by scale, making the sum numerically stable.

**sqrt:**

$$
B_{\mathrm{rot}} = \operatorname{symmetric\_sqrt}\!\left(S_\lambda^{\mathrm{rot}}\right)
\;\Longrightarrow\;
B_{\mathrm{orig}} = B_{\mathrm{rot}}\, Q^\top
$$

because $B_{\mathrm{orig}}^\top B_{\mathrm{orig}}
  = Q B_{\mathrm{rot}}^\top B_{\mathrm{rot}} Q^\top
  = Q S_\lambda^{\mathrm{rot}} Q^\top = S_\lambda \;\checkmark$

**log-det:** already handled by `log_det_slam` / `transform_slam` as currently implemented.

**Current implementation:** `transform_slam_with_Q` tracks the accumulated
rotation matrix $Q$ alongside `S_i_out`, so the sqrt can be rotated back to
the original basis after the stable App.B transform.

---

## Implementation

### `PenaltyHandler` (`_penalty_handler.py`)

The stable approaches above are implemented through `PenaltyHandler`, which
detects structure at construction time, pre-caches eigendecompositions, and
exposes two pure callables suitable for `jax.jit`.

#### Method enum

```python
class SqrtMethod(enum.Enum):
    SINGLE             = 0   # M=1, no null-space penalty
    SINGLE_WITH_NULL   = 1   # M=1, plus a scalar λ on the null space
    KRONECKER          = 2   # Kronecker-sum, no null-space penalty
    KRONECKER_WITH_NULL= 3   # Kronecker-sum, plus a scalar λ on the null space
    GENERAL            = 4   # overlapping multi-penalty → Wood App.B rotation
```

Mapping to the four cases described above:

| Document case | `SqrtMethod` variant(s) |
|---|---|
| Case 1 — single penalty | `SINGLE` |
| Case 2 — orthogonal (null-space pair) | `SINGLE_WITH_NULL` (1D); `GENERAL` (M>2) |
| Case 3 — Kronecker-sum | `KRONECKER`, `KRONECKER_WITH_NULL` |
| Case 4 — general overlapping | `GENERAL` |

Note: the document's Case 2 "stacked-row" construction is replaced by
`SINGLE_WITH_NULL`, which avoids forming the stacked matrix at all — it
reuses the same eigenvectors as SINGLE and applies `sqrt(lam[1])` uniformly
to the null-space modes.

#### Construction: `add()` and `add_kron()`

`ph.add(S_tensor, penalize_null_space, identifiability_fn)` accepts:

- `(q, q)` or `(1, q, q)` → SINGLE or SINGLE_WITH_NULL
- `(k, q, q)` with k > 1 → GENERAL

`ph.add_kron(factor_list, ...)` accepts a list of 1D factor matrices and
selects KRONECKER or KRONECKER_WITH_NULL. The Kronecker-sum eigenvalues are
never materialized as a full `(prod_q, prod_q)` matrix.

All eigendecompositions are computed eagerly at `add()` time and stored in
`_cache`.  Nothing shape-dynamic runs inside `jit`.

#### `build()` — pure callables for JIT

```python
compute_sqrt, compute_log_det_and_grad = ph.build()
```

`build()` snapshots the handler state (caches, method tags, group
membership) into two pure closures.  Smooths that share the same method,
shape, and identifiability function are batched with `jax.vmap`
(pre-constructed at `build()` time to avoid late-binding issues).

- `compute_sqrt(rhos) → (total_rows, n_params)` block-diagonal B matrix
- `compute_log_det_and_grad(rhos) → (list[scalar], list[array])`
  where list element `i` is log|S_lam_i|_+ and its gradient w.r.t. rho_i

Both callables close only over JAX arrays (the pre-stacked caches) and
vmapped functions, so they are `jax.jit`-safe without any static args.

#### Restricted-basis log-det and Schur correction

Identifiability constraints remove the unidentifiable constant from each
smooth.  The default constraint, `_drop_last_col`, drops the last column of
the design matrix; `compute_sqrt` mirrors this by dropping the last column of
the square-root factor, so

$$
B^\top B = S_\lambda[:-1, :-1].
$$

The score path therefore needs $\log|S_\lambda[:-1, :-1]|_+$, not the
full-basis $\log|S_\lambda|_+$ that the cached eigendecomposition gives
directly.  Recomputing an eigendecomposition of the restricted matrix at
every `rho` would defeat the point of caching.

When $S_\lambda$ is invertible, cofactor expansion along the last row gives
the standard identity

$$
\det(S_\lambda[:-1, :-1]) = \det(S_\lambda)\,(S_\lambda^{-1})[-1, -1],
\qquad
\log\det(S_\lambda[:-1, :-1]) = \log\det(S_\lambda) + \log(S_\lambda^{-1})[-1, -1].
$$

Both terms on the right are closed-form in `rho` once the eigendecomposition
of $S_\lambda$ is cached, so the restricted log-det costs no extra
factorization.

`PenaltyHandler` uses this Schur correction for the two null-penalized
methods, where the explicit null-space penalty makes $S_\lambda$ invertible.
For the other methods, $S_\lambda$ is singular and Schur does not apply;
they precompute a restricted eigendecomposition at `add()` time instead.

**`SINGLE_WITH_NULL`**

Let $S = U \operatorname{diag}(d) U^\top$, with positive eigenvalues
$d_i > 0$ for the penalized range and zero eigenvalues for the null space.
The full-basis penalty adds a scalar penalty on the null space,

$$
S_\lambda = \lambda_0 S + \lambda_1 P_0,
\qquad
P_0 = \sum_{d_i = 0} U[:, i]\, U[:, i]^\top,
$$

where $P_0$ is the orthogonal projector onto the null space of $S$.  Because
$S$ and $P_0$ share eigenvectors, $S_\lambda$ is diagonal in the basis $U$
with eigenvalues $\lambda_0 d_i$ on the range and $\lambda_1$ on the null
space.  In particular $S_\lambda$ is invertible for $\lambda_1 > 0$.

The full-basis log-determinant is

$$
\log\det(S_\lambda) = r\,\rho_0 + \log|S|_+ + r_0\,\rho_1,
$$

where $r$ is the rank of $S$, $r_0$ is the null-space dimension, and
$\lambda_j = \exp(\rho_j)$.  The inverse diagonal entry needed for Schur is

$$
(S_\lambda^{-1})[-1, -1]
  = \sum_i \frac{U[-1, i]^2}{\lambda_0 d_i + \lambda_1\,[d_i = 0]}
  = \alpha\, e^{-\rho_0} + \beta\, e^{-\rho_1},
$$

with

$$
\alpha = \sum_{d_i > 0} \frac{U[-1, i]^2}{d_i},
\qquad
\beta = \sum_{d_i = 0} U[-1, i]^2.
$$

`alpha` and `beta` are computed at `add()` time and cached as `log_alpha`
and `log_beta` (via `_safe_log`, which returns `-inf` when the underlying
sum is exactly zero; this is harmless because `logsumexp` treats `-inf` as a
dropped term).  The run-time Schur term and its gradient are

$$
c = \log\!\left(\alpha\, e^{-\rho_0} + \beta\, e^{-\rho_1}\right)
  = \operatorname{logsumexp}\!\left(\log\alpha - \rho_0,\ \log\beta - \rho_1\right),
$$

$$
\nabla_\rho c = -\operatorname{softmax}
  \left(\log\alpha - \rho_0,\ \log\beta - \rho_1\right).
$$

The function returns `log_det_full + c` and `grad_full - weights`, where
`weights` is the softmax vector above.

**`KRONECKER_WITH_NULL`**

For a tensor-product smooth with an explicit null-space penalty, the
full-basis penalty is diagonal in the Kronecker eigenbasis
$U_\otimes = U_1 \otimes \cdots \otimes U_M$.  Index the basis by a
multi-index $m = (m_1, \ldots, m_M)$, and write $d^j_{m_j}$ for the $m_j$-th
eigenvalue of factor $j$.  Call a multi-index a **positive mode** if at least
one $d^j_{m_j} > 0$, and a **null mode** if $d^j_{m_j} = 0$ for every $j$
(the joint null space of all factors).  The eigenvalue of $S_\lambda$ at
mode $m$ is then

$$
d_m =
\begin{cases}
\sum_j \lambda_j\, d^j_{m_j}, & m \text{ is a positive mode},\\
\lambda_{\mathrm{null}}, & m \text{ is a null mode}.
\end{cases}
$$

$S_\lambda$ is invertible for $\lambda_{\mathrm{null}} > 0$, so Schur applies.
The cache stores the squared last row of $U_\otimes$,

$$
w_m = U_\otimes[-1, m]^2,
$$

reshaped to the factor dimensions so that per-factor sums can broadcast
along one axis (same trick as `_kron_log_det_factor_grads`).  The Schur
term is

$$
D = (S_\lambda^{-1})[-1, -1] = \sum_m \frac{w_m}{d_m},
\qquad
c = \log D.
$$

Differentiating $c = \log D$ uses $\partial d_m / \partial \rho_j = \lambda_j\, d^j_{m_j}$
on positive modes (and zero on null modes), giving, for each non-null
factor $j$,

$$
\frac{\partial c}{\partial \rho_j}
  = -\frac{1}{D}
    \sum_{m\ \text{positive}}
    \frac{w_m\,\lambda_j\,d^j_{m_j}}{d_m^2}.
$$

On null modes $d_m = \lambda_{\mathrm{null}}$ and $\partial d_m / \partial \rho_{\mathrm{null}} = \lambda_{\mathrm{null}}$,
so

$$
\frac{\partial c}{\partial \rho_{\mathrm{null}}}
  = -\frac{\beta}{\lambda_{\mathrm{null}} D},
\qquad
\beta = \sum_{m\ \text{null}} w_m.
$$

The returned value and gradient are the full-basis Kronecker log-det and
gradient plus these Schur corrections.

**Methods without a Schur path**

When $S_\lambda$ is singular the Schur identity does not apply: $S_\lambda^{-1}$
does not exist, and $\det(S_\lambda[:-1, :-1])$ has to be obtained some other
way.

- `SINGLE` (no null penalty): $S$ is singular by assumption.  At `add()` time
  the handler does a one-off `eigh` of `S[:-1, :-1]` and caches its rank and
  pseudo-log-det.  At run-time the restricted log-det is just
  `rank_r * rho + log_det_S_r`, with gradient `rank_r`.
- `GENERAL`: $\sum_k \lambda_k S_k$ may be singular (any joint null direction
  of the $S_k$ remains unpenalized).  At `add()` time the full-rank precompute
  is repeated on `S_tensor[:, :-1, :-1]`, producing `full_rank_S_r`, and the
  run-time path routes through the usual `transform_slam` /
  `log_det_and_grad_slam` on this restricted tensor.  `compute_sqrt` is
  unaffected.
- `KRONECKER` (no null penalty): raises `NotImplementedError` under
  `_drop_last_col`.  Two separate obstructions hit at once.  First,
  restricting $\sum_k \lambda_k S_k$ to `[:-1, :-1]` destroys the
  Kronecker-sum factorization, so the cheap closed-form path via
  `_kron_log_det_factor_grads` no longer applies.  Second, $S_\lambda$ is
  still singular, so Schur cannot rescue it either.  Falling back to a
  generic `eigh` of the restricted matrix would defeat the point of the
  Kronecker path.  In practice, callers should opt into a null-space penalty
  via `add_kron(..., penalize_null_space=True)`, which routes to
  `KRONECKER_WITH_NULL` and is supported.

#### Integration with GCV / REML

`_pql_gcv._compute_gcv_and_states` and `_pql_reml._compute_reml_and_states`
now receive `compute_sqrt` (and `compute_log_det_and_grad` for REML) as
static callable arguments.  `tree_compute_sqrt_penalty` is no longer called
on the hot path.

#### `transform_slam_with_Q` (`_slam_compute.py`)

The previously noted gap — `transform_slam` not tracking the accumulated
rotation matrix Q — has been fixed.  `transform_slam_with_Q` returns
`(S_i_out, Q)` and is used by `PenaltyHandler._sqrt` for the GENERAL case
to rotate the sqrt back to the original basis:

```python
B_rot = sqrt_d[:, None] * U.T          # sqrt in rotated basis
return (B_rot @ Q.T) @ id_fn(U_keep.T) # back to original basis
```

### Precision

The masking thresholds use `jnp.finfo(float).eps`, which resolves to the
*active* JAX float precision.  Any meaningful use of multi-penalty smooths
requires float64.  The library should warn at fit time when float64 is not
active and multi-penalty blocks are present.

---

## Diagnostic: Case 3 failure and Kronecker-stable fix

```{code-cell} ipython3
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pgam_jax.penalty_utils import compute_weighted_penalty, symmetric_sqrt
```

### Build a 2D tensor-product 2nd-derivative penalty

```{code-cell} ipython3
nx, ny = 8, 7
p = nx * ny


def _diff2(n):
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0; D[i, i + 1] = -2.0; D[i, i + 2] = 1.0
    return D


Sx = _diff2(nx).T @ _diff2(nx)
Sy = _diff2(ny).T @ _diff2(ny)

S1 = np.kron(Sx, np.eye(ny))
S2 = np.kron(np.eye(nx), Sy)
S_tensor = jnp.stack([jnp.array(S1), jnp.array(S2)])


def _clean(eigs, rel_tol=np.finfo(float).eps ** 0.5):
    """Zero out numerical null-space modes before they contaminate large-λ products."""
    thresh = np.abs(eigs).max() * rel_tol
    return np.maximum(np.where(np.abs(eigs) < thresh, 0.0, eigs), 0.0)


dx_eig, Ux = np.linalg.eigh(Sx)
dy_eig, Uy = np.linalg.eigh(Sy)
dx_eig = _clean(dx_eig)
dy_eig = _clean(dy_eig)
U_kron = np.kron(Ux, Uy)
R_id   = jnp.eye(p)
```

### Exact ground truth from Kronecker structure

For $H = I$:

$$
\log\lvert H + S_\lambda\rvert = \sum_{i,j} \log(1 + \lambda_1 d^x_i + \lambda_2 d^y_j)
$$

```{code-cell} ipython3
def truth(lam1, lam2):
    eigs = np.add.outer(lam1 * dx_eig, lam2 * dy_eig).ravel()
    return float(np.sum(np.log(1.0 + eigs)))
```

### Method A: naive (old path, replaced by `PenaltyHandler`)

Both methods compute $\log|H + S_\lambda|$ via SVD of the stacked matrix
$A = \begin{bmatrix} R \\ B \end{bmatrix}$, with $R = I$ (i.e. $H = I$):

$$\log|A^\top A| = 2\sum_k \log \sigma_k$$

```{code-cell} ipython3
def _logdet_from_B(B):
    """log|I + B^T B| via SVD of [I; B] — the same path used in production."""
    sv = jnp.linalg.svd(jnp.vstack((R_id, B)), compute_uv=False)
    return float(2.0 * jnp.sum(jnp.log(sv)))

def logdet_naive(lam1, lam2):
    rho = jnp.log(jnp.array([lam1, lam2]))
    B = symmetric_sqrt(compute_weighted_penalty(S_tensor, rho, positive_mon_func=jnp.exp))
    return _logdet_from_B(B)
```

### Method B: Kronecker-stable (Case 3)

Build $B$ from the 1D eigenspectra; never form $\lambda_1 S_1 + \lambda_2 S_2$.

```{code-cell} ipython3
def logdet_kron_stable(lam1, lam2):
    eigs = np.add.outer(lam1 * dx_eig, lam2 * dy_eig).ravel()
    pos  = eigs > 0
    B    = jnp.array((U_kron[:, pos] * np.sqrt(eigs[pos])).T)
    return _logdet_from_B(B)
```

### Sweep $\rho_1$, fix $\lambda_2 = 1$

```{code-cell} ipython3
import pandas as pd

rho1_vals = np.arange(0.0, 46.0, 5.0)
lam2      = 1.0

rows = []
for rho1 in rho1_vals:
    lam1 = np.exp(rho1)
    t = truth(lam1, lam2)
    a = logdet_naive(lam1, lam2)
    b = logdet_kron_stable(lam1, lam2)
    rows.append(dict(rho1=int(rho1), truth=t, naive=a, kron_stable=b,
                     err_A=abs(a - t), err_B=abs(b - t)))

df = pd.DataFrame(rows).set_index("rho1")
df.style.format("{:.4e}", subset=["err_A", "err_B"]).format("{:.4f}", subset=["truth", "naive", "kron_stable"])
```

**Reading the table:**

- At moderate $\rho_1$ (≤ 20) both methods agree.
- From $\rho_1 \approx 25$ the naive method starts deviating: the masking
  threshold $\varepsilon \cdot \sigma_{\max}$ exceeds the valid small singular
  values that come from $S_2$ in the null space of $S_1$.
- Method B (Kronecker-stable) stays accurate to $\sim 10^{-6}$ across the
  whole range because it never forms the ill-conditioned sum.

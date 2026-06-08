---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
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

See **Restricted-basis log-det and Schur correction** for corrections to these formulas when column-dropping for identifiability is applied.

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

## Restricted-basis log-det and Schur correction

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

### `SINGLE_WITH_NULL`

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
  = \sum_{d_i > 0} \frac{U[-1, i]^2}{\lambda_0 d_i} + \sum_{d_i = 0} \frac{U[-1, i]^2}{\lambda_1}
  = \alpha\, e^{-\rho_0} + \beta\, e^{-\rho_1},
$$

with

$$
\alpha = \sum_{d_i > 0} \frac{U[-1, i]^2}{d_i},
\qquad
\beta = \sum_{d_i = 0} U[-1, i]^2.
$$

:::{admonition} Proof
:class: dropdown

Order the eigenpairs so that $d_1, \ldots, d_r > 0$ (the range) and
$d_{r+1} = \cdots = d_n = 0$ (the null space).  We are free to choose this
ordering: reordering the columns of $U$ is an orthogonal permutation, and since
$U$ is otherwise unspecified it leaves $S$, $S_\lambda$, and $(S_\lambda^{-1})[-1,-1]$
unchanged.  With this ordering $i \le r \Leftrightarrow d_i > 0$, and

$$
\begin{aligned}
S_{\lambda} &= 
U \cdot \begin{bmatrix}
\lambda_0 \cdot d_1 & 0                   & \cdots & 0                   & 0           & \cdots & 0           \\
0                   & \lambda_0 \cdot d_2 &        & \vdots              & \vdots      &        & \vdots      \\
\vdots              &                     & \ddots & 0                   &             &        &             \\
0                   & \cdots              & 0      & \lambda_0 \cdot d_r & 0           & \cdots & 0           \\
0                   & \cdots              &        & 0                   & \lambda_1 &        & 0           \\
\vdots              &        &              & \vdots              &             & \ddots &             \\
0                   & \cdots              &        & 0                   & 0           &        & \lambda_1
\end{bmatrix} \cdot U^{\top} \\
&= U \cdot D \cdot U^{\top}
\end{aligned}
$$

So $S_{\lambda}^{-1} =U \cdot D^{-1} \cdot U^{\top} $, and $D^{-1}_{ii} = \begin{cases} \frac{1}{\lambda_0 \cdot d_i} & i \le r \\  \frac{1}{\lambda_1} & \text{otherwise}\end{cases}$.

If we compute the element we need ($[S_{\lambda}^{-1}]_{-1-1}$), we obtain,

$$
\begin{aligned}
\left[S_{\lambda}^{-1}\right]_{-1-1} &= \sum_j u_{-1j} \sum_k D^{-1}_{jk} u_{-1k} \overset{\mathrm{diag}}{=} \sum_j u_{-1j} D^{-1}_{jj} u_{-1j} \\
&= \sum_{j=1}^r \frac{u_{-1j}^2}{\lambda_0 d_j} + \sum_{j=r+1}^n \frac{u_{-1j}^2}{\lambda_1}
\end{aligned}
$$

Which is exactly the identity above.

:::

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

### `KRONECKER_WITH_NULL`

For a tensor-product smooth with an explicit null-space penalty, the
full-basis penalty is diagonal in the Kronecker eigenbasis
$U_\otimes = U_1 \otimes \cdots \otimes U_M$.  Index the basis by a
amulti-index $\mathbf{m} = (m_1, \ldots, m_M)$, and write $d^j_{m_j}$ for the
$m_j$-th eigenvalue of factor $j$.  Call a multi-index a **positive mode** if at
least one $d^j_{m_j} > 0$, and a **null mode** if $d^j_{m_j} = 0$ for every $j$
(the joint null space of all factors).  The eigenvalue of $S_\lambda$ at
mode $\mathbf{m}$ is then

$$
d_{\mathbf{m}} =
\begin{cases}
\sum_j \lambda_j\, d^j_{m_j}, & \mathbf{m} \text{ is a positive mode},\\
\lambda_{\mathrm{null}}, & \mathbf{m} \text{ is a null mode}.
\end{cases}
$$

:::{admonition} Proof — Kronecker-sum eigendecomposition
:class: dropdown

Write each factor's eigendecomposition as
$S^{(j)} = U_j \operatorname{diag}(d^j) U_j^\top$ and each identity block as
$I_{q_k} = U_k\, I\, U_k^\top$ (valid because $U_k$ is orthogonal).  The
$M$-fold mixed-product identity

$$
(A_1 \otimes \cdots \otimes A_M)(B_1 \otimes \cdots \otimes B_M)
  = (A_1 B_1) \otimes \cdots \otimes (A_M B_M)
$$

(induction on the two-factor rule $(A \otimes B)(C \otimes D) = AC \otimes BD$)
pulls the orthogonal factors out of the slot-$j$ penalty
$S_j = I \otimes \cdots \otimes S^{(j)} \otimes \cdots \otimes I$:

$$
S_j
  = U_\otimes\,
    \bigl(I \otimes \cdots \otimes \operatorname{diag}(d^j) \otimes \cdots \otimes I\bigr)\,
    U_\otimes^\top,
\qquad U_\otimes = U_1 \otimes \cdots \otimes U_M.
$$

The inner matrix is a Kronecker product of diagonal matrices, hence itself
diagonal.  Make the index bookkeeping explicit, since that is what carries the
argument.  For $A \in \mathbb{R}^{n_A \times n_A}$ and
$B \in \mathbb{R}^{n_B \times n_B}$, a flat row index
$\mathbf{r} \in \{0, \ldots, n_A n_B - 1\}$ decomposes *uniquely* by Euclidean
division by $n_B$ as $\mathbf{r} = r_1 n_B + r_2$ with $r_1 \in \{0, \ldots, n_A - 1\}$
and $r_2 \in \{0, \ldots, n_B - 1\}$; write this bijection as
$\mathbf{r} \mapsto (r_1, r_2)$, and likewise $\mathbf{c} \mapsto (c_1, c_2)$ for
columns.  In this notation the Kronecker product is *defined* entrywise by

$$
(A \otimes B)_{\mathbf{r}\mathbf{c}} = A_{r_1 c_1}\, B_{r_2 c_2}.
$$

For diagonal $A = \operatorname{diag}(a)$, $B = \operatorname{diag}(b)$ this reads

$$
(A \otimes B)_{\mathbf{r}\mathbf{c}}
  = a_{r_1}\,\delta_{r_1 c_1}\; b_{r_2}\,\delta_{r_2 c_2}.
$$

If $\mathbf{r} \neq \mathbf{c}$ then $(r_1, r_2) \neq (c_1, c_2)$ — because
$\mathbf{r} \mapsto (r_1, r_2)$ is a bijection — so at least one of the two
Kronecker deltas vanishes and the entry is $0$.  If $\mathbf{r} = \mathbf{c}$
then $r_1 = c_1$, $r_2 = c_2$ and the entry is $a_{r_1} b_{r_2}$.  Hence
$A \otimes B$ is diagonal, carrying $a_{r_1} b_{r_2}$ at flat index
$(r_1 n_B + r_2,\ r_1 n_B + r_2)$.  Iterating over the $M$ factors —
a flat index $\mathbf{m} \leftrightarrow (m_1, \ldots, m_M)$ now corresponding in
mixed radix to its $M$ components — the product
$D_1 \otimes \cdots \otimes D_M$ is diagonal with entry $\prod_k (D_k)_{m_k m_k}$.
In our case every factor is the identity except slot $j$, so this collapses to
$1 \cdots d^j_{m_j} \cdots 1 = d^j_{m_j}$.  Summing with weights $\lambda_j$,

$$
\sum_j \lambda_j S_j
  = U_\otimes\,
    \operatorname{diag}\!\Bigl(\textstyle\sum_j \lambda_j d^j_{m_j}\Bigr)\,
    U_\otimes^\top,
$$

which vanishes exactly on the joint null modes ($d^j_{m_j} = 0$ for all $j$).
The null-space penalty is the orthogonal projector onto those modes,
$P_0 = \sum_{\mathbf{m}\ \text{null}} v_{\mathbf{m}} v_{\mathbf{m}}^\top$ with
$v_{\mathbf{m}} = U_\otimes[:, \mathbf{m}]$; since the $v_{\mathbf{m}}$ are
columns of the orthogonal $U_\otimes$,
$P_0 = U_\otimes \operatorname{diag}(\mathbf{1}[\mathbf{m}\ \text{null}]) U_\otimes^\top$.
Adding $\lambda_{\mathrm{null}} P_0$ sets the eigenvalue to
$\lambda_{\mathrm{null}}$ on the null modes and leaves the positive modes
untouched, giving $d_{\mathbf{m}}$ as stated.  $U_\otimes$ is orthogonal (a Kronecker
product of orthogonals), so this is a genuine eigendecomposition.

:::

$S_\lambda$ is invertible for $\lambda_{\mathrm{null}} > 0$, so Schur applies.
The cache stores the squared last row of $U_\otimes$,

$$
w_{\mathbf{m}} = U_\otimes[-1, \mathbf{m}]^2,
$$

reshaped to the factor dimensions so that per-factor sums can broadcast
along one axis (same trick as `_kron_log_det_factor_grads`).  The Schur
term is

$$
D = (S_\lambda^{-1})[-1, -1] = \sum_{\mathbf{m}} \frac{w_{\mathbf{m}}}{d_{\mathbf{m}}},
\qquad
c = \log D.
$$

:::{admonition} Proof
:class: dropdown

By the eigendecomposition above $S_\lambda = U_\otimes \operatorname{diag}(d_{\mathbf{m}}) U_\otimes^\top$
with every $d_{\mathbf{m}} > 0$, so $S_\lambda^{-1} = U_\otimes \operatorname{diag}(d_{\mathbf{m}}^{-1}) U_\otimes^\top$.
Reading off the last diagonal entry (the same step as in the
`SINGLE_WITH_NULL` proof, with the scalar index $i$ replaced by the
multi-index $\mathbf{m}$),

$$
\left[S_\lambda^{-1}\right]_{-1,-1}
  = \sum_{\mathbf{m}} U_\otimes[-1, \mathbf{m}] \, d_{\mathbf{m}}^{-1} \, U_\otimes[-1, \mathbf{m}]
  = \sum_{\mathbf{m}} \frac{U_\otimes[-1, \mathbf{m}]^2}{d_{\mathbf{m}}}
  = \sum_{\mathbf{m}} \frac{w_{\mathbf{m}}}{d_{\mathbf{m}}}.
$$

Splitting the modes into positive ($d_{\mathbf{m}} = \sum_j \lambda_j d^j_{m_j}$) and null
($d_{\mathbf{m}} = \lambda_{\mathrm{null}}$) recovers the two pieces of $D$.

:::

Differentiating $c = \log D$ uses $\partial d_{\mathbf{m}} / \partial \rho_j = \lambda_j\, d^j_{m_j}$
on positive modes (and zero on null modes), giving, for each non-null
factor $j$,

$$
\frac{\partial c}{\partial \rho_j}
  = -\frac{1}{D}
    \sum_{\mathbf{m}\ \text{positive}}
    \frac{w_{\mathbf{m}}\,\lambda_j\,d^j_{m_j}}{d_{\mathbf{m}}^2}.
$$

On null modes $d_{\mathbf{m}} = \lambda_{\mathrm{null}}$ and $\partial d_{\mathbf{m}} / \partial \rho_{\mathrm{null}} = \lambda_{\mathrm{null}}$,
so

$$
\frac{\partial c}{\partial \rho_{\mathrm{null}}}
  = -\frac{\beta}{\lambda_{\mathrm{null}} D},
\qquad
\beta = \sum_{\mathbf{m}\ \text{null}} w_{\mathbf{m}}.
$$

:::{admonition} Proof
:class: dropdown

Differentiate $c = \log D$ with $D = \sum_{\mathbf{m}} w_{\mathbf{m}} / d_{\mathbf{m}}$.
Since $\lambda_j = e^{\rho_j}$ gives $\partial \lambda_j / \partial \rho_j = \lambda_j$,
on a positive mode $d_{\mathbf{m}} = \sum_k \lambda_k d^k_{m_k}$ we have,

$$\partial d_{\mathbf{m}} / \partial \rho_j = \lambda_j d^j_{m_j},$$
and a mode that does not involve factor $j$ contributes nothing.  Applying the chain rule
through $c = \log D$ and then through $D = \sum_{\mathbf{m}} w_{\mathbf{m}} / d_{\mathbf{m}}$,

$$
\frac{\partial c}{\partial \rho_j}
  = \frac{1}{D}\,\frac{\partial D}{\partial \rho_j}
  = \frac{1}{D} \sum_{\mathbf{m}} w_{\mathbf{m}} \Bigl(-\frac{1}{d_{\mathbf{m}}^2}\Bigr)\frac{\partial d_{\mathbf{m}}}{\partial \rho_j}
  = -\frac{1}{D} \sum_{\mathbf{m}\ \text{positive}} \frac{w_{\mathbf{m}}\,\lambda_j\,d^j_{m_j}}{d_{\mathbf{m}}^2}.
$$

On a null mode $d_{\mathbf{m}} = \lambda_{\mathrm{null}}$ and
$\partial d_{\mathbf{m}} / \partial \rho_{\mathrm{null}} = \lambda_{\mathrm{null}}$, so each
null term contributes
$-w_{\mathbf{m}}\,\lambda_{\mathrm{null}} / (D\,\lambda_{\mathrm{null}}^2)
 = -w_{\mathbf{m}} / (\lambda_{\mathrm{null}} D)$.  
Summing over null modes gives
$$\frac{\partial c}{\partial \rho_{\mathrm{null}}} = -\beta / (\lambda_{\mathrm{null}} D),$$

with $\beta = \sum_{\mathbf{m}\ \text{null}} w_{\mathbf{m}}$.

:::

The returned value and gradient are the full-basis Kronecker log-det and
gradient plus these Schur corrections.

### Methods without a Schur path

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

---

## Worked example: 2D tensor-product Schur correction

The KRONECKER and KRONECKER_WITH_NULL math above is written for $M$ factors.
This section walks through the two-factor (2D tensor product) case end to
end, covering both the penalty structure and the Schur correction, to make
the indexing concrete.

Set $S_x \in \mathbb{R}^{q_x \times q_x}$ and $S_y \in \mathbb{R}^{q_y \times q_y}$
as the marginal 1D penalties, with eigendecompositions
$S_x = U_x \operatorname{diag}(d^x) U_x^\top$ and
$S_y = U_y \operatorname{diag}(d^y) U_y^\top$.  The penalty tensor for the
2D smooth has two slices,

$$
S_1 = S_x \otimes I_{q_y}, \qquad S_2 = I_{q_x} \otimes S_y,
$$

and the total penalty is $S_\lambda = \lambda_1 S_1 + \lambda_2 S_2$.

### Why both slices diagonalize in $U_x \otimes U_y$

The goal is to write the eigendecomposition of *both* slices in the *same*
outer basis $U_x \otimes U_y$.  Once we have that, their inner diagonal parts
add directly and the joint spectrum can be read off without ever forming the
big tensor.

Two facts about Kronecker products do all the work.

**Mixed-product identity.**  $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$.
Applying it twice splits a Kronecker product of two triple-products into a
triple-product of Kronecker products:

$$
(A_1 A_2 A_3) \otimes (B_1 B_2 B_3)
  = (A_1 \otimes B_1)\,(A_2 \otimes B_2)\,(A_3 \otimes B_3).
$$

(Check: $(A_1 \otimes B_1)(A_2 \otimes B_2) = (A_1 A_2) \otimes (B_1 B_2)$ by
the mixed-product rule, then right-multiply by $A_3 \otimes B_3$.)

**Transpose distributes.**  $(A \otimes B)^\top = A^\top \otimes B^\top$, so
$U_x^\top \otimes U_y^\top = (U_x \otimes U_y)^\top$.

Now the one trick.  For *any* orthogonal $Q$ we have $Q\,I\,Q^\top = I$, so
the identity block in $S_1 = S_x \otimes I$ can be replaced by the trivial
eigendecomposition $I = U_y\, I\, U_y^\top$ at no cost. Substituting that alongside
$S_x = U_x \operatorname{diag}(d^x) U_x^\top$,

$$
S_1 = S_x \otimes I
    = \bigl(U_x\, \operatorname{diag}(d^x)\, U_x^\top\bigr)
      \otimes
      \bigl(U_y\, I\, U_y^\top\bigr).
$$

Each block is a triple product, so the triple-product identity applies with
$(A_1, A_2, A_3) = (U_x, \operatorname{diag}(d^x), U_x^\top)$ and
$(B_1, B_2, B_3) = (U_y, I, U_y^\top)$.  Using the transpose rule on the last
factor,

$$
S_1
  = (U_x \otimes U_y)\,(\operatorname{diag}(d^x) \otimes I)\,(U_x^\top \otimes U_y^\top)
  = (U_x \otimes U_y)\,(\operatorname{diag}(d^x) \otimes I)\,(U_x \otimes U_y)^\top.
$$

This is a genuine eigendecomposition: $U_x \otimes U_y$ is orthogonal
(a Kronecker product of orthogonals), and $\operatorname{diag}(d^x) \otimes I$
is diagonal, with mode $(i, j)$ carrying eigenvalue $d^x_i$ for every $j$.

Doing the mirror image on $S_2 = I \otimes S_y$ (write the *left* identity as
$I = U_x\, I\, U_x^\top$ and keep the real eigendecomposition of $S_y$ on the
right) lands the same outer basis:

$$
S_2 = I \otimes S_y
    = (U_x \otimes U_y)\,(I \otimes \operatorname{diag}(d^y))\,(U_x \otimes U_y)^\top.
$$

Because both slices share the identical outer factor, the weighted sum
collapses onto its inner diagonal parts:

$$
S_\lambda
  = \lambda_1 S_1 + \lambda_2 S_2
  = (U_x \otimes U_y)\,
    \operatorname{diag}\!\bigl(\lambda_1 d^x_i + \lambda_2 d^y_j\bigr)\,
    (U_x \otimes U_y)^\top.
$$

The takeaway: the full $(q_x q_y) \times (q_x q_y)$ penalty is *jointly*
diagonal in a basis whose columns we can write down from the two small 1D
eigendecompositions, without ever forming the big tensor.

### Modes $(i, j)$

The columns of $U_x \otimes U_y$ are the joint eigenvectors of $S_\lambda$.
By the column rule that $A \otimes B$ has columns $A[:, i] \otimes B[:, j]$,
column $(i, j)$ is

$$
v_{ij} = U_x[:, i] \otimes U_y[:, j] \in \mathbb{R}^{q_x q_y}.
$$

We call $(i, j)$ the "mode" indexing this eigenvector, with eigenvalue

$$
d_{ij} = \lambda_1 d^x_i + \lambda_2 d^y_j.
$$

A mode is **positive** when at least one of $d^x_i, d^y_j$ is nonzero
(curvature already penalizes it).  A mode is **null** when
$d^x_i = d^y_j = 0$; it sits in the joint null space of both marginal
penalties.  For a 2nd-derivative penalty the marginal null space is
$\{1, x\}$ (dimension 2 per axis), so the 2D joint null space has
dimension 4: constants, linears in $x$, linears in $y$, and the bilinear
term $xy$.

### Schur correction step by step

For KRONECKER_WITH_NULL the eigenvalues of $S_\lambda$ in the basis
$U_x \otimes U_y$ are

$$
d_{ij} =
\begin{cases}
\lambda_1 d^x_i + \lambda_2 d^y_j, & (i, j) \text{ is a positive mode},\\
\lambda_{\mathrm{null}},           & (i, j) \text{ is a null mode}.
\end{cases}
$$

$S_\lambda$ is invertible (every $d_{ij} > 0$), so

$$
\log\det(S_\lambda[:-1, :-1])
  = \log\det(S_\lambda) + \log (S_\lambda^{-1})[-1, -1].
$$

The full-basis log-det is the sum of $\log d_{ij}$ over all modes,
computable from the two 1D spectra plus the null-mode count:

$$
\log\det(S_\lambda)
  = \sum_{(i, j)\ \text{positive}} \log\!\bigl(\lambda_1 d^x_i + \lambda_2 d^y_j\bigr)
  + n_{\mathrm{null}}\, \rho_{\mathrm{null}}.
$$

For the inverse diagonal, use the spectral identity
$M[-1, -1] = \sum_k V[-1, k]^2\, e_k$ when $M = V \operatorname{diag}(e) V^\top$,
applied to $S_\lambda^{-1}$ in the basis $V = U_x \otimes U_y$ with
$e_k = 1 / d_{ij}$.  The last row of a Kronecker product factors,

$$
(U_x \otimes U_y)[-1,\, (i, j)] = U_x[-1, i]\, U_y[-1, j],
$$

so the squared last row factors too.  Define

$$
w_{ij} = U_x[-1, i]^2\, U_y[-1, j]^2.
$$

This is the cache entry `u_kron_last_sq`, computed once at `add()` time
(it does not depend on the $\lambda$s).  Then

$$
D = (S_\lambda^{-1})[-1, -1]
  = \sum_{(i, j)\ \text{positive}}
      \frac{w_{ij}}{\lambda_1 d^x_i + \lambda_2 d^y_j}
  + \sum_{(i, j)\ \text{null}}
      \frac{w_{ij}}{\lambda_{\mathrm{null}}},
\qquad
c = \log D,
$$

and the restricted log-det is $\log\det(S_\lambda) + c$.

The gradient pieces, with $\rho_k = \log \lambda_k$, use
$\partial d_{ij} / \partial \rho_1 = \lambda_1 d^x_i$ on positive modes,
$\partial d_{ij} / \partial \rho_2 = \lambda_2 d^y_j$ on positive modes,
and $\partial d_{ij} / \partial \rho_{\mathrm{null}} = \lambda_{\mathrm{null}}$
on null modes (and zero otherwise):

$$
\frac{\partial c}{\partial \rho_1}
  = -\frac{1}{D} \sum_{(i, j)\ \text{positive}}
    \frac{w_{ij}\,\lambda_1\,d^x_i}{d_{ij}^2},
$$

$$
\frac{\partial c}{\partial \rho_2}
  = -\frac{1}{D} \sum_{(i, j)\ \text{positive}}
    \frac{w_{ij}\,\lambda_2\,d^y_j}{d_{ij}^2},
$$

$$
\frac{\partial c}{\partial \rho_{\mathrm{null}}}
  = -\frac{\beta}{\lambda_{\mathrm{null}} D},
\qquad
\beta = \sum_{(i, j)\ \text{null}} w_{ij}.
$$

These are exactly `factor_grad_corrs` and `null_grad_corr` in
`_log_det_and_grad` for KRONECKER_WITH_NULL, specialized to two factors.

**Cost.**  Nothing in this path ever materializes the
$(q_x q_y) \times (q_x q_y)$ matrix $S_\lambda$ or its inverse.  The work
runs over the two 1D spectra $(d^x, d^y)$ and the precomputed
$(q_x, q_y)$ array $w_{ij}$, with cost $O(q_x q_y)$ instead of the
$O((q_x q_y)^3)$ that a generic `eigh` of the restricted matrix would need.

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

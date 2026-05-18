---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Derivative of an upper-triangular Cholesky factor

This note derives the closed form used by `_chol_deriv.py:grad_cholesky`. It
is the JAX-native replacement for the Wood (2017) Appendix B.7 sequential
recurrence that the legacy PGAM uses in `grad_cholesky` /
`grad_chol_Vb_rho`. We need it for the corrected AIC (Wood 2017 eq. 6.32):

$$
V''_\beta \;=\; \sum_{k,l}\, (\partial_k R)^{\!\top}\, V_\rho[k,l]\, (\partial_l R),
\qquad R^{\!\top} R \;=\; V_\beta^{-1} \;=\; H + S_\lambda/\phi.
$$

---

## 1. Setup

Let $D = D(x)$ be a symmetric positive-definite matrix depending smoothly on a
scalar parameter $x \in \mathbb{R}$, and let $R = R(x)$ be the
**upper-triangular** Cholesky factor satisfying

$$
R^{\!\top} R \;=\; D, \qquad R_{ii} > 0.
$$

We want $\dot R := \mathrm{d}R / \mathrm{d}x$ given $\dot D := \mathrm{d}D / \mathrm{d}x$
(both depending implicitly on $x$).

> **Convention.** $R$ is upper-triangular, so $R^{\!\top}$ is lower-triangular.
> Most textbooks state the formula for the *lower* Cholesky $L L^{\!\top} = D$.
> The derivation below mirrors that one but in the upper triangle.

---

## 2. Differentiating the factorisation

Apply $\mathrm{d}/\mathrm{d}x$ to $R^{\!\top} R = D$:

$$
\dot R^{\!\top} R \;+\; R^{\!\top} \dot R \;=\; \dot D. \tag{1}
$$

Multiply both sides on the left by $R^{-\top}$ and on the right by $R^{-1}$:

$$
R^{-\top} \dot R^{\!\top} \;+\; \dot R\, R^{-1}
   \;=\; R^{-\top}\, \dot D\, R^{-1}.
$$

Define

$$
Y \;:=\; \dot R\, R^{-1}.
$$

Since $\dot R$ and $R^{-1}$ are both upper-triangular, $Y$ is also
upper-triangular. In terms of $Y$ the equation reads

$$
\boxed{\; Y^{\!\top} + Y \;=\; R^{-\top}\, \dot D\, R^{-1}. \;} \tag{2}
$$

The right-hand side is symmetric (it equals its own transpose because
$\dot D$ is symmetric). So (2) is a *"given a symmetric matrix $S$, find an
upper-triangular $Y$ with $Y + Y^{\!\top} = S$"* problem.

---

## 3. Recovering $Y$ from the symmetric sum

For any matrix $A$ define the **upper-half projector**

$$
\Phi(A) \;:=\; \mathrm{triu}(A) \;-\; \tfrac{1}{2}\,\mathrm{diag}(A),
$$

i.e. keep the strict upper triangle, halve the diagonal, zero the strict
lower triangle. Element-wise:

$$
\Phi(A)_{ij} \;=\;
\begin{cases}
A_{ij},     & i < j, \\[2pt]
A_{ii}/2,   & i = j, \\[2pt]
0,          & i > j.
\end{cases}
$$

**Claim.** If $S$ is symmetric, $Y := \Phi(S)$ is the unique
upper-triangular matrix with $Y + Y^{\!\top} = S$.

*Verification* — check the three cases.

$$
\begin{aligned}
(i < j) \qquad
   & (Y + Y^{\!\top})_{ij}
   \;=\; Y_{ij} + Y_{ji}
   \;=\; S_{ij} + 0
   \;=\; S_{ij}, \\[2pt]
(i = j) \qquad
   & (Y + Y^{\!\top})_{ii}
   \;=\; 2\, Y_{ii}
   \;=\; S_{ii}, \\[2pt]
(i > j) \qquad
   & (Y + Y^{\!\top})_{ij}
   \;=\; 0 + Y_{ji}
   \;=\; S_{ji}
   \;=\; S_{ij}
   \quad \text{(by symmetry of $S$)}.
\end{aligned}
$$

Applying this to (2):

$$
Y \;=\; \Phi\!\left( R^{-\top}\, \dot D\, R^{-1} \right). \tag{3}
$$

---

## 4. The closed form

From the definition $Y = \dot R\, R^{-1}$,

$$
\boxed{\;
\dot R \;=\; \Phi\!\left( R^{-\top}\, \dot D\, R^{-1} \right)\, R.
\;} \tag{4}
$$

That's it — one symmetric sandwich solve, one upper-half projection, one
matrix multiply. All three steps are $\mathcal{O}(p^3)$ BLAS-level
operations.

---

## 5. Numerical implementation

Forming $R^{-\top} \dot D\, R^{-1}$ never instantiates $R^{-1}$. Instead use
two triangular solves:

$$
\begin{aligned}
1. \quad & A := R^{-\top}\, \dot D
   &&\Longleftrightarrow\quad
   \text{solve } R^{\!\top} A = \dot D
   \quad \text{(lower-triangular solve).} \\[2pt]
2. \quad & M := A\, R^{-1}
   &&\Longleftrightarrow\quad
   \text{solve } R^{\!\top} M^{\!\top} = A^{\!\top}
   \quad \text{(lower-triangular solve), then transpose.} \\[2pt]
3. \quad & Y := \Phi(M). && \\[2pt]
4. \quad & \dot R := Y\, R. &&
\end{aligned}
$$

That is exactly what `_chol_deriv.py:grad_cholesky` does:

```python
def grad_cholesky(grad_D, R):
    A = solve_triangular(R.T, grad_D, lower=True)
    M = solve_triangular(R.T, A.T, lower=True).T
    upper_half = jnp.triu(M) - 0.5 * jnp.diag(jnp.diag(M))
    return upper_half @ R
```

A small numerical sanity check, reproducing the unit-test logic:

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

jax.config.update("jax_enable_x64", True)

def grad_cholesky(grad_D, R):
    A = solve_triangular(R.T, grad_D, lower=True)
    M = solve_triangular(R.T, A.T, lower=True).T
    upper_half = jnp.triu(M) - 0.5 * jnp.diag(jnp.diag(M))
    return upper_half @ R

rng = np.random.default_rng(0)
p = 6
A = rng.standard_normal((p, p))
D = jnp.asarray(A @ A.T + 0.5 * np.eye(p))
B = rng.standard_normal((p, p))
dD = jnp.asarray(B + B.T)            # symmetric

R = jnp.linalg.cholesky(D).T          # upper-triangular factor
dR = grad_cholesky(dD, R)

# Defining identity:  Ṙ^T R + R^T Ṙ = Ḋ
residual = float(jnp.max(jnp.abs(dR.T @ R + R.T @ dR - dD)))
print(f"|defining identity residual|_inf = {residual:.2e}")

# Central difference of chol(D + x · dD) at x = 0
eps = 1e-6
R_plus  = jnp.linalg.cholesky(D + eps * dD).T
R_minus = jnp.linalg.cholesky(D - eps * dD).T
dR_fd = (R_plus - R_minus) / (2 * eps)
fd_diff = float(jnp.max(jnp.abs(dR - dR_fd)))
print(f"|dR − dR_fd|_inf            = {fd_diff:.2e}")
```

---

## 6. Comparison with the Wood Appendix B.7 recurrence

The recurrence in Wood (2017) Appendix B.7 / Smith (1995) gives an
element-wise update equivalent to (4):

$$
\begin{aligned}
R_{ii}\, \dot R_{ii}
   &\;=\; \tfrac{1}{2}\!\left(
      \dot D_{ii} \;-\; 2 \sum_{m < i} R_{mi}\, \dot R_{mi}
   \right) \\[6pt]
R_{ii}\, \dot R_{ij}
   &\;=\; \dot D_{ij} \;-\; R_{ij}\, \dot R_{ii}
        \;-\; \sum_{m < i}\!\left(
              \dot R_{mi}\, R_{mj} + R_{mi}\, \dot R_{mj}
            \right)
   \qquad (j > i).
\end{aligned}
$$

These are precisely the entries of $\dot R = \Phi(R^{-\top} \dot D R^{-1})\, R$
computed by Gauss–Jordan elimination in column-major order. The closed
form (4) computes the same result but in BLAS-level operations, so it
`jit`-compiles and `vmap`-batches cleanly — which the per-element
recurrence does not.

The test
`tests/test_chol_deriv.py::TestGradCholesky::test_matches_legacy_recurrence`
asserts agreement to $\sim 10^{-10}$ between the closed form and a verbatim
numpy transcription of the recurrence; `test_matches_finite_difference`
confirms agreement with central differences of $\mathrm{chol}(D(x))$.

---

## 7. Application: covariance-Cholesky derivative for the corrected AIC

The corrected AIC $V''$ term (Wood 2017 eq. 6.32; legacy
`compute_AIC` / `grad_chol_Vb_rho`) wants $\partial R / \partial \rho_k$ where

$$
R^{\!\top} R \;=\; V_\beta,
$$

the **covariance**, not the precision. (The legacy PGAM variable named
`Vb_inv` *is* in fact $V_\beta$ — `Vbeta_rho_all` returns `(precision,
covariance)` in that order, and the caller binds them as `Vb, Vb_inv`.)

The upstream Laplace-REML machinery directly produces

$$
\frac{\partial V_\beta^{-1}}{\partial \rho_k}
   \;=\; \frac{\partial H}{\partial \rho_k}
       \;+\; \frac{\lambda_k S_k}{\phi},
$$

i.e. the *precision* derivative. Converting to the covariance derivative
uses the matrix-inverse derivative identity

$$
\frac{\partial V_\beta}{\partial \rho_k}
   \;=\; -\, V_\beta\,
         \frac{\partial V_\beta^{-1}}{\partial \rho_k}\, V_\beta, \tag{5}
$$

which the wrapper `_chol_deriv.py:grad_U_Vbeta` *avoids* — it operates on
$V_\beta^{-1}$ throughout (the well-conditioned object in penalised
regression) and returns $\mathrm{d}U/\mathrm{d}\rho$ where $U U^T = V_\beta$,
via the mgcv-style transform $\mathrm{d}U = -\tilde R^{-1}\,\mathrm{d}\tilde R\,\tilde R^{-1}$
with $\tilde R = \mathrm{chol}(V_\beta^{-1})$.  See the
`grad_U_Vbeta` docstring for the stability rationale and the convention
switch from the textbook $R^T R = V_\beta$ form.

---

## References

- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). CRC Press. §B.7 (Cholesky derivative recurrence); §6.11.2
  (corrected AIC).
- Smith, S. P. (1995). *Differentiation of the Cholesky Algorithm*. Journal of
  Computational and Graphical Statistics, 4(2), 134–147.
- Kass, R. E., & Steffey, D. (1989). Approximate Bayesian inference in
  conditionally independent hierarchical models. JASA, 84(407), 717–726.
- Wood, S. N., Pya, N., & Säfken, B. (2016). Smoothing parameter and model
  selection for general smooth models. JASA, 111(516), 1548–1563.
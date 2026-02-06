---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Penalized Generalized Additive Models (PGAM): A Technical Overview

This document provides a pedagogical introduction to the PGAM implementation,
intended for collaborators porting this code to nemos.

## 1. What is a GAM?

A **Generalized Additive Model** extends the linear model by replacing linear
terms with smooth functions:

**Linear Model:**
$$E[Y] = \beta_0 + \beta_1 x_1 + \beta_2 x_2$$

**Additive Model:**
$$E[Y] = \beta_0 + f_1(x_1) + f_2(x_2)$$

Each $f_j$ is a smooth (nonlinear) function of its predictor. This gives us
flexibility to capture nonlinear relationships while maintaining interpretability.

For **Generalized** Additive Models, we add a link function $g$ (just like in GLMs):
$$g(E[Y]) = \beta_0 + f_1(x_1) + f_2(x_2)$$

For Poisson regression (spike count data), we use the log link:
$$\log(E[Y]) = \beta_0 + f_1(x_1) + f_2(x_2)$$

## 2. Representing Smooth Functions with B-Splines

We represent each smooth function as a linear combination of **basis functions**:

$$f_j(x) = \sum_{k=1}^{K} \beta_{jk} \cdot B_k(x)$$

where $B_k(x)$ are B-spline basis functions and $\beta_{jk}$ are the coefficients
we need to estimate.

This converts our smooth function estimation problem into a linear regression problem
in the basis coefficients!

```python
# Example: Creating a B-spline basis with 12 functions
from pgam_clean.basis import GAMBSplineEval
bas = GAMBSplineEval(n_basis_funcs=12, order=4)  # order=4 means cubic splines
```

## 3. The Overfitting Problem

With many basis functions, the model can become too "wiggly" and overfit the data.
Consider fitting a smooth function through noisy data:

- **Too few basis functions**: Can't capture the true shape (underfitting)
- **Too many basis functions**: Fits the noise (overfitting)

The solution: **penalized estimation**.

## 4. Smoothing Penalties

We add a penalty term to the loss function that discourages "wiggliness":

$$\text{Loss} = \text{Deviance}(\beta) + \lambda \cdot \int [f''(x)]^2 dx$$

The penalty term $\int [f''(x)]^2 dx$ measures the integrated squared second
derivative—essentially, how curved the function is.

- $\lambda = 0$: No penalty, maximum flexibility (may overfit)
- $\lambda \to \infty$: Heavy penalty, forces $f$ to be linear (may underfit)
- $\lambda$ "just right": Balances fit and smoothness

### 4.1 Computing the Penalty Matrix

In basis function representation, the penalty becomes a quadratic form:

$$\int [f''(x)]^2 dx = \boldsymbol{\beta}^T \mathbf{S} \boldsymbol{\beta}$$

where $\mathbf{S}$ is the **penalty matrix**:

$$S_{ij} = \int B_i''(x) \cdot B_j''(x) \, dx$$

This integral is computed numerically using Simpson's rule in our implementation.

```python
# Computing the energy penalty matrix
import pgam_clean.penalty_utils as pen_utils

# For a single smooth term
penalty_tensor = pen_utils.compute_energy_penalty_tensor_additive_component(bas)
```

### 4.2 Multiple Penalty Components

For each smooth, we typically have multiple penalty matrices:

1. **Energy penalty**: Penalizes curvature (the main smoothing penalty)
2. **Null space penalty**: Penalizes the functions that escape the energy penalty

For second-derivative penalties, the null space consists of linear functions
($a + bx$), which have zero second derivative. Without a null space penalty,
these would be unpenalized and could take arbitrarily large values.

## 5. The Identifiability Constraint

### 5.1 The Problem

In an additive model with an intercept:
$$E[Y] = \beta_0 + f_1(x_1) + f_2(x_2)$$

There's an ambiguity: we can add a constant $c$ to $f_1$ and subtract it from
$\beta_0$ without changing the predictions:
$$E[Y] = (\beta_0 - c) + (f_1(x_1) + c) + f_2(x_2)$$

The individual smooth functions are **not uniquely identified**.

### 5.2 The Solution

We impose a constraint on each smooth:
$$\sum_i f_j(x_i) = 0$$

This "centers" each smooth to have zero mean, making the decomposition unique.

### 5.3 Implementation: Dropping a Column

In practice, we enforce this by **dropping one basis function** (column) from
each smooth term. That basis function's effect gets absorbed into the intercept.

```python
# The apply_identifiability function drops the last column
apply_identifiability = lambda x: x[..., :-1]

# For penalty matrices (which are square), we drop both last row and column
apply_identifiability_penalty = lambda x: x[..., :-1, :-1]
```

## 6. Generalized Cross-Validation (GCV)

### 6.1 The Challenge

How do we choose the regularization strength $\lambda$?

- Too small: Overfitting
- Too large: Underfitting

We need a data-driven method to select $\lambda$.

### 6.2 Leave-One-Out Cross-Validation

Ideally, we'd use leave-one-out CV:

$$\text{CV} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{f}_{-i}(x_i))^2$$

where $\hat{f}_{-i}$ is the model fitted without observation $i$.
But this requires fitting $n$ models—too expensive!

### 6.3 GCV: An Efficient Approximation

GCV provides a closed-form approximation:

$$\text{GCV} = \frac{n \cdot \text{RSS}}{(n - \gamma \cdot \text{tr}(\mathbf{A}))^2}$$

where:
- $\text{RSS} = \sum_i (y_i - \hat{y}_i)^2$ is the residual sum of squares
- $\mathbf{A}$ is the "hat matrix" (influence matrix)
- $\text{tr}(\mathbf{A})$ is the effective degrees of freedom
- $\gamma \geq 1$ is a correction factor (we use $\gamma = 1.5$)

The $\gamma > 1$ correction biases toward smoother fits, reducing overfitting risk.

### 6.4 Computing GCV Efficiently

The key insight is that $\text{tr}(\mathbf{A})$ can be computed without forming
the full $n \times n$ hat matrix:

1. Augment the design matrix: $\mathbf{X}_\text{aug} = [\mathbf{X}; \sqrt{\lambda}\mathbf{S}^{1/2}]$
2. Compute QR decomposition of the augmented matrix
3. Use SVD tricks to compute $\text{tr}(\mathbf{A})$ efficiently

```python
from pgam_clean.gcv_compute import gcv_compute_factory

# Create a GCV scoring function with custom VJP for efficient gradients
gcv_scorer = gcv_compute_factory(
    positive_mon_func=jnp.exp,           # λ = exp(θ) ensures positivity
    apply_identifiability_columns=...,    # for sqrt penalty
    apply_identifiability=...,            # for full penalty
    gamma=1.5                             # GCV correction factor
)
```

## 7. The PQL Algorithm

### 7.1 Overview

We fit the PGAM using **Penalized Quasi-Likelihood (PQL)**, which combines:

1. **IRLS** (Iteratively Reweighted Least Squares) for the GLM part
2. **GCV optimization** for selecting smoothing parameters

### 7.2 Algorithm Structure

```
Initialize: coefficients β, regularization strengths λ

REPEAT (outer loop):
    1. Compute fitted values: μ = g⁻¹(Xβ)
    2. Compute IRLS weights and pseudo-response (linearize the GLM)
    3. Form augmented system: [√W·X; √λ·S^½]
    4. Solve weighted least squares for new β

    5. INNER LOOP: Optimize λ by minimizing GCV
       - Use L-BFGS-B with gradients from custom VJP

    6. Check convergence
UNTIL convergence
```

### 7.3 Key Implementation Details

**Augmented System:**
Instead of solving the penalized problem directly, we augment the design matrix:

$$\begin{bmatrix} \sqrt{\mathbf{W}} \mathbf{X} \\ \sqrt{\lambda} \mathbf{S}^{1/2} \end{bmatrix} \boldsymbol{\beta} = \begin{bmatrix} \sqrt{\mathbf{W}} \mathbf{z} \\ \mathbf{0} \end{bmatrix}$$

where $\mathbf{z}$ is the pseudo-response from IRLS.

**Custom VJP:**
We use JAX's custom VJP (vector-Jacobian product) to compute GCV gradients
efficiently without repeated SVD computations.

## 8. Code Structure

### 8.1 Module Overview

| Module | Purpose |
|--------|---------|
| `penalty_utils.py` | Penalty matrix computation, block assembly |
| `gcv_compute.py` | GCV score and gradients via custom VJP |
| `iterative_optim.py` | PQL outer loop, IRLS, WLS |
| `basis/` | B-spline basis evaluation |

### 8.2 Key Functions

```python
# Compute penalty tensors for an additive basis
penalty_tree = compute_energy_penalty_tensor(basis, n_samples, penalize_null_space=True)

# Compute weighted sqrt penalty for augmented system
sqrt_penalty = tree_compute_sqrt_penalty(penalty_tree, reg_strength, ...)

# GCV computation with efficient gradients
gcv_score = gcv_compute_factory(...)(reg_strength, penalty_tree, X, Q, R, y)

# Main PQL fitting loop
coeffs, opt_lambda, n_iter = pql_outer_iteration(
    init_lambda, init_coeffs, X, y, penalty_tree, obs_model, ...
)
```

## 9. Porting Considerations for nemos

### 9.1 What to Preserve

1. **Penalty matrix computation**: The Simpson integration and tensor product
   construction are numerically stable and well-tested.

2. **GCV with custom VJP**: The gradient computation is carefully optimized
   to avoid redundant SVD calls.

3. **Block-diagonal structure**: The penalty assembly respects the additive
   structure of the model.

### 9.2 Critical Integration Notes

1. **Custom VJP is REQUIRED for GCV**: Standard `jax.grad()` produces incorrect
   gradients for the GCV computation—they fail numerical gradient checks. The
   custom VJP in `gcv_compute.py` is essential and must be preserved.

2. **Basis derivatives via nemos methods**: The penalty computation requires
   evaluating basis function derivatives. These should be exposed through nemos
   basis methods (e.g., a `derivative()` method or `der` parameter) rather than
   computed externally. This is cleaner than the current approach.

3. **Identifiability constraints stay at PGAM level**: The column-dropping for
   identifiability is specific to additive models with intercepts. Do NOT add
   this to nemos basis classes—it would clutter the API for non-GAM use cases.
   Handle it at the PGAM/penalty level during model fitting.

### 9.3 Simplifications

1. **Remove debug paths**: The `config.DEBUG` paths were for validating against
   the original implementation; they're no longer needed.

2. **Integrate with nemos basis**: Replace `GAMBSplineEval` with nemos's
   existing B-spline basis infrastructure.

3. **Observation model integration**: Use nemos's `ObservationModel` classes
   consistently throughout.

### 9.4 Testing Strategy

1. Verify penalty matrices match for simple cases (1D, 2D smooths)
2. Verify GCV values and gradients against numerical differentiation
3. Verify coefficient recovery on synthetic data
4. Compare against R's `mgcv` package for real-world validation

## 10. References

- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC Press.
- Eilers, P. H. C. and Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89-121.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society: Series B*, 73(1), 3-36.

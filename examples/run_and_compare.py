"""
PGAM Comparison Script: Original vs JAX Implementation
=======================================================

This script compares three approaches to fitting a Poisson regression model:
1. Original PGAM implementation (from the PGAM package)
2. New JAX-based PGAM implementation (pgam_jax)
3. Standard GLM without smoothing penalties (nemos)

The goal is to validate that the new JAX implementation produces equivalent
results to the original, while being faster and more maintainable.

Key Notes for nemos Integration
-------------------------------
1. **Custom VJP Required**: The GCV gradient computation uses a custom VJP
   (vector-Jacobian product). Standard jax.grad() produces incorrect gradients
   that fail numerical gradient checks. See _pql_gcv.py for the implementation.

2. **Basis Derivatives**: The penalty computation requires basis function
   derivatives. These should be exposed via nemos basis methods (e.g., a
   `derivative()` method or parameter) rather than computed externally.

3. **Identifiability Constraints**: The column-dropping for identifiability is
   specific to additive models with intercepts. This should NOT be added to
   nemos basis classes, as it would clutter the API for non-GAM use cases.
   Instead, handle it at the PGAM/penalty level during model fitting.

See docs/pgam_overview.md for a detailed pedagogical explanation of the
PGAM algorithm and its components.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import nemos as nmo
import numpy as np
import statsmodels.api as sm
from nemos.observation_models import PoissonObservations

# Original PGAM implementation (for comparison)
from PGAM.GAM_library import *

# New JAX-based PGAM implementation
import pgam_jax.penalty_utils as pen_utils
from pgam_jax._penalty_handler import PenaltyHandler, _drop_last_col
from pgam_jax._pql_gcv import gcv_compute_factory
from pgam_jax.iterative_optim import pql_outer_iteration

# =============================================================================
# Configuration
# =============================================================================

# Enable 64-bit precision for numerical stability in eigenvalue computations
jax.config.update("jax_enable_x64", True)

# =============================================================================
# Step 1: Define the basis and penalty structure
# =============================================================================

# Create a B-spline basis with 12 basis functions. The current pgam_jax port
# uses nemos bases directly.
bas = nmo.basis.BSplineEval(12, bounds=(0.0, 1.0))

# Create an additive model with two smooth terms: f1(x1) + f2(x2)
# Each smooth uses the same B-spline basis structure
add = bas + bas

# Compute the penalty tensor for this additive basis.
# This creates penalty matrices that penalize the "wiggliness" (second derivative)
# of each smooth function. The penalty has the form: λ * ∫ f''(x)² dx
#
# The tensor contains:
# - Energy penalties for each dimension (penalize curvature)
# - Null space penalties (penalize the unpenalized subspace, e.g., linear functions)
#
# penalize_null_space=True adds a penalty on the null space of the main penalty,
# which for second derivatives means penalizing linear functions (a + bx).
penalty_tree = pen_utils.compute_energy_penalty_tensor(
    add, 10**4, penalize_null_space=True
)

# =============================================================================
# Step 2: Generate synthetic data
# =============================================================================

np.random.seed(123)

# Two predictor variables (covariates)
x1, x2 = np.random.randn(3000), np.random.randn(3000)

# -----------------------------------------------------------------------------
# Set up the ORIGINAL PGAM model (for comparison)
# -----------------------------------------------------------------------------
sm_handler = smooths_handler()
knots = np.linspace(0, 1, 10)

# Add smooth terms to the original PGAM.
# Key parameters:
# - ord=4: cubic B-splines (order = degree + 1)
# - penalty_type="der", der=2: penalize second derivative (curvature)
# - lam=[1, 2]: initial regularization strengths for the two penalty components
#   (energy penalty and null space penalty)
sm_handler.add_smooth(
    "linspace",
    [x1],
    knots=[knots],
    ord=4,
    is_temporal_kernel=False,
    trial_idx=None,
    is_cyclic=[False],
    penalty_type="der",
    der=2,
    lam=[1, 2],
    knots_num=10,
    kernel_length=None,
    kernel_direction=None,
    time_bin=0.006,
    knots_percentiles=(0, 100),
)

sm_handler.add_smooth(
    "linspace2",
    [x2],
    knots=[knots],
    ord=4,
    is_temporal_kernel=False,
    trial_idx=None,
    is_cyclic=[False],
    penalty_type="der",
    der=2,
    lam=[1, 2],
    knots_num=10,
    kernel_length=None,
    kernel_direction=None,
    time_bin=0.006,
    knots_percentiles=(0, 100),
)

# Get the design matrix from the original PGAM.
# [:, 1:] removes the intercept column (handled separately in our implementation)
X = jnp.asarray(sm_handler.get_exog_mat_fast(sm_handler.smooths_var)[0])[:, 1:]

# True coefficients: first smooth has non-zero effect, second is zero
# This tests whether the model correctly shrinks the second smooth to zero
w = np.hstack(
    [np.random.randn(11), np.zeros(11)]
)  # 11 = 12 basis - 1 (identifiability)
intercept = np.array([0.0])

# Generate Poisson-distributed spike counts
# y ~ Poisson(exp(X @ w + intercept))
y = np.random.poisson(np.exp(X.dot(w)))

# =============================================================================
# Step 3: Configure the new JAX PGAM implementation
# =============================================================================

# Inverse link function: for Poisson GLM, this is exp()
# Maps linear predictor η to mean μ: μ = exp(η)
inv_link_func = jax.numpy.exp

# Variance function: for Poisson, Var(Y) = μ
# This is used in IRLS weight computation
variance_func = lambda x: x

# Build a PenaltyHandler from the penalty tensor list and snapshot it into
# pure callables. ``compute_sqrt(rhos)`` returns the block-diagonal sqrt of the
# total penalty (used to augment the WLS system as [X; sqrt(penalty)]); the
# log-det callable is unused by GCV but is required by REML-style scorers.
#
# - non_linearity=exp: regularization strengths are parameterized as exp(ρ) to
#   ensure positivity during optimization.
# - identifiability_fn=_drop_last_col: drops the last column of each smooth's
#   sqrt factor, matching the column drop applied to the design matrix.
# - penalize_null_space=False here because ``compute_energy_penalty_tensor``
#   above already stacked the null-space penalty into each (k, q, q) tensor.
ph = PenaltyHandler(non_linearity=jax.numpy.exp)
for S_tensor in penalty_tree:
    ph.add(S_tensor, penalize_null_space=False, identifiability_fn=_drop_last_col)
compute_sqrt, compute_log_det_and_grad = ph.build()

# Factory function that creates the GCV (Generalized Cross-Validation) scorer.
# GCV is used to select optimal smoothing parameters without explicit cross-validation.
#
# IMPORTANT: This uses a custom VJP (vector-Jacobian product) implementation.
# Standard jax.grad() would fail here - it produces gradients that don't match
# numerical gradient checks. The custom VJP in _pql_gcv.py is carefully
# designed to compute correct gradients through the SVD-based GCV computation.
#
# Arguments:
# 1. compute_sqrt: handler-built sqrt-penalty function (rhos -> sqrt matrix)
# 2. positive_mon_func: exp() to ensure λ > 0
# 3. apply_identifiability for sqrt penalty (drop last column)
# 4. apply_identifiability for full penalty (drop last row AND column)
# 5. gamma=1.5: GCV correction factor (>1 gives more smoothing, reduces overfitting)
inner_func = gcv_compute_factory(
    compute_sqrt,
    jax.numpy.exp,
    lambda x: x[..., :-1],
    lambda x: x[..., :-1, :-1],
    1.5,
)

# =============================================================================
# Step 4: Fit the ORIGINAL PGAM model
# =============================================================================

link = sm.genmod.families.links.Log()
poissFam = sm.genmod.families.family.Poisson(link=link)

# Create and fit the original PGAM model
pgam = general_additive_model(
    sm_handler,
    sm_handler.smooths_var,  # list of covariates to include
    np.asarray(y),  # response variable (spike counts)
    poissFam,  # Poisson family with log link
)

res = pgam.optim_gam(
    sm_handler.smooths_var,
    max_iter=10**2,
    use_dgcv=True,  # use derivative-based GCV optimization
    method="L-BFGS-B",
    fit_initial_beta=True,
    filter_trials=np.ones(len(y), dtype=bool),
)

# =============================================================================
# Step 5: Fit the NEW JAX PGAM model
# =============================================================================

# pql_outer_iteration implements the Penalized Quasi-Likelihood algorithm:
#
# Outer loop (PQL):
#   1. Given current coefficients, compute IRLS weights and pseudo-response
#   2. Solve weighted least squares with penalty augmentation
#   3. Update coefficients
#
# Inner loop (GCV optimization):
#   1. Given current weights/pseudo-response, optimize regularization λ
#   2. Minimize GCV score using L-BFGS-B
#
# Arguments:
# - Initial regularization: log([1, 2]) for each of the two smooths
# - Initial coefficients: zeros
# - X: design matrix
# - y: response
# - penalty_tree: penalty matrices for each smooth
# - PoissonObservations(): observation model (provides inverse link)
# - variance_func: Var(Y|μ) = μ for Poisson
# - inner_func: GCV scorer for λ optimization
# - compute_sqrt: handler-built function (rhos -> block-diagonal sqrt penalty)

opt_coef, opt_pen, niter = pql_outer_iteration(
    [jax.numpy.log(jax.numpy.array([1.0, 2.0]))] * 2,  # initial log(λ) for each smooth
    jtu.tree_map(jnp.zeros_like, (w, intercept)),  # initial (coefficients, intercept)
    X,
    y,
    penalty_tree,
    PoissonObservations(),
    variance_func,
    inner_func,
    compute_sqrt,
    fisher_scoring=False,  # use observed information, not expected
    max_iter=100,
    tol_update=10**-6,  # convergence tolerance for coefficient updates
    tol_optim=10**-10,  # tolerance for inner GCV optimization
)

# =============================================================================
# Step 6: Fit a standard GLM (no smoothing) for comparison
# =============================================================================

from nemos.glm import GLM

model = GLM().fit(X, y)

print(f"JAX PGAM converged in {niter} iterations")

# =============================================================================
# Step 7: Visualize and compare results
# =============================================================================

import matplotlib.pyplot as plt


def show_or_close():
    """Avoid GUI/show warnings when the script is run with a non-interactive backend."""
    if "agg" in plt.get_backend().lower():
        plt.close("all")
    else:
        plt.show()


plt.close("all")
fig, axs = plt.subplots(1, 4, figsize=(10, 3), sharey=True, sharex=True)

# Plot 1: Original PGAM vs JAX PGAM (should lie on diagonal if equivalent)
axs[0].set_xlabel("orig gam")
axs[0].set_ylabel("jax gam")
axs[0].scatter(res.beta, np.hstack((opt_coef[1], opt_coef[0])))
axs[0].plot([-3, 3], [-3, 3], "k")
axs[0].set_aspect("equal")
axs[0].set_title("Implementation\nComparison")

# Plot 2: True coefficients vs Original PGAM
axs[1].scatter(np.hstack((intercept, w)), res.beta)
axs[1].plot([-3, 3], [-3, 3], "k")
axs[1].set_xlabel("true")
axs[1].set_ylabel("orig gam")
axs[1].set_title("Original PGAM\nvs Truth")

# Plot 3: True coefficients vs JAX PGAM
axs[2].scatter(np.hstack((intercept, w)), np.hstack((opt_coef[1], opt_coef[0])))
axs[2].plot([-3, 3], [-3, 3], "k")
axs[2].set_xlabel("true")
axs[2].set_ylabel("jax gam")
axs[2].set_title("JAX PGAM\nvs Truth")

# Plot 4: True coefficients vs unpenalized GLM
# The GLM should overfit more since it has no smoothing penalty
axs[3].scatter(
    np.hstack((intercept, w)),
    np.hstack((model.intercept_, model.coef_)),
    color="orange",
)
axs[3].plot([-3, 3], [-3, 3], "k")
axs[3].set_xlabel("true")
axs[3].set_ylabel("GLM")
axs[3].set_title("GLM (no penalty)\nvs Truth")

fig.tight_layout()
show_or_close()

# Plot coefficient profiles
plt.figure()
plt.plot(res.beta[1:], label="orig-gam")
plt.plot(opt_coef[0], ls="--", label="jax-gam")
plt.plot(model.coef_, label="glm")
plt.scatter(np.arange(len(model.coef_)), w, label="true", zorder=5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient value")
plt.title("Coefficient Recovery Comparison")
plt.legend()
show_or_close()

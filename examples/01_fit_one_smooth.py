# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fit a single smooth
#
# A GLM with a B-spline basis can model a tuning curve of unknown shape.
# But the fit depends on the number of basis functions.
# Too few functions miss the shape. Too many functions follow the noise.
# The usual fix is to cross-validate over many basis sizes, which costs many fits.
#
# A penalized GAM takes a different route.
# It keeps one large basis and adds a penalty on the curvature of the fitted curve.
# It estimates the strength of that penalty from the data, in one fit.
#
# This notebook fits both models to the same data and compares them.

# %%
import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# %%
import nemos as nmo
import pandas as pd
from example_data import make_position_tuned_data
from example_plots import (
    ORANGE,
    PURPLE,
    add_rate_curve,
    plot_counts_and_truth,
    plot_design_recorded_and_sorted,
    set_plot_style,
)
from scipy.stats import poisson

from pgam_jax import GAM

set_plot_style()

# %% [markdown]
# ## The data
#
# One neuron and one input, the position $x$.
# We draw 320 positions $x \sim \mathrm{Uniform}(0, 1)$ and one spike count $y$ for each:
#
# $$\log \lambda(x) = -1.25 + 2.25 \exp\left(-\frac{(x - 0.57)^2}{2 \cdot 0.13^2}\right) + 0.65 \exp\left(-\frac{(x - 0.20)^2}{2 \cdot 0.075^2}\right)$$
#
# $$y \sim \mathrm{Poisson}(\lambda(x))$$
#
# The log-rate is a baseline plus two Gaussian bumps: a main field near 0.57 and a smaller shoulder near 0.2.

# %%
data = make_position_tuned_data()

# %%
for name in ["spikes", "position", "position_grid"]:
    label = f"Shape of data.{name}:"
    print(f"{label:<30}{getattr(data, name).shape}")
print()
print(f"Number of spikes: {data.spikes.sum()}")


# %%
fig, ax = plot_counts_and_truth(data)
plt.show()

# %% [markdown]
# ## Unpenalized GLM with a large basis

# %% [markdown]
# ### Basis definition

# %%
# The number of B-spline functions. Shared by GLM and GAM.
N_BASIS = 30

# %%
basis = nmo.basis.BSplineEval(
    n_basis_funcs=N_BASIS,
    bounds=(0.0, 1.0),
    label="position",
)

# %% [markdown]
# **NOTE:** `pgam_jax` requires `bounds` to be set on evaluation bases.

# %% [markdown]
# ### The design matrix

# %%
design = basis.compute_features(data.position)

print(f"Design matrix shape: {design.shape}")


# %%
fig = plot_design_recorded_and_sorted(design, data.position)
plt.show()

# %% [markdown]
# ### Unpenalized GLM

# %%
glm = nmo.glm.GLM()

glm.fit(design, data.spikes)

# %%
design_grid = basis.compute_features(data.position_grid)

glm_rate = glm.predict(design_grid)

# %%
fig, ax = plot_counts_and_truth(data)
add_rate_curve(
    ax, data.position_grid, glm_rate, f"GLM, K = {N_BASIS}, no penalty", PURPLE
)
plt.show()

# %% [markdown]
# ## Fit the GAM

# %%
gam = GAM(
    basis,  # uses the same nemos basis
    use_scipy=True,  # often faster on CPU
)

# %%
gam.fit(
    (data.position,),  # fit expects inputs as tuples
    data.spikes,  # the output/target is a single vector
)

# %% [markdown]
# ## GAM recovers the true tuning curve
#
# The penalty decides how much of the flexibility the fit actually uses.
# That amount is the effective degrees of freedom.

# %%
gam_rate = gam.predict((data.position_grid,))

# %%
fig, ax = plot_counts_and_truth(data)
add_rate_curve(
    ax,
    data.position_grid,
    glm_rate,
    f"GLM, K = {N_BASIS}, no penalty",
    PURPLE,
    alpha=0.85,
)
add_rate_curve(
    ax,
    data.position_grid,
    gam_rate,
    f"PGAM, K = {N_BASIS}, effective d.f. = {float(gam.edf_):.1f}",
    ORANGE,
)
plt.show()

# %% [markdown]
# ## Score on new data
#
# `gam.score` returns the mean log-likelihood of the data under the fitted model. Higher is better.
#
# A good fit must also predict data that it did not see.
# Draw a new sample from the same model, with a different seed, and score both models on the training data and on the new data.
# The true model gives a reference: no fit can be expected to do better on new data.

# %%
test_data = make_position_tuned_data(seed=123)
test_design = basis.compute_features(test_data.position)


def true_model_score(d):
    """Mean log-likelihood of the counts under the true rate."""
    return poisson.logpmf(d.spikes, d.true_rate).mean()


scores = pd.DataFrame(
    {
        "train": [
            float(glm.score(design, data.spikes)),
            float(gam.score((data.position,), data.spikes)),
            true_model_score(data),
        ],
        "test": [
            float(glm.score(test_design, test_data.spikes)),
            float(gam.score((test_data.position,), test_data.spikes)),
            true_model_score(test_data),
        ],
    },
    index=["GLM", "PGAM", "true model"],
)
scores.style.format("{:.3f}")

# %% [markdown]
# The GLM scores best on the training data but worst on the new data: it fits the noise of the training sample.
# The PGAM scores lower than the GLM on the training data, because the penalty keeps it from following that noise.
# On the new data it scores close to the true model.

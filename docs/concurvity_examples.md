---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Concurvity in practice: three worked examples

This notebook walks through three simulated Poisson GAMs that exhibit
qualitatively different flavors of concurvity:

1. **Independent covariates** — no concurvity problem. A baseline showing what
   "no problem" looks like in the table.
2. **One covariate is a noisy smooth function of another** — the textbook
   high-concurvity case (the example from `?mgcv::concurvity`). Two terms
   compete for the same signal direction; pairwise concurvity is symmetric.
3. **Three smooths, one jointly determined by the other two** — concurvity
   with structure: the pairwise matrix is *asymmetric*, revealing which term
   absorbs which. This kind of detail only shows up with three or more terms
   and is invisible in the two-term case above.

The magnitudes are not meant to march low → medium → high — scenario 3 will
in fact carry higher numbers than scenario 2 in this parameterization, because
joint determination by two predictors plus low noise constrains the focal
smooth more tightly than the simple noisy-function setup of scenario 2 does.
The point of scenario 3 is the asymmetric pairwise *structure*, not the
magnitude per se.

For each scenario we state the data-generating equations, fit the model, read
the three concurvity indices off `GAM.concurvity`, and explain why the numbers
land where they do.

The mathematical derivation of `worst` / `observed` / `estimate` and the QR
machinery underneath is in `docs/concurvity_mgcv.md`. Numerical agreement with
mgcv has been validated at the design-matrix-and-β level: an independent R
script saves the design matrix, fitted coefficients, and three concurvity
matrices from a battery of `mgcv::gam` fits, and the harness at
`_scripts/validate_concurvity.py` confirms `pgam_jax.GAM.concurvity` reproduces
them to ~1e-14. The values you see below are what mgcv would print on the
*same fitted model*; they are not directly the values mgcv would print if you
ran `mgcv::gam` itself on the same data, because the two packages differ in
basis construction and smoothing-parameter optimization. For numerically closer
agreement with mgcv you can pass the REML smoothing-parameter optimizer to
`GAM(...)` — its name has been changing across branches, so the cells below
use the package default to stay runnable across versions.

```python
import jax
jax.config.update("jax_enable_x64", True)  # concurvity needs float64

import numpy as np
import matplotlib.pyplot as plt
import nemos as nmo
from pgam_jax import GAM

rng = np.random.default_rng(20260522)
n = 600
```

## Scenario 1 — Independent covariates → low concurvity

The data-generating process:

$$
x_1,\, x_2 \stackrel{\text{i.i.d.}}{\sim} \mathrm{Uniform}(0, 1)
$$

$$
\eta(x_1, x_2) = 1.0 + 0.6 \sin(4\pi x_1) + 0.8\,(x_2 - 0.5)^2
$$

$$
y \mid x_1, x_2 \sim \mathrm{Poisson}\!\left(e^{\eta(x_1, x_2)}\right)
$$

Because $x_1$ and $x_2$ are sampled independently, the empirical correlation
between any column of the $s(x_1)$ basis and any column of the $s(x_2)$ basis
is small. Equivalently, the column spaces $\mathrm{col}(\mathbf{X}_1)$ and
$\mathrm{col}(\mathbf{X}_2)$ are nearly orthogonal once the intercept is
removed. The projection $\mathbf{g}_i$ of one smooth onto the other's span is
therefore small, and so is $\|\mathbf{g}_i\|^2 / \|\mathbf{f}_i\|^2$.

```python
x1 = rng.uniform(0, 1, n)
x2 = rng.uniform(0, 1, n)
eta = 1.0 + 0.6 * np.sin(4 * np.pi * x1) + 0.8 * (x2 - 0.5) ** 2
y = rng.poisson(np.exp(eta))

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(x1, x2, s=8, alpha=0.5)
ax.set(xlabel="x1", ylabel="x2", title="Scenario 1 — x1 and x2 sampled independently")
plt.show()
```

```python
basis = (nmo.basis.BSplineEval(12, bounds=(0., 1.), label="s(x1)")
         + nmo.basis.BSplineEval(12, bounds=(0., 1.), label="s(x2)"))
gam = GAM(basis, use_scipy=True, maxiter=15).fit((x1, x2), y)
gam.concurvity((x1, x2), as_dataframe=True)
```

All three indices stay well below 0.1. The `para` row sits at machine
epsilon (~1e-30) because the centered smooth bases are orthogonal to the
intercept column by construction — there is no parametric contamination
to worry about here. `worst` is the same for both smooths: in the
two-term case it is a property of the *pair* of subspaces (the largest
squared cosine of a principal angle between them), so swapping focal and
rest leaves it unchanged.

## Scenario 2 — One covariate is a smooth function of another → high concurvity

This is the example from the help page of `mgcv::concurvity`. Define a
deliberately wiggly function

$$
f(t) = 0.2\, t^{11}\,\bigl(10(1-t)\bigr)^6 + 10\,(10t)^3\,(1-t)^{10}
$$

and sample

$$
t \sim \mathrm{Uniform}(0, 1), \qquad x = f(t) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 3^2)
$$

$$
\eta(t, x) = 0.5 + 0.8 \sin(4\pi t) + 0.03 \, x, \qquad y \sim \mathrm{Poisson}\!\left(e^{\eta(t, x)}\right)
$$

Even after adding noise with $\sigma = 3$, $x$ is essentially a smooth
function of $t$ (the dynamic range of $f$ is much larger than the noise).
That means the $s(x)$ basis can be well approximated by some smooth in $t$,
and vice versa — the two terms compete for the same explanatory directions.
The fit still converges, but it cannot reliably attribute the signal to $t$
versus $x$.

```python
def f2(z):
    return 0.2 * z**11 * (10 * (1 - z))**6 + 10 * (10 * z)**3 * (1 - z)**10

t = np.sort(rng.uniform(0, 1, n))
x = f2(t) + rng.normal(0, 3, n)
eta = 0.5 + 0.8 * np.sin(4 * np.pi * t) + 0.03 * x
y = rng.poisson(np.exp(eta))

fig, ax = plt.subplots(figsize=(5, 4))
tt = np.linspace(0, 1, 400)
ax.plot(tt, f2(tt), color="C3", lw=2, label="f(t)  —  true mean of x")
ax.scatter(t, x, s=8, alpha=0.5, label="observed (t, x)")
ax.set(xlabel="t", ylabel="x", title="Scenario 2 — x is a noisy smooth function of t")
ax.legend()
plt.show()
```

```python
basis = (nmo.basis.BSplineEval(15, bounds=(float(t.min()), float(t.max())), label="s(t)")
         + nmo.basis.BSplineEval(15, bounds=(float(x.min()), float(x.max())), label="s(x)"))
gam = GAM(basis, use_scipy=True, maxiter=20).fit((t, x), y)
gam.concurvity((t, x), as_dataframe=True)
```

`worst` jumps to ~0.5 for both smooths: there is a coefficient direction in
which roughly half of either smooth lives in the other's span. Note that
`observed` is much higher for $s(x)$ than for $s(t)$ — that's the data
talking. The *true* $\eta$ depends mostly on $\sin(4\pi t)$ and only weakly
on $x$, so the fit allocates most of the signal to $s(t)$, leaving $s(x)$
small *and* largely re-explainable by $s(t)$. `estimate` (the Frobenius-norm
ratio) sits between the two and is closer to symmetric because it depends
only on the basis matrices, not the coefficients.

This is the canonical signature of high concurvity: a high `worst`, an
`observed` that may be either high or low depending on which way the
shrinkage broke, and a moderate-to-high `estimate`.

## Scenario 3 — Joint determination with asymmetric structure

$$
x_1,\, x_2 \stackrel{\text{i.i.d.}}{\sim} \mathrm{Uniform}(-1, 1)
$$

$$
x_3 = 0.8\, x_1 + 0.5\, \sin(2 x_2) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 0.3^2)
$$

$$
\eta(x_1, x_2, x_3) = x_1 + x_2^2 + \sin(2 x_3), \qquad y \sim \mathrm{Poisson}\!\left(e^{\eta}\right)
$$

Here $x_3$ is *jointly* recoverable from $(x_1, x_2)$ — with a linear
dependence on $x_1$ (weight 0.8) and a nonlinear dependence on $x_2$
(through $\sin 2x_2$) — but neither $x_1$ nor $x_2$ alone is enough.
With noise σ=0.3 against a signal range of ~2.6 in $x_3$, the joint
determination is *tighter* than the noisy-function setup of scenario 2,
so concurvity magnitudes will actually be higher than there. The
pedagogical point of this scenario is not the magnitude but the
**asymmetric pairwise structure**, which only emerges with three or
more terms. Predictions for the diagnostics:

- $s(x_3)$ should show the highest **full-model** concurvity, because the
  rest of the model $\{s(x_1), s(x_2)\}$ together can absorb most of it.
- $s(x_1)$ should also show appreciable full-model concurvity, because
  $x_3$ — already in the model — carries an $x_1$ component, so $s(x_1)$
  is partially re-explainable from $s(x_3)$.
- The **pairwise** matrix for `observed`/`estimate` should be
  asymmetric: $s(x_3)$ on its own can absorb a lot of $s(x_1)$ (since
  $x_3$ contains $x_1$ scaled by 0.8), while $s(x_1)$ on its own only
  accounts for a fraction of $s(x_3)$ because $x_3$ also carries the
  $\sin 2x_2$ piece plus noise. This directional information is the
  payoff of looking at the full pairwise table.

```python
x1 = rng.uniform(-1, 1, n)
x2 = rng.uniform(-1, 1, n)
x3 = 0.8 * x1 + 0.5 * np.sin(2 * x2) + rng.normal(0, 0.3, n)
eta = x1 + x2 ** 2 + np.sin(2 * x3)
y = rng.poisson(np.exp(eta))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (xa, xb, la, lb) in zip(
    axes,
    [(x1, x3, "x1", "x3"), (x2, x3, "x2", "x3"), (x1, x2, "x1", "x2")],
):
    ax.scatter(xa, xb, s=8, alpha=0.5)
    ax.set(xlabel=la, ylabel=lb)
fig.suptitle("Scenario 3 — pairwise relationships among (x1, x2, x3)")
plt.show()
```

```python
basis = (nmo.basis.BSplineEval(10, bounds=(-1., 1.), label="s(x1)")
         + nmo.basis.BSplineEval(10, bounds=(-1., 1.), label="s(x2)")
         + nmo.basis.BSplineEval(10, bounds=(float(x3.min()), float(x3.max())), label="s(x3)"))
gam = GAM(basis, use_scipy=True, maxiter=20).fit((x1, x2, x3), y)
gam.concurvity((x1, x2, x3), as_dataframe=True)
```

The full-model `worst` for $s(x_3)$ is the largest of the three (~0.8) and
$s(x_1)$ comes second (~0.7), exactly matching the prediction: $x_3$ is the
covariate that the other two can most fully approximate, with $s(x_1)$
elevated because $x_3$ contains a linear copy of it. Both numbers are
higher than scenario 2's symmetric ~0.5 — joint determination with low
noise is a more pathological setup than the two-term competing-signal
case. $s(x_2)$ also shows moderate `worst` (~0.65) because $\sin 2x_2$
enters $x_3$, but its `observed` is small: the fit on this dataset
happens to put the $s(x_2)$ weight in a direction that the other smooths
can't easily mimic.

The pairwise table is more revealing:

```python
pair = gam.concurvity((x1, x2, x3), full=False, as_dataframe=True)
pair["estimate"].round(3)
```

Reading rows-as-explainers, columns-as-focal:

- The $s(x_1) \to s(x_3)$ entry is sizeable: $x_1$ alone explains a
  meaningful fraction of $s(x_3)$ because $x_3 = 0.8 x_1 + \ldots$.
- The reverse $s(x_3) \to s(x_1)$ entry is smaller: $x_3$ on its own is
  $x_1$ plus a $\sin 2x_2$ contamination plus noise, so it's a noisier
  proxy for $x_1$ than $x_1$ is for $x_3$.
- $s(x_1) \leftrightarrow s(x_2)$ entries are near zero — the two
  generating variables were sampled independently.

```python
pair["worst"].round(3)
```

The `worst` matrix is symmetric (largest squared principal-angle cosine
of each pair of subspaces) — useful as a quick "is there any shared
direction at all?" check, but it can't distinguish which way the
information flows. That's why `observed` and `estimate` are necessary
complements.

## Diagnosing before you fit

Two of the three indices — `worst` and `estimate` — depend only on the
design matrix, not on the fitted coefficients. They can therefore be
read off *before* any fitting happens, just from the basis and the
covariates. This is useful when fitting is expensive (large Poisson
GAMs over long time series, hierarchical models with many smooths) and
you want to rule out a design that is fundamentally non-identifiable
before paying the optimization cost.

`GAM.concurvity` accepts being called on an unfitted model and returns
only the two coefficient-free measures in that case. The values it
reports for `worst` and `estimate` are numerically identical to what
you'd get after `fit` on the same inputs — they're a property of the
basis evaluated at those inputs, not of the optimizer.

```python
# Pre-fit diagnostic on the scenario 2 (concurve) covariates.
basis = (nmo.basis.BSplineEval(15, bounds=(float(t.min()), float(t.max())), label="s(t)")
         + nmo.basis.BSplineEval(15, bounds=(float(x.min()), float(x.max())), label="s(x)"))
gam_unfit = GAM(basis, use_scipy=True, maxiter=20)
gam_unfit.concurvity((t, x), as_dataframe=True)
```

The `worst` column already shows the ~0.5 we saw post-fit in scenario
2: the identifiability problem is visible without ever solving for
`β̂`. If you see this kind of value at the pre-fit stage, no amount of
clever optimization will give you stable coefficients — the right
response is to revise the model (drop a term, combine covariates, or
use a tensor-product interaction) before fitting.

Caveat: calling `concurvity` on an unfitted model has a small side
effect — it sets up the basis on the inputs you pass in (so it can be
evaluated at all). A subsequent `fit(xi_train, …)` re-runs `setup_basis`
with the training inputs and overwrites this state, so the diagnostic
won't influence the fit.

## Reading the numbers

A practical reading guide (consistent with `?mgcv::concurvity` and Wood
2017 §5.6.3):

- All indices are in $[0, 1]$ with 0 = identifiable and 1 = fully
  redundant.
- Worry seriously when any index exceeds ~0.8 for a smooth you care
  about interpreting.
- A high `worst` with a low `observed` means *this particular dataset*
  happens to dodge the unstable direction in coefficient space, but a
  small perturbation to the data could push the fit toward it. Treat
  this as a stability warning, not an all-clear.
- For tracking down which terms are causing the problem, the pairwise
  view (`full=False`) is more diagnostic than the full-model view: a
  high full-model concurvity tells you a term is redundant *somehow*,
  but the pairwise row-vs-column pattern tells you with whom.

Common remedies when concurvity is high: drop one of the colliding
smooths if its scientific meaning is duplicative; combine the covariates
into a single tensor-product smooth `te(x1, x2)` if the joint surface is
what you actually care about; or, if the covariates are causally
ordered, regress one on the other and use the residual as the second
predictor.

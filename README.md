# pgam_jax

`pgam_jax` is an alpha-stage, JAX-based implementation of the Poisson Generalized Additive Model (PGAM).

PGAM represents neural responses with B-splines regularized by smoothing penalties.

## Installation

This project currently targets Python 3.12, 3.13, and 3.14.

```bash
uv sync
```

For local development and comparisons against the original implementation:

```bash
uv sync --extra dev
```

For notebook-based experimentation:

```bash
uv sync --extra dev --extra notebooks
```

## Quick Start

The easiest way to try `pgam_jax` is the walkthrough notebook in `examples/quick_start.ipynb`.
It generates synthetic nonlinear Poisson data, fits a `GAM`, and visualizes the recovered rate surfaces.

Install with the required dependencies, then launch Jupyter Lab:
```bash
uv sync --extra dev --extra notebooks

uv run jupyter lab
```

Or if you don't want to install anything, just launch Jupyter Lab with:

```bash
uv run --extra dev --extra notebooks jupyter lab
```

## Tests
Run tests with:

```bash
uv run pytest tests
```

The API is still evolving while the package is prepared for early testing.

## Reference

Balzani, Edoardo, et al. "Efficient estimation of neural tuning during naturalistic behavior." *Advances in Neural Information Processing Systems 33* (2020): 12604-12614.

Paper: https://proceedings.neurips.cc/paper/2020/hash/94d2a3c6dd19337f2511cdf8b4bf907e-Abstract.html

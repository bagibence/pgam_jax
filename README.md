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

Run tests with:

```bash
uv run pytest tests
```

The API is still evolving while the package is prepared for early testing.

## Reference

Balzani, Edoardo, et al. "Efficient estimation of neural tuning during naturalistic behavior." *Advances in Neural Information Processing Systems 33* (2020): 12604-12614.

Paper: https://proceedings.neurips.cc/paper/2020/hash/94d2a3c6dd19337f2511cdf8b4bf907e-Abstract.html

# pgam_jax

`pgam_jax` is an alpha-stage, JAX-based implementation of the Poisson Generalized Additive Model (PGAM).

PGAM represents neural responses with B-splines regularized by smoothing penalties.

## Installation

### Supported platforms

This project currently targets Python 3.12, 3.13, and 3.14.

Tested on Linux and macOS. Native Windows isn't supported due to JAX's experimental Windows builds
and our experience with crashes in the CI.
Use [WSL](https://learn.microsoft.com/windows/wsl/) instead.

### Fresh install

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

### Existing environment

If you'd rather install into an existing environment:

```bash
pip install -e .
```

To include the development and notebook extras:

```bash
pip install -e ".[dev,notebooks]"
```

## Quick Start

The easiest way to try `pgam_jax` is the walkthrough notebook in `examples/quick_start.ipynb`.
It generates synthetic nonlinear Poisson data, fits a `GAM`, and visualizes the recovered rate surfaces.

To see how to use model temporal responses, see `examples/quick_start_temporal.ipynb` which is similar to the [original implementation's tutorial](https://github.com/BalzaniEdoardo/PGAM/blob/main/PGAM_Tutorial.ipynb).

Install with the required dependencies, then launch Jupyter Lab:
```bash
uv sync --extra dev --extra notebooks

uv run jupyter lab
```

Or if you don't want to install anything, just launch Jupyter Lab with:

```bash
uv run --extra dev --extra notebooks jupyter lab
```

The Quick Start commands assume `uv`. If you installed with `pip`, run `jupyter lab` (and `pytest`) directly in your environment instead.

## Performance

Consider turning on JAX's persistent compilation cache. It stores compiled kernels on disk, so a new session loads them instead of recompiling.

```python
import os

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax")) # or wherever you want to store it
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_compilation_cache_max_size", 1024**3)  # bytes, 1 GiB
```

JAX does not expand `~` itself. Without `os.path.expanduser` it creates a directory named `~` in the working directory.

Keep the minimum compile time at zero. A fit compiles a few hundred small kernels, and no single one takes more than a quarter of a second. The size cap bounds the cache directory instead. JAX evicts the least recently used entries once the directory exceeds it.

## Tests
Run tests with:

```bash
uv run pytest tests
```

The API is still evolving while the package is prepared for early testing.

## Reference

[Balzani, Edoardo, et al. "Efficient estimation of neural tuning during naturalistic behavior." *Advances in Neural Information Processing Systems 33* (2020): 12604-12614.](https://proceedings.neurips.cc/paper/2020/hash/94d2a3c6dd19337f2511cdf8b4bf907e-Abstract.html)

[Wood, S. N. (2017). Generalized additive models: An introduction with R (2nd ed.). Chapman and Hall/CRC.](https://doi.org/10.1201/9781315370279)

# PGAM Benchmarks

This directory contains the CPU benchmark harness for comparing `pgam_jax` with
the Dockerized legacy PGAM implementation.

The default legacy backend is the published Docker image:

```bash
edoardobalzani87/pgam:1.2
```

## Setup

Install the development environment from the repository root:

```bash
uv sync --extra dev
```

Start Docker before running benchmarks. The first legacy run may pull the image
if it is not already available; image pull time is not part of the in-container
fit timing.

## Run

Fast smoke run:

```bash
uv run python -m benchmarks.run_matrix --suite smoke
```

Larger run:

```bash
uv run python -m benchmarks.run_matrix --suite full --repetitions 5
```

JAX-only local smoke run:

```bash
uv run python -m benchmarks.run_matrix --suite smoke --skip-legacy
```

JAX-only local smoke run using SciPy's L-BFGS-B path:

```bash
uv run python -m benchmarks.run_matrix --suite smoke --skip-legacy --jax-use-scipy
```

Outputs are written under suite-specific directories in `benchmarks/artifacts/`:

- `smoke/cases/` and `full/cases/`: generated `.npz` arrays and JSON case metadata
- `smoke/results/` and `full/results/`: one JSON result and one prediction `.npz` per backend/case/repetition
- `smoke/summaries/` and `full/summaries/`: aggregate CSV and Markdown tables

`benchmarks/artifacts/` is ignored by git.

## Timing Columns

Legacy PGAM reports `fit`, the in-container `optim_gam` fit time. The host
wrapper also records `docker_wall`, which includes Docker process overhead.
The `model_total` timing includes legacy model setup plus `optim_gam`.

`pgam_jax` reports `fit_cold` and `fit_warm`. Use `fit_warm` for the primary
speedup ratio because the first JAX fit may include compilation overhead.
The `model_total_warm` timing includes nemos basis construction, `GAM`
construction, and the warm `GAM.fit` call.

When `--jax-use-scipy` is set, the JAX benchmark is written under the
`pgam_jax_scipy_cpu` backend name and summary tables include separate
`jax_scipy_*` columns.

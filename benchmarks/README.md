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

Fast smoke run (legacy Docker plus both pgam_jax variants, jaxopt and scipy):

```bash
uv run python -m benchmarks.run_matrix --suite smoke
```

Larger run:

```bash
uv run python -m benchmarks.run_matrix --suite full --repetitions 5
```

Restrict to a single pgam_jax variant:

```bash
uv run python -m benchmarks.run_matrix --suite smoke --jax-variants jaxopt
uv run python -m benchmarks.run_matrix --suite smoke --jax-variants scipy
```

JAX-only local smoke run:

```bash
uv run python -m benchmarks.run_matrix --suite smoke --skip-legacy
```

## Result Reuse

Existing result files are reused instead of re-run. pgam_jax results record
the repository commit in `runtime.git_commit`; when the current commit
differs, the stored result is treated as stale and re-run automatically.
Legacy results are always reused once present because the Docker image is
pinned. Pass `--overwrite-results` to force re-running everything. Running
from a repository with uncommitted changes prints a warning because the
commit stamp then cannot reproduce the benchmarked code.

## Failure Handling

Legacy PGAM can crash on large, flexible models (for example 10 smooths with
24 basis functions): its GCV objective overflows and feeds a non-finite
matrix to an SVD, which raises. Rather than aborting the whole matrix, the
runner captures such a crash as a result with `status: "failed"` and the
trailing stderr, then continues to the next case. Successful results carry
`status: "ok"`. The summary tables add a `legacy_status` column, and failed
runs are excluded from the timing medians and speedup ratios.

A failed result still counts as an existing result, so it is not retried on a
later run unless you pass `--overwrite-results`. Genuine environment problems
(Docker daemon unreachable, image missing: exit codes 125/126/127) are not
treated as fit failures and still abort the run.

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

The scipy variant is written under the `pgam_jax_scipy_cpu` backend name and
summary tables include separate `jax_scipy_*` columns. Summary tables also
report the short git commit behind each pgam_jax backend's results so mixed
code versions are visible at a glance.

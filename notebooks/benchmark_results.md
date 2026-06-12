---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Benchmark Results

Raw benchmark timing and score views from the per-run JSON artifacts.

```python
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```

```python
REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "benchmarks").exists():
    REPO_ROOT = REPO_ROOT.parent

ARTIFACT_SUITE = "full"  # "smoke", "full", or None for legacy un-namespaced artifacts
ARTIFACTS_DIR = REPO_ROOT / "benchmarks" / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / ARTIFACT_SUITE / "results" if ARTIFACT_SUITE is not None else ARTIFACTS_DIR / "results"
RESULT_RE = re.compile(r"(?P<case_id>.+)__(?P<backend>.+)__rep(?P<repetition>\d+)$")

RESULTS_DIR
```

```python
JAX_TIMING_MODE = "warm"  # "warm" or "cold"
JAX_BACKENDS = {"pgam_jax_cpu", "pgam_jax_scipy_cpu"}

if JAX_TIMING_MODE not in {"warm", "cold"}:
    raise ValueError("JAX_TIMING_MODE must be 'warm' or 'cold'.")
```

```python
def primary_fit_time_s(result: dict, jax_timing_mode: str = JAX_TIMING_MODE) -> float:
    timings = result["timings_s"]
    if result["backend"] in JAX_BACKENDS:
        return float(timings[f"fit_{jax_timing_mode}"])
    return float(timings["fit"])


def primary_model_total_time_s(result: dict, jax_timing_mode: str = JAX_TIMING_MODE) -> float:
    timings = result["timings_s"]
    if result["backend"] in JAX_BACKENDS:
        return float(timings.get(f"model_total_{jax_timing_mode}", timings[f"fit_{jax_timing_mode}"]))
    return float(timings.get("model_total", timings.get("setup", 0.0) + timings["fit"]))


def backend_label(backend: str) -> str:
    labels = {
        "pgam_jax_cpu": "pgam_jax",
        "pgam_jax_scipy_cpu": "pgam_jax + SciPy",
        "legacy_pgam_docker_cpu": "legacy PGAM",
    }
    return labels.get(backend, backend)


def load_result_rows(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text())
        match = RESULT_RE.match(path.stem)
        case = result["case"]
        timings = result["timings_s"]
        metrics = result.get("metrics", {})
        rows.append(
            {
                "path": path.relative_to(REPO_ROOT),
                "case_id": case["case_id"],
                "backend": result["backend"],
                "backend_label": backend_label(result["backend"]),
                "repetition": int(match["repetition"]) if match else None,
                "seed": case["seed"],
                "n_observations": case["n_observations"],
                "n_smooths": case["n_smooths"],
                "n_basis": case["n_basis"],
                "fit_time_s": primary_fit_time_s(result),
                "model_total_time_s": primary_model_total_time_s(result),
                "jax_timing_mode": JAX_TIMING_MODE if result["backend"] in JAX_BACKENDS else "legacy",
                "use_scipy": result.get("options", {}).get("use_scipy", False),
                "load_time_s": timings.get("load"),
                "setup_time_s": timings.get("setup", timings.get("setup_warm")),
                "predict_time_s": timings.get("predict"),
                "docker_wall_time_s": timings.get("docker_wall"),
                "prediction_mean": metrics.get("prediction_mean"),
                "prediction_std": metrics.get("prediction_std"),
                "prediction_min": metrics.get("prediction_min"),
                "prediction_max": metrics.get("prediction_max"),
                "mean_poisson_nll_no_constant": metrics.get("mean_poisson_nll_no_constant"),
                "corr_y_prediction": metrics.get("corr_y_prediction"),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No benchmark result JSONs found in {results_dir}. " "Set ARTIFACT_SUITE to 'smoke', 'full', or None."
        )

    df = pd.DataFrame(rows)
    ordered_backends = ["legacy PGAM", "pgam_jax", "pgam_jax + SciPy"]
    df["backend_label"] = pd.Categorical(df["backend_label"], categories=ordered_backends, ordered=True)
    for column in ["n_observations", "n_smooths", "n_basis", "seed", "repetition"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["n_observations", "n_basis", "n_smooths", "seed", "backend_label", "repetition"])


results = load_result_rows()
```

```python
results.groupby(["backend_label", "n_observations", "n_basis"], observed=True).size().rename("runs").reset_index()
```

```python
def timing_long(df: pd.DataFrame) -> pd.DataFrame:
    return df.melt(
        id_vars=[
            "case_id",
            "backend",
            "backend_label",
            "repetition",
            "seed",
            "n_observations",
            "n_smooths",
            "n_basis",
        ],
        value_vars=["fit_time_s", "model_total_time_s"],
        var_name="timing_kind",
        value_name="time_s",
    ).dropna(subset=["time_s"])


times = timing_long(results)
```

```python
def score_long(df: pd.DataFrame) -> pd.DataFrame:
    score_columns = [
        "mean_poisson_nll_no_constant",
        "corr_y_prediction",
        "prediction_mean",
        "prediction_std",
    ]
    labels = {
        "mean_poisson_nll_no_constant": "mean Poisson NLL",
        "corr_y_prediction": "corr(y, prediction)",
        "prediction_mean": "prediction mean",
        "prediction_std": "prediction std",
    }
    long = df.melt(
        id_vars=[
            "case_id",
            "backend",
            "backend_label",
            "repetition",
            "seed",
            "n_observations",
            "n_smooths",
            "n_basis",
        ],
        value_vars=score_columns,
        var_name="score_kind",
        value_name="score",
    ).dropna(subset=["score"])
    long["score_label"] = long["score_kind"].map(labels)
    return long


scores = score_long(results)
```

## Primary Fit Time

Legacy uses `timings_s.fit`; both `pgam_jax` and `pgam_jax + SciPy` use
`timings_s.fit_cold` or `timings_s.fit_warm`, depending on `JAX_TIMING_MODE`.

```python
def stripplot_grid(data: pd.DataFrame, title: str) -> sns.FacetGrid:
    grid = sns.catplot(
        data=data,
        kind="strip",
        row="n_observations",
        col="n_smooths",
        x="n_basis",
        y="time_s",
        hue="backend_label",
        dodge=True,
        jitter=0.18,
        alpha=0.85,
        # s=10,
        height=3.2,
        aspect=1.25,
        sharey=False,
        margin_titles=True,
    )
    grid.set_axis_labels("basis functions", "time (s)")
    grid.set_titles(row_template="n = {row_name}", col_template="smooths = {col_name}")
    grid.figure.subplots_adjust(top=0.92)
    grid.figure.suptitle(title)
    return grid


sns.set_theme(style="whitegrid", context="notebook")

fit_times = times[times["timing_kind"] == "fit_time_s"]
stripplot_grid(fit_times, f"Primary fit time by benchmark case; JAX={JAX_TIMING_MODE}")
plt.show()
```

## Model Total Time

Legacy uses `timings_s.model_total`; both `pgam_jax` and `pgam_jax + SciPy` use
`timings_s.model_total_cold` or `timings_s.model_total_warm`, depending on
`JAX_TIMING_MODE`.

```python
model_total_times = times[times["timing_kind"] == "model_total_time_s"]
stripplot_grid(model_total_times, f"Model total time by benchmark case; JAX={JAX_TIMING_MODE}")
plt.show()
```

## Score Comparison

These panels compare the fitted solutions using metrics stored in each result
JSON. Lower is better for mean Poisson NLL; higher is better for
`corr(y, prediction)`. The prediction mean and standard deviation are included
as quick sanity checks that the fitted rates are on the same scale.

```python
def score_stripplot_grid(data: pd.DataFrame, score_kind: str, title: str) -> sns.FacetGrid:
    plot_data = data[data["score_kind"] == score_kind]
    grid = sns.catplot(
        data=plot_data,
        kind="strip",
        row="n_observations",
        col="n_smooths",
        x="n_basis",
        y="score",
        hue="backend_label",
        dodge=True,
        jitter=0.18,
        alpha=0.85,
        height=3.2,
        aspect=1.25,
        sharey=False,
        margin_titles=True,
    )
    grid.set_axis_labels("basis functions", plot_data["score_label"].iloc[0])
    grid.set_titles(row_template="n = {row_name}", col_template="smooths = {col_name}")
    grid.figure.subplots_adjust(top=0.92)
    grid.figure.suptitle(title)
    return grid


score_stripplot_grid(scores, "mean_poisson_nll_no_constant", "Mean Poisson NLL by benchmark case")
plt.show()

score_stripplot_grid(scores, "corr_y_prediction", "Prediction correlation by benchmark case")
plt.show()
```

## Score Deltas From Legacy

These plots report the signed difference from legacy PGAM for the two
`pgam_jax` implementations. Values close to zero mean the implementations
reached similar fitted-rate summaries for that case; positive and negative
values show which side of legacy each implementation landed on.

```python
score_summary = (
    results.groupby(
        ["case_id", "n_observations", "n_smooths", "n_basis", "backend_label"],
        observed=True,
    )[
        [
            "mean_poisson_nll_no_constant",
            "corr_y_prediction",
            "prediction_mean",
            "prediction_std",
        ]
    ]
    .median()
    .reset_index()
)

score_wide = score_summary.pivot(
    index=["case_id", "n_observations", "n_smooths", "n_basis"],
    columns="backend_label",
)
score_wide.columns = [f"{metric}__{backend}" for metric, backend in score_wide.columns]
score_wide = score_wide.reset_index()

for metric in [
    "mean_poisson_nll_no_constant",
    "corr_y_prediction",
    "prediction_mean",
    "prediction_std",
]:
    legacy_col = f"{metric}__legacy PGAM"
    for backend in ["pgam_jax", "pgam_jax + SciPy"]:
        backend_col = f"{metric}__{backend}"
        if legacy_col in score_wide and backend_col in score_wide:
            score_wide[f"{metric}__delta_{backend}"] = score_wide[backend_col] - score_wide[legacy_col]

```

```python
delta_labels = {
    "mean_poisson_nll_no_constant": "delta in mean Poisson NLL",
    "corr_y_prediction": "delta in corr(y, prediction)",
    "prediction_mean": "delta in prediction mean",
    "prediction_std": "delta in prediction std",
}

delta_rows = []
for metric, label in delta_labels.items():
    for backend in ["pgam_jax", "pgam_jax + SciPy"]:
        column = f"{metric}__delta_{backend}"
        if column not in score_wide:
            continue
        for _, row in score_wide.iterrows():
            value = row[column]
            if pd.isna(value):
                continue
            delta_rows.append(
                {
                    "case_id": row["case_id"],
                    "n_observations": row["n_observations"],
                    "n_smooths": row["n_smooths"],
                    "n_basis": row["n_basis"],
                    "backend_label": backend,
                    "metric": metric,
                    "metric_label": label,
                    "delta_from_legacy": value,
                }
            )

score_deltas = pd.DataFrame(delta_rows)


def delta_stripplot_grid(data: pd.DataFrame, metric: str, title: str) -> sns.FacetGrid:
    plot_data = data[data["metric"] == metric]
    grid = sns.catplot(
        data=plot_data,
        kind="strip",
        row="n_observations",
        col="n_smooths",
        x="n_basis",
        y="delta_from_legacy",
        hue="backend_label",
        dodge=True,
        jitter=0.18,
        alpha=0.85,
        height=3.2,
        aspect=1.25,
        sharey=False,
        margin_titles=True,
    )
    grid.refline(y=0.0, color="0.35", linestyle="--", linewidth=1)
    grid.set_axis_labels("basis functions", plot_data["metric_label"].iloc[0])
    grid.set_titles(row_template="n = {row_name}", col_template="smooths = {col_name}")
    grid.figure.subplots_adjust(top=0.92)
    grid.figure.suptitle(title)
    return grid


delta_stripplot_grid(score_deltas, "mean_poisson_nll_no_constant", "NLL delta from legacy PGAM")
plt.show()

delta_stripplot_grid(score_deltas, "corr_y_prediction", "Prediction-correlation delta from legacy PGAM")
plt.show()
```

```python
scale_deltas = score_deltas[score_deltas["metric"].isin(["prediction_mean", "prediction_std"])]
grid = sns.catplot(
    data=scale_deltas,
    kind="strip",
    row="metric_label",
    col="n_smooths",
    x="n_basis",
    y="delta_from_legacy",
    hue="backend_label",
    dodge=True,
    jitter=0.18,
    alpha=0.85,
    height=3.2,
    aspect=1.25,
    sharey=False,
    margin_titles=True,
)
grid.refline(y=0.0, color="0.35", linestyle="--", linewidth=1)
grid.set_axis_labels("basis functions", "delta from legacy")
grid.set_titles(row_template="{row_name}", col_template="smooths = {col_name}")
grid.figure.subplots_adjust(top=0.9)
grid.figure.suptitle("Prediction-scale deltas from legacy PGAM")
plt.show()
```

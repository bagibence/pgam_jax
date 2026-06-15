from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common import RESULTS_DIR, SUMMARIES_DIR, artifact_dirs, read_json

JAX_BACKENDS = {
    "pgam_jax_cpu",
    "pgam_jax_scipy_cpu",
    "pgam_jax_noglm_cpu",
    "pgam_jax_scipy_noglm_cpu",
}


def _primary_fit_time(result: dict[str, Any]) -> float:
    timings = result["timings_s"]
    if result["backend"] in JAX_BACKENDS:
        return float(timings["fit_warm"])
    return float(timings["fit"])


def _primary_model_total_time(result: dict[str, Any]) -> float:
    timings = result["timings_s"]
    if result["backend"] in JAX_BACKENDS:
        return float(timings.get("model_total_warm", timings["fit_warm"]))
    return float(timings.get("model_total", timings.get("setup", 0.0) + timings["fit"]))


def _short_commits(runs: list[tuple[Path, dict[str, Any]]]) -> str | None:
    """Return the distinct short commits behind a backend's runs, or None."""
    if not runs:
        return None
    commits = set()
    for _path, result in runs:
        runtime = result.get("runtime")
        commit = runtime.get("git_commit") if isinstance(runtime, dict) else None
        commits.add("unknown" if commit is None else str(commit)[:9])
    return ",".join(sorted(commits))


def _ok_runs(
    runs: list[tuple[Path, dict[str, Any]]],
) -> list[tuple[Path, dict[str, Any]]]:
    """Return runs that did not record a failure (a missing status counts as ok)."""
    return [
        (path, result)
        for path, result in runs
        if result.get("status", "ok") != "failed"
    ]


def _backend_status(runs: list[tuple[Path, dict[str, Any]]]) -> str | None:
    """Summarize a backend's per-case statuses as ok/failed/partial, or None."""
    if not runs:
        return None
    n_ok = len(_ok_runs(runs))
    n_failed = len(runs) - n_ok
    if n_failed == 0:
        return "ok"
    if n_ok == 0:
        return "failed"
    return f"partial:{n_ok}ok/{n_failed}failed"


def _prediction_path(result: dict[str, Any], result_path: Path) -> Path | None:
    raw_path = result.get("prediction_path")
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    candidate = result_path.parent / path.name
    return candidate if candidate.exists() else None


def _prediction_rmse(left: Path | None, right: Path | None) -> float | None:
    if left is None or right is None:
        return None
    left_pred = np.load(left)["prediction"]
    right_pred = np.load(right)["prediction"]
    return float(np.sqrt(np.mean((left_pred - right_pred) ** 2)))


def collect_results(
    results_dir: Path = RESULTS_DIR,
) -> list[tuple[Path, dict[str, Any]]]:
    """Read all result JSON files in a directory."""
    return [(path, read_json(path)) for path in sorted(results_dir.glob("*.json"))]


def _jax_variant_summary(
    runs: list[tuple[Path, dict[str, Any]]],
    legacy_ok: list[tuple[Path, dict[str, Any]]],
    legacy_fit_median: float | None,
    legacy_model_total_median: float | None,
    prefix: str,
) -> dict[str, Any]:
    """
    Summarize one pgam_jax variant's runs as prefixed columns.

    Mirrors the inline jax/jax_scipy aggregation so additional variants (e.g.
    the GLM-init-disabled backends) report the same per-backend timing,
    speedup, and prediction-RMSE columns under their own prefix.
    """
    ok = _ok_runs(runs)
    fit_times = [_primary_fit_time(result) for _path, result in ok]
    model_total_times = [_primary_model_total_time(result) for _path, result in ok]
    fit_median = statistics.median(fit_times) if fit_times else None
    model_total_median = (
        statistics.median(model_total_times) if model_total_times else None
    )
    speedup = (
        legacy_fit_median / fit_median
        if legacy_fit_median is not None and fit_median
        else None
    )
    model_total_speedup = (
        legacy_model_total_median / model_total_median
        if legacy_model_total_median is not None and model_total_median
        else None
    )
    pred_rmse = None
    if legacy_ok and ok:
        pred_rmse = _prediction_rmse(
            _prediction_path(legacy_ok[0][1], legacy_ok[0][0]),
            _prediction_path(ok[0][1], ok[0][0]),
        )
    return {
        f"{prefix}_runs": len(runs),
        f"{prefix}_status": _backend_status(runs),
        f"{prefix}_git_commit": _short_commits(runs),
        f"{prefix}_fit_warm_median_s": fit_median,
        f"{prefix}_model_total_warm_median_s": model_total_median,
        f"speedup_legacy_over_{prefix}": speedup,
        f"model_total_speedup_legacy_over_{prefix}": model_total_speedup,
        f"prediction_rmse_{prefix}_first_rep": pred_rmse,
    }


def summarize_results(
    results: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate result JSON payloads into one row per case."""
    grouped: dict[str, dict[str, list[tuple[Path, dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path, result in results:
        grouped[result["case"]["case_id"]][result["backend"]].append((path, result))

    rows: list[dict[str, Any]] = []
    for case_id, by_backend in sorted(grouped.items()):
        legacy_runs = by_backend.get("legacy_pgam_docker_cpu", [])
        legacy_ok = _ok_runs(legacy_runs)
        legacy_status = _backend_status(legacy_runs)
        jax_runs = by_backend.get("pgam_jax_cpu", [])
        jax_ok = _ok_runs(jax_runs)
        jax_status = _backend_status(jax_runs)
        jax_scipy_runs = by_backend.get("pgam_jax_scipy_cpu", [])
        jax_scipy_ok = _ok_runs(jax_scipy_runs)
        jax_scipy_status = _backend_status(jax_scipy_runs)
        legacy_times = [_primary_fit_time(result) for _path, result in legacy_ok]
        jax_times = [_primary_fit_time(result) for _path, result in jax_ok]
        jax_scipy_times = [_primary_fit_time(result) for _path, result in jax_scipy_ok]
        legacy_model_total_times = [
            _primary_model_total_time(result) for _path, result in legacy_ok
        ]
        jax_model_total_times = [
            _primary_model_total_time(result) for _path, result in jax_ok
        ]
        jax_scipy_model_total_times = [
            _primary_model_total_time(result) for _path, result in jax_scipy_ok
        ]
        legacy_median = statistics.median(legacy_times) if legacy_times else None
        jax_median = statistics.median(jax_times) if jax_times else None
        jax_scipy_median = (
            statistics.median(jax_scipy_times) if jax_scipy_times else None
        )
        legacy_model_total_median = (
            statistics.median(legacy_model_total_times)
            if legacy_model_total_times
            else None
        )
        jax_model_total_median = (
            statistics.median(jax_model_total_times) if jax_model_total_times else None
        )
        jax_scipy_model_total_median = (
            statistics.median(jax_scipy_model_total_times)
            if jax_scipy_model_total_times
            else None
        )
        speedup = (
            legacy_median / jax_median
            if legacy_median is not None and jax_median
            else None
        )
        scipy_speedup = (
            legacy_median / jax_scipy_median
            if legacy_median is not None and jax_scipy_median
            else None
        )
        model_total_speedup = (
            legacy_model_total_median / jax_model_total_median
            if legacy_model_total_median is not None and jax_model_total_median
            else None
        )
        scipy_model_total_speedup = (
            legacy_model_total_median / jax_scipy_model_total_median
            if legacy_model_total_median is not None and jax_scipy_model_total_median
            else None
        )

        pred_rmse = None
        if legacy_ok and jax_ok:
            pred_rmse = _prediction_rmse(
                _prediction_path(legacy_ok[0][1], legacy_ok[0][0]),
                _prediction_path(jax_ok[0][1], jax_ok[0][0]),
            )

        scipy_pred_rmse = None
        if legacy_ok and jax_scipy_ok:
            scipy_pred_rmse = _prediction_rmse(
                _prediction_path(legacy_ok[0][1], legacy_ok[0][0]),
                _prediction_path(jax_scipy_ok[0][1], jax_scipy_ok[0][0]),
            )

        jax_noglm_summary = _jax_variant_summary(
            by_backend.get("pgam_jax_noglm_cpu", []),
            legacy_ok,
            legacy_median,
            legacy_model_total_median,
            "jax_noglm",
        )
        jax_scipy_noglm_summary = _jax_variant_summary(
            by_backend.get("pgam_jax_scipy_noglm_cpu", []),
            legacy_ok,
            legacy_median,
            legacy_model_total_median,
            "jax_scipy_noglm",
        )

        representative = (
            legacy_runs
            or jax_runs
            or jax_scipy_runs
            or by_backend.get("pgam_jax_noglm_cpu", [])
            or by_backend.get("pgam_jax_scipy_noglm_cpu", [])
        )[0][1]
        rows.append(
            {
                "case_id": case_id,
                "n_observations": representative["case"]["n_observations"],
                "n_smooths": representative["case"]["n_smooths"],
                "n_basis": representative["case"]["n_basis"],
                "legacy_runs": len(legacy_runs),
                "legacy_status": legacy_status,
                "jax_runs": len(jax_runs),
                "jax_status": jax_status,
                "jax_scipy_runs": len(jax_scipy_runs),
                "jax_scipy_status": jax_scipy_status,
                "jax_git_commit": _short_commits(jax_runs),
                "jax_scipy_git_commit": _short_commits(jax_scipy_runs),
                "legacy_fit_median_s": legacy_median,
                "jax_fit_warm_median_s": jax_median,
                "jax_scipy_fit_warm_median_s": jax_scipy_median,
                "speedup_legacy_over_jax": speedup,
                "speedup_legacy_over_jax_scipy": scipy_speedup,
                "legacy_model_total_median_s": legacy_model_total_median,
                "jax_model_total_warm_median_s": jax_model_total_median,
                "jax_scipy_model_total_warm_median_s": jax_scipy_model_total_median,
                "model_total_speedup_legacy_over_jax": model_total_speedup,
                "model_total_speedup_legacy_over_jax_scipy": scipy_model_total_speedup,
                "prediction_rmse_first_rep": pred_rmse,
                "prediction_rmse_scipy_first_rep": scipy_pred_rmse,
                **jax_noglm_summary,
                **jax_scipy_noglm_summary,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write summary rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a compact Markdown summary table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = (
        "case_id",
        "n_observations",
        "n_smooths",
        "n_basis",
        "legacy_status",
        "jax_status",
        "jax_scipy_status",
        "jax_noglm_status",
        "jax_scipy_noglm_status",
        "legacy_fit_median_s",
        "jax_fit_warm_median_s",
        "jax_scipy_fit_warm_median_s",
        "jax_noglm_fit_warm_median_s",
        "jax_scipy_noglm_fit_warm_median_s",
        "speedup_legacy_over_jax",
        "speedup_legacy_over_jax_scipy",
        "speedup_legacy_over_jax_noglm",
        "speedup_legacy_over_jax_scipy_noglm",
        "legacy_model_total_median_s",
        "jax_model_total_warm_median_s",
        "jax_scipy_model_total_warm_median_s",
        "jax_noglm_model_total_warm_median_s",
        "jax_scipy_noglm_model_total_warm_median_s",
        "model_total_speedup_legacy_over_jax",
        "model_total_speedup_legacy_over_jax_scipy",
        "model_total_speedup_legacy_over_jax_noglm",
        "model_total_speedup_legacy_over_jax_scipy_noglm",
        "prediction_rmse_first_rep",
        "prediction_rmse_scipy_first_rep",
        "prediction_rmse_jax_noglm_first_rep",
        "prediction_rmse_jax_scipy_noglm_first_rep",
        "jax_git_commit",
        "jax_scipy_git_commit",
        "jax_noglm_git_commit",
        "jax_scipy_noglm_git_commit",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for key in headers:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize pgam_jax benchmark results."
    )
    parser.add_argument("--suite", choices=("smoke", "full"))
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dirs = artifact_dirs(args.suite) if args.suite else None
    results_dir = args.results_dir or (dirs.results if dirs else RESULTS_DIR)
    output_csv = args.output_csv or (
        (dirs.summaries if dirs else SUMMARIES_DIR) / "summary.csv"
    )
    output_md = args.output_md or (
        (dirs.summaries if dirs else SUMMARIES_DIR) / "summary.md"
    )
    rows = summarize_results(collect_results(results_dir))
    write_csv(rows, output_csv)
    write_markdown(rows, output_md)
    print(output_csv)
    print(output_md)


if __name__ == "__main__":
    main()

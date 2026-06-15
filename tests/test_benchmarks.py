import json

import numpy as np
import pytest

from benchmarks.common import (
    ARTIFACTS_DIR,
    CaseSpec,
    artifact_dirs,
    git_commit,
    jax_backend_name,
    pgam_jax_tree,
    prediction_summary,
    read_json,
    result_stem,
)
from benchmarks.make_cases import generate_case, write_case
from benchmarks.restamp_results import REPO_ROOT, restamp_result
from benchmarks.run_legacy_pgam import build_docker_command
from benchmarks.run_matrix import _should_run_jax
from benchmarks.summarize import summarize_results, write_csv


def test_generate_case_is_deterministic():
    spec = CaseSpec("tiny", 20, 2, 12, 123, active_smooths=1)
    left = generate_case(spec)
    right = generate_case(spec)
    assert np.array_equal(left["x"], right["x"])
    assert np.array_equal(left["y"], right["y"])


def test_write_case_round_trips_metadata(tmp_path):
    spec = CaseSpec("tiny", 20, 2, 12, 123, active_smooths=1)
    case_path, metadata_path = write_case(spec, tmp_path)
    assert case_path.exists()
    metadata = read_json(metadata_path)
    assert metadata["case_id"] == "tiny"
    assert metadata["n_knots"] == 10


def test_artifact_dirs_can_be_namespaced_by_suite():
    dirs = artifact_dirs("smoke")
    assert dirs.root == ARTIFACTS_DIR / "smoke"
    assert dirs.cases == ARTIFACTS_DIR / "smoke" / "cases"
    assert dirs.results == ARTIFACTS_DIR / "smoke" / "results"
    assert dirs.summaries == ARTIFACTS_DIR / "smoke" / "summaries"


def test_build_docker_command_mounts_benchmarks(tmp_path):
    case_path = tmp_path / "case.npz"
    metadata_path = tmp_path / "case.json"
    output_path = tmp_path / "results" / "out.json"
    prediction_path = tmp_path / "results" / "out.npz"
    command = build_docker_command(
        case_path=case_path,
        metadata_path=metadata_path,
        output_path=output_path,
        prediction_output_path=prediction_path,
    )
    assert command[:3] == ["docker", "run", "--rm"]
    assert "edoardobalzani87/pgam:1.2" in command
    assert "benchmarks.legacy_worker" in command
    assert f"/artifacts/cases/{case_path.name}" in command


def test_prediction_summary_handles_constant_predictions():
    summary = prediction_summary(np.array([0, 1, 2]), np.ones(3))
    assert np.isnan(summary["corr_y_prediction"])
    assert summary["prediction_mean"] == 1.0


def test_should_run_jax_reruns_results_from_other_pgam_jax_trees(tmp_path):
    result_path = tmp_path / "result.json"

    assert _should_run_jax(result_path, "tree_a", overwrite=False)

    result_path.write_text(
        json.dumps({"runtime": {"pgam_jax_tree": "tree_a"}}), encoding="utf-8"
    )
    assert not _should_run_jax(result_path, "tree_a", overwrite=False)
    assert _should_run_jax(result_path, "tree_b", overwrite=False)
    assert _should_run_jax(result_path, "tree_a", overwrite=True)
    # Unknown current library tree cannot invalidate existing results.
    assert not _should_run_jax(result_path, None, overwrite=False)

    result_path.write_text(json.dumps({"backend": "pgam_jax_cpu"}), encoding="utf-8")
    assert _should_run_jax(result_path, "tree_a", overwrite=False)


def test_restamp_result_backfills_tree_from_commit(tmp_path):
    head = git_commit(REPO_ROOT)
    expected = pgam_jax_tree(REPO_ROOT, ref="HEAD")
    if head is None or expected is None:
        pytest.skip("git metadata unavailable")

    path = tmp_path / "result.json"
    path.write_text(
        json.dumps({"backend": "pgam_jax_cpu", "runtime": {"git_commit": head}}),
        encoding="utf-8",
    )
    assert restamp_result(path) == expected
    assert read_json(path)["runtime"]["pgam_jax_tree"] == expected
    # Idempotent: an already stamped result is skipped.
    assert restamp_result(path) is None

    # Non-pgam_jax backends are left untouched.
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {"backend": "legacy_pgam_docker_cpu", "runtime": {"git_commit": head}}
        ),
        encoding="utf-8",
    )
    assert restamp_result(legacy) is None


def test_summarize_marks_failed_legacy_and_skips_its_timing(tmp_path):
    case = {"case_id": "case", "n_observations": 20, "n_smooths": 2, "n_basis": 12}
    failed_legacy = {
        "backend": "legacy_pgam_docker_cpu",
        "status": "failed",
        "case": case,
        "timings_s": {"docker_wall": 1.0},
        "error": {"returncode": 1, "stderr_tail": "overflow"},
    }
    jax = {
        "backend": "pgam_jax_cpu",
        "case": case,
        "timings_s": {"fit_warm": 0.5},
    }
    failed_path = tmp_path / f"{result_stem('case', 'legacy_pgam_docker_cpu', 0)}.json"
    jax_path = tmp_path / f"{result_stem('case', 'pgam_jax_cpu', 0)}.json"
    failed_path.write_text(json.dumps(failed_legacy), encoding="utf-8")
    jax_path.write_text(json.dumps(jax), encoding="utf-8")

    rows = summarize_results([(failed_path, failed_legacy), (jax_path, jax)])
    row = rows[0]
    assert row["legacy_status"] == "failed"
    assert row["legacy_fit_median_s"] is None
    assert row["speedup_legacy_over_jax"] is None
    assert row["jax_fit_warm_median_s"] == 0.5


def test_summarize_marks_failed_jax_and_skips_its_timing(tmp_path):
    case = {"case_id": "case", "n_observations": 20, "n_smooths": 2, "n_basis": 12}
    legacy = {
        "backend": "legacy_pgam_docker_cpu",
        "status": "ok",
        "case": case,
        "timings_s": {"fit": 2.0},
    }
    failed_jax = {
        "backend": "pgam_jax_cpu",
        "status": "failed",
        "case": case,
        "timings_s": {"wall": 1.0},
        "error": {"returncode": 1, "stderr_tail": "nan"},
    }
    legacy_path = tmp_path / f"{result_stem('case', 'legacy_pgam_docker_cpu', 0)}.json"
    failed_path = tmp_path / f"{result_stem('case', 'pgam_jax_cpu', 0)}.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    failed_path.write_text(json.dumps(failed_jax), encoding="utf-8")

    rows = summarize_results([(legacy_path, legacy), (failed_path, failed_jax)])
    row = rows[0]
    assert row["jax_status"] == "failed"
    assert row["jax_fit_warm_median_s"] is None
    assert row["speedup_legacy_over_jax"] is None
    assert row["legacy_fit_median_s"] == 2.0


def test_jax_backend_name_maps_scipy_and_glm_init():
    assert jax_backend_name(use_scipy=False, use_glm_init=True) == "pgam_jax_cpu"
    assert jax_backend_name(use_scipy=True, use_glm_init=True) == "pgam_jax_scipy_cpu"
    assert jax_backend_name(use_scipy=False, use_glm_init=False) == "pgam_jax_noglm_cpu"
    assert (
        jax_backend_name(use_scipy=True, use_glm_init=False)
        == "pgam_jax_scipy_noglm_cpu"
    )


def test_summarize_includes_noglm_backends(tmp_path):
    case = {"case_id": "case", "n_observations": 20, "n_smooths": 2, "n_basis": 12}
    legacy = {
        "backend": "legacy_pgam_docker_cpu",
        "case": case,
        "timings_s": {"fit": 2.0},
    }
    jax_noglm = {
        "backend": "pgam_jax_noglm_cpu",
        "case": case,
        "timings_s": {"fit_warm": 0.5},
    }
    jax_scipy_noglm = {
        "backend": "pgam_jax_scipy_noglm_cpu",
        "case": case,
        "timings_s": {"fit_warm": 1.0},
    }
    runs = []
    for backend, result in (
        ("legacy_pgam_docker_cpu", legacy),
        ("pgam_jax_noglm_cpu", jax_noglm),
        ("pgam_jax_scipy_noglm_cpu", jax_scipy_noglm),
    ):
        path = tmp_path / f"{result_stem('case', backend, 0)}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        runs.append((path, result))

    row = summarize_results(runs)[0]
    assert row["jax_noglm_status"] == "ok"
    assert row["jax_noglm_fit_warm_median_s"] == 0.5
    assert row["speedup_legacy_over_jax_noglm"] == 4.0
    assert row["jax_scipy_noglm_fit_warm_median_s"] == 1.0
    assert row["speedup_legacy_over_jax_scipy_noglm"] == 2.0


def test_summarize_results_and_write_csv(tmp_path):
    case = {"case_id": "case", "n_observations": 20, "n_smooths": 2, "n_basis": 12}
    legacy = {
        "backend": "legacy_pgam_docker_cpu",
        "case": case,
        "timings_s": {"fit": 2.0},
    }
    jax = {
        "backend": "pgam_jax_cpu",
        "case": case,
        "timings_s": {"fit_warm": 0.5},
    }
    legacy_path = tmp_path / f"{result_stem('case', 'legacy_pgam_docker_cpu', 0)}.json"
    jax_path = tmp_path / f"{result_stem('case', 'pgam_jax_cpu', 0)}.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    jax_path.write_text(json.dumps(jax), encoding="utf-8")

    rows = summarize_results([(legacy_path, legacy), (jax_path, jax)])
    assert rows[0]["speedup_legacy_over_jax"] == 4.0

    output = tmp_path / "summary.csv"
    write_csv(rows, output)
    assert "speedup_legacy_over_jax" in output.read_text(encoding="utf-8")

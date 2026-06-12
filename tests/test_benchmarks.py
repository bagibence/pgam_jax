import json

import numpy as np

from benchmarks.common import (
    ARTIFACTS_DIR,
    CaseSpec,
    artifact_dirs,
    prediction_summary,
    read_json,
    result_stem,
)
from benchmarks.make_cases import generate_case, write_case
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


def test_should_run_jax_reruns_results_from_other_commits(tmp_path):
    result_path = tmp_path / "result.json"

    assert _should_run_jax(result_path, "commit_a", overwrite=False)

    result_path.write_text(json.dumps({"runtime": {"git_commit": "commit_a"}}), encoding="utf-8")
    assert not _should_run_jax(result_path, "commit_a", overwrite=False)
    assert _should_run_jax(result_path, "commit_b", overwrite=False)
    assert _should_run_jax(result_path, "commit_a", overwrite=True)
    # Unknown current commit cannot invalidate existing results.
    assert not _should_run_jax(result_path, None, overwrite=False)

    result_path.write_text(json.dumps({"backend": "pgam_jax_cpu"}), encoding="utf-8")
    assert _should_run_jax(result_path, "commit_a", overwrite=False)


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

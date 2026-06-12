from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.common import artifact_dirs, default_cpu_env, read_json, result_stem, write_json
from benchmarks.make_cases import SUITES, write_suite
from benchmarks.run_legacy_pgam import build_docker_command
from benchmarks.summarize import collect_results, summarize_results, write_csv, write_markdown


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pgam_jax benchmark matrix.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-jax", action="store_true")
    parser.add_argument("--overwrite-cases", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--jax-use-scipy", action="store_true")
    parser.add_argument("--summary-name", default="summary")
    return parser.parse_args()


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _run_legacy(command: list[str], output_path: Path) -> None:
    t0 = time.perf_counter()
    _run(command, env=default_cpu_env())
    payload = read_json(output_path)
    payload.setdefault("timings_s", {})["docker_wall"] = time.perf_counter() - t0
    write_json(output_path, payload)


def _run_jax(
    case_path: Path,
    metadata_path: Path,
    output_path: Path,
    prediction_output_path: Path,
    max_iter: int,
    use_scipy: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "benchmarks.run_pgam_jax",
        "--case",
        str(case_path),
        "--metadata",
        str(metadata_path),
        "--output",
        str(output_path),
        "--prediction-output",
        str(prediction_output_path),
        "--backend",
        "cpu",
        "--max-iter",
        str(max_iter),
    ]
    if use_scipy:
        command.append("--use-scipy")
    _run(command, env=default_cpu_env())


def main() -> None:
    args = _parse_args()
    dirs = artifact_dirs(args.suite)
    cases = write_suite(args.suite, dirs.cases, overwrite=args.overwrite_cases)

    for repetition in range(args.repetitions):
        for case_path, metadata_path in cases:
            case_id = case_path.stem
            if not args.skip_legacy:
                stem = result_stem(case_id, "legacy_pgam_docker_cpu", repetition)
                output_path = dirs.results / f"{stem}.json"
                prediction_path = dirs.results / f"{stem}.npz"
                if args.overwrite_results or not output_path.exists():
                    _run_legacy(
                        build_docker_command(
                            case_path=case_path,
                            metadata_path=metadata_path,
                            output_path=output_path,
                            prediction_output_path=prediction_path,
                            max_iter=args.max_iter,
                        ),
                        output_path,
                    )
            if not args.skip_jax:
                jax_backend = "pgam_jax_scipy_cpu" if args.jax_use_scipy else "pgam_jax_cpu"
                stem = result_stem(case_id, jax_backend, repetition)
                output_path = dirs.results / f"{stem}.json"
                prediction_path = dirs.results / f"{stem}.npz"
                if args.overwrite_results or not output_path.exists():
                    _run_jax(
                        case_path,
                        metadata_path,
                        output_path,
                        prediction_path,
                        args.max_iter,
                        args.jax_use_scipy,
                    )

    rows = summarize_results(collect_results(dirs.results))
    write_csv(rows, dirs.summaries / f"{args.summary_name}.csv")
    write_markdown(rows, dirs.summaries / f"{args.summary_name}.md")
    print(dirs.summaries / f"{args.summary_name}.md")


if __name__ == "__main__":
    main()

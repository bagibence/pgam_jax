from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.common import (
    artifact_dirs,
    default_cpu_env,
    git_commit,
    git_dirty,
    read_json,
    result_stem,
    write_json,
)
from benchmarks.make_cases import SUITES, write_suite
from benchmarks.run_legacy_pgam import build_docker_command
from benchmarks.summarize import (
    collect_results,
    summarize_results,
    write_csv,
    write_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

JAX_VARIANT_BACKENDS = {
    "jaxopt": "pgam_jax_cpu",
    "scipy": "pgam_jax_scipy_cpu",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pgam_jax benchmark matrix.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-jax", action="store_true")
    parser.add_argument(
        "--jax-variants",
        nargs="+",
        choices=sorted(JAX_VARIANT_BACKENDS),
        default=sorted(JAX_VARIANT_BACKENDS),
        help="pgam_jax solver variants to benchmark (default: all).",
    )
    parser.add_argument("--overwrite-cases", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--max-iter", type=int, default=100)
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


def _stored_git_commit(output_path: Path) -> str | None:
    """Return the commit recorded in an existing result file, or None."""
    try:
        payload = read_json(output_path)
    except (OSError, ValueError):
        return None
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        return None
    return runtime.get("git_commit")


def _should_run_jax(
    output_path: Path, current_commit: str | None, overwrite: bool
) -> bool:
    """Decide whether a pgam_jax result needs (re-)running.

    Existing results are reused only when they were produced from the current
    repository commit; results from another commit are stale and re-run.
    """
    if overwrite or not output_path.exists():
        return True
    if current_commit is None:
        return False
    stored_commit = _stored_git_commit(output_path)
    if stored_commit == current_commit:
        return False
    print(
        f"{output_path.name}: existing result is from commit {stored_commit}; re-running at {current_commit}.",
        flush=True,
    )
    return True


def _run_case_legacy(
    args: argparse.Namespace,
    dirs,
    case_path: Path,
    metadata_path: Path,
    repetition: int,
) -> None:
    stem = result_stem(case_path.stem, "legacy_pgam_docker_cpu", repetition)
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


def _run_case_jax_variants(
    args: argparse.Namespace,
    dirs,
    case_path: Path,
    metadata_path: Path,
    repetition: int,
    current_commit: str | None,
) -> None:
    for variant in args.jax_variants:
        if variant == "jaxopt":
            use_scipy = False
        elif variant == "scipy":
            use_scipy = True
        else:
            raise ValueError(f"Unknown jax variant {variant!r}.")
        stem = result_stem(case_path.stem, JAX_VARIANT_BACKENDS[variant], repetition)
        output_path = dirs.results / f"{stem}.json"
        prediction_path = dirs.results / f"{stem}.npz"
        if _should_run_jax(output_path, current_commit, args.overwrite_results):
            _run_jax(
                case_path,
                metadata_path,
                output_path,
                prediction_path,
                args.max_iter,
                use_scipy,
            )


def main() -> None:
    args = _parse_args()
    dirs = artifact_dirs(args.suite)
    cases = write_suite(args.suite, dirs.cases, overwrite=args.overwrite_cases)

    current_commit = git_commit(REPO_ROOT)
    if git_dirty(REPO_ROOT):
        print(
            f"WARNING: uncommitted changes in {REPO_ROOT}; results will be stamped "
            f"with commit {current_commit} but may not be reproducible from it.",
            flush=True,
        )

    for repetition in range(args.repetitions):
        for case_path, metadata_path in cases:
            if not args.skip_legacy:
                _run_case_legacy(args, dirs, case_path, metadata_path, repetition)
            if not args.skip_jax:
                _run_case_jax_variants(
                    args, dirs, case_path, metadata_path, repetition, current_commit
                )

    rows = summarize_results(collect_results(dirs.results))
    write_csv(rows, dirs.summaries / f"{args.summary_name}.csv")
    write_markdown(rows, dirs.summaries / f"{args.summary_name}.md")
    print(dirs.summaries / f"{args.summary_name}.md")


if __name__ == "__main__":
    main()

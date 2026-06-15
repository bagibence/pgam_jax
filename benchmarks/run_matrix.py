from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.common import (
    artifact_dirs,
    default_cpu_env,
    jax_backend_name,
    pgam_jax_dirty,
    pgam_jax_tree,
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

# Each pgam_jax variant maps to its (use_scipy, use_glm_init) options; the
# backend name is derived from that combination via jax_backend_name. The
# "_noglm" variants benchmark the same solvers with the GLM warm-start disabled.
JAX_VARIANT_OPTIONS = {
    "jaxopt": (False, True),
    "scipy": (True, True),
    "jaxopt_noglm": (False, False),
    "scipy_noglm": (True, False),
}
DEFAULT_JAX_VARIANTS = ["jaxopt", "scipy"]

# Docker/CLI exit codes (daemon unreachable, image missing, command not
# executable). These signal an environment problem affecting every case, not a
# legacy-fit crash, so they abort the run instead of being recorded as failures.
DOCKER_INFRA_EXIT_CODES = frozenset({125, 126, 127})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pgam_jax benchmark matrix.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-jax", action="store_true")
    parser.add_argument(
        "--jax-variants",
        nargs="+",
        choices=sorted(JAX_VARIANT_OPTIONS),
        default=DEFAULT_JAX_VARIANTS,
        help="pgam_jax solver variants to benchmark (default: jaxopt and scipy "
        "with the GLM warm-start enabled; the '_noglm' variants disable it).",
    )
    parser.add_argument("--overwrite-cases", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--summary-name", default="summary")
    return parser.parse_args()


def _tail(text: str, max_chars: int) -> str:
    """Return the trailing portion of ``text`` for compact error storage."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return "...\n" + text[-max_chars:]


def _write_failure_result(
    output_path: Path,
    *,
    backend: str,
    metadata: dict,
    returncode: int,
    error_tail: str,
    wall_s: float,
    wall_key: str,
    pgam_jax_tree: str | None,
) -> None:
    """Record a crashed fit as a result so the matrix can continue.

    The recorded ``pgam_jax_tree`` matters for pgam_jax: a failure stamped with
    the current library tree is reused (not retried) until the ``src/pgam_jax``
    source changes or ``--overwrite-results`` is passed. Legacy passes ``None``
    because its image is pinned and failures are reused unconditionally.
    """
    write_json(
        output_path,
        {
            "backend": backend,
            "status": "failed",
            "case": metadata,
            "timings_s": {wall_key: wall_s},
            "error": {"returncode": returncode, "stderr_tail": error_tail},
            "runtime": {"pgam_jax_tree": pgam_jax_tree, "git_dirty": None},
        },
    )


def _run_legacy(command: list[str], output_path: Path, metadata: dict) -> None:
    """Run one legacy case, recording fit crashes instead of aborting.

    A non-zero worker exit (the legacy fit raising, e.g. an overflow in the GCV
    objective) is captured as a ``status: "failed"`` result so the rest of the
    matrix still runs. Docker/CLI failures (exit 125/126/127) signal an
    environment problem affecting every case and re-raise.
    """
    print(" ".join(command), flush=True)
    t0 = time.perf_counter()
    completed = subprocess.run(
        command, env=default_cpu_env(), stderr=subprocess.PIPE, text=True
    )
    docker_wall_s = time.perf_counter() - t0
    stderr = completed.stderr or ""

    if completed.returncode != 0:
        if completed.returncode in DOCKER_INFRA_EXIT_CODES:
            print(stderr, flush=True)
            raise subprocess.CalledProcessError(
                completed.returncode, command, stderr=stderr
            )
        error_tail = _tail(stderr, max_chars=4000)
        print(
            f"{output_path.name}: legacy fit failed (exit {completed.returncode}); "
            "recording failure and continuing.",
            flush=True,
        )
        if error_tail:
            print(error_tail, flush=True)
        _write_failure_result(
            output_path,
            backend="legacy_pgam_docker_cpu",
            metadata=metadata,
            returncode=completed.returncode,
            error_tail=error_tail,
            wall_s=docker_wall_s,
            wall_key="docker_wall",
            pgam_jax_tree=None,
        )
        return

    payload = read_json(output_path)
    payload.setdefault("status", "ok")
    payload.setdefault("timings_s", {})["docker_wall"] = docker_wall_s
    write_json(output_path, payload)


def _run_jax(
    case_path: Path,
    metadata_path: Path,
    output_path: Path,
    prediction_output_path: Path,
    max_iter: int,
    use_scipy: bool,
    use_glm_init: bool,
    backend: str,
    current_tree: str | None,
) -> None:
    """Run one pgam_jax case, recording fit crashes instead of aborting.

    A non-zero worker exit (the fit raising, e.g. a non-finite loss) is captured
    as a ``status: "failed"`` result stamped with the current library tree, so
    the rest of the matrix still runs and the failure is retried only when the
    ``src/pgam_jax`` source changes or ``--overwrite-results`` is passed.
    """
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
    if not use_glm_init:
        command.append("--no-glm-init")

    print(" ".join(command), flush=True)
    t0 = time.perf_counter()
    completed = subprocess.run(
        command, env=default_cpu_env(), stderr=subprocess.PIPE, text=True
    )
    wall_s = time.perf_counter() - t0

    if completed.returncode != 0:
        error_tail = _tail(completed.stderr or "", max_chars=4000)
        print(
            f"{output_path.name}: pgam_jax fit failed (exit {completed.returncode}); "
            "recording failure and continuing.",
            flush=True,
        )
        if error_tail:
            print(error_tail, flush=True)
        _write_failure_result(
            output_path,
            backend=backend,
            metadata=read_json(metadata_path),
            returncode=completed.returncode,
            error_tail=error_tail,
            wall_s=wall_s,
            wall_key="wall",
            pgam_jax_tree=current_tree,
        )


def _stored_pgam_jax_tree(output_path: Path) -> str | None:
    """Return the pgam_jax library tree recorded in an existing result file, or None."""
    try:
        payload = read_json(output_path)
    except (OSError, ValueError):
        return None
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        return None
    return runtime.get("pgam_jax_tree")


def _should_run_jax(
    output_path: Path, current_tree: str | None, overwrite: bool
) -> bool:
    """Decide whether a pgam_jax result needs (re-)running.

    Existing results are reused only when they were produced from the current
    ``src/pgam_jax`` library source (identified by its git tree hash); results
    from another library tree are stale and re-run. Benchmark-harness-only
    changes leave the tree unchanged, so they no longer trigger re-runs.
    """
    if overwrite or not output_path.exists():
        return True
    if current_tree is None:
        return False
    stored_tree = _stored_pgam_jax_tree(output_path)
    if stored_tree == current_tree:
        return False
    print(
        f"{output_path.name}: existing result is from pgam_jax tree {stored_tree}; re-running at {current_tree}.",
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
            read_json(metadata_path),
        )


def _run_case_jax_variants(
    args: argparse.Namespace,
    dirs,
    case_path: Path,
    metadata_path: Path,
    repetition: int,
    current_tree: str | None,
) -> None:
    for variant in args.jax_variants:
        options = JAX_VARIANT_OPTIONS.get(variant)
        if options is None:
            raise ValueError(f"Unknown jax variant {variant!r}.")
        use_scipy, use_glm_init = options
        backend = jax_backend_name(use_scipy, use_glm_init)
        stem = result_stem(case_path.stem, backend, repetition)
        output_path = dirs.results / f"{stem}.json"
        prediction_path = dirs.results / f"{stem}.npz"
        if _should_run_jax(output_path, current_tree, args.overwrite_results):
            _run_jax(
                case_path,
                metadata_path,
                output_path,
                prediction_path,
                args.max_iter,
                use_scipy,
                use_glm_init,
                backend,
                current_tree,
            )


def main() -> None:
    args = _parse_args()
    dirs = artifact_dirs(args.suite)
    cases = write_suite(args.suite, dirs.cases, overwrite=args.overwrite_cases)

    current_tree = pgam_jax_tree(REPO_ROOT)
    if pgam_jax_dirty(REPO_ROOT):
        print(
            f"WARNING: uncommitted changes under src/pgam_jax in {REPO_ROOT}; results "
            f"will be stamped with library tree {current_tree} but may not be "
            "reproducible from it.",
            flush=True,
        )

    for repetition in range(args.repetitions):
        for case_path, metadata_path in cases:
            if not args.skip_legacy:
                _run_case_legacy(args, dirs, case_path, metadata_path, repetition)
            if not args.skip_jax:
                _run_case_jax_variants(
                    args, dirs, case_path, metadata_path, repetition, current_tree
                )

    rows = summarize_results(collect_results(dirs.results))
    write_csv(rows, dirs.summaries / f"{args.summary_name}.csv")
    write_markdown(rows, dirs.summaries / f"{args.summary_name}.md")
    print(dirs.summaries / f"{args.summary_name}.md")


if __name__ == "__main__":
    main()

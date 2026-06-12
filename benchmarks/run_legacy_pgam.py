from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from benchmarks.common import (
    BENCHMARK_DIR,
    LEGACY_DOCKER_IMAGE,
    default_cpu_env,
    read_json,
    write_json,
)


def build_docker_command(
    *,
    case_path: Path,
    metadata_path: Path,
    output_path: Path,
    prediction_output_path: Path,
    image: str = LEGACY_DOCKER_IMAGE,
    max_iter: int = 100,
    tol: float = 1e-5,
    gcv_sel_tol: float = 1e-10,
) -> list[str]:
    """Build the Docker command that runs one legacy PGAM benchmark."""
    artifact_dir = output_path.resolve().parents[1]
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{BENCHMARK_DIR.resolve()}:/benchmarks:ro",
        "-v",
        f"{artifact_dir}:/artifacts",
        "-w",
        "/",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-e",
        "MKL_NUM_THREADS=1",
        image,
        "python",
        "-m",
        "benchmarks.legacy_worker",
        "--case",
        f"/artifacts/cases/{case_path.name}",
        "--metadata",
        f"/artifacts/cases/{metadata_path.name}",
        "--output",
        f"/artifacts/results/{output_path.name}",
        "--prediction-output",
        f"/artifacts/results/{prediction_output_path.name}",
        "--max-iter",
        str(max_iter),
        "--tol",
        str(tol),
        "--gcv-sel-tol",
        str(gcv_sel_tol),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one legacy PGAM Docker benchmark case."
    )
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--image", default=LEGACY_DOCKER_IMAGE)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--gcv-sel-tol", type=float, default=1e-10)
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    command = build_docker_command(
        case_path=args.case,
        metadata_path=args.metadata,
        output_path=args.output,
        prediction_output_path=args.prediction_output,
        image=args.image,
        max_iter=args.max_iter,
        tol=args.tol,
        gcv_sel_tol=args.gcv_sel_tol,
    )
    if args.print_command:
        print(" ".join(command))
        return

    t0 = time.perf_counter()
    subprocess.run(command, check=True, env=default_cpu_env())
    docker_wall_s = time.perf_counter() - t0

    payload = read_json(args.output)
    payload.setdefault("timings_s", {})["docker_wall"] = docker_wall_s
    write_json(args.output, payload)


if __name__ == "__main__":
    main()

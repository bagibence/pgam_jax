from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks.common import CASES_DIR, CaseSpec, artifact_dirs, case_metadata, ensure_artifact_dirs, write_json


def _case_id(n_observations: int, n_smooths: int, n_basis: int, seed: int) -> str:
    return f"n{n_observations}_s{n_smooths}_b{n_basis}_seed{seed}"


SMOKE_SPECS = (
    CaseSpec(_case_id(3_000, 2, 12, 123), 3_000, 2, 12, 123, active_smooths=1),
    CaseSpec(_case_id(10_000, 5, 12, 124), 10_000, 5, 12, 124, active_smooths=3),
)

FULL_SPECS = tuple(
    CaseSpec(
        _case_id(n_observations, n_smooths, n_basis, seed),
        n_observations,
        n_smooths,
        n_basis,
        seed,
        active_smooths=max(1, (n_smooths + 1) // 2),
    )
    for n_observations in (3_000, 10_000, 30_000)
    for n_smooths in (2, 5, 10)
    for n_basis in (12, 24)
    for seed in (123, 124, 125)
)

SUITES = {
    "smoke": SMOKE_SPECS,
    "full": FULL_SPECS,
}


def generate_case(spec: CaseSpec) -> dict[str, np.ndarray]:
    """Generate deterministic nonlinear Poisson data for a benchmark case."""
    rng = np.random.default_rng(spec.seed)
    x = rng.uniform(spec.lower_bound, spec.upper_bound, size=(spec.n_smooths, spec.n_observations))

    eta = np.full(spec.n_observations, spec.true_intercept, dtype=float)
    for idx in range(spec.n_smooths):
        if idx >= spec.active_smooths:
            continue
        amplitude = 0.35 / np.sqrt(idx + 1)
        phase = 0.3 * idx
        eta += amplitude * np.sin(2.0 * np.pi * (idx + 1) * x[idx] + phase)
        eta += 0.15 * amplitude * np.cos(2.0 * np.pi * x[idx] - phase)

    eta = np.clip(eta, -4.0, 4.0)
    rate = np.exp(eta)
    y = rng.poisson(rate).astype(float)
    return {
        "x": x.astype(float),
        "y": y,
        "eta_true": eta,
        "rate_true": rate,
    }


def write_case(spec: CaseSpec, output_dir: Path = CASES_DIR, overwrite: bool = False) -> tuple[Path, Path]:
    """Write one case's arrays and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    case_path = output_dir / f"{spec.case_id}.npz"
    metadata_path = output_dir / f"{spec.case_id}.json"
    if not overwrite and case_path.exists() and metadata_path.exists():
        return case_path, metadata_path

    arrays = generate_case(spec)
    np.savez_compressed(case_path, **arrays)
    write_json(metadata_path, case_metadata(spec))
    return case_path, metadata_path


def write_suite(suite: str, output_dir: Path = CASES_DIR, overwrite: bool = False) -> list[tuple[Path, Path]]:
    """Write all cases for a named benchmark suite."""
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}. Expected one of {sorted(SUITES)}.")
    suite_dirs = artifact_dirs(suite)
    if output_dir == suite_dirs.cases:
        ensure_artifact_dirs(suite_dirs)
    elif output_dir == CASES_DIR:
        ensure_artifact_dirs()
    return [write_case(spec, output_dir, overwrite=overwrite) for spec in SUITES[suite]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pgam_jax benchmark cases.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or artifact_dirs(args.suite).cases
    generated = write_suite(args.suite, output_dir, overwrite=args.overwrite)
    for case_path, metadata_path in generated:
        print(f"{case_path} {metadata_path}")


if __name__ == "__main__":
    main()

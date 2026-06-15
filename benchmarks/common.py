from __future__ import annotations

import json
import math
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BENCHMARK_DIR / "artifacts"


@dataclass(frozen=True)
class ArtifactDirs:
    """Benchmark artifact paths rooted under one directory."""

    root: Path
    cases: Path
    results: Path
    summaries: Path


def artifact_dirs(name: str | None = None) -> ArtifactDirs:
    """Return artifact directories, optionally namespaced by suite/run name."""
    root = ARTIFACTS_DIR / name if name else ARTIFACTS_DIR
    return ArtifactDirs(
        root=root,
        cases=root / "cases",
        results=root / "results",
        summaries=root / "summaries",
    )


DEFAULT_ARTIFACT_DIRS = artifact_dirs()
CASES_DIR = DEFAULT_ARTIFACT_DIRS.cases
RESULTS_DIR = DEFAULT_ARTIFACT_DIRS.results
SUMMARIES_DIR = DEFAULT_ARTIFACT_DIRS.summaries

LEGACY_DOCKER_IMAGE = "edoardobalzani87/pgam:1.2"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class CaseSpec:
    """Deterministic synthetic benchmark case."""

    case_id: str
    n_observations: int
    n_smooths: int
    n_basis: int
    seed: int
    active_smooths: int
    order: int = 4
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    true_intercept: float = -0.2

    @property
    def n_knots(self) -> int:
        """Number of explicit knots used by legacy PGAM for this basis size."""
        return self.n_basis - 2


def default_cpu_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a deterministic CPU-oriented environment for benchmark subprocesses."""
    env = dict(os.environ if base_env is None else base_env)
    env["JAX_PLATFORM_NAME"] = "cpu"
    for name in THREAD_ENV_VARS:
        env.setdefault(name, "1")
    return env


def ensure_artifact_dirs(dirs: ArtifactDirs | None = None) -> None:
    """Create the benchmark artifact directory tree."""
    dirs = DEFAULT_ARTIFACT_DIRS if dirs is None else dirs
    dirs.cases.mkdir(parents=True, exist_ok=True)
    dirs.results.mkdir(parents=True, exist_ok=True)
    dirs.summaries.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def package_versions(package_names: tuple[str, ...]) -> dict[str, str | None]:
    """Collect installed package versions without failing when a package is absent."""
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit(repo_root: Path) -> str | None:
    """Return the current git commit for ``repo_root`` if available."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def git_dirty(repo_root: Path) -> bool | None:
    """Return whether ``repo_root`` has uncommitted tracked changes, or None if unknown."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.stdout.strip())


PGAM_JAX_SOURCE = "src/pgam_jax"


def pgam_jax_tree(repo_root: Path, ref: str = "HEAD") -> str | None:
    """
    Return the git tree hash of the pgam_jax library source at ``ref``.

    This is the staleness key for cached pgam_jax results: the tree hash changes
    only when files under ``src/pgam_jax`` change, so benchmark-harness-only
    commits no longer invalidate results. Passing a specific commit ``ref``
    yields the library tree that produced a result recorded at that commit,
    which is how older results are re-stamped.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", f"{ref}:{PGAM_JAX_SOURCE}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def pgam_jax_dirty(repo_root: Path) -> bool | None:
    """Return whether the pgam_jax library source has uncommitted changes, or None if unknown."""
    try:
        out = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                PGAM_JAX_SOURCE,
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.stdout.strip())


def runtime_metadata(repo_root: Path, packages: tuple[str, ...]) -> dict[str, Any]:
    """Build shared runtime metadata for benchmark result files."""
    return {
        "created_at": utc_now(),
        "git_commit": git_commit(repo_root),
        "git_dirty": git_dirty(repo_root),
        "pgam_jax_tree": pgam_jax_tree(repo_root),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "package_versions": package_versions(packages),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write indented JSON with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def case_metadata(spec: CaseSpec) -> dict[str, Any]:
    """Serialize a case spec with derived legacy parameters."""
    payload = asdict(spec)
    payload["n_knots"] = spec.n_knots
    payload["bounds"] = [spec.lower_bound, spec.upper_bound]
    payload["family"] = "poisson"
    return payload


def load_case(case_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load case arrays from an ``.npz`` case file."""
    data = np.load(case_path)
    return data["x"], data["y"], data["eta_true"], data["rate_true"]


def poisson_deviance_like_nll(y: np.ndarray, rate: np.ndarray) -> float:
    """Return mean Poisson negative log likelihood without the constant log-factorial."""
    safe_rate = np.clip(rate.astype(float), np.finfo(float).tiny, np.inf)
    return float(np.mean(safe_rate - y * np.log(safe_rate)))


def prediction_summary(y: np.ndarray, rate: np.ndarray) -> dict[str, float]:
    """Build compact prediction metrics for one fitted model."""
    rate = np.asarray(rate, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(rate) == 0 or np.std(y) == 0:
        corr = math.nan
    else:
        corr = float(np.corrcoef(y, rate)[0, 1])
    return {
        "prediction_mean": float(np.mean(rate)),
        "prediction_std": float(np.std(rate)),
        "prediction_min": float(np.min(rate)),
        "prediction_max": float(np.max(rate)),
        "mean_poisson_nll_no_constant": poisson_deviance_like_nll(y, rate),
        "corr_y_prediction": corr,
    }


def result_stem(case_id: str, backend: str, repetition: int) -> str:
    """Return the stable stem for one benchmark result."""
    return f"{case_id}__{backend}__rep{repetition:02d}"


def jax_backend_name(use_scipy: bool, use_glm_init: bool) -> str:
    """
    Return the backend name for a pgam_jax solver/init combination.

    Shared by the worker and the matrix so the success-record name and the
    failure-record name cannot drift.
    """
    if use_scipy and use_glm_init:
        return "pgam_jax_scipy_cpu"
    if use_scipy and not use_glm_init:
        return "pgam_jax_scipy_noglm_cpu"
    if not use_scipy and use_glm_init:
        return "pgam_jax_cpu"
    if not use_scipy and not use_glm_init:
        return "pgam_jax_noglm_cpu"
    raise ValueError(
        f"Unhandled backend combination use_scipy={use_scipy!r}, use_glm_init={use_glm_init!r}."
    )

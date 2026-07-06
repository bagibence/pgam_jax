"""
Generate mgcv golden fixtures for `test_matches_mgcv_golden`.

Run once (from the repo root) to produce JSON files in tests/data/:
    python scripts/generate_concurvity_test_data.py

Requires R on the PATH (`Rscript`) with the mgcv and jsonlite packages.
For each of the three concurvity scenarios it:

1. regenerates the scenario data with the same generators, seed, and
   sample size as the fixtures in tests/test_concurvity.py (keep them in
   sync),
2. writes the data to a temporary CSV,
3. calls scripts/generate_concurvity_test_data.R, which fits the
   corresponding `mgcv::gam`, runs `mgcv::concurvity` (full=TRUE and
   full=FALSE), and saves the model matrix, coefficients, term blocks,
   and both concurvity outputs to tests/data/concurvity_mgcv_<name>.json.

The golden test feeds mgcv's own design matrix and coefficients through
`pgam_jax.concurvity.concurvity`, so basis construction differences
between nemos and mgcv never enter: the comparison isolates the
concurvity linear algebra and holds to near machine precision.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

SEED = 123
N = 600

REPO_ROOT = Path(__file__).resolve().parents[1]
R_SCRIPT = REPO_ROOT / "scripts" / "generate_concurvity_test_data.R"
OUT_DIR = REPO_ROOT / "tests" / "data"


def scenario1():
    """Independent covariates."""
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    eta = 1.0 + 0.6 * np.sin(4 * np.pi * x1) + 0.8 * (x2 - 0.5) ** 2
    y = rng.poisson(np.exp(eta))
    return {"x1": x1, "x2": x2, "y": y}


def scenario2():
    """One covariate a noisy smooth function of another."""
    rng = np.random.default_rng(SEED)

    def f2(z):
        return 0.2 * z**11 * (10 * (1 - z)) ** 6 + 10 * (10 * z) ** 3 * (1 - z) ** 10

    t = np.sort(rng.uniform(0, 1, N))
    x = f2(t) + rng.normal(0, 3, N)
    eta = 0.5 + 0.8 * np.sin(4 * np.pi * t) + 0.03 * x
    y = rng.poisson(np.exp(eta))
    return {"t": t, "x": x, "y": y}


def scenario3():
    """Three smooths, one jointly determined by the other two."""
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(-1, 1, N)
    x2 = rng.uniform(-1, 1, N)
    x3 = 0.8 * x1 + 0.5 * np.sin(2 * x2) + rng.normal(0, 0.3, N)
    eta = x1 + x2**2 + np.sin(2 * x3)
    y = rng.poisson(np.exp(eta))
    return {"x1": x1, "x2": x2, "x3": x3, "y": y}


SCENARIOS = {"scenario1": scenario1, "scenario2": scenario2, "scenario3": scenario3}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, gen in SCENARIOS.items():
            cols = gen()
            csv_path = Path(tmp) / f"{name}.csv"
            np.savetxt(
                csv_path,
                np.column_stack(list(cols.values())),
                fmt="%.17g",
                delimiter=",",
                header=",".join(cols.keys()),
                comments="",
            )
            out_path = OUT_DIR / f"concurvity_mgcv_{name}.json"
            subprocess.run(
                ["Rscript", str(R_SCRIPT), name, str(csv_path), str(out_path)],
                check=True,
            )
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

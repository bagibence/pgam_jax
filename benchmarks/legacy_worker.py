from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import statsmodels.api as sm

from benchmarks.common import load_case, prediction_summary, read_json, runtime_metadata, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one legacy PGAM benchmark inside Docker.")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--gcv-sel-tol", type=float, default=1e-10)
    return parser.parse_args()


def _build_smooth_handler(x: np.ndarray, metadata: dict[str, Any]):
    from PGAM.GAM_library import smooths_handler

    sm_handler = smooths_handler()
    knots = np.linspace(metadata["lower_bound"], metadata["upper_bound"], metadata["n_knots"])
    for idx in range(metadata["n_smooths"]):
        sm_handler.add_smooth(
            f"x{idx}",
            [x[idx]],
            knots=[knots],
            ord=metadata["order"],
            is_temporal_kernel=False,
            trial_idx=None,
            is_cyclic=[False],
            penalty_type="der",
            der=2,
            lam=[1, 2],
            knots_num=metadata["n_knots"],
            kernel_length=None,
            kernel_direction=None,
            time_bin=0.006,
            knots_percentiles=(0, 100),
        )
    return sm_handler


def main() -> None:
    args = _parse_args()

    from PGAM.GAM_library import general_additive_model

    metadata = read_json(args.metadata)
    load_t0 = time.perf_counter()
    x, y, _eta_true, rate_true = load_case(args.case)
    load_s = time.perf_counter() - load_t0

    setup_t0 = time.perf_counter()
    sm_handler = _build_smooth_handler(x, metadata)
    link = sm.genmod.families.links.Log()
    poisson_family = sm.genmod.families.family.Poisson(link=link)
    pgam = general_additive_model(
        sm_handler,
        sm_handler.smooths_var,
        np.asarray(y),
        poisson_family,
    )
    setup_s = time.perf_counter() - setup_t0

    fit_t0 = time.perf_counter()
    result = pgam.optim_gam(
        sm_handler.smooths_var,
        max_iter=args.max_iter,
        tol=args.tol,
        use_dgcv=True,
        method="L-BFGS-B",
        gcv_sel_tol=args.gcv_sel_tol,
        fit_initial_beta=True,
        filter_trials=np.ones(len(y), dtype=bool),
    )
    fit_s = time.perf_counter() - fit_t0

    predict_t0 = time.perf_counter()
    exog = sm_handler.get_exog_mat_fast(sm_handler.smooths_var)[0]
    prediction = np.exp(np.clip(exog.dot(result.beta), -30.0, 30.0))
    predict_s = time.perf_counter() - predict_t0

    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.prediction_output, prediction=prediction, rate_true=rate_true)

    payload = {
        "backend": "legacy_pgam_docker_cpu",
        "case": metadata,
        "timings_s": {
            "load": load_s,
            "setup": setup_s,
            "fit": fit_s,
            "model_total": setup_s + fit_s,
            "predict": predict_s,
        },
        "fit": {
            "converged": bool(getattr(pgam, "converged", False)),
            "smooth_pen": np.asarray(result.smooth_pen, dtype=float).tolist(),
            "beta_size": int(np.asarray(result.beta).size),
        },
        "metrics": prediction_summary(y, prediction),
        "prediction_path": str(args.prediction_output),
        "runtime": runtime_metadata(Path("/benchmarks").resolve().parent, ("PGAM", "numpy", "scipy", "statsmodels")),
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()

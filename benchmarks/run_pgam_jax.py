from __future__ import annotations

import argparse
import os
import time
from functools import reduce
from operator import add
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common import (
    default_cpu_env,
    load_case,
    prediction_summary,
    read_json,
    runtime_metadata,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one pgam_jax CPU benchmark case.")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu",), default="cpu")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol-update", type=float, default=1e-6)
    parser.add_argument("--tol-optim", type=float, default=1e-10)
    parser.add_argument("--use-scipy", action="store_true")
    return parser.parse_args()


def _basis(n_smooths: int, n_basis: int, bounds: tuple[float, float]):
    import nemos as nmo

    bases = [nmo.basis.BSplineEval(n_basis, bounds=bounds) for _ in range(n_smooths)]
    return reduce(add, bases)


def _block_until_ready(model: Any) -> None:
    import jax

    jax.block_until_ready(model.coef_)
    jax.block_until_ready(model.intercept_)


def _fit_once(
    x: np.ndarray, y: np.ndarray, metadata: dict[str, Any], args: argparse.Namespace
) -> tuple[Any, float, float]:
    from nemos.observation_models import PoissonObservations

    from pgam_jax import GAM

    setup_t0 = time.perf_counter()
    basis = _basis(
        metadata["n_smooths"],
        metadata["n_basis"],
        (metadata["lower_bound"], metadata["upper_bound"]),
    )
    model = GAM(
        basis,
        observation_model=PoissonObservations(),
        maxiter=args.max_iter,
        tol_update=args.tol_update,
        tol_optim=args.tol_optim,
        use_scipy=args.use_scipy,
    )
    setup_s = time.perf_counter() - setup_t0
    t0 = time.perf_counter()
    model.fit(tuple(x[idx] for idx in range(x.shape[0])), y)
    _block_until_ready(model)
    return model, setup_s, time.perf_counter() - t0


def _predict(model: Any, x: np.ndarray) -> np.ndarray:
    import jax

    rate = model.predict(tuple(x[idx] for idx in range(x.shape[0])))
    rate = jax.block_until_ready(rate)
    return np.asarray(rate, dtype=float)


def main() -> None:
    args = _parse_args()
    os.environ.update(default_cpu_env(os.environ))

    import jax

    jax.config.update("jax_enable_x64", True)
    device_platforms = sorted({device.platform for device in jax.devices()})
    if args.backend not in device_platforms:
        raise RuntimeError(
            f"Requested {args.backend!r}, but JAX devices are {device_platforms!r}."
        )

    metadata = read_json(args.metadata)
    load_t0 = time.perf_counter()
    x, y, _eta_true, rate_true = load_case(args.case)
    load_s = time.perf_counter() - load_t0

    cold_model, setup_cold_s, fit_cold_s = _fit_once(x, y, metadata, args)
    warm_model, setup_warm_s, fit_warm_s = _fit_once(x, y, metadata, args)

    predict_t0 = time.perf_counter()
    prediction = _predict(warm_model, x)
    predict_s = time.perf_counter() - predict_t0

    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.prediction_output, prediction=prediction, rate_true=rate_true
    )

    backend_name = "pgam_jax_scipy_cpu" if args.use_scipy else "pgam_jax_cpu"
    runtime_packages = ("pgam_jax", "jax", "jaxlib", "nemos", "numpy")
    if args.use_scipy:
        runtime_packages += ("scipy",)

    result = {
        "backend": backend_name,
        "case": metadata,
        "timings_s": {
            "load": load_s,
            "setup_cold": setup_cold_s,
            "fit_cold": fit_cold_s,
            "model_total_cold": setup_cold_s + fit_cold_s,
            "setup_warm": setup_warm_s,
            "fit_warm": fit_warm_s,
            "model_total_warm": setup_warm_s + fit_warm_s,
            "predict": predict_s,
        },
        "fit": {
            "cold_n_iter": int(cold_model.n_iter_),
            "warm_n_iter": int(warm_model.n_iter_),
            "regularizer_strength": [
                np.asarray(value, dtype=float).tolist()
                for value in warm_model.regularizer_strength_
            ],
            "intercept": np.asarray(warm_model.intercept_, dtype=float).tolist(),
        },
        "options": {
            "use_scipy": args.use_scipy,
        },
        "metrics": prediction_summary(y, prediction),
        "prediction_path": str(args.prediction_output),
        "runtime": runtime_metadata(
            Path(__file__).resolve().parents[1], runtime_packages
        ),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()

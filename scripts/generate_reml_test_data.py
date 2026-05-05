"""
Generate regression fixtures for test_pql_reml.py.

Run once (from the repo root) to produce JSON files in tests/data/:
    python _script/generate_reml_test_data.py

This script generates regression from the original BalzaniEdoardo/PGAM implementation.

https://github.com/BalzaniEdoardo/PGAM

Each file is self-contained: it stores the augmented whitened matrices,
per-smooth penalty blocks, rho, and the reference REML val + grad produced
by the numpy PGAM implementation.  The test loads these without importing PGAM.
"""

import json
import numpy as np
from pathlib import Path
from itertools import groupby

from PGAM.gam_data_handlers import smooths_handler, compute_Sjs
from PGAM._pql_gcv import weights_and_data
from PGAM._pql_reml import linearized_reml_objective as reml_objective
from statsmodels.genmod.families import Poisson


def _block_range(S):
    rows = np.where(np.abs(S).sum(axis=1) > 1e-15)[0]
    return int(rows[0]), int(rows[-1]) + 1


def generate(sm_handler, var_list, n_obs, rng, out_path):
    S_all = compute_Sjs(sm_handler, var_list)
    X, _ = sm_handler.get_exog_mat_fast(var_list)

    w_coef = np.zeros(X.shape[1])
    y_resp = rng.poisson(np.exp(X.dot(w_coef)))
    fam = Poisson()
    f_wd = weights_and_data(y_resp, fam, fisher_scoring=True)
    mu = f_wd.family.fitted(X.dot(w_coef))
    z, wts = f_wd.get_params(mu)

    pen_matrix = sm_handler.get_penalty_agumented(var_list)
    Xagu = np.vstack((X, pen_matrix))
    yagu = np.zeros(Xagu.shape[0])
    yagu[:n_obs] = z
    wagu = np.ones(Xagu.shape[0])
    wagu[:n_obs] = wts
    sqrt_w_X = np.sqrt(wagu)[:, None] * Xagu
    sqrt_w_y = np.sqrt(wagu) * yagu
    Q, R = np.linalg.qr(sqrt_w_X, "reduced")
    rho = np.log(np.concatenate([sm_handler[v].lam for v in var_list]))

    reml_val, reml_grad = reml_objective(
        rho, sqrt_w_X, Q, R, sqrt_w_y, sm_handler,
        S_all=S_all, var_list=var_list,
    )

    # Group S_all by block range → one group per smooth
    ranges = [_block_range(S) for S in S_all]
    groups = []
    for rng_key, grp in groupby(enumerate(ranges), key=lambda iv: iv[1]):
        idxs = [i for i, _ in grp]
        groups.append((rng_key, idxs))

    penalty_blocks = []
    offset = 0
    for (r0, r1), idxs in groups:
        block = np.stack([S_all[i][r0:r1, r0:r1] for i in idxs])
        n = len(idxs)
        penalty_blocks.append({
            "S":   block.tolist(),
            "rho": rho[offset:offset + n].tolist(),
        })
        offset += n

    data = {
        "sqrt_w_X":      sqrt_w_X.tolist(),
        "Q":             Q.tolist(),
        "R":             R.tolist(),
        "sqrt_w_y":      sqrt_w_y.tolist(),
        "penalty_blocks": penalty_blocks,
        "reml_val":      float(reml_val),
        "reml_grad":     reml_grad.tolist(),
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {out_path}  (val={reml_val:.6f}, {len(penalty_blocks)} blocks)")


if __name__ == "__main__":
    out_dir = Path("tests/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_obs = 200
    rng = np.random.default_rng(42)

    x_cov  = rng.standard_normal(n_obs)
    y_cov  = rng.standard_normal(n_obs)
    x2_cov = rng.standard_normal(n_obs)
    y2_cov = rng.standard_normal(n_obs)

    sm = smooths_handler()
    sm.add_smooth("x",   [x_cov],  ord=4, knots_num=10, perc_out_range=0.,
                  penalty_type="der", der=2)
    sm.add_smooth("y",   [y_cov],  ord=4, knots_num=11, perc_out_range=0.,
                  penalty_type="der", der=2)
    sm.add_smooth("pos", [x2_cov, y2_cov], knots_num=5,
                  is_cyclic=np.array([False, False]), penalty_type="EqSpaced")

    generate(sm, ["x", "y"],        n_obs, rng, out_dir / "reml_1d_smooths.json")
    generate(sm, ["x", "y", "pos"], n_obs, rng, out_dir / "reml_1d_and_2d_smooth.json")
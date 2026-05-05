"""
Regression tests for _pql_reml.

Reference val + grad were produced by the numpy PGAM implementation and saved
to tests/data/ by _script/generate_reml_test_data.py.  No PGAM import here.

float64 is enabled globally in conftest.py.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgam_jax._pql_reml import reml_compute_factory

DATA_DIR = Path(__file__).parent / "data"


# module-level identity — stable object identity for JIT static-arg caching
def _identity(x):
    return x


def _load_and_run(filename):
    data = json.loads((DATA_DIR / filename).read_text())

    sqrt_w_X = jnp.array(data["sqrt_w_X"])
    Q        = jnp.array(data["Q"])
    R        = jnp.array(data["R"])
    sqrt_w_y = jnp.array(data["sqrt_w_y"])

    penalty_tree = [jnp.array(b["S"])   for b in data["penalty_blocks"]]
    reg_strength = [jnp.array(b["rho"]) for b in data["penalty_blocks"]]

    n_smooths = len(penalty_tree)
    reml_fn = reml_compute_factory(
        penalty_tree=penalty_tree,
        positive_mon_func=jnp.exp,
        apply_identifiability_columns=tuple(_identity for _ in range(n_smooths)),
        apply_identifiability=tuple(_identity for _ in range(n_smooths)),
    )

    f, g_tree = jax.value_and_grad(reml_fn)(
        reg_strength, penalty_tree, sqrt_w_X, Q, R, sqrt_w_y,
    )
    g_flat = np.concatenate([np.array(g) for g in g_tree])
    return float(f), g_flat, data["reml_val"], np.array(data["reml_grad"])


# ---------------------------------------------------------------------------

def test_reml_1d_smooths():
    f_jax, g_jax, f_ref, g_ref = _load_and_run("reml_1d_smooths.json")
    assert abs(f_jax - f_ref) < 1e-10, f"val diff = {abs(f_jax - f_ref):.2e}"
    np.testing.assert_allclose(g_jax, g_ref, atol=1e-10)


def test_reml_1d_and_2d_smooth():
    f_jax, g_jax, f_ref, g_ref = _load_and_run("reml_1d_and_2d_smooth.json")
    assert abs(f_jax - f_ref) < 1e-10, f"val diff = {abs(f_jax - f_ref):.2e}"
    np.testing.assert_allclose(g_jax, g_ref, atol=1e-10)
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

from pgam_jax._penalty_handler import PenaltyHandler
from pgam_jax._pql_reml import reml_compute_factory
from nemos.inverse_link_function_utils import identity as _identity

DATA_DIR = Path(__file__).parent / "data"


def _build_ph_and_factory(penalty_tree):
    """Construct PenaltyHandler from a list of (k, q, q) tensors and return reml_fn."""
    n_smooths = len(penalty_tree)
    ph = PenaltyHandler(non_linearity=jnp.exp)
    for S in penalty_tree:
        ph.add(S, penalize_null_space=False, identifiability_fn=_identity)
    compute_sqrt, compute_log_det_and_grad, _ = ph.build()

    id_fns = tuple(_identity for _ in range(n_smooths))
    return reml_compute_factory(
        compute_sqrt=compute_sqrt,
        compute_log_det_and_grad=compute_log_det_and_grad,
        positive_mon_func=jnp.exp,
        apply_identifiability_columns=id_fns,
        apply_identifiability=id_fns,
    ), compute_sqrt, compute_log_det_and_grad


def _load_and_run(filename):
    data = json.loads((DATA_DIR / filename).read_text())

    sqrt_w_X = jnp.array(data["sqrt_w_X"])
    Q = jnp.array(data["Q"])
    R = jnp.array(data["R"])
    sqrt_w_y = jnp.array(data["sqrt_w_y"])

    penalty_tree = [jnp.array(b["S"]) for b in data["penalty_blocks"]]
    reg_strength = [jnp.array(b["rho"]) for b in data["penalty_blocks"]]

    reml_fn, _, _ = _build_ph_and_factory(penalty_tree)

    f, g_tree = jax.value_and_grad(reml_fn)(
        reg_strength,
        penalty_tree,
        sqrt_w_X,
        Q,
        R,
        sqrt_w_y,
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


# ---------------------------------------------------------------------------
# Helpers shared by the numerical-gradient tests
# ---------------------------------------------------------------------------


def _load_inputs(filename):
    """Return (reml_fn, reg_strength, penalty_tree, sqrt_w_X, Q, R, sqrt_w_y)."""
    data = json.loads((DATA_DIR / filename).read_text())

    sqrt_w_X = jnp.array(data["sqrt_w_X"])
    Q = jnp.array(data["Q"])
    R = jnp.array(data["R"])
    sqrt_w_y = jnp.array(data["sqrt_w_y"])

    penalty_tree = [jnp.array(b["S"]) for b in data["penalty_blocks"]]
    reg_strength = [jnp.array(b["rho"]) for b in data["penalty_blocks"]]

    reml_fn, _, _ = _build_ph_and_factory(penalty_tree)
    return reml_fn, reg_strength, penalty_tree, sqrt_w_X, Q, R, sqrt_w_y


# ---------------------------------------------------------------------------
# Val consistency: bare call == value from value_and_grad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename", ["reml_1d_smooths.json", "reml_1d_and_2d_smooth.json"]
)
def test_val_equals_val_from_value_and_grad(filename):
    reml_fn, reg_strength, penalty_tree, X, Q, R, y = _load_inputs(filename)

    f_direct = float(reml_fn(reg_strength, penalty_tree, X, Q, R, y))
    f_vg, _ = jax.value_and_grad(reml_fn)(reg_strength, penalty_tree, X, Q, R, y)

    assert (
        abs(f_direct - float(f_vg)) < 1e-12
    ), f"{filename}: val mismatch = {abs(f_direct - float(f_vg)):.2e}"


# ---------------------------------------------------------------------------
# Gradient finite-difference check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename", ["reml_1d_smooths.json", "reml_1d_and_2d_smooth.json"]
)
def test_gradient_finite_diff(filename):
    reml_fn, reg_strength, penalty_tree, X, Q, R, y = _load_inputs(filename)

    # analytic gradient (flat)
    _, g_tree = jax.value_and_grad(reml_fn)(reg_strength, penalty_tree, X, Q, R, y)
    g_analytic = np.concatenate([np.array(g) for g in g_tree])

    # flatten reg_strength for perturbation
    rho_flat = np.concatenate([np.array(r) for r in reg_strength])
    split_sizes = [len(r) for r in reg_strength]

    def _reml_from_flat(rho_f):
        rho_f_splits = np.split(rho_f, np.cumsum(split_sizes[:-1]))
        rs = [jnp.array(r) for r in rho_f_splits]
        return float(reml_fn(rs, penalty_tree, X, Q, R, y))

    h = 1e-5
    g_fd = np.zeros_like(rho_flat)
    for j in range(len(rho_flat)):
        rp = rho_flat.copy()
        rp[j] += h
        rm = rho_flat.copy()
        rm[j] -= h
        g_fd[j] = (_reml_from_flat(rp) - _reml_from_flat(rm)) / (2 * h)

    np.testing.assert_allclose(
        g_analytic,
        g_fd,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"{filename}: analytic vs FD gradient mismatch",
    )

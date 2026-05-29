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
import nemos as nmo
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from nemos.inverse_link_function_utils import identity as _identity

from pgam_jax import GAM
from pgam_jax._penalty_handler import PenaltyHandler
from pgam_jax._pql_reml import reml_compute_factory

DATA_DIR = Path(__file__).parent / "data"


def _build_ph_and_factory(penalty_tree):
    """Construct PenaltyHandler from a list of (k, q, q) tensors and return reml_fn."""
    n_smooths = len(penalty_tree)
    ph = PenaltyHandler()
    for S in penalty_tree:
        ph.add(S, penalize_null_space=False, identifiability_fn=_identity)
    compute_sqrt, compute_log_det_and_grad = ph.build()

    id_fns = tuple(_identity for _ in range(n_smooths))
    return (
        reml_compute_factory(
            compute_sqrt=compute_sqrt,
            compute_log_det_and_grad=compute_log_det_and_grad,
            apply_identifiability_columns=id_fns,
            apply_identifiability=id_fns,
        ),
        compute_sqrt,
        compute_log_det_and_grad,
    )


def _bsp(n):
    return nmo.basis.BSplineEval(n_basis_funcs=n, order=4, bounds=(-1.0, 1.0))


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
    g_analytic, _ = ravel_pytree(g_tree)

    rho_flat, unravel_rho = ravel_pytree(reg_strength)

    def _reml_from_flat(rho_f):
        return float(reml_fn(unravel_rho(rho_f), penalty_tree, X, Q, R, y))

    h = 1e-5
    g_fd = np.zeros_like(np.array(rho_flat))
    for j in range(rho_flat.size):
        up, down = rho_flat.at[j].add(h), rho_flat.at[j].add(-h)
        g_fd[j] = (_reml_from_flat(up) - _reml_from_flat(down)) / (2 * h)

    np.testing.assert_allclose(
        np.array(g_analytic),
        g_fd,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"{filename}: analytic vs FD gradient mismatch",
    )


# ---------------------------------------------------------------------------
# Routed GAM-style REML gradient check
# ---------------------------------------------------------------------------


_ROUTED_BASIS_CASES = [
    pytest.param(lambda: _bsp(8), id="eval"),
    pytest.param(lambda: _bsp(8) + _bsp(7), id="eval+eval"),
    pytest.param(lambda: _bsp(6) * _bsp(5), id="eval*eval"),
    pytest.param(lambda: _bsp(7) + (_bsp(5) * _bsp(4)), id="eval+(eval*eval)"),
]


def _routed_reml_setup(make_basis, seed=0, n_obs=300):
    """
    Build REML exactly through GAM's penalty routing.

    The fixture/reference tests above construct every block with ``ph.add(S)``,
    which exercises the GENERAL path.  This setup goes through
    ``GAM._build_penalty_handler`` so the finite-difference check covers the
    production SINGLE_WITH_NULL and KRONECKER_WITH_NULL routes.
    """
    rng = np.random.default_rng(seed)
    gam = GAM(make_basis(), method="reml")
    penalty_tree = gam._get_penalty_tree()
    ph = gam._build_penalty_handler(penalty_tree)
    compute_sqrt, compute_log_det_and_grad = ph.build()

    rho = [jnp.array(rng.uniform(-1.0, 1.0, size=p.rho_len)) for p in ph._penalties]

    n_params = compute_sqrt(rho).shape[1] + 1
    Xw = jnp.array(rng.standard_normal((n_obs, n_params)))
    Q, R = jnp.linalg.qr(Xw)
    y = jnp.array(rng.standard_normal(n_obs))

    reml_fn = reml_compute_factory(
        compute_sqrt=compute_sqrt,
        compute_log_det_and_grad=compute_log_det_and_grad,
        apply_identifiability_columns=gam._apply_identifiability_column,
        apply_identifiability=gam._apply_identifiability_square,
    )
    return reml_fn, rho, penalty_tree, Xw, Q, R, y


@pytest.mark.parametrize("make_basis", _ROUTED_BASIS_CASES)
def test_routed_reml_gradient_matches_finite_diff(make_basis):
    reml_fn, rho, penalty_tree, X, Q, R, y = _routed_reml_setup(make_basis)

    _, g_tree = jax.value_and_grad(reml_fn)(rho, penalty_tree, X, Q, R, y)
    g_analytic, _ = ravel_pytree(g_tree)

    def _from_flat(flat_rho):
        return float(reml_fn(unravel_rho(flat_rho), penalty_tree, X, Q, R, y))

    flat, unravel_rho = ravel_pytree(rho)
    h = 1e-5
    g_fd = np.zeros_like(np.array(flat))
    for j in range(flat.size):
        up, down = flat.at[j].add(h), flat.at[j].add(-h)
        g_fd[j] = (_from_flat(up) - _from_flat(down)) / (2 * h)

    np.testing.assert_allclose(np.array(g_analytic), g_fd, atol=1e-7, rtol=1e-5)

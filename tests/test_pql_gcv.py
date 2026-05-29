"""
Finite-difference check of the analytic GCV gradient.

``gcv_compute_factory`` builds a custom-VJP objective whose forward pass uses
``PenaltyHandler.compute_sqrt`` and whose backward pass reconstructs the
gradient from ``compute_penalty_blocks(penalty_tree)``.  Those are two
independent representations of the same penalty, so a mismatch (lambda
ordering, identifiability drop, block offset) only shows up as a gradient
error.  This pins the forward/backward pairing in the exact configuration
``GAM.fit`` uses: ``penalize_null_space=True`` with per-component column drop,
exercising SINGLE_WITH_NULL and KRONECKER_WITH_NULL.
"""

import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest

from pgam_jax import GAM
from pgam_jax._pql_gcv import gcv_compute_factory

jax.config.update("jax_enable_x64", True)

GAMMA = 1.5


def _bsp(n):
    return nmo.basis.BSplineEval(n_basis_funcs=n, order=4, bounds=(-1.0, 1.0))


_BASIS_CASES = [
    pytest.param(lambda: _bsp(8), id="eval"),
    pytest.param(lambda: _bsp(8) + _bsp(7), id="eval+eval"),
    pytest.param(lambda: _bsp(6) * _bsp(5), id="eval*eval"),
    pytest.param(lambda: _bsp(7) + (_bsp(5) * _bsp(4)), id="eval+(eval*eval)"),
]


def _gcv_setup(make_basis, seed=0, n_obs=300):
    """
    Build the GCV objective the way ``GAM.fit`` does and synthesise consistent
    QR inputs sized to the penalty's column count.
    """
    rng = np.random.default_rng(seed)
    gam = GAM(make_basis(), method="gcv")
    penalty_tree = gam._get_penalty_tree()
    ph = gam._build_penalty_handler(penalty_tree)
    compute_sqrt, _ = ph.build()

    rho = [jnp.array(rng.uniform(-1.0, 1.0, size=p.rho_len)) for p in ph._penalties]

    # n_params = intercept column + columns of the block-diagonal sqrt penalty
    n_params = compute_sqrt(rho).shape[1] + 1
    Xw = jnp.array(rng.standard_normal((n_obs, n_params)))
    Q, R = jnp.linalg.qr(Xw)
    y = jnp.array(rng.standard_normal(n_obs))

    gcv_fn = gcv_compute_factory(
        compute_sqrt,
        gam._apply_identifiability_column,
        gam._apply_identifiability_square,
        GAMMA,
    )
    return gcv_fn, rho, penalty_tree, Xw, Q, R, y


@pytest.mark.parametrize("make_basis", _BASIS_CASES)
def test_gcv_gradient_matches_finite_diff(make_basis):
    gcv_fn, rho, penalty_tree, X, Q, R, y = _gcv_setup(make_basis)

    _, g_tree = jax.value_and_grad(gcv_fn)(rho, penalty_tree, X, Q, R, y)
    g_analytic = np.concatenate([np.array(g) for g in g_tree])

    flat = np.concatenate([np.array(r) for r in rho])
    sizes = [len(r) for r in rho]

    def _from_flat(flat_rho):
        splits = np.split(flat_rho, np.cumsum(sizes[:-1]))
        rs = [jnp.array(s) for s in splits]
        return float(gcv_fn(rs, penalty_tree, X, Q, R, y))

    h = 1e-6
    g_fd = np.zeros_like(flat)
    for j in range(len(flat)):
        up, down = flat.copy(), flat.copy()
        up[j] += h
        down[j] -= h
        g_fd[j] = (_from_flat(up) - _from_flat(down)) / (2 * h)

    np.testing.assert_allclose(g_analytic, g_fd, atol=1e-6, rtol=1e-4)

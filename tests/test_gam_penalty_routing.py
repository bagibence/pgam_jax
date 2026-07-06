"""
Pins the penalty class that ``gam._build_penalty_handler`` selects per basis component.

The production route dispatches one-dimensional smooths to SINGLE_WITH_NULL and
tensor-product smooths to KRONECKER_WITH_NULL.  The tests also keep an
always-GENERAL baseline so routing changes cannot silently alter lambda
ordering, lambda count, factor order, or the resulting penalty math.

Two test classes:

* ``TestRoutedMethod``:
  Asserts the penalty class that each basis component is dispatched to.
* ``TestPenaltyMathInvariant``:
  Compares the routed handler's ``compute_sqrt`` and
  ``compute_log_det_and_grad`` outputs against a manually-constructed
  "always GENERAL" baseline.
"""

import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest

from pgam_jax import GAM
from pgam_jax._identifiable_features import _should_drop_basis_col
from pgam_jax._penalty_handler import (
    PenaltyHandler,
    _KroneckerWithNullPenalty,
    _SingleWithNullPenalty,
)
from pgam_jax.penalty_utils import DROP_LAST_COL, IDENTITY


def _bsp_eval(n=10):
    return nmo.basis.BSplineEval(n_basis_funcs=n, order=4, bounds=(-1.0, 1.0))


def _bsp_conv(n=10, window=51):
    return nmo.basis.BSplineConv(n_basis_funcs=n, window_size=window)


def _routed_ph(basis, *, drop_conv_basis_col=False):
    """Handler built the way GAM.fit() builds it, without fitting."""
    gam = GAM(basis, drop_conv_basis_col=drop_conv_basis_col)
    return gam._build_penalty_handler(gam._get_penalty_tree())


def _baseline_general_ph(basis, *, drop_conv_basis_col=False):
    """Today's behaviour: every component routed through ph.add as GENERAL."""
    gam = GAM(basis, drop_conv_basis_col=drop_conv_basis_col)
    penalty_tree = gam._get_penalty_tree()
    ph = PenaltyHandler()
    for b, S in zip(basis, penalty_tree):
        id_fn = (
            DROP_LAST_COL
            if _should_drop_basis_col(b, drop_conv_basis_col)
            else IDENTITY
        )
        ph.add(S, penalize_null_space=False, identifiability_fn=id_fn)
    return ph


def _rhos_for(basis):
    """One rho vector per component sized to match the stacked tensor's M+1 slices."""
    rng = np.random.default_rng(0)
    rhos = []
    for b in basis:
        if isinstance(b, nmo.basis.MultiplicativeBasis):
            k = sum(1 for _ in b._iterate_over_components())
        else:
            k = 1
        rhos.append(jnp.array(rng.uniform(-1.0, 1.0, size=k + 1)))
    return rhos


# ---------------------------------------------------------------------------
# Class A: dispatch assertions. Currently RED; GREEN after routing fix.
# ---------------------------------------------------------------------------


class TestRoutedMethod:
    def test_single_eval_routes_to_single_with_null(self):
        ph = _routed_ph(_bsp_eval())
        assert [type(p) for p in ph._penalties] == [_SingleWithNullPenalty]

    def test_additive_eval_eval_routes_to_two_single_with_null(self):
        ph = _routed_ph(_bsp_eval() + _bsp_eval())
        assert [type(p) for p in ph._penalties] == [_SingleWithNullPenalty] * 2

    def test_multiplicative_eval_eval_routes_to_kronecker_with_null(self):
        ph = _routed_ph(_bsp_eval() * _bsp_eval())
        assert [type(p) for p in ph._penalties] == [_KroneckerWithNullPenalty]

    def test_mixed_eval_plus_multiplicative_routes_one_of_each(self):
        ph = _routed_ph(_bsp_eval() + (_bsp_eval() * _bsp_eval()))
        assert [type(p) for p in ph._penalties] == [
            _SingleWithNullPenalty,
            _KroneckerWithNullPenalty,
        ]

    def test_conv_routes_to_single_with_null(self):
        ph = _routed_ph(_bsp_conv(), drop_conv_basis_col=False)
        assert [type(p) for p in ph._penalties] == [_SingleWithNullPenalty]


# ---------------------------------------------------------------------------
# Class B: penalty-value equivalence. GREEN before AND after the fix.
# Catches a fix that changes lambda ordering, lambda count, factor order, etc.
# ---------------------------------------------------------------------------


_BASIS_CASES = [
    pytest.param(lambda: _bsp_eval(), False, id="eval"),
    pytest.param(lambda: _bsp_eval() + _bsp_eval(), False, id="eval+eval"),
    pytest.param(lambda: _bsp_eval() * _bsp_eval(), False, id="eval*eval"),
    pytest.param(
        lambda: _bsp_eval() + (_bsp_eval() * _bsp_eval()),
        False,
        id="eval+(eval*eval)",
    ),
    pytest.param(lambda: _bsp_conv(), False, id="conv"),
]


@pytest.mark.parametrize("make_basis,drop_conv_basis_col", _BASIS_CASES)
class TestPenaltyMathInvariant:
    def test_compute_sqrt_BtB_matches_general_baseline(
        self, make_basis, drop_conv_basis_col
    ):
        basis = make_basis()
        ph_routed = _routed_ph(basis, drop_conv_basis_col=drop_conv_basis_col)
        ph_baseline = _baseline_general_ph(
            basis, drop_conv_basis_col=drop_conv_basis_col
        )
        rhos = _rhos_for(basis)
        B_r = ph_routed.compute_sqrt(rhos)
        B_b = ph_baseline.compute_sqrt(rhos)
        # Compare B.T @ B, not B: eigh sign ambiguity flips eigenvector columns
        # between routes while the penalty matrix itself is invariant.
        np.testing.assert_allclose(
            np.array(B_r.T @ B_r), np.array(B_b.T @ B_b), atol=1e-9
        )

    def test_log_det_and_grad_matches_general_baseline(
        self, make_basis, drop_conv_basis_col
    ):
        basis = make_basis()
        ph_routed = _routed_ph(basis, drop_conv_basis_col=drop_conv_basis_col)
        ph_baseline = _baseline_general_ph(
            basis, drop_conv_basis_col=drop_conv_basis_col
        )
        rhos = _rhos_for(basis)
        ld_r, grad_r = ph_routed.compute_log_det_and_grad(rhos)
        ld_b, grad_b = ph_baseline.compute_log_det_and_grad(rhos)
        assert len(ld_r) == len(ld_b)
        for r, b in zip(ld_r, ld_b):
            np.testing.assert_allclose(np.array(r), np.array(b), atol=1e-9)
        for r, b in zip(grad_r, grad_b):
            np.testing.assert_allclose(np.array(r), np.array(b), atol=1e-9)

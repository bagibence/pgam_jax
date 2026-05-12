"""Tests for _penalty_handler.py — PenaltyHandler sqrt methods."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import block_diag

from pgam_jax._penalty_handler import PenaltyHandler, SqrtMethod
from pgam_jax.penalty_utils import (
    compute_penalty_null_space,
    ndim_tensor_product_basis_penalty,
)

jax.config.update("jax_enable_x64", True)

ATOL = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diff2_penalty(n):
    """n×n second-derivative penalty, rank n-2, null space = {constants, linear}."""
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return jnp.array(D.T @ D, dtype=float)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def S1():
    """10×10 second-derivative penalty, rank 8, 2-dim null space."""
    return _diff2_penalty(10)


@pytest.fixture(scope="module")
def S1_null(S1):
    """Projector onto the 2-dim null space of S1."""
    return jnp.array(compute_penalty_null_space(np.array(S1)[None]))


@pytest.fixture(scope="module")
def S_full_rank():
    """8×8 full-rank SPD matrix (no null space)."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((8, 8))
    return jnp.array(A @ A.T + 0.1 * np.eye(8))


@pytest.fixture(scope="module")
def S_kron(S1):
    """Kronecker-sum tensor for S1 ⊗ S1, shape (2, 100, 100)."""
    return jnp.array(ndim_tensor_product_basis_penalty(np.array(S1), np.array(S1)))


@pytest.fixture(scope="module")
def S_kron_null(S_kron):
    """Projector onto the null space of S_kron."""
    return jnp.array(compute_penalty_null_space(np.array(S_kron)))


# ---------------------------------------------------------------------------
# SINGLE
# ---------------------------------------------------------------------------

class TestSINGLE:
    def test_method_selected(self, S1):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        assert SqrtMethod.SINGLE in [m for m, _ in ph._groups]

    def test_full_rank_stays_single_with_penalize_null(self, S_full_rank):
        ph = PenaltyHandler()
        ph.add(S_full_rank, penalize_null_space=True)
        assert SqrtMethod.SINGLE in [m for m, _ in ph._groups]
        assert SqrtMethod.SINGLE_WITH_NULL not in [m for m, _ in ph._groups]

    @pytest.mark.parametrize("rho", [-2.0, 0.0, 2.0])
    def test_round_trip(self, S1, rho):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        B = ph.compute_sqrt([jnp.array([rho])])
        np.testing.assert_allclose(
            np.array(B.T @ B), np.array(jnp.exp(rho) * S1), atol=ATOL
        )

    def test_identifiability_shape(self, S1):
        q = S1.shape[0]
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False, identifiability_fn=lambda x: x[..., :-1])
        B = ph.compute_sqrt([jnp.array([0.0])])
        assert B.shape == (q, q - 1)

    def test_identifiability_equivalent_to_column_drop(self, S1):
        ph_full = PenaltyHandler()
        ph_full.add(S1, penalize_null_space=False)
        B_full = ph_full.compute_sqrt([jnp.array([1.0])])

        ph_id = PenaltyHandler()
        ph_id.add(S1, penalize_null_space=False, identifiability_fn=lambda x: x[..., :-1])
        B_id = ph_id.compute_sqrt([jnp.array([1.0])])

        np.testing.assert_allclose(np.array(B_id), np.array(B_full[:, :-1]), atol=1e-12)


# ---------------------------------------------------------------------------
# SINGLE_WITH_NULL
# ---------------------------------------------------------------------------

class TestSINGLE_WITH_NULL:
    def test_method_selected(self, S1):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=True)
        assert SqrtMethod.SINGLE_WITH_NULL in [m for m, _ in ph._groups]

    @pytest.mark.parametrize("rho_pen,rho_null", [(-1.0, 0.0), (0.5, -0.5), (2.0, 1.0)])
    def test_round_trip(self, S1, S1_null, rho_pen, rho_null):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=True)
        B = ph.compute_sqrt([jnp.array([rho_pen, rho_null])])
        S_expected = jnp.exp(rho_pen) * S1 + jnp.exp(rho_null) * S1_null
        np.testing.assert_allclose(np.array(B.T @ B), np.array(S_expected), atol=ATOL)

    def test_identifiability_shape(self, S1):
        q = S1.shape[0]
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=True, identifiability_fn=lambda x: x[..., :-1])
        B = ph.compute_sqrt([jnp.array([0.0, 0.0])])
        assert B.shape == (q, q - 1)


# ---------------------------------------------------------------------------
# KRONECKER
# ---------------------------------------------------------------------------

class TestKRONECKER:
    def test_method_selected(self, S1):
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=False)
        assert SqrtMethod.KRONECKER in [m for m, _ in ph._groups]

    def test_not_fired_when_one_factor_full_rank(self, S1, S_full_rank):
        ph = PenaltyHandler()
        ph.add_kron([S1, S_full_rank], penalize_null_space=True)
        methods = [m for m, _ in ph._groups]
        assert SqrtMethod.KRONECKER in methods
        assert SqrtMethod.KRONECKER_WITH_NULL not in methods

    @pytest.mark.parametrize("rho0,rho1", [(-1.0, 0.5), (0.0, 0.0), (1.5, -0.5)])
    def test_round_trip(self, S1, S_kron, rho0, rho1):
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=False)
        B = ph.compute_sqrt([jnp.array([rho0, rho1])])
        lams = jnp.exp(jnp.array([rho0, rho1]))
        S_lam = jnp.einsum("k,kij->ij", lams, S_kron)
        np.testing.assert_allclose(np.array(B.T @ B), np.array(S_lam), atol=ATOL)

    def test_identifiability_shape(self, S1):
        q = S1.shape[0]
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=False, identifiability_fn=lambda x: x[..., :-1])
        B = ph.compute_sqrt([jnp.array([0.0, 0.0])])
        q_kron = q * q
        assert B.shape == (q_kron, q_kron - 1)


# ---------------------------------------------------------------------------
# KRONECKER_WITH_NULL
# ---------------------------------------------------------------------------

class TestKRONECKER_WITH_NULL:
    def test_method_selected(self, S1):
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=True)
        assert SqrtMethod.KRONECKER_WITH_NULL in [m for m, _ in ph._groups]

    def test_requires_all_factors_rank_deficient(self, S1, S_full_rank):
        ph = PenaltyHandler()
        ph.add_kron([S1, S_full_rank], penalize_null_space=True)
        assert SqrtMethod.KRONECKER_WITH_NULL not in [m for m, _ in ph._groups]

    @pytest.mark.parametrize("rho0,rho1,rho_null", [(-1.0, 0.5, 0.0), (0.0, 0.0, -1.0)])
    def test_round_trip(self, S1, S_kron, S_kron_null, rho0, rho1, rho_null):
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=True)
        B = ph.compute_sqrt([jnp.array([rho0, rho1, rho_null])])
        lams = jnp.exp(jnp.array([rho0, rho1]))
        S_lam = jnp.einsum("k,kij->ij", lams, S_kron) + jnp.exp(rho_null) * S_kron_null
        np.testing.assert_allclose(np.array(B.T @ B), np.array(S_lam), atol=ATOL)


# ---------------------------------------------------------------------------
# GENERAL (k > 1 via add(), null space dropped by precompute projection)
# ---------------------------------------------------------------------------

class TestGENERAL:
    def test_method_selected(self, S_kron):
        ph = PenaltyHandler()
        ph.add(S_kron)
        assert SqrtMethod.GENERAL in [m for m, _ in ph._groups]

    @pytest.mark.parametrize("rho0,rho1", [(-1.0, 0.5), (0.5, 2.0), (0.0, 0.0)])
    def test_round_trip(self, S_kron, rho0, rho1):
        ph = PenaltyHandler()
        ph.add(S_kron)
        B = ph.compute_sqrt([jnp.array([rho0, rho1])])
        lams = jnp.exp(jnp.array([rho0, rho1]))
        S_lam = jnp.einsum("k,kij->ij", lams, S_kron)
        np.testing.assert_allclose(np.array(B.T @ B), np.array(S_lam), atol=ATOL)

    def test_matches_kronecker_round_trip(self, S1, S_kron):
        """GENERAL and KRONECKER agree on B.T @ B."""
        rho0, rho1 = -1.0, 0.5

        ph_kron = PenaltyHandler()
        ph_kron.add_kron([S1, S1], penalize_null_space=False)
        B_kron = ph_kron.compute_sqrt([jnp.array([rho0, rho1])])

        ph_gen = PenaltyHandler()
        ph_gen.add(S_kron)
        B_gen = ph_gen.compute_sqrt([jnp.array([rho0, rho1])])

        np.testing.assert_allclose(
            np.array(B_gen.T @ B_gen), np.array(B_kron.T @ B_kron), atol=ATOL
        )

    def test_identifiability_shape(self, S_kron):
        q = S_kron.shape[-1]
        ph = PenaltyHandler()
        ph.add(S_kron, identifiability_fn=lambda x: x[..., :-1])
        B = ph.compute_sqrt([jnp.array([0.0, 0.0])])
        assert B.shape[1] == q - 1


# ---------------------------------------------------------------------------
# Block-diagonal assembly (multiple penalties)
# ---------------------------------------------------------------------------

class TestBlockDiagonal:
    def test_shape_single_plus_kronecker(self, S1, S_kron):
        q1 = S1.shape[0]
        q_kron = S_kron.shape[-1]
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        ph.add_kron([S1, S1], penalize_null_space=False)
        B = ph.compute_sqrt([jnp.array([0.0]), jnp.array([0.0, 0.0])])
        assert B.shape == (q1 + q_kron, q1 + q_kron)

    def test_round_trip_block_structure(self, S1, S_kron):
        rho1 = 1.0
        rho2, rho3 = -0.5, 1.5

        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        ph.add_kron([S1, S1], penalize_null_space=False)
        B = ph.compute_sqrt([jnp.array([rho1]), jnp.array([rho2, rho3])])

        lams_kron = jnp.exp(jnp.array([rho2, rho3]))
        S_expected = block_diag(
            jnp.exp(rho1) * S1,
            jnp.einsum("k,kij->ij", lams_kron, S_kron),
        )
        np.testing.assert_allclose(np.array(B.T @ B), np.array(S_expected), atol=ATOL)

    def test_shape_with_identifiability(self, S1):
        q = S1.shape[0]
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False, identifiability_fn=lambda x: x[..., :-1])
        ph.add(S1, penalize_null_space=False, identifiability_fn=lambda x: x[..., :-1])
        B = ph.compute_sqrt([jnp.array([0.0]), jnp.array([0.0])])
        assert B.shape == (2 * q, 2 * (q - 1))

    def test_ordering_preserved(self, S1):
        """compute_sqrt returns blocks in the same order as add() calls."""
        rho0, rho1 = 1.0, -1.0
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        ph.add(S1, penalize_null_space=False)
        B = ph.compute_sqrt([jnp.array([rho0]), jnp.array([rho1])])

        q = S1.shape[0]
        B0 = B[:q, :q]
        B1 = B[q:, q:]
        np.testing.assert_allclose(np.array(B0.T @ B0), np.array(jnp.exp(rho0) * S1), atol=ATOL)
        np.testing.assert_allclose(np.array(B1.T @ B1), np.array(jnp.exp(rho1) * S1), atol=ATOL)
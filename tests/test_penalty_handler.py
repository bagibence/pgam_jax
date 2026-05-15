"""Tests for _penalty_handler.py — PenaltyHandler sqrt methods."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import block_diag

from conftest import central_diff, diff2_penalty

from pgam_jax._penalty_handler import PenaltyHandler, SqrtMethod, _drop_last_col
from pgam_jax.penalty_utils import (
    compute_penalty_null_space,
    ndim_tensor_product_basis_penalty,
)

jax.config.update("jax_enable_x64", True)

ATOL = 1e-12


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def S1():
    """10×10 second-derivative penalty, rank 8, 2-dim null space."""
    return diff2_penalty(10)


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
        assert SqrtMethod.SINGLE in [m for m, *_ in ph._groups]

    def test_full_rank_stays_single_with_penalize_null(self, S_full_rank):
        ph = PenaltyHandler()
        ph.add(S_full_rank, penalize_null_space=True)
        assert SqrtMethod.SINGLE in [m for m, *_ in ph._groups]
        assert SqrtMethod.SINGLE_WITH_NULL not in [m for m, *_ in ph._groups]

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
        ph.add(S1, penalize_null_space=False, identifiability_fn=_drop_last_col)
        B = ph.compute_sqrt([jnp.array([0.0])])
        assert B.shape == (q, q - 1)

    def test_identifiability_equivalent_to_column_drop(self, S1):
        ph_full = PenaltyHandler()
        ph_full.add(S1, penalize_null_space=False)
        B_full = ph_full.compute_sqrt([jnp.array([1.0])])

        ph_id = PenaltyHandler()
        ph_id.add(S1, penalize_null_space=False, identifiability_fn=_drop_last_col)
        B_id = ph_id.compute_sqrt([jnp.array([1.0])])

        np.testing.assert_allclose(np.array(B_id), np.array(B_full[:, :-1]), atol=ATOL)


# ---------------------------------------------------------------------------
# SINGLE_WITH_NULL
# ---------------------------------------------------------------------------

class TestSINGLE_WITH_NULL:
    def test_method_selected(self, S1):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=True)
        assert SqrtMethod.SINGLE_WITH_NULL in [m for m, *_ in ph._groups]

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
        ph.add(S1, penalize_null_space=True, identifiability_fn=_drop_last_col)
        B = ph.compute_sqrt([jnp.array([0.0, 0.0])])
        assert B.shape == (q, q - 1)


# ---------------------------------------------------------------------------
# KRONECKER
# ---------------------------------------------------------------------------

class TestKRONECKER:
    def test_method_selected(self, S1):
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=False)
        assert SqrtMethod.KRONECKER in [m for m, *_ in ph._groups]

    def test_not_fired_when_one_factor_full_rank(self, S1, S_full_rank):
        ph = PenaltyHandler()
        ph.add_kron([S1, S_full_rank], penalize_null_space=True)
        methods = [m for m, *_ in ph._groups]
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
        ph.add_kron([S1, S1], penalize_null_space=False, identifiability_fn=_drop_last_col)
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
        assert SqrtMethod.KRONECKER_WITH_NULL in [m for m, *_ in ph._groups]

    def test_requires_all_factors_rank_deficient(self, S1, S_full_rank):
        ph = PenaltyHandler()
        ph.add_kron([S1, S_full_rank], penalize_null_space=True)
        assert SqrtMethod.KRONECKER_WITH_NULL not in [m for m, *_ in ph._groups]

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
        assert SqrtMethod.GENERAL in [m for m, *_ in ph._groups]

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
        ph.add(S_kron, identifiability_fn=_drop_last_col)
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
        ph.add(S1, penalize_null_space=False, identifiability_fn=_drop_last_col)
        ph.add(S1, penalize_null_space=False, identifiability_fn=_drop_last_col)
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


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJIT:
    """compute_sqrt must be JIT-compilable; eager and JIT results must agree."""

    def _check(self, ph, rhos):
        B_eager = ph.compute_sqrt(rhos)
        B_jit = jax.jit(ph.compute_sqrt)(rhos)
        # Compare B.T @ B, not B: eigh has sign ambiguity so eigenvector columns
        # may flip between eager and JIT, while the penalty matrix is invariant.
        np.testing.assert_allclose(
            np.array(B_jit.T @ B_jit), np.array(B_eager.T @ B_eager), atol=ATOL
        )

    def test_singleton_groups_all_methods(self, S1, S_kron):
        """One penalty per method — every group is a singleton (linear path)."""
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)            # SINGLE
        ph.add(S1, penalize_null_space=True)             # SINGLE_WITH_NULL
        ph.add_kron([S1, S1], penalize_null_space=False) # KRONECKER
        ph.add_kron([S1, S1], penalize_null_space=True)  # KRONECKER_WITH_NULL
        ph.add(S_kron)                                   # GENERAL
        rhos = [
            jnp.array([0.5]),
            jnp.array([0.5, -0.5]),
            jnp.array([0.5, -0.5]),
            jnp.array([0.5, -0.5, 0.0]),
            jnp.array([0.5, -0.5]),
        ]
        self._check(ph, rhos)

    def test_vmap_groups_all_methods(self, S1, S_kron):
        """Two penalties per method — every group has two members (vmap path)."""
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        ph.add(S1, penalize_null_space=False)            # SINGLE × 2
        ph.add(S1, penalize_null_space=True)
        ph.add(S1, penalize_null_space=True)             # SINGLE_WITH_NULL × 2
        ph.add_kron([S1, S1], penalize_null_space=False)
        ph.add_kron([S1, S1], penalize_null_space=False) # KRONECKER × 2
        ph.add_kron([S1, S1], penalize_null_space=True)
        ph.add_kron([S1, S1], penalize_null_space=True)  # KRONECKER_WITH_NULL × 2
        ph.add(S_kron)
        ph.add(S_kron)                                   # GENERAL × 2
        rhos = [
            jnp.array([0.5]),  jnp.array([-0.5]),
            jnp.array([0.5, -0.5]),  jnp.array([-0.5, 0.5]),
            jnp.array([0.5, -0.5]),  jnp.array([-0.5, 0.5]),
            jnp.array([0.5, -0.5, 0.0]),  jnp.array([-0.5, 0.5, 0.0]),
            jnp.array([0.5, -0.5]),  jnp.array([-0.5, 0.5]),
        ]
        self._check(ph, rhos)


# ---------------------------------------------------------------------------
# compute_log_det_and_grad
# ---------------------------------------------------------------------------

def _ref_log_det(S_lam):
    """Numpy reference: log|S_lam|_+ via eigenvalues."""
    eig = np.linalg.eigvalsh(np.array(S_lam))
    return float(np.sum(np.log(eig[eig > 1e-10])))


class TestLogDetAndGrad:

    def _ph_ld_grad(self, ph, rhos, idx=0):
        lds, gs = ph.compute_log_det_and_grad(rhos)
        return float(lds[idx]), np.array(gs[idx])

    # ---- SINGLE ------------------------------------------------------------

    @pytest.mark.parametrize("rho", [-1.0, 0.0, 1.5])
    def test_single_value(self, S1, rho):
        rho_arr = jnp.array([rho])
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        ld, _ = self._ph_ld_grad(ph, [rho_arr])
        S_lam = float(jnp.exp(rho_arr[0])) * np.array(S1)
        np.testing.assert_allclose(ld, _ref_log_det(S_lam), atol=ATOL)

    @pytest.mark.parametrize("rho", [-1.0, 0.0, 1.5])
    def test_single_grad(self, S1, rho):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        rho = jnp.array([rho])
        _, g = self._ph_ld_grad(ph, [rho])
        g_cd = central_diff(lambda r: self._ph_ld_grad(ph, [jnp.asarray(r)])[0], rho)
        np.testing.assert_allclose(g, g_cd, rtol=1e-5)

    # ---- SINGLE_WITH_NULL --------------------------------------------------

    @pytest.mark.parametrize("rho_pen,rho_null", [(-1.0, 0.0), (0.5, -0.5)])
    def test_single_with_null_value(self, S1, S1_null, rho_pen, rho_null):
        rho_arr = jnp.array([rho_pen, rho_null])
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=True)
        ld, _ = self._ph_ld_grad(ph, [rho_arr])
        S_lam = (float(jnp.exp(rho_arr[0])) * np.array(S1)
                 + float(jnp.exp(rho_arr[1])) * np.array(S1_null))
        np.testing.assert_allclose(ld, _ref_log_det(S_lam), atol=ATOL)

    @pytest.mark.parametrize("rho_pen,rho_null", [(-1.0, 0.0), (0.5, -0.5)])
    def test_single_with_null_grad(self, S1, rho_pen, rho_null):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=True)
        rho = jnp.array([rho_pen, rho_null])
        _, g = self._ph_ld_grad(ph, [rho])
        g_cd = central_diff(lambda r: self._ph_ld_grad(ph, [jnp.asarray(r)])[0], rho)
        np.testing.assert_allclose(g, g_cd, rtol=1e-5)

    # ---- KRONECKER ---------------------------------------------------------

    @pytest.mark.parametrize("rho0,rho1", [(-1.0, 0.5), (0.0, 0.0)])
    def test_kronecker_value(self, S1, S_kron, rho0, rho1):
        rho_arr = jnp.array([rho0, rho1])
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=False)
        ld, _ = self._ph_ld_grad(ph, [rho_arr])
        lams = np.exp([rho0, rho1])
        S_lam = lams[0] * np.array(S_kron[0]) + lams[1] * np.array(S_kron[1])
        np.testing.assert_allclose(ld, _ref_log_det(S_lam), atol=1e-8)

    @pytest.mark.parametrize("rho0, rho1", [(-1.0, 0.5), (0.0, 0.0)])
    def test_kronecker_grad(self, S1, rho0, rho1):
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=False)
        rho = jnp.array([rho0, rho1])
        _, g = self._ph_ld_grad(ph, [rho])
        g_cd = central_diff(lambda r: self._ph_ld_grad(ph, [jnp.asarray(r)])[0], rho)
        np.testing.assert_allclose(g, g_cd, rtol=1e-5)

    # ---- KRONECKER_WITH_NULL -----------------------------------------------

    @pytest.mark.parametrize("rho0,rho1,rho_null", [(-1.0, 0.5, 0.0), (0.0, 0.0, -1.0)])
    def test_kronecker_with_null_value(self, S1, S_kron, S_kron_null, rho0, rho1, rho_null):
        rho_arr = jnp.array([rho0, rho1, rho_null])
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=True)
        ld, _ = self._ph_ld_grad(ph, [rho_arr])
        lams = np.exp([rho0, rho1])
        S_lam = (lams[0] * np.array(S_kron[0]) + lams[1] * np.array(S_kron[1])
                 + np.exp(rho_null) * np.array(S_kron_null))
        np.testing.assert_allclose(ld, _ref_log_det(S_lam), atol=1e-8)

    @pytest.mark.parametrize("rho0,rho1,rho_null", [(-1.0, 0.5, 0.0), (0.0, 0.0, -1.0)])
    def test_kronecker_with_null_grad(self, S1, rho0, rho1, rho_null):
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=True)
        rho = jnp.array([rho0, rho1, rho_null])
        _, g = self._ph_ld_grad(ph, [rho])
        g_cd = central_diff(lambda r: self._ph_ld_grad(ph, [jnp.asarray(r)])[0], rho)
        np.testing.assert_allclose(g, g_cd, rtol=1e-5)

    # ---- GENERAL -----------------------------------------------------------

    @pytest.mark.parametrize("rho0,rho1", [(-1.0, 0.5), (0.0, 0.0)])
    def test_general_value(self, S_kron, rho0, rho1):
        rho_arr = jnp.array([rho0, rho1])
        ph = PenaltyHandler()
        ph.add(S_kron)
        ld, _ = self._ph_ld_grad(ph, [rho_arr])
        lams = np.exp([rho0, rho1])
        S_lam = lams[0] * np.array(S_kron[0]) + lams[1] * np.array(S_kron[1])
        np.testing.assert_allclose(ld, _ref_log_det(S_lam), atol=1e-8)

    @pytest.mark.parametrize("rho0, rho1", [(-1.0, 0.5), (0.0, 0.0)])
    def test_general_grad(self, S_kron, rho0, rho1):
        ph = PenaltyHandler()
        ph.add(S_kron)
        rho = jnp.array([rho0, rho1])
        _, g = self._ph_ld_grad(ph, [rho])
        g_cd = central_diff(lambda r: self._ph_ld_grad(ph, [jnp.asarray(r)])[0], rho)
        np.testing.assert_allclose(g, g_cd, rtol=1e-5)

    # ---- GENERAL matches KRONECKER on same tensor --------------------------

    def test_general_matches_kronecker(self, S1, S_kron):
        rho = jnp.array([0.7, -0.3])
        ph_kron = PenaltyHandler()
        ph_kron.add_kron([S1, S1], penalize_null_space=False)
        ph_gen = PenaltyHandler()
        ph_gen.add(S_kron)
        ld_kron, g_kron = ph_kron.compute_log_det_and_grad([rho])
        ld_gen, g_gen = ph_gen.compute_log_det_and_grad([rho])
        np.testing.assert_allclose(float(ld_kron[0]), float(ld_gen[0]), atol=1e-8)
        np.testing.assert_allclose(np.array(g_kron[0]), np.array(g_gen[0]), atol=1e-5)

    # ---- JIT ---------------------------------------------------------------

    def test_jit(self, S1, S_kron):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        ph.add(S1, penalize_null_space=True)
        ph.add_kron([S1, S1], penalize_null_space=False)
        ph.add_kron([S1, S1], penalize_null_space=True)
        ph.add(S_kron)
        rhos = [
            jnp.array([0.5]),
            jnp.array([0.5, -0.5]),
            jnp.array([0.5, -0.5]),
            jnp.array([0.5, -0.5, 0.0]),
            jnp.array([0.5, -0.5]),
        ]
        lds_e, gs_e = ph.compute_log_det_and_grad(rhos)
        lds_j, gs_j = jax.jit(ph.compute_log_det_and_grad)(rhos)
        for ld_e, ld_j in zip(lds_e, lds_j):
            np.testing.assert_allclose(float(ld_j), float(ld_e), atol=ATOL)
        for g_e, g_j in zip(gs_e, gs_j):
            np.testing.assert_allclose(np.array(g_j), np.array(g_e), atol=ATOL)

    # ---- vmap (two members per group) --------------------------------------

    def test_vmap_matches_singletons(self, S1):
        rho_a, rho_b = jnp.array([0.5, -0.5]), jnp.array([-0.3, 0.7])
        ph_group = PenaltyHandler()
        ph_group.add_kron([S1, S1], penalize_null_space=False)
        ph_group.add_kron([S1, S1], penalize_null_space=False)
        lds_g, gs_g = ph_group.compute_log_det_and_grad([rho_a, rho_b])

        for k, rho in enumerate([rho_a, rho_b]):
            ph = PenaltyHandler()
            ph.add_kron([S1, S1], penalize_null_space=False)
            ld, g = ph.compute_log_det_and_grad([rho])
            np.testing.assert_allclose(float(lds_g[k]), float(ld[0]), atol=ATOL)
            np.testing.assert_allclose(np.array(gs_g[k]), np.array(g[0]), atol=ATOL)


# ---------------------------------------------------------------------------
# build() — pre-built closures
# ---------------------------------------------------------------------------

class TestBuild:
    """build() returns callables equivalent to compute_sqrt / compute_log_det_and_grad."""

    def _make_mixed_ph(self, S1, S_kron):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        ph.add(S1, penalize_null_space=True)
        ph.add_kron([S1, S1], penalize_null_space=False)
        ph.add_kron([S1, S1], penalize_null_space=True)
        ph.add(S_kron)
        return ph

    def _mixed_rhos(self):
        return [
            jnp.array([0.5]),
            jnp.array([0.5, -0.5]),
            jnp.array([0.5, -0.5]),
            jnp.array([0.5, -0.5, 0.0]),
            jnp.array([0.5, -0.5]),
        ]

    def test_compute_sqrt_matches_method(self, S1, S_kron):
        ph = self._make_mixed_ph(S1, S_kron)
        rhos = self._mixed_rhos()
        compute_sqrt, _, _ = ph.build()
        B_built = compute_sqrt(rhos)
        B_method = ph.compute_sqrt(rhos)
        np.testing.assert_allclose(
            np.array(B_built.T @ B_built), np.array(B_method.T @ B_method), atol=ATOL
        )

    def test_compute_log_det_matches_method(self, S1, S_kron):
        ph = self._make_mixed_ph(S1, S_kron)
        rhos = self._mixed_rhos()
        _, compute_ld, _ = ph.build()
        lds_built, gs_built = compute_ld(rhos)
        lds_method, gs_method = ph.compute_log_det_and_grad(rhos)
        for ld_b, ld_m in zip(lds_built, lds_method):
            np.testing.assert_allclose(float(ld_b), float(ld_m), atol=ATOL)
        for g_b, g_m in zip(gs_built, gs_method):
            np.testing.assert_allclose(np.array(g_b), np.array(g_m), atol=ATOL)

    def test_compute_sqrt_jittable(self, S1, S_kron):
        ph = self._make_mixed_ph(S1, S_kron)
        rhos = self._mixed_rhos()
        compute_sqrt, _, _ = ph.build()
        B_eager = compute_sqrt(rhos)
        B_jit = jax.jit(compute_sqrt)(rhos)
        np.testing.assert_allclose(
            np.array(B_jit.T @ B_jit), np.array(B_eager.T @ B_eager), atol=ATOL
        )

    def test_compute_log_det_jittable(self, S1, S_kron):
        ph = self._make_mixed_ph(S1, S_kron)
        rhos = self._mixed_rhos()
        _, compute_ld, _ = ph.build()
        lds_e, gs_e = compute_ld(rhos)
        lds_j, gs_j = jax.jit(compute_ld)(rhos)
        for ld_e, ld_j in zip(lds_e, lds_j):
            np.testing.assert_allclose(float(ld_j), float(ld_e), atol=ATOL)
        for g_e, g_j in zip(gs_e, gs_j):
            np.testing.assert_allclose(np.array(g_j), np.array(g_e), atol=ATOL)

    def test_vmap_group_sqrt_matches_singleton(self, S1):
        """build() vmap path matches per-singleton results."""
        rho_a, rho_b = jnp.array([0.5]), jnp.array([-0.3])
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        ph.add(S1, penalize_null_space=False)
        compute_sqrt, _, _ = ph.build()
        B_full = compute_sqrt([rho_a, rho_b])
        q = S1.shape[0]
        B0 = B_full[:q, :q]
        B1 = B_full[q:, q:]

        ph0 = PenaltyHandler()
        ph0.add(S1, penalize_null_space=False)
        ph1 = PenaltyHandler()
        ph1.add(S1, penalize_null_space=False)
        np.testing.assert_allclose(
            np.array(B0.T @ B0), np.array(ph0.compute_sqrt([rho_a]).T @ ph0.compute_sqrt([rho_a])), atol=ATOL
        )
        np.testing.assert_allclose(
            np.array(B1.T @ B1), np.array(ph1.compute_sqrt([rho_b]).T @ ph1.compute_sqrt([rho_b])), atol=ATOL
        )


# ---------------------------------------------------------------------------
# compute_log_det_hess — Hessian of log|S_lam|+ wrt rho (block-diagonal)
# ---------------------------------------------------------------------------

class TestLogDetHess:
    """Hessian of ``Σᵢ log|S_lam_i|_+`` wrt the flattened ρ.

    The handler's gradient ``compute_log_det_and_grad`` is itself an analytical
    expression; differentiating it once more by central differences gives a
    high-precision FD reference.  Tests compare the analytical Hessian against
    this reference for every method (SINGLE, SINGLE_WITH_NULL, KRONECKER,
    KRONECKER_WITH_NULL, GENERAL) plus block-diagonal mixtures.
    """

    @staticmethod
    def _fd_hess_from_grad(ph, rhos, i_pen):
        """Numerical Hessian of log_det[i_pen] wrt rhos[i_pen] from grads."""
        def grad_fn(r):
            _, gs = ph.compute_log_det_and_grad(
                [jnp.asarray(r) if j == i_pen else rhos[j] for j in range(len(rhos))]
            )
            return np.asarray(gs[i_pen])

        return central_diff(grad_fn, rhos[i_pen])

    # ---- SINGLE — Hessian is identically zero -----------------------------

    @pytest.mark.parametrize("rho", [-1.5, 0.0, 1.5])
    def test_single_zero(self, S1, rho):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        _, _, compute_ldh = ph.build()
        H = compute_ldh([jnp.array([rho])])
        np.testing.assert_allclose(np.array(H), np.zeros((1, 1)), atol=ATOL)

    # ---- SINGLE_WITH_NULL — Hessian is identically zero -------------------

    @pytest.mark.parametrize("rho_pen,rho_null", [(-1.0, 0.0), (0.5, -0.5)])
    def test_single_with_null_zero(self, S1, rho_pen, rho_null):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=True)
        _, _, compute_ldh = ph.build()
        H = compute_ldh([jnp.array([rho_pen, rho_null])])
        np.testing.assert_allclose(np.array(H), np.zeros((2, 2)), atol=ATOL)

    # ---- KRONECKER --------------------------------------------------------

    @pytest.mark.parametrize("rho0,rho1", [(-1.0, 0.5), (0.0, 0.0), (1.2, -0.8)])
    def test_kronecker_matches_fd(self, S1, rho0, rho1):
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=False)
        _, _, compute_ldh = ph.build()
        rho = jnp.array([rho0, rho1])
        H = np.asarray(compute_ldh([rho]))
        H_fd = self._fd_hess_from_grad(ph, [rho], i_pen=0)
        np.testing.assert_allclose(H, H_fd, rtol=1e-5, atol=1e-7)

    def test_kronecker_symmetric(self, S1):
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=False)
        _, _, compute_ldh = ph.build()
        H = np.asarray(compute_ldh([jnp.array([0.3, -0.4])]))
        np.testing.assert_allclose(H, H.T, atol=ATOL)

    # ---- KRONECKER_WITH_NULL ---------------------------------------------

    @pytest.mark.parametrize("rho0,rho1,rho_null",
                             [(-1.0, 0.5, 0.0), (0.0, 0.0, -1.0), (0.7, -0.3, 0.4)])
    def test_kronecker_with_null_matches_fd(self, S1, rho0, rho1, rho_null):
        ph = PenaltyHandler()
        ph.add_kron([S1, S1], penalize_null_space=True)
        _, _, compute_ldh = ph.build()
        rho = jnp.array([rho0, rho1, rho_null])
        H = np.asarray(compute_ldh([rho]))
        H_fd = self._fd_hess_from_grad(ph, [rho], i_pen=0)
        np.testing.assert_allclose(H, H_fd, rtol=1e-5, atol=1e-7)
        # Null-space coordinate (last row/col) must be exactly zero —
        # log|S_lam|+ is linear in ρ_null, so all second derivatives vanish.
        np.testing.assert_allclose(H[-1, :], np.zeros(3), atol=ATOL)
        np.testing.assert_allclose(H[:, -1], np.zeros(3), atol=ATOL)

    # ---- GENERAL ----------------------------------------------------------

    @pytest.mark.parametrize("rho0,rho1", [(-1.0, 0.5), (0.0, 0.0), (1.2, -0.8)])
    def test_general_matches_fd(self, S_kron, rho0, rho1):
        ph = PenaltyHandler()
        ph.add(S_kron)
        _, _, compute_ldh = ph.build()
        rho = jnp.array([rho0, rho1])
        H = np.asarray(compute_ldh([rho]))
        H_fd = self._fd_hess_from_grad(ph, [rho], i_pen=0)
        np.testing.assert_allclose(H, H_fd, rtol=1e-5, atol=1e-7)

    def test_general_matches_kronecker(self, S1, S_kron):
        """GENERAL on a Kronecker tensor matches add_kron's analytical Hessian."""
        rho = jnp.array([0.7, -0.3])
        ph_kron = PenaltyHandler()
        ph_kron.add_kron([S1, S1], penalize_null_space=False)
        ph_gen = PenaltyHandler()
        ph_gen.add(S_kron)
        _, _, ldh_kron = ph_kron.build()
        _, _, ldh_gen = ph_gen.build()
        H_kron = np.asarray(ldh_kron([rho]))
        H_gen = np.asarray(ldh_gen([rho]))
        np.testing.assert_allclose(H_kron, H_gen, atol=1e-5)

    # ---- Block-diagonal mixture ------------------------------------------

    def test_block_diagonal_structure(self, S1, S_kron):
        """Mixed handler: rho of penalty i has no influence on log|S_lam_j| for j != i."""
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)          # SINGLE
        ph.add(S1, penalize_null_space=True)           # SINGLE_WITH_NULL
        ph.add_kron([S1, S1], penalize_null_space=False)  # KRONECKER
        ph.add_kron([S1, S1], penalize_null_space=True)   # KRONECKER_WITH_NULL
        ph.add(S_kron)                                  # GENERAL
        rhos = [
            jnp.array([0.5]),
            jnp.array([0.5, -0.5]),
            jnp.array([0.5, -0.5]),
            jnp.array([0.5, -0.5, 0.0]),
            jnp.array([0.5, -0.5]),
        ]
        _, _, compute_ldh = ph.build()
        H = np.asarray(compute_ldh(rhos))

        sizes = [r.shape[0] for r in rhos]
        offsets = np.cumsum([0] + sizes)
        total = offsets[-1]
        assert H.shape == (total, total)

        # Off-diagonal blocks must be exactly zero (penalties are independent).
        for i in range(len(sizes)):
            for j in range(len(sizes)):
                if i == j:
                    continue
                block = H[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]]
                np.testing.assert_allclose(block, 0.0, atol=ATOL)

        # On-diagonal blocks must match each penalty's standalone Hessian.
        cases = [
            (PenaltyHandler(), lambda p: p.add(S1, penalize_null_space=False)),
            (PenaltyHandler(), lambda p: p.add(S1, penalize_null_space=True)),
            (PenaltyHandler(), lambda p: p.add_kron([S1, S1], penalize_null_space=False)),
            (PenaltyHandler(), lambda p: p.add_kron([S1, S1], penalize_null_space=True)),
            (PenaltyHandler(), lambda p: p.add(S_kron)),
        ]
        for i, (ph_i, builder) in enumerate(cases):
            builder(ph_i)
            _, _, ldh_i = ph_i.build()
            H_i = np.asarray(ldh_i([rhos[i]]))
            diag = H[offsets[i]:offsets[i + 1], offsets[i]:offsets[i + 1]]
            np.testing.assert_allclose(diag, H_i, atol=ATOL)

    # ---- vmap path (group with two members) ------------------------------

    def test_vmap_matches_singletons(self, S1):
        """Two penalties with the same method share a vmap group; result matches."""
        rho_a, rho_b = jnp.array([0.5, -0.5]), jnp.array([-0.3, 0.7])
        ph_group = PenaltyHandler()
        ph_group.add_kron([S1, S1], penalize_null_space=False)
        ph_group.add_kron([S1, S1], penalize_null_space=False)
        _, _, ldh_group = ph_group.build()
        H_group = np.asarray(ldh_group([rho_a, rho_b]))

        # Per-member singleton handlers
        ph_a = PenaltyHandler()
        ph_a.add_kron([S1, S1], penalize_null_space=False)
        _, _, ldh_a = ph_a.build()
        H_a = np.asarray(ldh_a([rho_a]))

        ph_b = PenaltyHandler()
        ph_b.add_kron([S1, S1], penalize_null_space=False)
        _, _, ldh_b = ph_b.build()
        H_b = np.asarray(ldh_b([rho_b]))

        m = rho_a.shape[0]
        np.testing.assert_allclose(H_group[:m, :m], H_a, atol=ATOL)
        np.testing.assert_allclose(H_group[m:, m:], H_b, atol=ATOL)
        # Cross-block: still zero
        np.testing.assert_allclose(H_group[:m, m:], 0.0, atol=ATOL)
        np.testing.assert_allclose(H_group[m:, :m], 0.0, atol=ATOL)

    # ---- JIT --------------------------------------------------------------

    def test_jittable(self, S1, S_kron):
        ph = PenaltyHandler()
        ph.add(S1, penalize_null_space=False)
        ph.add_kron([S1, S1], penalize_null_space=True)
        ph.add(S_kron)
        rhos = [
            jnp.array([0.5]),
            jnp.array([0.5, -0.5, 0.0]),
            jnp.array([0.5, -0.5]),
        ]
        _, _, compute_ldh = ph.build()
        H_e = np.asarray(compute_ldh(rhos))
        H_j = np.asarray(jax.jit(compute_ldh)(rhos))
        np.testing.assert_allclose(H_j, H_e, atol=ATOL)

"""Tests for penalty handling when empty columns are dropped."""

import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest
from conftest import any_columns_dropped, n_columns_dropped

from pgam_jax import GAM
from pgam_jax._penalty_handler import (
    PenaltyHandler,
    _GeneralPenalty,
    _KroneckerWithNullPenalty,
    _SinglePenalty,
    _SingleWithNullPenalty,
)
from pgam_jax.penalty_utils import DROP_LAST_COL, IDENTITY

jax.config.update("jax_enable_x64", True)


def _bspline(k):
    return nmo.basis.BSplineEval(k, bounds=(0.0, 1.0))


def _island_inputs(n=600, radius=0.22, center=(0.3, 0.35), seed=0):
    """Positions confined to a disc inside the unit square."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    counts = rng.poisson(1.0, n).astype(float)
    return (x, y), jnp.asarray(counts)


def _spread_inputs(n=600, seed=1):
    """Positions covering the whole unit square."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.02, 0.98, n)
    y = rng.uniform(0.02, 0.98, n)
    counts = rng.poisson(1.0, n).astype(float)
    return (x, y), jnp.asarray(counts)


def _prepared(basis, xi, counts, drop_empty_columns):
    """Run detection only, without a full fit."""
    gam = GAM(basis, drop_empty_columns=drop_empty_columns)
    gam._fit_design_matrix(xi, counts)
    return gam


@pytest.fixture
def tensor_basis():
    return _bspline(8) * _bspline(8)


class TestDetectionOnATensorBasis:
    def test_the_island_leaves_columns_empty(self, tensor_basis):
        xi, counts = _island_inputs()
        gam = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        assert any_columns_dropped(gam)
        assert n_columns_dropped(gam) > 10
        assert len(gam.component_infos_) == 1

    def test_spread_data_leaves_nothing_empty(self, tensor_basis):
        xi, counts = _spread_inputs()
        gam = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        assert not any_columns_dropped(gam)

    def test_the_flag_off_keeps_every_column(self, tensor_basis):
        xi, counts = _island_inputs()
        gam = _prepared(tensor_basis, xi, counts, drop_empty_columns=False)
        assert not any_columns_dropped(gam)

    def test_the_design_matrix_shrinks(self, tensor_basis):
        xi, counts = _island_inputs()
        on = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        off = _prepared(tensor_basis, xi, counts, drop_empty_columns=False)
        X_on, _ = on._fit_design_matrix(xi, counts)
        X_off, _ = off._fit_design_matrix(xi, counts)
        n_kept = on.component_infos_[0].n_kept
        assert X_on.shape[1] == n_kept - 1
        assert X_off.shape[1] == 64 - 1
        assert X_on.shape[1] < X_off.shape[1]


class TestMaskedPenaltyTree:
    def test_tree_is_sliced_to_the_kept_columns(self, tensor_basis):
        xi, counts = _island_inputs()
        gam = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        k = gam.component_infos_[0].n_kept
        tree = gam._get_penalty_tree()
        assert len(tree) == 1
        assert tree[0].shape[1:] == (k, k)

    def test_tree_is_untouched_without_a_mask(self, tensor_basis):
        xi, counts = _spread_inputs()
        gam = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        tree = gam._get_penalty_tree()
        assert tree[0].shape[1:] == (64, 64)

    def test_slicing_keeps_the_penalty_symmetric_and_psd(self, tensor_basis):
        xi, counts = _island_inputs()
        gam = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        for S in gam._get_penalty_tree()[0]:
            np.testing.assert_allclose(S, S.T, atol=1e-10)
            assert np.min(np.linalg.eigvalsh(S)) > -1e-8


class TestPenaltyHandlerRouting:
    def test_unmasked_tensor_keeps_the_kronecker_path(self, tensor_basis):
        xi, counts = _spread_inputs()
        gam = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        ph = gam._build_penalty_handler(gam._get_penalty_tree())
        assert isinstance(ph._penalties[0], _KroneckerWithNullPenalty)

    def test_masked_tensor_falls_back_to_the_general_path(self, tensor_basis):
        xi, counts = _island_inputs()
        gam = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        ph = gam._build_penalty_handler(gam._get_penalty_tree())
        assert isinstance(ph._penalties[0], _GeneralPenalty)

    def test_unmasked_one_dimensional_keeps_the_single_path(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0.02, 0.98, 400)
        counts = jnp.asarray(rng.poisson(1.0, 400).astype(float))
        gam = _prepared(_bspline(8), (x,), counts, drop_empty_columns=True)
        assert not any_columns_dropped(gam)
        ph = gam._build_penalty_handler(gam._get_penalty_tree())
        assert isinstance(ph._penalties[0], _SingleWithNullPenalty)

    def test_masked_one_dimensional_drops_the_redundant_null_term(self):
        """A reduced 1-D energy penalty is full rank, so the null lambda goes."""
        rng = np.random.default_rng(4)
        x = rng.uniform(0.02, 0.35, 400)  # covers only part of the range
        counts = jnp.asarray(rng.poisson(1.0, 400).astype(float))
        gam = _prepared(_bspline(10), (x,), counts, drop_empty_columns=True)
        assert any_columns_dropped(gam)
        tree = gam._get_penalty_tree()
        assert tree[0].shape[0] == 1
        ph = gam._build_penalty_handler(tree)
        assert isinstance(ph._penalties[0], _SinglePenalty)
        assert ph._penalties[0].rho_len == 1

    @pytest.mark.parametrize("drop_identifiability", [False, True])
    @pytest.mark.parametrize("rho", [[0.3, -0.7], [-4.0, 3.0], [3.0, -4.0]])
    def test_masked_one_dimensional_with_null_matches_general(
        self, drop_identifiability, rho
    ):
        """
        Removing one column leaves one unpenalized linear direction.

        Zero-only masking retains the identifiability drop. Threshold masking
        removes a nonempty column and keeps all surviving columns instead.
        """
        x = np.linspace(0.02, 0.79, 400)
        if drop_identifiability:
            min_obs = True
            id_fn = DROP_LAST_COL
        else:
            x = np.append(x, 0.9)
            min_obs = 2
            id_fn = IDENTITY
        gam = _prepared(_bspline(8), (x,), np.ones(len(x)), min_obs)
        info = gam.component_infos_[0]
        assert info.n_dropped == 1
        assert info.drops_identifiability_column == drop_identifiability
        tree = gam._get_penalty_tree()
        assert tree[0].shape == (2, 7, 7)

        routed = gam._build_penalty_handler(tree)
        penalty = routed._penalties[0]
        assert isinstance(penalty, _SingleWithNullPenalty)
        assert penalty.rank_null == 1
        assert penalty.rho_len == gam._init_regularizer_strength(tree)[0].size == 2

        baseline = PenaltyHandler()
        baseline.add(tree[0], penalize_null_space=False, identifiability_fn=id_fn)
        assert isinstance(baseline._penalties[0], _GeneralPenalty)
        rhos = [jnp.asarray(rho)]
        B = np.asarray(routed.compute_sqrt(rhos))
        B_general = np.asarray(baseline.compute_sqrt(rhos))
        expected = np.einsum("i,ijk->jk", np.exp(rho), tree[0])
        if drop_identifiability:
            expected = expected[:-1, :-1]
        np.testing.assert_allclose(B.T @ B, expected, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(B.T @ B, B_general.T @ B_general, atol=1e-8)

        values, gradients = routed.compute_log_det_and_grad(rhos)
        general_values, general_gradients = baseline.compute_log_det_and_grad(rhos)
        np.testing.assert_allclose(values, general_values, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(gradients, general_gradients, rtol=1e-8, atol=1e-8)


class TestRhoLengthsAgree:
    @pytest.mark.parametrize("inputs_fn", [_island_inputs, _spread_inputs])
    def test_handler_rho_len_matches_the_tree(self, tensor_basis, inputs_fn):
        xi, counts = inputs_fn()
        gam = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        tree = gam._get_penalty_tree()
        ph = gam._build_penalty_handler(tree)
        init = gam._init_regularizer_strength(tree)
        assert [p.rho_len for p in ph._penalties] == [r.shape[0] for r in init]

    def test_masking_drops_the_redundant_null_lambda(self, tensor_basis):
        """
        The null-space term is rebuilt from the reduced penalty, not carried over.

        A null-space vector of the full tensor penalty is a global polynomial.
        It does not vanish outside the kept columns, so the reduced penalty is
        full rank and needs no separate null-space smoothing parameter. Keeping
        one leaves a parameter with nothing to penalize, and selection drives it
        to infinity.
        """
        xi, counts = _island_inputs()
        on = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        off = _prepared(tensor_basis, xi, counts, drop_empty_columns=False)
        on_tree = on._get_penalty_tree()
        off_tree = off._get_penalty_tree()
        assert off_tree[0].shape[0] == 3  # two energy terms plus the null term
        assert on_tree[0].shape[0] == 2  # the null term is gone
        reduced = np.asarray(on_tree[0]).sum(axis=0)
        assert np.linalg.matrix_rank(reduced) == reduced.shape[0]


def _penalty_from_tree(S_tensor, rho):
    """Reference weighted penalty with the identifiability row and column gone."""
    lams = np.exp(np.asarray(rho))
    S = np.einsum("i,ijk->jk", lams, np.asarray(S_tensor))
    return S[:-1, :-1]


class TestMaskedSqrtIsCorrect:
    """``compute_sqrt`` must factor the masked, identifiability-reduced penalty."""

    def test_threshold_tensor_penalty_keeps_all_survivors(self):
        rng = np.random.default_rng(0)
        xi = tuple(rng.uniform(0.02, 0.45, 400) for _ in range(2))
        gam = _prepared(_bspline(8) * _bspline(8), xi, np.ones(400), 100)
        info = gam.component_infos_[0]
        assert not info.drops_identifiability_column
        tree = gam._get_penalty_tree()
        sqrt, logdet = gam._build_penalty_handler(tree).build()
        rho = [jnp.asarray([0.3, -0.7])]
        S = np.einsum("i,ijk->jk", np.exp(rho[0]), tree[0])
        B = np.asarray(sqrt(rho))
        assert B.shape[1] == info.n_kept
        np.testing.assert_allclose(B.T @ B, S, atol=1e-8)
        values, gradients = logdet(rho)
        np.testing.assert_allclose(values[0], np.linalg.slogdet(S)[1])
        expected_grad = np.exp(rho[0]) * np.einsum(
            "ab,iba->i", np.linalg.inv(S), tree[0]
        )
        np.testing.assert_allclose(gradients[0], expected_grad, rtol=1e-8)

    @pytest.mark.parametrize("lam", [1.0, 1e-6])
    def test_nearly_dependent_survivors_match_dense_penalized_solve(self, lam):
        x = np.random.default_rng(0).uniform(0.02, 0.45, 400)
        gam = GAM(_bspline(10), drop_empty_columns=26)
        X, _ = gam._fit_design_matrix((x,), np.ones(len(x)))
        info = gam.component_infos_[0]
        assert not info.drops_identifiability_column
        assert np.linalg.cond(X) > 1e4

        full = _prepared(_bspline(10), (x,), np.ones(len(x)), False)
        # The full tree's first entry is the energy penalty, before its null term.
        S = np.asarray(full._get_penalty_tree()[0][0])
        S = S[np.ix_(info.nonempty_mask, info.nonempty_mask)]
        tree = gam._get_penalty_tree()
        sqrt, logdet = gam._build_penalty_handler(tree).build()
        rho = [jnp.asarray([np.log(lam)])]
        B = np.asarray(sqrt(rho))
        np.testing.assert_allclose(B.T @ B, lam * S, rtol=1e-9, atol=1e-10)
        values, gradients = logdet(rho)
        np.testing.assert_allclose(values[0], np.linalg.slogdet(lam * S)[1])
        np.testing.assert_allclose(gradients[0], [S.shape[0]])

        target = np.sin(5 * x)
        target -= target.mean()
        augmented = np.vstack((X, B))
        beta = np.linalg.lstsq(
            augmented, np.concatenate((target, np.zeros(B.shape[0]))), rcond=None
        )[0]
        expected = np.linalg.solve(
            np.asarray(X).T @ X + lam * S, np.asarray(X).T @ target
        )
        np.testing.assert_allclose(beta, expected, rtol=1e-7, atol=1e-8)

    @pytest.mark.parametrize("inputs_fn", [_island_inputs, _spread_inputs])
    def test_sqrt_reproduces_the_penalty(self, tensor_basis, inputs_fn):
        xi, counts = inputs_fn()
        gam = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        tree = gam._get_penalty_tree()
        ph = gam._build_penalty_handler(tree)
        compute_sqrt, _ = ph.build()

        rho = [jnp.asarray([0.3, -0.7, 1.1][: tree[0].shape[0]])]
        B = np.asarray(compute_sqrt(rho))
        got = B.T @ B
        expected = _penalty_from_tree(tree[0], rho[0])
        np.testing.assert_allclose(got, expected, atol=1e-8)

    def test_log_det_matches_the_dense_pseudo_determinant(self, tensor_basis):
        xi, counts = _island_inputs()
        gam = _prepared(tensor_basis, xi, counts, drop_empty_columns=True)
        tree = gam._get_penalty_tree()
        ph = gam._build_penalty_handler(tree)
        _, compute_log_det_and_grad = ph.build()

        rho = [jnp.asarray([0.3, -0.7, 1.1][: tree[0].shape[0]])]
        log_dets, _ = compute_log_det_and_grad(rho)

        S = _penalty_from_tree(tree[0], rho[0])
        eig = np.linalg.eigvalsh(S)
        tol = eig.max() * S.shape[0] * np.finfo(S.dtype).eps
        expected = np.sum(np.log(eig[eig > tol]))
        np.testing.assert_allclose(float(log_dets[0]), expected, rtol=1e-8)

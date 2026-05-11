import enum
import jax.numpy as jnp
import jax
from collections import defaultdict
from typing import Any, Callable

from nemos.basis import MultiplicativeBasis

from .penalty_utils import sym_sqrt
from functools import reduce
from .penalty_utils import compute_energy_penalty, ndim_tensor_product_basis_penalty, compute_penalty_null_space

class SqrtMethod(enum.Enum):
    SINGLE = "single"
    ORTHOGONAL = "orthogonal"
    KRONECKER = "kronecker"
    KRONECKER_WITH_NULL = "kronecker_with_null"
    GENERAL = "general"


def _preprocess_kron(S):
    S = 0.5 * (S + S.T)
    eig, U = jnp.linalg.eigh(S)
    thresh = jnp.finfo(float).eps ** 0.7 * jnp.maximum(jnp.abs(eig).max(), 1e-300)
    pos = eig > thresh
    rank = jnp.sum(pos)
    # Keep static shapes: zero out sub-threshold eigenvalues instead of selecting.
    # U is returned in full (q × q); eigenvectors are columns, as per eigh convention.
    return jnp.where(pos, eig, 0.0), U, rank


class PenaltyHandler:

    def __init__(self, n_samples: int=10**4, non_linearity: Callable=jnp.exp):
        self._n_samples = n_samples
        self._groups = defaultdict(list)
        self._penalty_tensors = []
        self._cache: list[dict[SqrtMethod, Any]] = []
        self._non_linearity = non_linearity

    def _energy_penalty(self, basis) -> list[jax.Array]:
        return [
            compute_energy_penalty(
            self._n_samples,
            b.derivative,
            getattr(b, "bounds", None) or (0.0, 1.0),
        )
        for b in basis._iterate_over_components()
    ]

    @staticmethod
    def _get_sqrt_method(n_penalties: int, has_null:bool) -> SqrtMethod:
        # if len(penalty_list) == 1, then it is single penalty
        if n_penalties == 1 and has_null:
            method = SqrtMethod.ORTHOGONAL
        elif n_penalties == 1 and not has_null:
            method = SqrtMethod.SINGLE
        elif n_penalties > 1 and has_null:
            method = SqrtMethod.KRONECKER_WITH_NULL
        elif n_penalties > 1 and not has_null:
            method = SqrtMethod.KRONECKER
        else:
            # should never happen
            raise ValueError("Empty penalty list.")
        return method

    def add(self, basis, penalize_null_space: bool=True):
        """

        Parameters
        ----------
        penalty_tensor :
            Shape (M, q, q).
        penalize_null_space :
           True if null space should be penalized.

        """
        if basis._n_inputs > 1 and not isinstance(basis, MultiplicativeBasis):
            raise ValueError("PenaltyHandler 1D bases or products of 1D basis.")

        penalty_list = self._energy_penalty(basis)

        # compute kron product for multi-dimensional bases
        penalty_tensor = ndim_tensor_product_basis_penalty(*penalty_list)
        n_penalties = len(penalty_tensor)

        null_pen = None
        has_null = False
        if penalize_null_space:
            _null = compute_penalty_null_space(penalty_tensor)
            if ~jnp.all(_null == 0):
                null_pen = _null
                penalty_tensor = jnp.concatenate((penalty_tensor, null_pen[None]), axis=0)
                has_null = True

        method = self._get_sqrt_method(n_penalties, has_null)

        # assign group and store penalty
        self._groups[(method, penalty_tensor.shape)].append(len(self._penalty_tensors))
        self._penalty_tensors.append(penalty_tensor)
        self._cache.append(self._compute_cache(penalty_list, null_pen, method))


    @staticmethod
    def _compute_cache(penalty_list: list, null_pen, method) -> dict:
        """
        Parameters
        ----------
        penalty_list :
            1-D factor penalties, one (q_k, q_k) matrix per factor dimension.
            For 1-D bases this is a single-element list; for multiplicative bases
            it has one entry per multiplicative component.
        null_pen :
            Full (prod_q, prod_q) null-space penalty matrix, or None when the
            null space is empty / not penalized.
        method :
            Algebraic structure label.
        """
        match method:
            case SqrtMethod.SINGLE:
                B, rank, log_det_S = sym_sqrt(penalty_list[0])
                return {"B": B, "rank": rank, "log_det_S": log_det_S}

            case SqrtMethod.ORTHOGONAL:
                # Stack the 1-D energy penalty with the null penalty so that both
                # are processed in a single vmapped call (same q, same shape).
                stacked = jnp.stack([penalty_list[0], null_pen])
                B, rank, log_det_S = jax.vmap(sym_sqrt)(stacked)
                return {"B": B, "rank": rank, "log_det_S": log_det_S}

            case SqrtMethod.KRONECKER:
                # Decompose each 1-D factor independently (Python loop: factors can
                # have different sizes so vmap is not applicable in general).
                results = [_preprocess_kron(S) for S in penalty_list]
                eigs  = [eig  for eig, _, _    in results]  # list of (q_k,) arrays
                Us    = [U    for _,   U, _    in results]  # list of (q_k, q_k) arrays
                ranks = [rank for _,   _, rank in results]  # list of scalars
                # (U_1 ⊗ U_2 ⊗ ...) precomputed; shape (prod_q, prod_q).
                # At sqrt time: B = diag(sqrt(outer_eig)) @ kron_U.T
                kron_U = reduce(jnp.kron, Us)
                return {"eigs": eigs, "kron_U": kron_U, "ranks": ranks}

            case SqrtMethod.KRONECKER_WITH_NULL:
                results = [_preprocess_kron(S) for S in penalty_list]
                eigs  = [eig  for eig, _, _    in results]
                Us    = [U    for _,   U, _    in results]
                ranks = [rank for _,   _, rank in results]
                kron_U = reduce(jnp.kron, Us)
                B, rank, log_det_S = sym_sqrt(null_pen)
                return {
                    "kron": {"eigs": eigs, "kron_U": kron_U, "ranks": ranks},
                    "null": {"B": B, "rank": rank, "log_det_S": log_det_S},
                }

            case SqrtMethod.GENERAL:
                # not yet implemented; must use the runtime S_lam
                return {}

        raise ValueError(f"Unrecognized penalty method ``{method}``.")

    @staticmethod
    def _kron_sqrt(eigs, kron_U, lams):
        # Outer sum: combined[i_1,...,i_n] = sum_k lams[k] * eigs[k][i_k]
        combined = reduce(jnp.add.outer, [lam * eig for lam, eig in zip(lams, eigs)])
        combined = combined.ravel()                       # (prod_q,)
        # B s.t. B.T @ B = sum_k lams[k] * S_k(kron); zero rows for null modes.
        return jnp.sqrt(combined)[:, None] * kron_U.T    # (prod_q, prod_q)

    def _sqrt(self, method, cache, lams):
        match method:
            case SqrtMethod.SINGLE:
                return jnp.sqrt(lams[0]) * cache["B"]

            case SqrtMethod.ORTHOGONAL:
                B = cache["B"]                              # (2, q, q)
                sqrt_lams = jnp.sqrt(lams)                 # (2,)
                return (sqrt_lams[:, None, None] * B).reshape(-1, B.shape[-1])

            case SqrtMethod.KRONECKER:
                return self._kron_sqrt(cache["eigs"], cache["kron_U"], lams)

            case SqrtMethod.KRONECKER_WITH_NULL:
                B_kron = self._kron_sqrt(
                    cache["kron"]["eigs"], cache["kron"]["kron_U"], lams[:-1]
                )
                B_null = jnp.sqrt(lams[-1]) * cache["null"]["B"]
                return jnp.concatenate([B_kron, B_null], axis=0)

            case _:
                raise NotImplementedError(f"_sqrt not implemented for {method}.")

    def compute_sqrt(self, rhos):
        lams = jax.tree_util.tree_map(self._non_linearity, rhos)
        # for now, just loop over members
        # for the future, stack all cache and vmap
        for (method, _), members in self._groups.items():
            for idx in members:
                self._sqrt(method, self._cache[idx], lams[idx])

import enum
import jax.numpy as jnp
import jax
from collections import defaultdict
from typing import Any, Callable

from nemos.basis import MultiplicativeBasis

from functools import reduce
from .penalty_utils import compute_energy_penalty, ndim_tensor_product_basis_penalty, compute_penalty_null_space

class SqrtMethod(enum.Enum):
    SINGLE = 0
    SINGLE_WITH_NULL = 1
    KRONECKER = 2
    KRONECKER_WITH_NULL = 3
    GENERAL = 4


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
            method = SqrtMethod.SINGLE_WITH_NULL
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
            Single-element list for 1-D bases; one entry per factor for
            multiplicative bases.
        null_pen :
            Full (prod_q, prod_q) null-space penalty matrix, or None when the
            null space is empty / not penalized.  null_pen need not be
            decomposed: compute_penalty_null_space projects onto the null
            eigenvectors of S_energy, so U_energy simultaneously diagonalises
            both S_energy and null_pen.  The null penalty has all active
            eigenvalues equal to 1, so only rank_null is needed for log-det.
        method :
            Algebraic structure label.
        """
        match method:
            case SqrtMethod.SINGLE:
                eig, U, rank = _preprocess_kron(penalty_list[0])
                log_det_S = jnp.sum(jnp.where(eig > 0, jnp.log(jnp.where(eig > 0, eig, 1.0)), 0.0))
                return {"eig": eig, "U": U, "rank": rank, "log_det_S": log_det_S}

            case SqrtMethod.SINGLE_WITH_NULL:
                eig, U, rank = _preprocess_kron(penalty_list[0])
                log_det_S = jnp.sum(jnp.where(eig > 0, jnp.log(jnp.where(eig > 0, eig, 1.0)), 0.0))
                rank_null = eig.shape[0] - rank   # null_pen eigenvalues are 1, no decomposition needed
                return {"eig": eig, "U": U, "rank": rank, "rank_null": rank_null, "log_det_S": log_det_S}

            case SqrtMethod.KRONECKER | SqrtMethod.KRONECKER_WITH_NULL:
                results   = [_preprocess_kron(S) for S in penalty_list]
                eigs      = [eig  for eig, _, _    in results]
                Us        = [U    for _,   U, _    in results]
                ranks     = [rank for _,   _, rank in results]
                log_det_S = [jnp.sum(jnp.where(eig > 0, jnp.log(jnp.where(eig > 0, eig, 1.0)), 0.0))
                             for eig in eigs]
                kron_U    = reduce(jnp.kron, Us)
                return {"eigs": eigs, "kron_U": kron_U, "ranks": ranks, "log_det_S": log_det_S}

            case SqrtMethod.GENERAL:
                # not yet implemented; must use the runtime S_lam
                return {}

        raise ValueError(f"Unrecognized penalty method ``{method}``.")

    def _sqrt(self, method, cache, lams):
        match method:
            case SqrtMethod.SINGLE:
                sqrt_d = jnp.sqrt(lams[0] * cache["eig"])
                return sqrt_d[:, None] * cache["U"].T                          # (q, q)

            case SqrtMethod.SINGLE_WITH_NULL:
                eig    = cache["eig"]
                sqrt_d = jnp.where(eig > 0,
                                   jnp.sqrt(lams[0] * eig),
                                   jnp.sqrt(lams[1]))
                return sqrt_d[:, None] * cache["U"].T                          # (q, q)

            case SqrtMethod.KRONECKER:
                combined = reduce(jnp.add.outer,
                                  [lam * eig for lam, eig in zip(lams, cache["eigs"])]).ravel()
                return jnp.sqrt(combined)[:, None] * cache["kron_U"].T        # (prod_q, prod_q)

            case SqrtMethod.KRONECKER_WITH_NULL:
                combined = reduce(
                    jnp.add.outer,
                    [lam * eig for lam, eig in zip(lams[:-1], cache["eigs"])]
                ).ravel()
                sqrt_d = jnp.where(combined > 0, jnp.sqrt(combined), jnp.sqrt(lams[-1]))
                return sqrt_d[:, None] * cache["kron_U"].T                     # (prod_q, prod_q)

            case _:
                raise NotImplementedError(f"_sqrt not implemented for {method}.")

    def compute_sqrt(self, rhos) -> list:
        lams = jax.tree_util.tree_map(self._non_linearity, rhos)
        # for now, just loop over members
        # for the future, stack all cache and vmap
        out = [None] * len(self._penalty_tensors)
        for (method, _), members in self._groups.items():
            for idx in members:
                out[idx] = self._sqrt(method, self._cache[idx], lams[idx])
        return out


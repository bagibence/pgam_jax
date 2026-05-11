import enum
import jax.numpy as jnp
import jax
from collections import defaultdict
from typing import Callable
from functools import reduce

from ._slam_compute import transform_slam_with_Q


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

    def __init__(self, non_linearity: Callable = jnp.exp):
        self._groups: dict = defaultdict(list)
        self._n_penalties: int = 0
        self._cache: list[dict] = []
        self._non_linearity = non_linearity

    def add(self, S_tensor, penalize_null_space: bool = True):
        """
        Parameters
        ----------
        S_tensor :
            Shape (k, q, q) or (q, q) stacked penalty matrices.  The penalty
            is ``sum_j lambda_j * S_tensor[j]``.
        penalize_null_space :
            Only relevant when k == 1.  For k > 1 the GENERAL method removes
            the null space implicitly via the full-rank precompute projection.
        """
        S_tensor = jnp.asarray(S_tensor)
        if S_tensor.ndim == 2:
            S_tensor = S_tensor[None]

        if S_tensor.shape[0] == 1:
            cache, method = self._compute_cache(S_tensor[0], penalize_null_space)
        else:
            cache, method = self._compute_cache(S_tensor, penalize_null_space)

        self._groups[(method, S_tensor.shape)].append(self._n_penalties)
        self._n_penalties += 1
        self._cache.append(cache)

    def add_kron(self, factor_list, penalize_null_space: bool = True):
        """
        Parameters
        ----------
        factor_list :
            List of (q_i, q_i) 1D factor penalty matrices.  Exploits the
            Kronecker-sum structure to compute the sqrt via per-factor
            eigendecompositions, avoiding the full (prod_q, prod_q) eigh.
        penalize_null_space :
            If True and every factor individually has a null space, the null
            space of the Kronecker sum is penalized via a separate lambda.
        """
        cache, method = self._compute_cache(factor_list, penalize_null_space)
        q = reduce(lambda a, b: a * b, [S.shape[0] for S in factor_list])
        shape = (len(factor_list), q, q)
        self._groups[(method, shape)].append(self._n_penalties)
        self._n_penalties += 1
        self._cache.append(cache)

    @staticmethod
    def _compute_cache(data, penalize_null_space: bool = True) -> tuple[dict, SqrtMethod]:
        """
        Compute the precomputed cache and select the sqrt method in a single pass.

        Parameters
        ----------
        data :
            - 2-D array (q, q)     → SINGLE / SINGLE_WITH_NULL
            - list of 2-D arrays   → KRONECKER / KRONECKER_WITH_NULL
            - 3-D array (k, q, q)  → GENERAL
        penalize_null_space :
            Ignored for GENERAL (null space dropped by the precompute projection).
        """
        if isinstance(data, list):
            # KRONECKER
            results   = [_preprocess_kron(S) for S in data]
            eigs      = [eig  for eig, _, _    in results]
            Us        = [U    for _,   U, _    in results]
            ranks     = [rank for _,   _, rank in results]
            # Kronecker-sum null space exists iff every factor has its own null space.
            has_null  = penalize_null_space and all(
                bool(rank < S.shape[0]) for rank, S in zip(ranks, data)
            )
            method    = SqrtMethod.KRONECKER_WITH_NULL if has_null else SqrtMethod.KRONECKER
            log_det_S = [jnp.sum(jnp.where(eig > 0, jnp.log(jnp.where(eig > 0, eig, 1.0)), 0.0))
                         for eig in eigs]
            kron_U    = reduce(jnp.kron, Us)
            return {"eigs": eigs, "kron_U": kron_U, "ranks": ranks, "log_det_S": log_det_S}, method

        if data.ndim == 2:
            # SINGLE
            eig, U, rank = _preprocess_kron(data)
            has_null  = penalize_null_space and bool(rank < data.shape[0])
            method    = SqrtMethod.SINGLE_WITH_NULL if has_null else SqrtMethod.SINGLE
            log_det_S = jnp.sum(jnp.where(eig > 0, jnp.log(jnp.where(eig > 0, eig, 1.0)), 0.0))
            cache     = {"eig": eig, "U": U, "rank": rank, "log_det_S": log_det_S}
            if method is SqrtMethod.SINGLE_WITH_NULL:
                cache["rank_null"] = data.shape[0] - rank
            return cache, method

        # GENERAL (ndim == 3)
        S        = data
        frobs    = jnp.sqrt(jnp.sum(S**2, axis=(1, 2), keepdims=True))
        norm_avg = jnp.sum(S / frobs, axis=0)
        ev, U    = jnp.linalg.eigh(norm_avg)
        eps      = float(jnp.finfo(float).eps**0.8)
        keep     = ev > ev[-1] * eps          # eager boolean index — precompute only
        U_keep   = U[:, keep]                 # (q, r)
        full_rank_S = U_keep.T[None] @ S @ U_keep[None]  # (k, r, r)
        return {"U_keep": U_keep, "full_rank_S": full_rank_S}, SqrtMethod.GENERAL

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

            case SqrtMethod.GENERAL:
                U_keep      = cache["U_keep"]           # (q, r)
                full_rank_S = cache["full_rank_S"]      # (k, r, r) — formally full rank
                lams_arr    = jnp.stack(lams)           # (k,)
                S_i_out, Q_s = transform_slam_with_Q(full_rank_S, lams_arr)
                S_full = jnp.einsum('i,ijk->jk', lams_arr, S_i_out)  # (r, r) stable
                S_full = 0.5 * (S_full + S_full.T)
                ev, U   = jnp.linalg.eigh(S_full)
                safe_ev = jnp.where(ev > 0, ev, 1.0)
                sqrt_d  = jnp.where(ev > 0, jnp.sqrt(safe_ev), 0.0)
                B_rot   = sqrt_d[:, None] * U.T         # (r, r) — sqrt in rotated basis
                return (B_rot @ Q_s.T) @ U_keep.T       # (r, q) — back to original basis

            case _:
                raise NotImplementedError(f"_sqrt not implemented for {method}.")

    def compute_sqrt(self, rhos) -> list:
        lams = jax.tree_util.tree_map(self._non_linearity, rhos)
        out = [None] * self._n_penalties
        for (method, _), members in self._groups.items():
            for idx in members:
                out[idx] = self._sqrt(method, self._cache[idx], lams[idx])
        return out
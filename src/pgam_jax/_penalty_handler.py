import enum
import jax.numpy as jnp
import jax
from collections import defaultdict
from typing import Callable
from functools import reduce

from nemos.inverse_link_function_utils import identity
from ._slam_compute import transform_slam_with_Q, transform_slam, log_det_and_grad_slam
from jax.scipy.linalg import block_diag


class SqrtMethod(enum.Enum):
    SINGLE = 0
    SINGLE_WITH_NULL = 1
    KRONECKER = 2
    KRONECKER_WITH_NULL = 3
    GENERAL = 4


def _drop_last_col(x):
    return x[..., :-1]


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
    def __init__(self):
        self._groups: dict = defaultdict(list)
        self._n_penalties: int = 0
        self._cache: list[dict] = []
        self._rho_len = []

    def add(self, S_tensor, penalize_null_space: bool = True, identifiability_fn: Callable = identity):
        """
        Parameters
        ----------
        S_tensor :
            Shape (k, q, q) or (q, q) stacked penalty matrices.  The penalty
            is ``sum_j lambda_j * S_tensor[j]``.
        penalize_null_space :
            Only relevant when k == 1.  For k > 1 the GENERAL method removes
            the null space implicitly via the full-rank precompute projection.
        identifiability_fn :
            Must be a module-level callable (not a lambda) so that object
            identity is stable and can be used as a group key for vmap batching.
            Use ``identity`` (no-op) or ``_drop_last_col``, or define your own
            at module level.
        """
        S_tensor = jnp.asarray(S_tensor)
        if S_tensor.ndim == 2:
            S_tensor = S_tensor[None]

        if S_tensor.shape[0] == 1:
            cache, method = self._compute_cache(S_tensor[0], penalize_null_space)
        else:
            cache, method = self._compute_cache(S_tensor, penalize_null_space)

        self._groups[(method, S_tensor.shape, identifiability_fn)].append(self._n_penalties)
        self._n_penalties += 1
        self._cache.append(cache)

    def add_kron(self, factor_list, penalize_null_space: bool = True, identifiability_fn: Callable = identity):
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
        identifiability_fn :
            See ``add``.
        """
        cache, method = self._compute_cache(factor_list, penalize_null_space)
        q = reduce(lambda a, b: a * b, [S.shape[0] for S in factor_list])
        shape = (len(factor_list), q, q)
        self._groups[(method, shape, identifiability_fn)].append(self._n_penalties)
        self._n_penalties += 1
        self._cache.append(cache)

    def _compute_cache(self, data, penalize_null_space: bool = True) -> tuple[dict, SqrtMethod]:
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
            self._rho_len.append(len(eigs) + has_null)
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
            self._rho_len.append(1 + has_null)
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
        self._rho_len.append(full_rank_S.shape[0])
        return {"U_keep": U_keep, "full_rank_S": full_rank_S}, SqrtMethod.GENERAL

    def _sqrt(self, method, cache, lams, id_fn):
        match method:
            case SqrtMethod.SINGLE:
                sqrt_d = jnp.sqrt(lams[0] * cache["eig"])
                return sqrt_d[:, None] * id_fn(cache["U"].T)  # (q, q)

            case SqrtMethod.SINGLE_WITH_NULL:
                eig    = cache["eig"]
                sqrt_d = jnp.where(eig > 0,
                                   jnp.sqrt(lams[0] * eig),
                                   jnp.sqrt(lams[1]))
                return sqrt_d[:, None] * id_fn(cache["U"].T)  # (q, q)

            case SqrtMethod.KRONECKER:
                combined = reduce(
                    jnp.add.outer,
                    [lam * eig for lam, eig in zip(lams, cache["eigs"])]
                ).ravel()
                return jnp.sqrt(combined)[:, None] * id_fn(cache["kron_U"].T)  # (prod_q, prod_q)

            case SqrtMethod.KRONECKER_WITH_NULL:
                combined = reduce(
                    jnp.add.outer,
                    [lam * eig for lam, eig in zip(lams[:-1], cache["eigs"])]
                ).ravel()
                sqrt_d = jnp.where(combined > 0, jnp.sqrt(combined), jnp.sqrt(lams[-1]))
                return sqrt_d[:, None] * id_fn(cache["kron_U"].T)  # (prod_q, prod_q)

            case SqrtMethod.GENERAL:
                U_keep      = cache["U_keep"]           # (q, r)
                full_rank_S = cache["full_rank_S"]      # (k, r, r) — formally full rank
                S_i_out, Q_s = transform_slam_with_Q(full_rank_S, lams)
                S_full = jnp.einsum('i,ijk->jk', lams, S_i_out)  # (r, r) stable
                S_full = 0.5 * (S_full + S_full.T)
                ev, U   = jnp.linalg.eigh(S_full)
                safe_ev = jnp.where(ev > 0, ev, 1.0)
                sqrt_d  = jnp.where(ev > 0, jnp.sqrt(safe_ev), 0.0)
                B_rot   = sqrt_d[:, None] * U.T          # (r, r) — sqrt in rotated basis
                return (B_rot @ Q_s.T) @ id_fn(U_keep.T) # (r, q) — back to original basis

            case _:
                raise NotImplementedError(f"_sqrt not implemented for {method}.")

    @staticmethod
    def _kron_log_det_factor_grads(lams, eigs):
        """
        Shared core for KRONECKER* log-det.

        Returns
        -------
        log_det_pos : scalar
            log|S_lam|_+ restricted to the positive-eigenvalue subspace.
        factor_grads : list[scalar]
            d log_det_pos / d rho[j] for each factor j.
        n_pos : scalar
            Number of positive eigenvalues (= total - null-space dim).
        """
        combined_nd = reduce(
            jnp.add.outer,
            [lam * eig for lam, eig in zip(lams, eigs)]
        )
        pos = combined_nd > 0
        safe = jnp.where(pos, combined_nd, 1.0)
        log_det_pos = jnp.sum(jnp.where(pos, jnp.log(safe), 0.0))
        # d/drho[j] = lam_j * sum_{pos} eig_j[i_j] / combined[multi-index]
        ndim = len(eigs)
        factor_grads = []
        for j, (lam, eig) in enumerate(zip(lams, eigs)):
            shape = [1] * ndim
            shape[j] = -1
            factor_grads.append(lam * jnp.sum(jnp.where(pos, eig.reshape(shape) / safe, 0.0)))
        return log_det_pos, factor_grads, jnp.sum(pos)

    def _log_det_and_grad(self, method, cache, rho):
        """Return (log|S_lam|_+, d log|S_lam|_+ / d rho) for one penalty."""
        match method:
            case SqrtMethod.SINGLE:
                # log|lam*S|_+ = rank*rho + log_det_S,  grad = rank
                rank = cache["rank"]
                log_det = rank * rho[0] + cache["log_det_S"]
                return log_det, jnp.full_like(rho, rank)

            case SqrtMethod.SINGLE_WITH_NULL:
                # log|lam0*S + lam1*P_null|_+ = rank*rho0 + log_det_S + rank_null*rho1
                rank, rank_null = cache["rank"], cache["rank_null"]
                log_det = rank * rho[0] + cache["log_det_S"] + rank_null * rho[1]
                return log_det, jnp.stack([rank, rank_null]).astype(rho.dtype)

            case SqrtMethod.KRONECKER:
                lams = jnp.exp(rho)
                log_det, factor_grads, _ = self._kron_log_det_factor_grads(
                    lams, cache["eigs"]
                )
                return log_det, jnp.stack(factor_grads)

            case SqrtMethod.KRONECKER_WITH_NULL:
                lams = jnp.exp(rho)
                log_det_pos, factor_grads, n_pos = self._kron_log_det_factor_grads(
                    lams[:-1], cache["eigs"]
                )
                total = reduce(lambda a, b: a * b, [eig.shape[0] for eig in cache["eigs"]])
                n_null = total - n_pos
                log_det = log_det_pos + n_null * rho[-1]
                return log_det, jnp.stack([*factor_grads, n_null.astype(rho.dtype)])

            case SqrtMethod.GENERAL:
                S_i_out = transform_slam(cache["full_rank_S"], rho)
                return log_det_and_grad_slam(rho, S_i_out)

            case _:
                raise NotImplementedError(f"_log_det_and_grad not implemented for {method}.")

    def compute_log_det_and_grad(self, rhos: list[jnp.ndarray]) -> tuple[list, list]:
        """
        Returns
        -------
        log_dets : list[scalar]
            log|S_lam_i|_+ for each registered penalty i.
        grads : list[jnp.ndarray]
            d log|S_lam_i|_+ / d rho for each i; shape matches rhos[i].
        """
        log_dets = [None] * self._n_penalties
        grads = [None] * self._n_penalties
        for (method, _, id_fn), members in self._groups.items():
            if len(members) == 1:
                idx = members[0]
                ld, g = self._log_det_and_grad(method, self._cache[idx], rhos[idx])
                log_dets[idx] = ld
                grads[idx] = g
            else:
                batched_cache = jax.tree_util.tree_map(
                    lambda *a: jnp.stack(a), *[self._cache[i] for i in members]
                )
                batched_rhos = jnp.stack([rhos[i] for i in members])
                batched_ld, batched_g = jax.vmap(
                    lambda c, r: self._log_det_and_grad(method, c, r)
                )(batched_cache, batched_rhos)
                for k, idx in enumerate(members):
                    log_dets[idx] = batched_ld[k]
                    grads[idx] = batched_g[k]
        return log_dets, grads

    def compute_sqrt(self, rhos: list[jnp.ndarray]) -> jnp.ndarray:
        lams = jax.tree_util.tree_map(jnp.exp, rhos)
        out = [None] * self._n_penalties
        for (method, _, id_fn), members in self._groups.items():
            if len(members) == 1:
                idx = members[0]
                out[idx] = self._sqrt(method, self._cache[idx], lams[idx], id_fn)
            else:
                # All members share method, shape, and id_fn → batch with vmap.
                batched_cache = jax.tree_util.tree_map(
                    lambda *a: jnp.stack(a), *[self._cache[i] for i in members]
                )
                batched_lams = jnp.stack([lams[i] for i in members])
                batched_out = jax.vmap(
                    lambda c, l: self._sqrt(method, c, l, id_fn)
                )(batched_cache, batched_lams)
                for k, idx in enumerate(members):
                    out[idx] = batched_out[k]
        return block_diag(*out)

    def build(self) -> tuple:
        """
        Snapshot the handler into two pure callables.

        Returns
        -------
        compute_sqrt : callable
            ``compute_sqrt(rhos) -> (total_pen_rows, n_params)`` block-diagonal matrix.
        compute_log_det_and_grad : callable
            ``compute_log_det_and_grad(rhos) -> (list[scalar], list[array])``
        """
        n = self._n_penalties
        caches = list(self._cache)
        _sqrt_fn = self._sqrt
        _ld_fn = self._log_det_and_grad

        group_data = []
        for (method, _shape, id_fn), members in self._groups.items():
            if len(members) == 1:
                group_data.append({
                    "singleton": True,
                    "method": method,
                    "id_fn": id_fn,
                    "idx": members[0],
                    "cache": caches[members[0]],
                })
            else:
                stacked = jax.tree_util.tree_map(
                    lambda *a: jnp.stack(a), *[caches[i] for i in members]
                )
                group_data.append({
                    "singleton": False,
                    "method": method,
                    "id_fn": id_fn,
                    "members": members,
                    "stacked_cache": stacked,
                    "vmapped_sqrt": jax.vmap(
                        lambda c, l, _m=method, _f=id_fn: _sqrt_fn(_m, c, l, _f)
                    ),
                    "vmapped_ld": jax.vmap(
                        lambda c, r, _m=method: _ld_fn(_m, c, r)
                    ),
                })

        def compute_sqrt(rhos):
            lams = [jnp.exp(r) for r in rhos]
            out = [None] * n
            for g in group_data:
                if g["singleton"]:
                    out[g["idx"]] = _sqrt_fn(g["method"], g["cache"], lams[g["idx"]], g["id_fn"])
                else:
                    batched_lams = jnp.stack([lams[i] for i in g["members"]])
                    batched_out = g["vmapped_sqrt"](g["stacked_cache"], batched_lams)
                    for k, idx in enumerate(g["members"]):
                        out[idx] = batched_out[k]
            return block_diag(*out)

        def compute_log_det_and_grad(rhos):
            log_dets = [None] * n
            grads = [None] * n
            for g in group_data:
                if g["singleton"]:
                    ld, gr = _ld_fn(g["method"], g["cache"], rhos[g["idx"]])
                    log_dets[g["idx"]] = ld
                    grads[g["idx"]] = gr
                else:
                    batched_rhos = jnp.stack([rhos[i] for i in g["members"]])
                    batched_ld, batched_g = g["vmapped_ld"](g["stacked_cache"], batched_rhos)
                    for k, idx in enumerate(g["members"]):
                        log_dets[idx] = batched_ld[k]
                        grads[idx] = batched_g[k]
            return log_dets, grads

        return compute_sqrt, compute_log_det_and_grad

    def __len__(self):
        return self._n_penalties

    def __repr__(self) -> str:
        string = self.__class__.__name__ + "("
        if len(self._rho_len) == 0:
            return string + ")"
        for rlen in self._rho_len:
            string += f"\n\tn_reg_strengths={rlen},"
        string = string[:-1] + "\n)"
        return string
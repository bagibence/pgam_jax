import enum
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from nemos.inverse_link_function_utils import identity

from ._slam_compute import log_det_and_grad_slam, transform_slam, transform_slam_with_Q


class SqrtMethod(enum.Enum):
    SINGLE = 0
    SINGLE_WITH_NULL = 1
    KRONECKER = 2
    KRONECKER_WITH_NULL = 3
    GENERAL = 4


def _drop_last_col(x):
    return x[..., :-1]


def _eigh_and_rank(S):
    S = 0.5 * (S + S.T)
    eig, U = jnp.linalg.eigh(S)
    thresh = jnp.finfo(float).eps ** 0.7 * jnp.maximum(jnp.abs(eig).max(), 1e-300)
    pos = eig > thresh
    rank = jnp.sum(pos)
    # Keep static shapes: zero out sub-threshold eigenvalues instead of selecting.
    # U is returned in full (q × q); eigenvectors are columns, as per eigh convention.
    return jnp.where(pos, eig, 0.0), U, rank


def _log_pseudo_det_from_eigvals(eig):
    """
    Sum of log of positive eigenvalues.

    jnp.where traces both branches, so a naive log(0) on the dead branch would
    still produce a NaN gradient under autograd. The inner where substitutes a
    safe 1.0 first so log only sees positive inputs. Gradients are currently
    analytic everywhere this is called, but it doesn't hurt to guard against
    this in case of future autograd use.
    """
    pos = eig > 0
    return jnp.sum(jnp.where(pos, jnp.log(jnp.where(pos, eig, 1.0)), 0.0))


def _safe_log(x):
    """log(x) for x > 0, -inf for x == 0, so no log(0) ever evaluated."""
    return jnp.where(x > 0, jnp.log(jnp.where(x > 0, x, 1.0)), -jnp.inf)


@dataclass(frozen=True, eq=False)
class _Penalty:
    # method for computing the sqrt penalty
    method: SqrtMethod
    # method-specific precomputed arrays
    cache: dict[str, jnp.ndarray | list[jnp.ndarray] | int]
    # identity or _drop_last_col
    id_fn: Callable
    # S_tensor shape
    shape: tuple[int, ...]

    @property
    def rho_len(self) -> int:
        # len(rhos[i]), so how many regularization strengths it needs
        match self.method:
            case SqrtMethod.SINGLE:
                return 1
            case SqrtMethod.SINGLE_WITH_NULL:
                return 2
            case SqrtMethod.KRONECKER:
                return len(self.cache["eigs"])
            case SqrtMethod.KRONECKER_WITH_NULL:
                return len(self.cache["eigs"]) + 1
            case SqrtMethod.GENERAL:
                return self.cache["full_rank_S"].shape[0]

    @property
    def _group_key(self):
        """Key for identifying stackable groups.

        Two penalties can be vmap-batched iff their caches have the same
        pytree structure with identical per-leaf shapes. That's what
        ``tree_map(jnp.stack)`` requires.
        Two penalties with identical cache structure might need different
        computations, so ``method`` and ``id_fn`` are also part of the key.
        E.g. KRONECKER and KRONECKER_WITH_NULL have the same cache structure,
        but do different computations.
        """
        leaves, treedef = jax.tree_util.tree_flatten(self.cache)
        leaf_shapes = tuple(jnp.shape(leaf) for leaf in leaves)
        return (self.method, self.id_fn, treedef, leaf_shapes)


class PenaltyHandler:
    def __init__(self):
        self._penalties = []

    def add(
        self,
        S_tensor,
        penalize_null_space: bool,
        identifiability_fn: Callable,
    ):
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
            S_tensor = S_tensor[None, :, :]

        # TODO: Give a more meaningful name to cache
        if S_tensor.shape[0] == 1:
            cache, method = self._compute_cache(
                S_tensor[0], penalize_null_space, identifiability_fn
            )
        else:
            cache, method = self._compute_cache(
                S_tensor, penalize_null_space, identifiability_fn
            )

        self._penalties.append(
            _Penalty(
                method=method,
                cache=cache,
                id_fn=identifiability_fn,
                shape=S_tensor.shape,
            )
        )

    def add_kron(
        self,
        factor_list,
        penalize_null_space: bool,
        identifiability_fn: Callable,
    ):
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
        cache, method = self._compute_cache(
            factor_list, penalize_null_space, identifiability_fn
        )
        q = reduce(lambda a, b: a * b, [S.shape[0] for S in factor_list])
        shape = (len(factor_list), q, q)
        self._penalties.append(
            _Penalty(
                method=method,
                cache=cache,
                id_fn=identifiability_fn,
                shape=shape,
            )
        )

    def _compute_cache(
        self,
        data,
        penalize_null_space: bool,
        id_fn: Callable,
    ) -> tuple[dict, SqrtMethod]:
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
            results = [_eigh_and_rank(S) for S in data]
            eigs = [eig for eig, _, _ in results]
            Us = [U for _, U, _ in results]
            ranks = [rank for _, _, rank in results]
            # Kronecker-sum null space exists iff every factor has its own null space.
            has_null = penalize_null_space and all(
                bool(rank < S.shape[0]) for rank, S in zip(ranks, data)
            )
            method = (
                SqrtMethod.KRONECKER_WITH_NULL if has_null else SqrtMethod.KRONECKER
            )
            log_det_S = [_log_pseudo_det_from_eigvals(eig) for eig in eigs]
            kron_U = reduce(jnp.kron, Us)
            # Schur-correction precompute for id_fn=_drop_last_col on KRONECKER_WITH_NULL:
            # store w[i] = U_kron[-1, i]^2 reshaped to factor dims so per-factor grads
            # can broadcast eig_j along axis j (same trick as _kron_log_det_factor_grads).
            factor_shape = tuple(eig.shape[0] for eig in eigs)
            u_kron_last_sq = (kron_U[-1, :] ** 2).reshape(factor_shape)

            return {
                "eigs": eigs,
                "kron_U": kron_U,
                "ranks": ranks,
                "log_det_S": log_det_S,
                "u_kron_last_sq": u_kron_last_sq,
            }, method

        if data.ndim == 2:
            # SINGLE
            eig, U, rank = _eigh_and_rank(data)
            has_null = penalize_null_space and bool(rank < data.shape[0])
            method = SqrtMethod.SINGLE_WITH_NULL if has_null else SqrtMethod.SINGLE
            log_det_S = _log_pseudo_det_from_eigvals(eig)
            cache = {
                "eig": eig,
                "U": U,
                "rank": rank,
                "log_det_S": log_det_S,
            }

            if method is SqrtMethod.SINGLE_WITH_NULL:
                cache["rank_null"] = data.shape[0] - rank

            # Schur-correction precompute for id_fn=_drop_last_col on SINGLE_WITH_NULL:
            # log|A[:-1,:-1]| = log|A| + log((A^-1)[-1,-1])
            #                 = log|A| + log(alpha * e^-rho0 + beta * e^-rho1)
            # alpha = sum_{eig>0} U[-1,i]^2 / eig[i], beta = sum_{eig=0} U[-1,i]^2.
            if method is SqrtMethod.SINGLE_WITH_NULL and id_fn is _drop_last_col:
                last_sq = U[-1, :] ** 2
                pos = eig > 0
                alpha = jnp.sum(jnp.where(pos, last_sq / jnp.where(pos, eig, 1.0), 0.0))
                beta = jnp.sum(jnp.where(pos, 0.0, last_sq))
                cache["log_alpha"] = _safe_log(alpha)
                cache["log_beta"] = _safe_log(beta)

            # Restricted precompute for id_fn=_drop_last_col on SINGLE only.
            # (SINGLE_WITH_NULL uses Schur instead; see _log_det_and_grad.)
            if method is SqrtMethod.SINGLE and id_fn is _drop_last_col:
                S_r = data[:-1, :-1]
                eig_r, _, rank_r = _eigh_and_rank(S_r)
                cache["rank_r"] = rank_r
                cache["log_det_S_r"] = _log_pseudo_det_from_eigvals(eig_r)

            return cache, method

        # GENERAL (ndim == 3)
        def _full_rank_precompute(S_in):
            """Frobenius-averaging + eigh + null-space drop. Returns (U_keep, full_rank_S)."""
            frobs = jnp.sqrt(jnp.sum(S_in**2, axis=(1, 2), keepdims=True))
            norm_avg = jnp.sum(S_in / frobs, axis=0)
            ev, U = jnp.linalg.eigh(norm_avg)
            eps = float(jnp.finfo(float).eps ** 0.8)
            keep = ev > ev[-1] * eps  # eager boolean index — precompute only
            U_keep = U[:, keep]
            full_rank_S = U_keep.T[None] @ S_in @ U_keep[None]
            return U_keep, full_rank_S

        S = data
        U_keep, full_rank_S = _full_rank_precompute(S)
        cache = {"U_keep": U_keep, "full_rank_S": full_rank_S}
        # Restricted precompute for id_fn=_drop_last_col on GENERAL.
        # Only full_rank_S_r is needed (log-det path); compute_sqrt is unaffected.
        if id_fn is _drop_last_col:
            _, full_rank_S_r = _full_rank_precompute(S[:, :-1, :-1])
            cache["full_rank_S_r"] = full_rank_S_r
        return cache, SqrtMethod.GENERAL

    def _sqrt(self, method, cache, lams, id_fn):
        match method:
            case SqrtMethod.SINGLE:
                sqrt_d = jnp.sqrt(lams[0] * cache["eig"])
                return sqrt_d[:, None] * id_fn(cache["U"].T)  # (q, q)

            case SqrtMethod.SINGLE_WITH_NULL:
                eig = cache["eig"]
                sqrt_d = jnp.where(eig > 0, jnp.sqrt(lams[0] * eig), jnp.sqrt(lams[1]))
                return sqrt_d[:, None] * id_fn(cache["U"].T)  # (q, q)

            case SqrtMethod.KRONECKER:
                combined = reduce(
                    jnp.add.outer, [lam * eig for lam, eig in zip(lams, cache["eigs"])]
                ).ravel()
                return jnp.sqrt(combined)[:, None] * id_fn(
                    cache["kron_U"].T
                )  # (prod_q, prod_q)

            case SqrtMethod.KRONECKER_WITH_NULL:
                combined = reduce(
                    jnp.add.outer,
                    [lam * eig for lam, eig in zip(lams[:-1], cache["eigs"])],
                ).ravel()
                sqrt_d = jnp.where(combined > 0, jnp.sqrt(combined), jnp.sqrt(lams[-1]))
                return sqrt_d[:, None] * id_fn(cache["kron_U"].T)  # (prod_q, prod_q)

            case SqrtMethod.GENERAL:
                U_keep = cache["U_keep"]  # (q, r)
                full_rank_S = cache["full_rank_S"]  # (k, r, r) — formally full rank
                S_i_out, Q_s = transform_slam_with_Q(full_rank_S, lams)
                S_full = jnp.einsum("i,ijk->jk", lams, S_i_out)  # (r, r) stable
                S_full = 0.5 * (S_full + S_full.T)
                ev, U = jnp.linalg.eigh(S_full)
                safe_ev = jnp.where(ev > 0, ev, 1.0)
                sqrt_d = jnp.where(ev > 0, jnp.sqrt(safe_ev), 0.0)
                B_rot = sqrt_d[:, None] * U.T  # (r, r) — sqrt in rotated basis
                return (B_rot @ Q_s.T) @ id_fn(
                    U_keep.T
                )  # (r, q) — back to original basis

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
        combined_nd = reduce(jnp.add.outer, [lam * eig for lam, eig in zip(lams, eigs)])
        pos = combined_nd > 0
        safe = jnp.where(pos, combined_nd, 1.0)
        log_det_pos = jnp.sum(jnp.where(pos, jnp.log(safe), 0.0))
        # d/drho[j] = lam_j * sum_{pos} eig_j[i_j] / combined[multi-index]
        ndim = len(eigs)
        factor_grads = []
        for j, (lam, eig) in enumerate(zip(lams, eigs)):
            shape = [1] * ndim
            shape[j] = -1
            factor_grads.append(
                lam * jnp.sum(jnp.where(pos, eig.reshape(shape) / safe, 0.0))
            )
        return log_det_pos, factor_grads, jnp.sum(pos)

    def _log_det_and_grad(self, method, cache, rho, id_fn):
        """Return (log|S_lam|_+, d log|S_lam|_+ / d rho) for one penalty.

        ``id_fn`` is the identifiability map (``identity`` or ``_drop_last_col``);
        methods that have a Schur correction for the restricted log-det use it
        to decide whether to apply the correction.
        """
        match method:
            case SqrtMethod.SINGLE:
                # log|lam*S_id|_+ = rank(S_id)*rho + log|S_id|_+, with S_id = id_fn'd S.
                if id_fn is _drop_last_col:
                    rank = cache["rank_r"]
                    log_det_S = cache["log_det_S_r"]
                elif id_fn is identity:
                    rank = cache["rank"]
                    log_det_S = cache["log_det_S"]
                else:
                    raise NotImplementedError(
                        f"SINGLE log_det not implemented for id_fn={id_fn}"
                    )
                log_det = rank * rho[0] + log_det_S
                return log_det, jnp.full_like(rho, rank)

            case SqrtMethod.SINGLE_WITH_NULL:
                # Full-basis: log|lam0*S + lam1*P_null| = rank*rho0 + log_det_S + rank_null*rho1
                rank, rank_null = cache["rank"], cache["rank_null"]
                log_det_full = rank * rho[0] + cache["log_det_S"] + rank_null * rho[1]
                grad_full = jnp.stack([rank, rank_null]).astype(rho.dtype)
                if id_fn is _drop_last_col:
                    # Schur correction: log|A[:-1,:-1]| = log|A| + log((A^-1)[-1,-1])
                    # (A^-1)[-1,-1] = alpha * exp(-rho0) + beta * exp(-rho1)
                    log_terms = jnp.stack(
                        [cache["log_alpha"] - rho[0], cache["log_beta"] - rho[1]]
                    )
                    c = jax.scipy.special.logsumexp(log_terms)
                    weights = jnp.exp(log_terms - c)  # softmax weights, sum to 1
                    return log_det_full + c, grad_full - weights
                elif id_fn is identity:
                    return log_det_full, grad_full
                else:
                    raise NotImplementedError(
                        f"SINGLE_WITH_NULL log_det not implemented for id_fn={id_fn}"
                    )

            case SqrtMethod.KRONECKER:
                if id_fn is _drop_last_col:
                    # Restricting Σ λ_k S_k to [:-1, :-1] destroys the Kron-sum
                    # eigenvalue factorization and the matrix is rank-deficient
                    # (Schur doesn't apply). The natural tensor-product smooth
                    # routes to KRONECKER_WITH_NULL instead, which is supported.
                    raise NotImplementedError(
                        "KRONECKER log_det not implemented under _drop_last_col; "
                        "use add_kron(..., penalize_null_space=True) for KRONECKER_WITH_NULL."
                    )
                elif id_fn is identity:
                    lams = jnp.exp(rho)
                    log_det, factor_grads, _ = self._kron_log_det_factor_grads(
                        lams, cache["eigs"]
                    )
                    return log_det, jnp.stack(factor_grads)
                else:
                    raise NotImplementedError(
                        f"KRONECKER log_det not implemented for id_fn={id_fn}"
                    )

            case SqrtMethod.KRONECKER_WITH_NULL:
                lams = jnp.exp(rho)
                log_det_pos, factor_grads, n_pos = self._kron_log_det_factor_grads(
                    lams[:-1], cache["eigs"]
                )
                total = reduce(
                    lambda a, b: a * b, [eig.shape[0] for eig in cache["eigs"]]
                )
                n_null = total - n_pos
                log_det_full = log_det_pos + n_null * rho[-1]
                grad_full = jnp.stack([*factor_grads, n_null.astype(rho.dtype)])
                if id_fn is _drop_last_col:
                    eigs = cache["eigs"]
                    w = cache["u_kron_last_sq"]  # shape = factor_shape
                    # Kron-sum eigenvalues, reshaped to factor dims.
                    combined = reduce(
                        jnp.add.outer,
                        [lam * eig for lam, eig in zip(lams[:-1], eigs)],
                    )  # shape = factor_shape
                    pos = combined > 0
                    # d_lam: lam_null on null modes, combined on positive modes.
                    d_lam = jnp.where(pos, jnp.where(pos, combined, 1.0), lams[-1])
                    D = jnp.sum(w / d_lam)  # = (A^-1)[-1, -1], strictly positive
                    c = jnp.log(D)
                    # Per-factor grad: -sum_{pos} w * lam_j * eig_j / d_lam^2 / D
                    ndim = len(eigs)
                    d_sq_inv = 1.0 / (d_lam**2)
                    factor_grad_corrs = []
                    for j, (lam, eig) in enumerate(zip(lams[:-1], eigs)):
                        shape = [1] * ndim
                        shape[j] = -1
                        term = jnp.where(
                            pos, w * lam * eig.reshape(shape) * d_sq_inv, 0.0
                        )
                        factor_grad_corrs.append(-jnp.sum(term) / D)
                    # Null grad: -beta / (lam_null * D) where beta = sum_{null} w
                    beta = jnp.sum(jnp.where(pos, 0.0, w))
                    null_grad_corr = -beta / (lams[-1] * D)
                    return (
                        log_det_full + c,
                        grad_full
                        + jnp.stack([*factor_grad_corrs, null_grad_corr]).astype(
                            rho.dtype
                        ),
                    )
                elif id_fn is identity:
                    return log_det_full, grad_full
                else:
                    raise NotImplementedError(
                        f"KRONECKER_WITH_NULL log_det not implemented for id_fn={id_fn}"
                    )

            case SqrtMethod.GENERAL:
                if id_fn is _drop_last_col:
                    full_rank_S = cache["full_rank_S_r"]
                elif id_fn is identity:
                    full_rank_S = cache["full_rank_S"]
                else:
                    raise NotImplementedError(
                        f"GENERAL log_det not implemented for id_fn={id_fn}"
                    )
                S_i_out = transform_slam(full_rank_S, rho)
                return log_det_and_grad_slam(rho, S_i_out)

            case _:
                raise NotImplementedError(
                    f"_log_det_and_grad not implemented for {method}."
                )

    def compute_log_det_and_grad(self, rhos: list[jnp.ndarray]) -> tuple[list, list]:
        _, _compute_log_det_and_grad = self.build()
        return _compute_log_det_and_grad(rhos)

    def compute_sqrt(self, rhos: list[jnp.ndarray]) -> jnp.ndarray:
        _compute_sqrt, _ = self.build()
        return _compute_sqrt(rhos)

    def build(self) -> tuple:
        """
        Snapshot the handler into two pure callables.

        Returns
        -------
        compute_sqrt : callable
            ``compute_sqrt(rhos) -> (total_pen_rows, n_params)`` block-diagonal matrix.
        compute_log_det_and_grad : callable
            ``compute_log_det_and_grad(rhos) -> (log_dets, grads)``
            where
                log_dets : list[scalar]
                    log|S_lam_i|_+ for each registered penalty i.
                grads : list[jnp.ndarray]
                    d log|S_lam_i|_+ / d rho for each i; shape matches rhos[i].
        """
        # create snapshots of these at the time of build
        penalties = self._penalties
        n = len(penalties)
        _sqrt_fn = self._sqrt
        _ld_fn = self._log_det_and_grad

        # build stack groups
        stack_groups = defaultdict(list)
        for i, p in enumerate(penalties):
            stack_groups[p._group_key].append(i)

        group_data = []
        for member_indices in stack_groups.values():
            sample_member = penalties[member_indices[0]]
            method, id_fn = sample_member.method, sample_member.id_fn
            if len(member_indices) == 1:
                i = member_indices[0]
                group_data.append(
                    {
                        "singleton": True,
                        "method": method,
                        "id_fn": id_fn,
                        "idx": i,
                        "cache": penalties[i].cache,
                    }
                )
            else:
                stacked = jax.tree_util.tree_map(
                    lambda *a: jnp.stack(a),
                    *[penalties[i].cache for i in member_indices],
                )
                group_data.append(
                    {
                        "singleton": False,
                        "method": method,
                        "id_fn": id_fn,
                        "member_indices": member_indices,
                        "stacked_cache": stacked,
                        "vmapped_sqrt": jax.vmap(
                            lambda cache, lams, _m=method, _f=id_fn: _sqrt_fn(
                                _m, cache, lams, _f
                            )
                        ),
                        "vmapped_ld": jax.vmap(
                            lambda cache, rho, _m=method, _f=id_fn: _ld_fn(
                                _m, cache, rho, _f
                            )
                        ),
                    }
                )

        def compute_sqrt(rhos):
            lams = [jnp.exp(r) for r in rhos]
            out = [None] * n
            for g in group_data:
                if g["singleton"]:
                    out[g["idx"]] = _sqrt_fn(
                        g["method"], g["cache"], lams[g["idx"]], g["id_fn"]
                    )
                else:
                    # all members share method, shape, and id_fn, so they can be batched with vmap
                    batched_lams = jnp.stack([lams[i] for i in g["member_indices"]])
                    batched_out = g["vmapped_sqrt"](g["stacked_cache"], batched_lams)
                    for k, idx in enumerate(g["member_indices"]):
                        out[idx] = batched_out[k]
            return block_diag(*out)

        def compute_log_det_and_grad(rhos):
            log_dets = [None] * n
            grads = [None] * n
            for g in group_data:
                if g["singleton"]:
                    ld, gr = _ld_fn(g["method"], g["cache"], rhos[g["idx"]], g["id_fn"])
                    log_dets[g["idx"]] = ld
                    grads[g["idx"]] = gr
                else:
                    batched_rhos = jnp.stack([rhos[i] for i in g["member_indices"]])
                    batched_ld, batched_g = g["vmapped_ld"](
                        g["stacked_cache"], batched_rhos
                    )
                    for k, idx in enumerate(g["member_indices"]):
                        log_dets[idx] = batched_ld[k]
                        grads[idx] = batched_g[k]
            return log_dets, grads

        return compute_sqrt, compute_log_det_and_grad

    def __len__(self):
        return len(self._penalties)

    def __repr__(self) -> str:
        string = self.__class__.__name__ + "("
        if len(self._penalties) == 0:
            return string + ")"
        for pen in self._penalties:
            string += f"\n\tn_reg_strengths={pen.rho_len},"
        string = string[:-1] + "\n)"
        return string

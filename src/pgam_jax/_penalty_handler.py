import abc
from collections import defaultdict
from functools import reduce
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

from ._slam_compute import log_det_and_grad_slam, transform_slam, transform_slam_with_Q
from .penalty_utils import DROP_LAST_COL, IDENTITY


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


def _full_rank_precompute(S_in):
    """
    Frobenius-averaging + eigh + null-space drop. Returns (U_keep, full_rank_S).
    """
    frobs = jnp.sqrt(jnp.sum(S_in**2, axis=(1, 2), keepdims=True))
    norm_avg = jnp.sum(S_in / frobs, axis=0)
    ev, U = jnp.linalg.eigh(norm_avg)
    eps = float(jnp.finfo(float).eps ** 0.8)
    keep = ev > ev[-1] * eps  # eager boolean index, precompute only
    U_keep = U[:, keep]
    full_rank_S = U_keep.T[None] @ S_in @ U_keep[None]
    return U_keep, full_rank_S


class _AbstractPenalty(eqx.Module):
    """
    Abstract base. Subclasses own their precompute + sqrt + log_det_and_grad.
    """

    @abc.abstractmethod
    def sqrt(self, rho): ...

    @abc.abstractmethod
    def log_det_and_grad(self, rho): ...

    @property
    @abc.abstractmethod
    def rho_len(self) -> int: ...

    @property
    def _group_key(self):
        leaves, treedef = jax.tree_util.tree_flatten(self)
        return (treedef, tuple(jnp.shape(leaf) for leaf in leaves))


class _SinglePenalty(_AbstractPenalty):
    id_fn: Callable = eqx.field(static=True)
    eig: jnp.ndarray
    U: jnp.ndarray
    rank: jnp.ndarray
    log_det_S: jnp.ndarray
    rank_r: Optional[jnp.ndarray] = None
    log_det_S_r: Optional[jnp.ndarray] = None

    @classmethod
    def from_S(cls, S, id_fn):
        eig, U, rank = _eigh_and_rank(S)
        log_det_S = _log_pseudo_det_from_eigvals(eig)
        rank_r = log_det_S_r = None
        if id_fn is DROP_LAST_COL:
            eig_r, _, rank_r = _eigh_and_rank(S[:-1, :-1])
            log_det_S_r = _log_pseudo_det_from_eigvals(eig_r)
        return cls(
            id_fn=id_fn,
            eig=eig,
            U=U,
            rank=rank,
            log_det_S=log_det_S,
            rank_r=rank_r,
            log_det_S_r=log_det_S_r,
        )

    @property
    def rho_len(self) -> int:
        return 1

    def sqrt(self, rho):
        sqrt_d = jnp.sqrt(jnp.exp(rho[0]) * self.eig)
        return sqrt_d[:, None] * self.id_fn(self.U.T)

    def log_det_and_grad(self, rho):
        if self.id_fn is DROP_LAST_COL:
            rank, log_det_S = self.rank_r, self.log_det_S_r
        elif self.id_fn is IDENTITY:
            rank, log_det_S = self.rank, self.log_det_S
        else:
            raise NotImplementedError(
                f"SINGLE log_det not implemented for id_fn={self.id_fn}"
            )
        return rank * rho[0] + log_det_S, jnp.full_like(rho, rank)


class _SingleWithNullPenalty(_AbstractPenalty):
    id_fn: Callable = eqx.field(static=True)
    eig: jnp.ndarray
    U: jnp.ndarray
    rank: jnp.ndarray
    rank_null: jnp.ndarray
    log_det_S: jnp.ndarray
    log_alpha: Optional[jnp.ndarray] = None
    log_beta: Optional[jnp.ndarray] = None

    @classmethod
    def from_S(cls, S, id_fn):
        eig, U, rank = _eigh_and_rank(S)
        log_det_S = _log_pseudo_det_from_eigvals(eig)
        rank_null = S.shape[0] - rank
        log_alpha = log_beta = None
        if id_fn is DROP_LAST_COL:
            last_sq = U[-1, :] ** 2
            pos = eig > 0
            alpha = jnp.sum(jnp.where(pos, last_sq / jnp.where(pos, eig, 1.0), 0.0))
            beta = jnp.sum(jnp.where(pos, 0.0, last_sq))
            log_alpha = _safe_log(alpha)
            log_beta = _safe_log(beta)
        return cls(
            id_fn=id_fn,
            eig=eig,
            U=U,
            rank=rank,
            rank_null=rank_null,
            log_det_S=log_det_S,
            log_alpha=log_alpha,
            log_beta=log_beta,
        )

    @property
    def rho_len(self) -> int:
        return 2

    def sqrt(self, rho):
        lams = jnp.exp(rho)
        sqrt_d = jnp.where(
            self.eig > 0, jnp.sqrt(lams[0] * self.eig), jnp.sqrt(lams[1])
        )
        return sqrt_d[:, None] * self.id_fn(self.U.T)

    def log_det_and_grad(self, rho):
        log_det_full = self.rank * rho[0] + self.log_det_S + self.rank_null * rho[1]
        grad_full = jnp.stack([self.rank, self.rank_null]).astype(rho.dtype)
        if self.id_fn is DROP_LAST_COL:
            log_det_corr, grad_corr = self._drop_last_col_correction(rho)
            return log_det_full + log_det_corr, grad_full + grad_corr
        elif self.id_fn is IDENTITY:
            return log_det_full, grad_full
        else:
            raise NotImplementedError(
                f"SINGLE_WITH_NULL log_det not implemented for id_fn={self.id_fn}"
            )

    def _drop_last_col_correction(self, rho):
        """
        Schur correction terms for DROP_LAST_COL on SINGLE_WITH_NULL.

        Returns ``(log_det_correction, grad_correction)`` to be added to the
        uncorrected full-space log-det and gradient.
        """
        log_terms = jnp.stack([self.log_alpha - rho[0], self.log_beta - rho[1]])
        log_det_corr = jax.scipy.special.logsumexp(log_terms)
        weights = jnp.exp(log_terms - log_det_corr)
        return log_det_corr, -weights


class _AbstractKronecker(_AbstractPenalty):
    """
    Abstract intermediate. Declares the shared fields as eqx.AbstractVar
    so the type-checker enforces presence in concrete leaves and the shared
    from_factors classmethod (inherited by both concrete leaves).
    """

    id_fn: eqx.AbstractVar[Callable]
    eigs: eqx.AbstractVar[tuple]
    kron_U: eqx.AbstractVar[jnp.ndarray]
    ranks: eqx.AbstractVar[tuple]
    log_det_S: eqx.AbstractVar[tuple]
    u_kron_last_sq: eqx.AbstractVar[jnp.ndarray]

    @classmethod
    def from_factors(cls, factor_list, id_fn):
        results = [_eigh_and_rank(S) for S in factor_list]
        eigs = tuple(eig for eig, _, _ in results)
        Us = [U for _, U, _ in results]
        ranks = tuple(rank for _, _, rank in results)
        log_det_S = tuple(_log_pseudo_det_from_eigvals(eig) for eig in eigs)
        kron_U = reduce(jnp.kron, Us)
        factor_shape = tuple(eig.shape[0] for eig in eigs)
        u_kron_last_sq = (kron_U[-1, :] ** 2).reshape(factor_shape)
        return cls(
            id_fn=id_fn,
            eigs=eigs,
            kron_U=kron_U,
            ranks=ranks,
            log_det_S=log_det_S,
            u_kron_last_sq=u_kron_last_sq,
        )


class _KroneckerPenalty(_AbstractKronecker):
    id_fn: Callable = eqx.field(static=True)
    eigs: tuple
    kron_U: jnp.ndarray
    ranks: tuple
    log_det_S: tuple
    u_kron_last_sq: jnp.ndarray

    @property
    def rho_len(self) -> int:
        return len(self.eigs)

    def sqrt(self, rho):
        lams = jnp.exp(rho)
        combined = reduce(
            jnp.add.outer, [lam * eig for lam, eig in zip(lams, self.eigs)]
        ).ravel()
        return jnp.sqrt(combined)[:, None] * self.id_fn(self.kron_U.T)

    def log_det_and_grad(self, rho):
        if self.id_fn is DROP_LAST_COL:
            raise NotImplementedError(
                "KRONECKER log_det not implemented under DROP_LAST_COL; "
                "use add_kron(..., penalize_null_space=True) for KRONECKER_WITH_NULL."
            )
        elif self.id_fn is IDENTITY:
            lams = jnp.exp(rho)
            log_det, factor_grads, _ = _kron_log_det_factor_grads(lams, self.eigs)
            return log_det, jnp.stack(factor_grads)
        else:
            raise NotImplementedError(
                f"KRONECKER log_det not implemented for id_fn={self.id_fn}"
            )


class _KroneckerWithNullPenalty(_AbstractKronecker):
    id_fn: Callable = eqx.field(static=True)
    eigs: tuple
    kron_U: jnp.ndarray
    ranks: tuple
    log_det_S: tuple
    u_kron_last_sq: jnp.ndarray

    @property
    def rho_len(self) -> int:
        return len(self.eigs) + 1

    def sqrt(self, rho):
        lams = jnp.exp(rho)
        combined = reduce(
            jnp.add.outer, [lam * eig for lam, eig in zip(lams[:-1], self.eigs)]
        ).ravel()
        sqrt_d = jnp.where(combined > 0, jnp.sqrt(combined), jnp.sqrt(lams[-1]))
        return sqrt_d[:, None] * self.id_fn(self.kron_U.T)

    def log_det_and_grad(self, rho):
        lams = jnp.exp(rho)
        log_det_pos, factor_grads, n_pos = _kron_log_det_factor_grads(
            lams[:-1], self.eigs
        )
        total = reduce(lambda a, b: a * b, [eig.shape[0] for eig in self.eigs])
        n_null = total - n_pos
        log_det_full = log_det_pos + n_null * rho[-1]
        grad_full = jnp.stack([*factor_grads, n_null.astype(rho.dtype)])
        if self.id_fn is DROP_LAST_COL:
            log_det_corr, grad_corr = self._drop_last_col_correction(lams)
            return log_det_full + log_det_corr, grad_full + grad_corr
        elif self.id_fn is IDENTITY:
            return log_det_full, grad_full
        else:
            raise NotImplementedError(
                f"KRONECKER_WITH_NULL log_det not implemented for id_fn={self.id_fn}"
            )

    def _drop_last_col_correction(self, lams):
        """
        Schur correction terms for DROP_LAST_COL on KRONECKER_WITH_NULL.

        Returns ``(log_det_correction, grad_correction)`` to be added to the
        uncorrected full-space log-det and gradient.
        """
        eigs = self.eigs
        w = self.u_kron_last_sq
        combined = reduce(
            jnp.add.outer, [lam * eig for lam, eig in zip(lams[:-1], eigs)]
        )
        pos = combined > 0
        d_lam = jnp.where(pos, jnp.where(pos, combined, 1.0), lams[-1])
        D = jnp.sum(w / d_lam)
        log_det_corr = jnp.log(D)
        ndim = len(eigs)
        d_sq_inv = 1.0 / (d_lam**2)
        factor_grad_corrs = []
        for j, (lam, eig) in enumerate(zip(lams[:-1], eigs)):
            shape = [1] * ndim
            shape[j] = -1
            term = jnp.where(pos, w * lam * eig.reshape(shape) * d_sq_inv, 0.0)
            factor_grad_corrs.append(-jnp.sum(term) / D)
        beta = jnp.sum(jnp.where(pos, 0.0, w))
        null_grad_corr = -beta / (lams[-1] * D)
        grad_corr = jnp.stack([*factor_grad_corrs, null_grad_corr]).astype(lams.dtype)
        return log_det_corr, grad_corr


class _GeneralPenalty(_AbstractPenalty):
    id_fn: Callable = eqx.field(static=True)
    U_keep: jnp.ndarray  # (q, r)
    full_rank_S: jnp.ndarray  # (k, r, r)
    full_rank_S_r: Optional[jnp.ndarray] = None  # (k, r', r') when id_fn=DROP_LAST_COL

    @classmethod
    def from_S(cls, S, id_fn):
        U_keep, full_rank_S = _full_rank_precompute(S)
        full_rank_S_r = None
        if id_fn is DROP_LAST_COL:
            _, full_rank_S_r = _full_rank_precompute(S[:, :-1, :-1])
        return cls(
            id_fn=id_fn,
            U_keep=U_keep,
            full_rank_S=full_rank_S,
            full_rank_S_r=full_rank_S_r,
        )

    @property
    def rho_len(self) -> int:
        return self.full_rank_S.shape[0]

    def sqrt(self, rho):
        lams = jnp.exp(rho)
        S_i_out, Q_s = transform_slam_with_Q(self.full_rank_S, lams)
        S_full = jnp.einsum("i,ijk->jk", lams, S_i_out)
        S_full = 0.5 * (S_full + S_full.T)
        ev, U = jnp.linalg.eigh(S_full)
        safe_ev = jnp.where(ev > 0, ev, 1.0)
        sqrt_d = jnp.where(ev > 0, jnp.sqrt(safe_ev), 0.0)
        B_rot = sqrt_d[:, None] * U.T
        return (B_rot @ Q_s.T) @ self.id_fn(self.U_keep.T)

    def log_det_and_grad(self, rho):
        if self.id_fn is DROP_LAST_COL:
            full_rank_S = self.full_rank_S_r
        elif self.id_fn is IDENTITY:
            full_rank_S = self.full_rank_S
        else:
            raise NotImplementedError(
                f"GENERAL log_det not implemented for id_fn={self.id_fn}"
            )
        S_i_out = transform_slam(full_rank_S, rho)
        return log_det_and_grad_slam(rho, S_i_out)


def _check_rho_lengths(rhos, penalties):
    """
    Fail loudly when ``rhos`` does not match the registered penalties.

    Without this, ``sqrt`` silently drops trailing entries via ``zip`` and
    ``log_det_and_grad`` returns a wrong-shaped gradient.
    """
    n = len(penalties)
    if len(rhos) != n:
        raise ValueError(
            f"PenaltyHandler has {n} penalties but got {len(rhos)} "
            "regularization-strength vectors."
        )
    for i, (rho, pen) in enumerate(zip(rhos, penalties)):
        shape = jnp.shape(rho)
        if len(shape) != 1 or shape[0] != pen.rho_len:
            raise ValueError(
                f"Penalty {i} ({type(pen).__name__}) expects a rho "
                f"vector of length {pen.rho_len}, got shape {tuple(shape)}."
            )


class PenaltyHandler:
    def __init__(self):
        self._penalties: list[_AbstractPenalty] = []

    def add(self, S_tensor, penalize_null_space: bool, identifiability_fn: Callable):
        """
        Register a single (possibly stacked) penalty matrix.

        Parameters
        ----------
        S_tensor :
            Shape (k, q, q) or (q, q) stacked penalty matrices. The penalty
            is ``sum_j lambda_j * S_tensor[j]``.
        penalize_null_space :
            Only relevant when k == 1. For k > 1 the general method removes
            the null space implicitly via the full-rank precompute projection.
        identifiability_fn :
            Must be a module-level callable (not a lambda) so that object
            identity is stable and can be used as a group key for vmap batching.
            Use ``penalty_utils.IDENTITY`` (no-op) or
            ``penalty_utils.DROP_LAST_COL``, or define your own at module level.
        """
        S = jnp.asarray(S_tensor)
        if S.ndim == 2 or S.shape[0] == 1:
            S2d = S if S.ndim == 2 else S[0]
            _, _, rank = _eigh_and_rank(S2d)
            has_null = penalize_null_space and bool(rank < S2d.shape[0])
            cls = _SingleWithNullPenalty if has_null else _SinglePenalty
            p = cls.from_S(S2d, identifiability_fn)
        else:
            p = _GeneralPenalty.from_S(S, identifiability_fn)
        self._penalties.append(p)

    def add_kron(
        self, factor_list, penalize_null_space: bool, identifiability_fn: Callable
    ):
        """
        Register a Kronecker-sum penalty from its factor matrices.

        Parameters
        ----------
        factor_list :
            List of (q_i, q_i) factor penalty matrices. Exploits the
            Kronecker-sum structure to compute the sqrt via per-factor
            eigendecompositions, avoiding the full (prod_q, prod_q) eigh.
        penalize_null_space :
            If True and every factor individually has a null space, the null
            space of the Kronecker sum is penalized via a separate lambda.
        identifiability_fn :
            See ``add``.
        """
        has_null = penalize_null_space and all(
            bool(_eigh_and_rank(S)[2] < S.shape[0]) for S in factor_list
        )
        cls = _KroneckerWithNullPenalty if has_null else _KroneckerPenalty
        self._penalties.append(cls.from_factors(factor_list, identifiability_fn))

    def compute_log_det_and_grad(self, rhos):
        _, fn = self.build()
        return fn(rhos)

    def compute_sqrt(self, rhos):
        fn, _ = self.build()
        return fn(rhos)

    def build(self):
        """
        Snapshot the handler into two pure callables.
        """
        n = len(self._penalties)
        penalties = self._penalties

        stack_groups = defaultdict(list)
        for i, p in enumerate(penalties):
            stack_groups[p._group_key].append(i)

        groups = []  # each: (tuple(member_indices), stacked_pytree, members)
        for member_indices in stack_groups.values():
            members = [penalties[i] for i in member_indices]
            stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *members)
            groups.append((tuple(member_indices), stacked, members))

        vmapped_sqrt = jax.vmap(lambda m, rho: m.sqrt(rho))
        vmapped_ld = jax.vmap(lambda m, rho: m.log_det_and_grad(rho))

        def compute_sqrt(rhos):
            _check_rho_lengths(rhos, penalties)
            out = [None] * n
            for member_indices, stacked, members in groups:
                # Singleton groups (the common case) call the penalty directly;
                # vmap-of-1 measurably regresses the singleton-heavy path.
                if len(member_indices) == 1:
                    idx = member_indices[0]
                    out[idx] = members[0].sqrt(rhos[idx])
                else:
                    batched_rhos = jnp.stack([rhos[i] for i in member_indices])
                    B_batch = vmapped_sqrt(stacked, batched_rhos)
                    for k, idx in enumerate(member_indices):
                        out[idx] = B_batch[k]
            return block_diag(*out)

        def compute_log_det_and_grad(rhos):
            _check_rho_lengths(rhos, penalties)
            log_dets = [None] * n
            grads = [None] * n
            for member_indices, stacked, members in groups:
                # Singleton fast-path: see compute_sqrt.
                if len(member_indices) == 1:
                    idx = member_indices[0]
                    log_dets[idx], grads[idx] = members[0].log_det_and_grad(rhos[idx])
                else:
                    batched_rhos = jnp.stack([rhos[i] for i in member_indices])
                    ld_batch, gr_batch = vmapped_ld(stacked, batched_rhos)
                    for k, idx in enumerate(member_indices):
                        log_dets[idx] = ld_batch[k]
                        grads[idx] = gr_batch[k]
            return log_dets, grads

        return compute_sqrt, compute_log_det_and_grad

    def __len__(self):
        return len(self._penalties)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        if not self._penalties:
            return s + ")"
        for p in self._penalties:
            s += f"\n\tn_reg_strengths={p.rho_len},"
        return s[:-1] + "\n)"

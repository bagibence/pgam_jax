"""
_slam_compute.py
-----------------
JAX port of log|Σᵢ λᵢ Sᵢ|_+ and its gradient / Hessian w.r.t. log-λ.

Reference: Demo_PGAM/PGAM/src/PGAM/deriv_det_Slam.py

Key design decisions vs. the numpy reference
---------------------------------------------
1. transform_Slam uses lax.scan with a fixed iteration budget of q steps.
   Each step after convergence (gamma_mask empty) is an exact no-op.

2. The Cholesky + try/except fallback in the reference is replaced by a
   single unconditional eigh path (_eigh_logdet_and_inv).  try/except cannot
   be traced by JAX JIT; eigh handles the singular/semi-definite case
   uniformly via jnp.where masking of non-positive eigenvalues.

3. Gradients and Hessian are computed analytically (not via autodiff).
   lax.scan through eigh does not reliably support reverse-mode AD, and the
   analytical expressions are cheap once S_lam^{-1} is available.

Public API
----------
    S_i_out               = transform_Slam(S_tensor, rho)
    log_det               = logDet_Slam(rho, S_i_out)
    grad   : (M,)         = grad_logDet_Slam(rho, S_i_out)
    hessian: (M, M)       = hes_logDet_Slam(rho, S_i_out)
"""

import warnings
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


def _warn_if_not_f64(fname: str) -> None:
    """Warn once if JAX is not running in float64 mode."""
    if jnp.finfo(float).dtype != np.dtype("float64"):
        warnings.warn(
            f"{fname}: JAX is operating in {jnp.finfo(float).dtype} precision. "
            "The Wood (2011) Appendix B algorithm uses machine-epsilon thresholds "
            "for the dominant/subdominant split and rank determination that are "
            "calibrated for float64 and will produce inaccurate results in lower "
            "precision.  Enable float64 with "
            "jax.config.update('jax_enable_x64', True).",
            UserWarning,
            stacklevel=3,
        )


# ===========================================================================
# Wood (2011) Appendix B — block-diagonal transform via lax.scan
# ===========================================================================

def _make_scan_body(lams, q):
    """
    Build the lax.scan body function for the Appendix B transform.

    lams is captured in the closure as a (potentially traced) JAX array.
    q is a static Python int derived from S_tensor.shape[1].
    M is derived from lams.shape[0] inside the body — also a static int.

    Epsilon constants are derived from jnp.finfo(float).eps at call time so
    they reflect whatever JAX precision mode is active (float32 or float64).

    State tuple: (S_bar, S_i_out, gamma_mask, K)
      S_bar      : (M, q, q)  working matrices; active block is [K:, K:]
      S_i_out    : (M, q, q)  individually transformed output matrices
      gamma_mask : (M,) bool  True for indices still in the active set gamma
      K          : ()  int32  accumulated block offset
    """
    M = lams.shape[0]   # static int at trace time

    # Derive thresholds from current JAX float precision at call time.
    _eps       = jnp.finfo(float).eps
    _eps_split = float(_eps ** (1.0 / 3.0))
    _eps_rank  = float(_eps ** 0.8)
    _big       = float(jnp.finfo(float).max) * 1e-10

    I_q   = jnp.eye(q)                            # dtype follows current default
    outer = jnp.arange(q, dtype=jnp.int32)         # [0, 1, ..., q-1]; static

    def body(state, _):
        S_bar, S_i_out, gamma_mask, K = state

        # Active-block mask: only rows/cols >= K are live this iteration.
        act2d = (outer[:, None] >= K) & (outer[None, :] >= K)   # (q, q)

        # Step 1: Frobenius norms restricted to active block [K:, K:].
        # S_bar has garbage in [:K, :] from previous step-8 embeds so we
        # must zero that region before computing the norm.
        S_bar_act = S_bar * act2d[None, :, :]                    # (M, q, q)
        frobs     = jnp.sqrt((S_bar_act ** 2).sum(axis=(1, 2))) # (M,)
        omegas    = frobs * lams * gamma_mask                    # 0 for i ∉ γ
        max_omega = jnp.where(gamma_mask, omegas, 0.0).max()    # masked max

        # Step 2: split into dominant alpha and subdominant gamma_prime.
        alpha       = (omegas >= _eps_split * max_omega) & gamma_mask
        gamma_prime = (omegas <  _eps_split * max_omega) & gamma_mask

        # Step 3: rank r of Frobenius-normalised dominant sum in active block.
        safe_norms = jnp.where(alpha & (frobs > 0), frobs, 1.0)
        weights    = jnp.where(alpha, 1.0 / safe_norms, 0.0)
        norm_S_act = jnp.einsum("ijk,i->jk", S_bar_act, weights)
        ev3        = jnp.linalg.eigvalsh(norm_S_act)            # ascending
        r          = jnp.sum(ev3 > ev3[-1] * _eps_rank, dtype=jnp.int32)

        # Step 4 (r == Q termination) encoded as a flag.
        # When True, all Step 5-8 results are discarded via jnp.where and
        # gamma_mask is zeroed so the scan body becomes a no-op hereafter.
        terminate = (r == q - K)

        # Step 5: eigendecomposition of dominant weighted sum.
        # -BIG diagonal trick: inactive dims (< K) sort to the end of U
        # after reversing, keeping the eigh input/output shape fixed at (q, q).
        S_lam_alpha = jnp.einsum("ijk,i->jk", S_bar_act, jnp.where(alpha, lams, 0.0))
        S_eigh      = S_lam_alpha - _big * jnp.diag((outer < K).astype(lams.dtype))
        _, U        = jnp.linalg.eigh(S_eigh)                   # ascending
        U           = U[:, ::-1]          # descending; inactive cols go last
        # U[:, :q-K] = active eigenvectors (non-zero only in rows [K:])
        # U[:, q-K:] = standard basis e_0..e_{K-1}

        # V = block_diag(I_K, U_active) = T_gp from reference
        V = jnp.where(act2d, U, I_q)                            # (q, q)

        # Step 7: transform S_i_out.
        # T_alpha = V with cols [K+r:] zeroed  (= T_al in reference)
        # T_gamma = V                           (= T_gp in reference)
        keep         = (outer < K + r)
        T_alpha      = jnp.where(keep[None, :], V, 0.0)
        S_from_alpha = T_alpha.T[None] @ S_i_out @ T_alpha[None]
        S_from_gamma = V.T[None]       @ S_i_out @ V[None]
        S_i_out_new  = jnp.where(alpha[:, None, None],       S_from_alpha,
                       jnp.where(gamma_prime[:, None, None], S_from_gamma, S_i_out))

        # Step 8: project S_bar for gamma' onto null space of dominant terms.
        # V_n: replace dominant cols [K:K+r] of V with standard basis vectors.
        # V_n.T @ S_bar @ V_n places  U_n.T S_bar_active U_n  in [K+r:, K+r:]
        # without changing the (q, q) shape.
        dom_cols  = (outer >= K) & (outer < K + r)
        V_n       = jnp.where(dom_cols[None, :], I_q, V)
        S_bar_gp  = V_n.T[None] @ S_bar @ V_n[None]
        S_bar_new = jnp.where(gamma_prime[:, None, None], S_bar_gp, S_bar)

        # Step 9: conditional updates.
        # When terminate=True every jnp.where selects the unchanged value,
        # so S_bar, S_i_out stay the same and K does not advance.
        # Setting gamma_mask = zeros stops subsequent iterations from doing work.
        S_i_out_out = jnp.where(terminate, S_i_out,    S_i_out_new)
        S_bar_out   = jnp.where(terminate, S_bar,      S_bar_new)
        K_out       = jnp.where(terminate, K,           K + r)
        gm_out      = jnp.where(terminate, jnp.zeros(M, dtype=bool), gamma_prime)

        return (S_bar_out, S_i_out_out, gm_out, K_out), None

    return body


def transform_Slam(S_tensor, rho):
    """
    Stable block-diagonal transform of Σᵢ λᵢ Sᵢ (Wood 2011, Appendix B).

    Runs lax.scan for exactly q iterations.  Iterations after convergence
    (gamma_mask empty or r == Q) are exact no-ops, so the budget is safe.

    Parameters
    ----------
    S_tensor : (M, q, q) JAX array — PSD penalty matrices
    rho      : (M,)      JAX array — log-smoothing parameters (lams = exp(rho))

    Returns
    -------
    S_i_out : (M, q, q) — individually transformed matrices.
        Each S_i_out[i] has support concentrated in a single diagonal block,
        so Σᵢ λᵢ S_i_out[i] has no cross-scale cancellation.
        Pass S_i_out to logDet_Slam / grad_logDet_Slam / hes_logDet_Slam.
    """
    _warn_if_not_f64("transform_Slam")
    lams = jnp.exp(rho)
    M, q = S_tensor.shape[0], S_tensor.shape[1]   # static ints at trace time

    body = _make_scan_body(lams, q)
    init = (
        S_tensor,
        S_tensor,
        jnp.ones(M, dtype=bool),
        jnp.zeros((), dtype=jnp.int32),
    )
    (_, S_i_out, _, _), _ = lax.scan(body, init, None, length=q)
    return S_i_out


# ===========================================================================
# Shared helper: log|S_lam|_+ and S_lam^{-1} via eigh
# ===========================================================================

def _eigh_logdet_and_inv(S_i_out, lams):
    """
    Compute log|S_lam|_+ and the Moore-Penrose pseudo-inverse of S_lam.

    Uses eigh unconditionally — no try/except needed.
    Non-positive eigenvalues (null space of S_lam) are masked out with
    jnp.where so both log_det and Sinv are numerically well-defined.

    Parameters
    ----------
    S_i_out : (M, q, q) transformed penalty matrices
    lams    : (M,)       smoothing parameters

    Returns
    -------
    log_det : scalar   — log|S_lam|_+
    Sinv    : (q, q)   — S_lam^{-1} (pseudo-inverse in the range space)
    """
    Slam = jnp.einsum("ijk,i->jk", S_i_out, lams)
    Slam = 0.5 * (Slam + Slam.T)                    # enforce symmetry

    ev, U   = jnp.linalg.eigh(Slam)                # ev ascending, U columns = eigenvectors
    pos     = ev > jnp.finfo(float).eps            # mask: positive eigenvalues only
    ev_safe = jnp.where(pos, ev, 1.0)              # avoid log(0) / div-by-zero

    log_det = jnp.sum(jnp.where(pos, jnp.log(ev_safe), 0.0))
    Sinv    = (U * jnp.where(pos, 1.0 / ev_safe, 0.0)[None, :]) @ U.T

    return log_det, Sinv


# ===========================================================================
# Log-det, gradient, Hessian
# ===========================================================================

def logDet_Slam(rho, S_i_out):
    """
    log|Σᵢ exp(ρᵢ) Sᵢ_out|_+   (scalar).

    Parameters
    ----------
    rho     : (M,) log-smoothing parameters
    S_i_out : (M, q, q) output of transform_Slam

    Returns
    -------
    log_det : scalar
    """
    _warn_if_not_f64("logDet_Slam")
    lams = jnp.exp(rho)
    log_det, _ = _eigh_logdet_and_inv(S_i_out, lams)
    return log_det


def grad_logDet_Slam(rho, S_i_out):
    """
    ∂log|S_lam|/∂ρⱼ = λⱼ tr(S_lam^{-1} Sⱼ_out)    (M,) vector.

    Derivation: d/dρⱼ log|S_lam| = tr(S_lam^{-1} dS_lam/dρⱼ)
                                  = tr(S_lam^{-1} λⱼ Sⱼ_out)
    """
    _warn_if_not_f64("grad_logDet_Slam")
    lams       = jnp.exp(rho)
    _, Sinv    = _eigh_logdet_and_inv(S_i_out, lams)

    # tr(Sinv @ S_i_out[j]) = Σ_kl Sinv_kl * S_i_out[j, l, k]
    grad = lams * jnp.einsum("kl,jlk->j", Sinv, S_i_out)
    return grad


def hes_logDet_Slam(rho, S_i_out):
    """
    ∂²log|S_lam|/∂ρᵢ∂ρⱼ   (M, M) matrix.

    H[i,j] = -λᵢ λⱼ tr(S_lam^{-1} Sᵢ_out S_lam^{-1} Sⱼ_out)
             + δᵢⱼ λᵢ tr(S_lam^{-1} Sᵢ_out)

    The diagonal term equals the gradient component, so no separate
    trace computation is needed.
    """
    _warn_if_not_f64("hes_logDet_Slam")
    lams    = jnp.exp(rho)
    _, Sinv = _eigh_logdet_and_inv(S_i_out, lams)

    # Sinv_S[i] = Sinv @ S_i_out[i]   shape (M, q, q)
    Sinv_S = jnp.einsum("kl,ilj->ikj", Sinv, S_i_out)

    # H_off[i,j] = tr(Sinv_S[i] @ Sinv_S[j])
    #            = Σ_kl  Sinv_S[i, k, l] * Sinv_S[j, l, k]
    H_off = jnp.einsum("ikl,jlk->ij", Sinv_S, Sinv_S)

    # Diagonal term = gradient
    grad_diag = lams * jnp.einsum("kl,jlk->j", Sinv, S_i_out)

    return -jnp.outer(lams, lams) * H_off + jnp.diag(grad_diag)


# ===========================================================================
# Smoke-test (run as: python -m pgam_jax._slam_compute)
# ===========================================================================

if __name__ == "__main__":
    import numpy as _np

    jax.config.update("jax_enable_x64", True)

    def _run(name, fn):
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()

    # 1. Toy extreme-lambda log-det ground truth
    def test_logdet_ground_truth():
        for seed in range(5):
            rng  = _np.random.default_rng(seed)
            rand = rng.standard_normal((5, 5))
            _, U = _np.linalg.eigh(rand.T @ rand)
            S    = jnp.stack([
                jnp.array(U[:, :3] @ U[:, :3].T),
                jnp.array(U[:, 3:] @ U[:, 3:].T),
            ])
            rho  = jnp.array([-20.0, 15.0])
            true = 3 * (-20.0) + 2 * 15.0   # = -30

            Si   = jax.jit(transform_Slam)(S, rho)
            ld   = float(logDet_Slam(rho, Si))
            err  = abs(ld - true)
            assert err < 1e-8, f"seed={seed}: err={err:.2e}"

    # 2. Gradient finite-difference check
    def test_grad_fd():
        rng  = _np.random.default_rng(42)
        q, M = 8, 3
        A    = [rng.standard_normal((q, q)) for _ in range(M)]
        S    = jnp.stack([jnp.array(a.T @ a) for a in A])
        rho  = jnp.array(rng.uniform(-3, 3, M))
        h    = 1e-5

        Si   = jax.jit(transform_Slam)(S, rho)
        g    = _np.array(grad_logDet_Slam(rho, Si))

        for j in range(M):
            rho_p = rho.at[j].add(h)
            rho_m = rho.at[j].add(-h)
            Si_p  = jax.jit(transform_Slam)(S, rho_p)
            Si_m  = jax.jit(transform_Slam)(S, rho_m)
            fd    = (float(logDet_Slam(rho_p, Si_p)) - float(logDet_Slam(rho_m, Si_m))) / (2 * h)
            err   = abs(g[j] - fd)
            assert err < 1e-5, f"grad[{j}]: analytic={g[j]:.6f} fd={fd:.6f} err={err:.2e}"

    # 3. Hessian finite-difference check (gradient of gradient)
    def test_hess_fd():
        rng  = _np.random.default_rng(7)
        q, M = 6, 3
        A    = [rng.standard_normal((q, q)) for _ in range(M)]
        S    = jnp.stack([jnp.array(a.T @ a) for a in A])
        rho  = jnp.array(rng.uniform(-2, 2, M))
        h    = 1e-4

        Si   = jax.jit(transform_Slam)(S, rho)
        H    = _np.array(hes_logDet_Slam(rho, Si))

        for j in range(M):
            rho_p = rho.at[j].add(h)
            rho_m = rho.at[j].add(-h)
            Si_p  = jax.jit(transform_Slam)(S, rho_p)
            Si_m  = jax.jit(transform_Slam)(S, rho_m)
            fd_col = (_np.array(grad_logDet_Slam(rho_p, Si_p))
                    - _np.array(grad_logDet_Slam(rho_m, Si_m))) / (2 * h)
            err = _np.max(_np.abs(H[:, j] - fd_col))
            assert err < 1e-4, f"hess col {j}: max|err|={err:.2e}"

    # 4. JIT compiles without error
    def test_jit():
        rng = _np.random.default_rng(99)
        q, M = 5, 2
        S   = jnp.stack([jnp.array(rng.standard_normal((q, q))) for _ in range(M)])
        S   = jnp.einsum("mij,mkj->mik", S, S)   # make PSD
        rho = jnp.array([-20.0, 15.0])
        _   = jax.jit(transform_Slam)(S, rho)     # must not raise

    print("=" * 48)
    print("_slam_compute smoke-tests")
    print("=" * 48)
    for name, fn in [
        ("log-det ground truth (extreme rho)", test_logdet_ground_truth),
        ("gradient finite-difference check",   test_grad_fd),
        ("Hessian finite-difference check",    test_hess_fd),
        ("JIT compilation",                    test_jit),
    ]:
        _run(name, fn)
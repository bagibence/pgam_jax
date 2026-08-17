import jax.numpy as jnp

from pgam_jax._chisqsum import psum_chisq

from ._utils import to_zero_dim_jax_array


def weighted_chisq_pval(t_r, weights, df, kappa) -> float:
    """
    Compute the upper tail probability of a weighted sum of chi-squares.

    Provide ``kappa`` when the scale is estimated. In this case, the
    reference probability is
        Pr(sum_j w_j chi2_j > t_r * chi2_kappa / kappa)
    This is evaluated by moving the term on the right over to the left
        Pr(sum_j w_j chi2_j - t_r * chi2_kappa / kappa > 0)
    so adding a new term with w = -t_r / kappa and df=kappa
    and evaluating at zero.
    """
    t_r = to_zero_dim_jax_array(t_r)
    weights = jnp.atleast_1d(jnp.asarray(weights))
    df = jnp.atleast_1d(jnp.asarray(df))

    if weights.size == 0:
        raise ValueError("At least one chi-squared weight is required.")

    if df.size == 1 and weights.size > 1:
        df = jnp.broadcast_to(df, weights.size)
    elif df.size != weights.size:
        raise ValueError(
            f"`df` must have size 1 or match `weights`. "
            f"Got {df.size} and {weights.size}"
        )

    # if the scale is not estimated, we don't need the additional term
    if kappa is None:
        p = psum_chisq(
            t_r,
            weights=weights,
            df=df,
            lower_tail=False,
        )
    else:
        kappa = to_zero_dim_jax_array(kappa)
        if not bool(jnp.isfinite(kappa)) or float(kappa) <= 0:
            raise ValueError(
                f"`kappa` must be finite and positive. Got {float(kappa)}."
            )

        # add the new term's weight and degrees of freedom
        weights = jnp.concatenate([weights, jnp.atleast_1d(-t_r / kappa)])
        df = jnp.concatenate([df, jnp.atleast_1d(kappa)])

        p = psum_chisq(
            0.0,
            weights=weights,
            df=df,
            lower_tail=False,
        )

    return float(p)

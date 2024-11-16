##################################################
# weights & Effective Sample Size
##################################################
import jax.numpy as jnp


def compute_weights(
    log_weights: jnp.ndarray,   # log weights
    ):
    """
    Compute normalized weights from log weights.
    """
    log_w = log_weights - jnp.max(log_weights)
    w = jnp.exp(log_w)
    w = w / jnp.sum(w)
    return w


def ess_log_weight(
        log_weights: jnp.ndarray,   # log weights
        ):
    """
    Computes ESS = 1 / sum(w_i**2) where w_i are the normalized weights.
    Note that: 1 <= ESS <= S where S is the number of samples.
    """
    log_w = log_weights - jnp.max(log_weights)
    w = jnp.exp(log_w)
    w = w / jnp.sum(w)
    ess = 1. / jnp.sum(w**2)
    return ess


def ess_normalized_log_weight(
        log_weights: jnp.ndarray,   # log weights
        ):
    """
    Computes:
        ESS_normalized = ESS / S
    with ESS = 1 / sum(w_i**2) and w_i are the normalized weights.
    We have: 0 <= ESS_normalized <= 1
    """
    S = len(log_weights)
    ess = ess_log_weight(log_weights)
    ess_normalized = ess / S
    return ess_normalized


def target_ess_normalized(
        log_weights: jnp.ndarray,       # log weights
        ess_normalized_target: float,   # target ess_normalized
        tmax: float = 1.,               # max temperature
        tol: float = 10**-5,            # tolerance for the bissection
        ):
    """
    Use a bissection algorithm to find the temperature `t`
    such that ESS_normalized(t*log_weights) = ess_normalized_target.
    If ESS_normalized(tmax*log_weights) >= ess_normalized_target, return tmax.
    """
    tmin = 0.

    ess_tmax = ess_normalized_log_weight(tmax*log_weights)
    if ess_tmax >= ess_normalized_target:
        return tmax

    while tmax - tmin > tol:
        tmid = (tmin + tmax)/2.
        ess_new = ess_normalized_log_weight(tmid*log_weights)

        if ess_new < ess_normalized_target:
            tmax = tmid
        else:
            tmin = tmid
    return (tmin + tmax)/2.

##################################################
# weights & Effective Sample Size
##################################################
import jax.numpy as jnp
from jax.nn import softmax


def compute_weights(log_weights: jnp.ndarray):
    """
    Converts log-weights to normalized importance weights using the softmax function.

    Parameters:
        log_weights (jnp.ndarray): Array of unnormalized log-importance weights.

    Returns:
        jnp.ndarray: Normalized importance weights summing to 1.
    """
    return softmax(log_weights)


def ess_log_weight(log_weights: jnp.ndarray):
    """
    Computes the Effective Sample Size (ESS) from a set of log-importance weights.

    ESS estimates the number of effectively independent samples represented by 
    a weighted set of particles. It is computed as:
        ESS = 1 / sum(w_i^2)
    where w_i are the normalized importance weights obtained via softmax.

    Parameters:
        log_weights (jnp.ndarray): Array of log-importance weights.

    Returns:
        float: The effective sample size.
    """
    w = softmax(log_weights)
    return 1.0 / jnp.sum(w ** 2)


def ess_normalized_log_weight(log_weights: jnp.ndarray):
    """
    Computes the normalized Effective Sample Size (ESS) from log-importance weights.

    Normalized ESS is defined as:
        ESS_normalized = ESS / S
    where ESS = 1 / sum(w_i^2) and w_i are the normalized importance weights (via softmax),
    and S is the number of samples.

    This yields a value in [0, 1], where 1 indicates uniform weights (ideal),
    and values near 0 indicate high weight concentration (poor sample diversity).

    Parameters:
        log_weights (jnp.ndarray): Array of log-importance weights.

    Returns:
        float: Normalized effective sample size in the range [0, 1].
    """
    ess = ess_log_weight(log_weights)
    return ess / log_weights.size


# def target_ess_normalized(
#         log_weights: jnp.ndarray,       # log weights
#         ess_normalized_target: float,   # target ess_normalized
#         tmax: float = 1.,               # max temperature
#         tol: float = 10**-5,            # tolerance for the bissection
#         ):
#     """
#     Use a bissection algorithm to find the temperature `t`
#     such that ESS_normalized(t*log_weights) = ess_normalized_target.
#     If ESS_normalized(tmax*log_weights) >= ess_normalized_target, return tmax.
#     """
#     tmin = 0.

#     ess_tmax = ess_normalized_log_weight(tmax*log_weights)
#     if ess_tmax >= ess_normalized_target:
#         return tmax

#     while tmax - tmin > tol:
#         tmid = (tmin + tmax)/2.
#         ess_new = ess_normalized_log_weight(tmid*log_weights)

#         if ess_new < ess_normalized_target:
#             tmax = tmid
#         else:
#             tmin = tmid
#     return (tmin + tmax)/2.

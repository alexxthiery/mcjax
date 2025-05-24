##################################################
# weights & Effective Sample Size
##################################################
import jax
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


def effective_sample_size(*, log_weights: jnp.ndarray):
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


def effective_sample_size_normalized(*, log_weights: jnp.ndarray):
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
    ess = effective_sample_size(log_weights)
    return ess / log_weights.size


def select_next_temperature(
    log_p0: jnp.ndarray,
    log_p1: jnp.ndarray,
    log_weights: jnp.ndarray,
    lambda_prev: float,
    ess_threshold: float,
    tol: float = 1e-4,
    max_iter: int = 20,
) -> float:
    """
    Adaptive selection of the next inverse temperature using bisection to meet a desired
    effective sample size (ESS) threshold.

    The importance weights are computed using a tempered distribution:
        log_weight = log_weights + (lambda - lambda_prev) * (log_p1 - log_p0),
    where:
        - log_p0 is the log-density under the base distribution,
        - log_p1 is the log-density under the target distribution,
        - lambda is the next temperature between lambda_prev and 1.0.

    Args:
        log_p0: Log-probabilities under base distribution, shape (N,).
        log_p1: Log-probabilities under target distribution, shape (N,).
        log_weights: Current log-weights of the particles, shape (N,).
        lambda_prev: Current inverse temperature.
        ess_threshold: Desired minimum effective sample size.
        tol: Tolerance for convergence of the bisection method.
        max_iter: Maximum bisection iterations (not used here, placeholder).

    Returns:
        lambda_next: The next inverse temperature in [lambda_prev, 1.0].
    """
    delta_full = 1.0 - lambda_prev
    log_w_full = log_weights + delta_full * (log_p1 - log_p0)
    ess_full = effective_sample_size(log_w_full)

    def body_fn(val):
        lower, upper, _ = val
        midpoint = 0.5 * (lower + upper)
        delta_lambda = midpoint - lambda_prev
        log_w_new = log_weights + delta_lambda * (log_p1 - log_p0)
        ess = effective_sample_size(log_w_new)
        cond = ess < ess_threshold
        new_lower = jnp.where(cond, lower, midpoint)
        new_upper = jnp.where(cond, midpoint, upper)
        return (new_lower, new_upper, midpoint)

    def cond_fn(val):
        lower, upper, _ = val
        return (upper - lower) > tol

    def return_one(_):
        return 1.0

    def run_bisect(_):
        init_val = (lambda_prev, 1.0, lambda_prev)
        final_val = jax.lax.while_loop(cond_fn, body_fn, init_val)
        return final_val[2]

    return jax.lax.cond(ess_full >= ess_threshold,
                        return_one,
                        run_bisect,
                        operand=None)

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

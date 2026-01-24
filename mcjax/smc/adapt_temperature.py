
import jax
import jax.numpy as jnp
from mcjax.util.weights import effective_sample_size_normalized


def temp_adaptive(
    log_p0: jnp.ndarray,
    log_p1: jnp.ndarray,
    log_weights: jnp.ndarray,
    temp: float,
    ess_threshold: float,
    tol: float = 1e-4,
) -> float:
    """
    Adaptive selection of the next inverse temperature using bisection to meet a desired
    effective sample size (ESS) threshold.

    The importance weights are computed using a tempered distribution:
        log_weight = log_weights + (lambda - temp) * (log_p1 - log_p0),
    where:
        - log_p0 is the log-density under the base distribution,
        - log_p1 is the log-density under the target distribution,
        - lambda is the next temperature between temp and 1.0.

    Args:
        log_p0: Log-probabilities under base distribution, shape (N,).
        log_p1: Log-probabilities under target distribution, shape (N,).
        log_weights: Current log-weights of the particles, shape (N,).
        temp: Current inverse temperature.
        ess_threshold: Desired minimum effective sample size.
        tol: Tolerance for convergence of the bisection method.
        max_iter: Maximum bisection iterations (not used here, placeholder).

    Returns:
        temp_next: The next inverse temperature in [temp, 1.0].
    """
    delta_full = 1.0 - temp
    log_w_full = log_weights + delta_full * (log_p1 - log_p0)
    ess_full = effective_sample_size_normalized(log_w_full)

    def body_fn(val):
        lower, upper, _ = val
        midpoint = 0.5 * (lower + upper)
        temp_delta = midpoint - temp
        log_w_new = log_weights + temp_delta * (log_p1 - log_p0)
        ess = effective_sample_size_normalized(log_w_new)
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
        init_val = (temp, 1.0, temp)
        final_val = jax.lax.while_loop(cond_fn, body_fn, init_val)
        return final_val[2]

    return jax.lax.cond(ess_full >= ess_threshold,
                        return_one,
                        run_bisect,
                        operand=None)


# def temp_deterministic(
#     temp: float,
#     temp_increment: float = None,
# ) -> float:
#     """
#     Deterministically increments the temperature by a fixed step size, clipped to 1.0.

#     This function is compatible with the interface of `select_next_temperature`
#     for use in modular SMC designs.

#     Args:
#         temp: Current inverse temperature.
#         temp_increment: Fixed amount to increment the temperature.

#     Returns:
#         temp_next: The next inverse temperature in [temp, 1.0].
#     """
#     # temp_next = jnp.minimum(1.0, temp + temp_increment)
#     # return temp_next
#     has_delta = temp_increment is not None
#     delta_value = 0.0 if temp_increment is None else temp_increment

#     def valid_case(delta):
#         return jnp.minimum(1.0, temp + delta)

#     def invalid_case(_):
#         return jnp.array(-1.0)

#     return jax.lax.cond(has_delta, valid_case, invalid_case, delta_value)

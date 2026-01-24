"""
Step-size adaptation for MCMC kernels.

Uses short batched MCMC runs to tune step_size for target acceptance rate.
"""

from typing import Callable, Any

import jax
import jax.numpy as jnp
import jax.random as jr

from .core import run_mcmc_batch


def adapt_step_size(
    step_fn: Callable,
    init_state_fn: Callable,
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    xs: jnp.ndarray,
    params: Any,
    key: jax.Array,
    *,
    n_iters: int = 10,
    n_steps: int = 5,
    target_accept_low: float = 0.2,
    target_accept_high: float = 0.8,
    increase_factor: float = 1.2,
    decrease_factor: float = 0.8,
) -> Any:
    """
    Adapt step_size to achieve acceptance rate in target range.

    Runs short MCMC chains from xs, measures acceptance, adjusts step_size.
    Stops early if acceptance is within [target_accept_low, target_accept_high].

    Conventions assumed:
    - params.step_size exists and can be updated via params.replace(step_size=...)
    - stats.is_accept exists (returned by step_fn)

    Parameters
    ----------
    step_fn:
        Kernel step function: step_fn(key, log_prob, state, params) -> (state, stats)
    init_state_fn:
        Function to create state: init_state_fn(log_prob, x) -> state
    log_prob:
        Target log-density function.
    xs:
        Batch of positions to run short chains from, shape (batch_size, dim).
    params:
        Initial kernel parameters. Must have step_size field.
    key:
        JAX PRNG key.
    n_iters:
        Maximum adaptation iterations.
    n_steps:
        Number of MCMC steps per iteration.
    target_accept_low:
        Lower bound for acceptable acceptance rate.
    target_accept_high:
        Upper bound for acceptable acceptance rate.
    increase_factor:
        Multiply step_size by this when acceptance too high.
    decrease_factor:
        Multiply step_size by this when acceptance too low.

    Returns
    -------
    Updated params with adapted step_size.
    """
    xs = jnp.asarray(xs)

    def cond_fn(state):
        iters_remaining, _, _, acc, _ = state
        too_low = acc < target_accept_low
        too_high = acc > target_accept_high
        needs_adapt = jnp.logical_or(too_low, too_high)
        return jnp.logical_and(iters_remaining > 0, needs_adapt)

    def body_fn(state):
        iters_remaining, step_size, params_curr, _, key_curr = state

        key_curr, subkey = jr.split(key_curr)

        # Update params with current step_size
        params_for_run = params_curr.replace(step_size=step_size)

        # Run short chains
        output = run_mcmc_batch(
            step_fn=step_fn,
            init_state_fn=init_state_fn,
            log_prob=log_prob,
            xs_init=xs,
            params=params_for_run,
            key=subkey,
            n_samples=n_steps,
        )

        # Compute mean acceptance rate across all chains and steps
        acc = jnp.mean(output.stats.is_accept)

        # Adjust step_size
        step_size_new = jnp.where(
            acc < target_accept_low,
            step_size * decrease_factor,
            step_size,
        )
        step_size_new = jnp.where(
            acc > target_accept_high,
            step_size_new * increase_factor,
            step_size_new,
        )

        params_new = params_curr.replace(step_size=step_size_new)

        return (iters_remaining - 1, step_size_new, params_new, acc, key_curr)

    # Initialize: force at least one iteration by setting acc outside target
    init_state = (
        n_iters,
        params.step_size,
        params,
        jnp.array(0.0),  # initial acc (outside target range)
        key,
    )

    _, _, final_params, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

    return final_params

import jax
import jax.numpy as jnp
from typing import Callable, Any

from .core import MarkovKernel


def adapt_step_size(
    kernel: MarkovKernel,
    params: Any,
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    xs: jnp.ndarray,
    key: jax.Array,
    *,
    max_steps: int = 10,
    short_mcmc_steps: int = 5,
    acceptance_threshold_up: float = 0.8,
    acceptance_threshold_down: float = 0.2,
    # traits
    get_step_size: Callable[[Any], float],
    set_step_size: Callable[[Any, float], Any],
    reduce_acceptance: Callable[[Any], jnp.ndarray],
) -> Any:
    """
    Generic step-size adaptation using short batched MCMC runs.

    This function is kernel-agnostic. It assumes:
      - `kernel` is a MarkovKernel with `run_mcmc_batch`.
      - `params` is a PyTree of kernel parameters.
      - `get_step_size(params) -> float` returns the current step size.
      - `set_step_size(params, step_size)` returns updated params.
      - `reduce_acceptance(summary) -> scalar` produces a scalar
        acceptance diagnostic from the MCMC summary.

    The adaptation logic:
      - Repeatedly runs short MCMC batches from initial states `xs`.
      - Measures acceptance via `reduce_acceptance`.
      - Multiplies step size by `DECREASE_FACTOR` if acceptance is too low, `INCREASE_FACTOR` if too high.
      - Stops early if acceptance is within [down, up] or when max_steps is reached.
    """
    xs = jnp.asarray(xs)
    INCREASE_FACTOR = 1.2
    DECREASE_FACTOR = 0.8

    step_size0 = get_step_size(params)
    params0 = params

    def cond_fn(state):
        steps_remaining, _, _, acc, _ = state
        too_low = acc < acceptance_threshold_down
        too_high = acc > acceptance_threshold_up
        adapt = jnp.logical_or(too_low, too_high)
        return jnp.logical_and(steps_remaining > 0, adapt)

    def body_fn(state):
        steps_remaining, step_size, params_curr, _, key_curr = state

        # Update PRNG key per iteration
        key_curr, subkey = jax.random.split(key_curr)

        # Use the current step size in the parameters for this probe run
        params_for_run = set_step_size(params_curr, step_size)

        run = kernel.run_mcmc_batch(
            log_prob=log_prob,
            xs_init=xs,
            params=params_for_run,
            key=subkey,
            n_samples=short_mcmc_steps,
        )

        # Extract scalar acceptance diagnostic from the summary
        acc = reduce_acceptance(run.summary)

        # Adjust step size based on thresholds
        step_size_new = jnp.where(
            acc < acceptance_threshold_down,
            step_size * DECREASE_FACTOR,
            step_size,
        )
        step_size_new = jnp.where(
            acc > acceptance_threshold_up,
            step_size_new * INCREASE_FACTOR,
            step_size_new,
        )

        # Update params with the new step size
        params_new = set_step_size(params_curr, step_size_new)

        return (
            steps_remaining - 1,
            step_size_new,
            params_new,
            acc,
            key_curr,
        )

    # Start with acceptance outside the desired range to force at least one iteration
    init_state = (
        max_steps,
        step_size0,
        params0,
        jnp.array(0.0),  # initial acc
        key,
    )

    _, _, final_params, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_params
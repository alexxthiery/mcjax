"""
Core MCMC infrastructure: output containers and chain drivers.

This module provides:
- MCMCOutput: container for MCMC trajectory and statistics
- run_mcmc: run a single chain given a step function
- run_mcmc_batch: run multiple chains in parallel via vmap
"""

from typing import Callable, Any
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct


@struct.dataclass
class MCMCOutput:
    """
    Output container for an MCMC run.

    Attributes
    ----------
    states:
        Trajectory of MCMC states (stacked PyTree with leading axis n_samples).
        Each state has at minimum `.x` (position) and `.log_prob` (log density).
    stats:
        Trajectory of per-step statistics (stacked PyTree with leading axis n_samples).
        Contents are kernel-specific (e.g., acceptance indicators, jump distances).
    """
    states: Any
    stats: Any


def run_mcmc(
    step_fn: Callable,
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    state: Any,
    params: Any,
    key: jax.Array,
    n_samples: int,
) -> MCMCOutput:
    """
    Run a single MCMC chain using lax.scan.

    Parameters
    ----------
    step_fn:
        Markov kernel step function with signature:
        step_fn(key, log_prob, state, params) -> (new_state, stats)
    log_prob:
        Log-density of target distribution. Takes position array, returns scalar.
    state:
        Initial state of the chain (created via kernel's init_state).
    params:
        Kernel parameters (e.g., step_size, scale).
    key:
        JAX PRNG key.
    n_samples:
        Number of MCMC steps to run.

    Returns
    -------
    MCMCOutput
        Contains `states` (trajectory) and `stats` (per-step diagnostics).
    """
    def one_step(carry, _):
        key, state = carry
        key, subkey = jr.split(key)
        new_state, stats = step_fn(subkey, log_prob, state, params)
        return (key, new_state), (new_state, stats)

    (_, _), (states, stats) = jax.lax.scan(
        one_step,
        (key, state),
        None,
        length=n_samples,
    )

    return MCMCOutput(states=states, stats=stats)


def run_mcmc_batch(
    step_fn: Callable,
    init_state_fn: Callable,
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    xs_init: jnp.ndarray,
    params: Any,
    key: jax.Array,
    n_samples: int,
) -> MCMCOutput:
    """
    Run multiple MCMC chains in parallel using vmap.

    Parameters
    ----------
    step_fn:
        Markov kernel step function with signature:
        step_fn(key, log_prob, state, params) -> (new_state, stats)
    init_state_fn:
        Function to create initial state from log_prob and position:
        init_state_fn(log_prob, x) -> state
    log_prob:
        Log-density of target distribution (non-batched).
    xs_init:
        Batch of initial positions, shape (batch_size, dim).
    params:
        Kernel parameters, shared across all chains.
    key:
        JAX PRNG key (will be split for each chain).
    n_samples:
        Number of MCMC steps per chain.

    Returns
    -------
    MCMCOutput
        Batched output with leading axis batch_size.
        states.x has shape (batch_size, n_samples, dim).
    """
    batch_size = xs_init.shape[0]
    keys = jr.split(key, batch_size)

    def run_single_chain(x_init, key):
        state = init_state_fn(log_prob, x_init)
        return run_mcmc(step_fn, log_prob, state, params, key, n_samples)

    return jax.vmap(run_single_chain)(xs_init, keys)

"""
Random Walk Metropolis (RWM) kernel.

Exports:
- State: chain state (position + cached log_prob)
- Params: kernel parameters (step_size, scale)
- Stats: per-step diagnostics (acceptance, jump distance)
- step: one RWM transition
- init_state: create initial state from position
- make_params: create params from step_size and covariance
"""

from typing import Callable, Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from .core import run_mcmc, MCMCOutput
from .adaptation import adapt_step_size
from mcjax.util.psd import empirical_cov


# =============================================================================
# Data structures
# =============================================================================

@struct.dataclass
class State:
    """
    RWM chain state.

    Attributes
    ----------
    x: Current position, shape (dim,).
    log_prob: Cached log-density at x.
    """
    x: jnp.ndarray
    log_prob: jnp.ndarray


@struct.dataclass
class Params:
    """
    RWM kernel parameters.

    Attributes
    ----------
    step_size:
        Global scale multiplier for proposals.
    scale:
        Proposal noise transform. Interpretation depends on shape:
        - 1D array (dim,): diagonal, proposal = x + step_size * (scale * noise)
        - 2D array (dim, dim): Cholesky factor, proposal = x + step_size * (scale @ noise)
    """
    step_size: float
    scale: jnp.ndarray


@struct.dataclass
class Stats:
    """
    Per-step RWM diagnostics.

    Attributes
    ----------
    is_accept: Whether proposal was accepted (bool).
    accept_prob: Metropolis-Hastings acceptance probability.
    sq_jump: Acceptance-weighted squared jump distance.
    """
    is_accept: jnp.ndarray
    accept_prob: jnp.ndarray
    sq_jump: jnp.ndarray


# =============================================================================
# Core functions
# =============================================================================

def init_state(
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
) -> State:
    """
    Create initial RWM state from position.

    Parameters
    ----------
    log_prob: Target log-density function.
    x: Initial position, shape (dim,).

    Returns
    -------
    State with position and cached log_prob.
    """
    return State(x=x, log_prob=log_prob(x))


def make_params(
    step_size: float,
    cov: jnp.ndarray,
) -> Params:
    """
    Create RWM parameters from step size and covariance.

    Parameters
    ----------
    step_size: Global scale multiplier for proposals.
    cov: Proposal covariance. Shape determines structure:
        - 1D (dim,): diagonal covariance (variances)
        - 2D (dim, dim): full covariance matrix

    Returns
    -------
    Params with precomputed scale (sqrt for diag, Cholesky for full).
    """
    cov = jnp.asarray(cov)

    if cov.ndim == 1:
        # Diagonal: scale = sqrt(variance)
        scale = jnp.sqrt(cov)
    elif cov.ndim == 2:
        # Full: scale = Cholesky factor
        scale = jnp.linalg.cholesky(cov)
    else:
        raise ValueError(f"cov must be 1D or 2D, got ndim={cov.ndim}")

    return Params(step_size=step_size, scale=scale)


def step(
    key: jax.Array,
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    state: State,
    params: Params,
) -> Tuple[State, Stats]:
    """
    One RWM transition.

    Parameters
    ----------
    key: JAX PRNG key.
    log_prob: Target log-density function.
    state: Current chain state.
    params: Kernel parameters.

    Returns
    -------
    (new_state, stats): Updated state and diagnostics.
    """
    key_prop, key_accept = jr.split(key)

    # Generate proposal
    noise = jr.normal(key_prop, shape=state.x.shape)
    if params.scale.ndim == 1:
        delta = params.step_size * (params.scale * noise)
    else:
        delta = params.step_size * (params.scale @ noise)

    x_prop = state.x + delta
    log_prob_prop = log_prob(x_prop)

    # Metropolis-Hastings acceptance
    log_ratio = log_prob_prop - state.log_prob
    accept_prob = jnp.exp(jnp.minimum(0.0, log_ratio))

    u = jr.uniform(key_accept)
    is_accept = u < accept_prob

    # Update state
    x_new = jnp.where(is_accept, x_prop, state.x)
    log_prob_new = jnp.where(is_accept, log_prob_prop, state.log_prob)

    # Diagnostics
    sq_jump = accept_prob * jnp.sum(delta ** 2)

    return (
        State(x=x_new, log_prob=log_prob_new),
        Stats(is_accept=is_accept, accept_prob=accept_prob, sq_jump=sq_jump),
    )


# =============================================================================
# Convenience API
# =============================================================================

def sample(
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    x_init: jnp.ndarray,
    key: jax.Array,
    n_samples: int,
    *,
    step_size: float,
    cov: Optional[jnp.ndarray] = None,
) -> MCMCOutput:
    """
    Run RWM and return samples (convenience wrapper).

    Parameters
    ----------
    log_prob: Target log-density function.
    x_init: Initial position, shape (dim,).
    key: JAX PRNG key.
    n_samples: Number of MCMC steps.
    step_size: Proposal step size.
    cov: Proposal covariance. Defaults to identity (ones).
        - 1D (dim,): diagonal covariance
        - 2D (dim, dim): full covariance

    Returns
    -------
    MCMCOutput with states and stats trajectories.

    Example
    -------
    >>> output = rwm.sample(log_prob, x0, key, 1000, step_size=0.1)
    >>> samples = output.states.x  # shape (1000, dim)
    """
    x_init = jnp.asarray(x_init)

    if cov is None:
        cov = jnp.ones(x_init.shape[-1])

    state = init_state(log_prob, x_init)
    params = make_params(step_size, cov)

    return run_mcmc(step, log_prob, state, params, key, n_samples)


# =============================================================================
# Adaptation
# =============================================================================

def adapt(
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    xs: jnp.ndarray,
    params: Params,
    key: jax.Array,
    *,
    adapt_cov: bool = True,
    n_iters: int = 10,
    n_steps: int = 5,
    target_accept_low: float = 0.2,
    target_accept_high: float = 0.8,
) -> Params:
    """
    Adapt RWM parameters (step_size and optionally covariance).

    Parameters
    ----------
    log_prob: Target log-density function.
    xs: Sample positions for covariance estimation and short chains, shape (n, dim).
    params: Current RWM parameters.
    key: JAX PRNG key.
    adapt_cov: If True, estimate covariance from xs and update params.
    n_iters: Max iterations for step_size adaptation.
    n_steps: MCMC steps per iteration.
    target_accept_low: Lower acceptance target.
    target_accept_high: Upper acceptance target.

    Returns
    -------
    Adapted Params with updated step_size (and scale if adapt_cov=True).
    """
    xs = jnp.asarray(xs)

    # Optionally update covariance from samples
    if adapt_cov:
        diag = params.scale.ndim == 1
        cov = empirical_cov(xs, diag=diag)
        params = make_params(params.step_size, cov)

    # Adapt step_size
    return adapt_step_size(
        step_fn=step,
        init_state_fn=init_state,
        log_prob=log_prob,
        xs=xs,
        params=params,
        key=key,
        n_iters=n_iters,
        n_steps=n_steps,
        target_accept_low=target_accept_low,
        target_accept_high=target_accept_high,
    )

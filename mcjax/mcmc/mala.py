"""
Metropolis-Adjusted Langevin Algorithm (MALA) kernel.

Exports:
- State: chain state (position + cached log_prob + gradient)
- Params: kernel parameters (step_size, scale, cov_inv, grad_clip)
- Stats: per-step diagnostics (acceptance, jump distance)
- step: one MALA transition
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
    MALA chain state.

    Attributes
    ----------
    x: Current position, shape (dim,).
    log_prob: Cached log-density at x.
    grad: Cached gradient of log-density at x.
    """
    x: jnp.ndarray
    log_prob: jnp.ndarray
    grad: jnp.ndarray


@struct.dataclass
class Params:
    """
    MALA kernel parameters.

    Attributes
    ----------
    step_size:
        Global scale for drift and diffusion.
    scale:
        Proposal noise transform (sqrt or Cholesky of covariance).
        - 1D (dim,): diagonal
        - 2D (dim, dim): Cholesky factor
    cov_inv:
        Inverse covariance for MH correction.
        - 1D (dim,): 1/variance for diagonal
        - 2D (dim, dim): full inverse
    grad_clip:
        Maximum L2 norm for gradient clipping. Use jnp.inf for no clipping.
    """
    step_size: float
    scale: jnp.ndarray
    cov_inv: jnp.ndarray
    grad_clip: float


@struct.dataclass
class Stats:
    """
    Per-step MALA diagnostics.

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
    Create initial MALA state from position.

    Parameters
    ----------
    log_prob: Target log-density function.
    x: Initial position, shape (dim,).

    Returns
    -------
    State with position, cached log_prob, and gradient.
    """
    value, grad = jax.value_and_grad(log_prob)(x)
    return State(x=x, log_prob=value, grad=grad)


def make_params(
    step_size: float,
    cov: jnp.ndarray,
    grad_clip: float = jnp.inf,
) -> Params:
    """
    Create MALA parameters from step size and covariance.

    Parameters
    ----------
    step_size: Global scale for drift and diffusion.
    cov: Proposal covariance. Shape determines structure:
        - 1D (dim,): diagonal covariance (variances)
        - 2D (dim, dim): full covariance matrix
    grad_clip: Maximum L2 norm for gradient clipping.

    Returns
    -------
    Params with precomputed scale and cov_inv.
    """
    cov = jnp.asarray(cov)

    if cov.ndim == 1:
        scale = jnp.sqrt(cov)
        cov_inv = 1.0 / cov
    elif cov.ndim == 2:
        scale = jnp.linalg.cholesky(cov)
        cov_inv = jnp.linalg.inv(cov)
    else:
        raise ValueError(f"cov must be 1D or 2D, got ndim={cov.ndim}")

    return Params(step_size=step_size, scale=scale, cov_inv=cov_inv, grad_clip=grad_clip)


def _clip_gradient(grad: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    """Clip gradient to have L2 norm at most max_norm."""
    norm = jnp.linalg.norm(grad)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-16))
    return grad * scale


def step(
    key: jax.Array,
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    state: State,
    params: Params,
) -> Tuple[State, Stats]:
    """
    One MALA transition.

    MALA proposal:
        x' = x + eps * cov @ grad(x) + sqrt(2*eps) * scale @ noise

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
    eps = params.step_size

    # Clipped gradient for drift
    grad_clipped = _clip_gradient(state.grad, params.grad_clip)

    # Generate proposal
    noise = jr.normal(key_prop, shape=state.x.shape)

    if params.scale.ndim == 1:
        # Diagonal covariance
        cov_diag = params.scale ** 2
        drift = eps * (cov_diag * grad_clipped)
        diffusion = jnp.sqrt(2.0 * eps) * (params.scale * noise)
    else:
        # Full covariance
        cov = params.scale @ params.scale.T
        drift = eps * (cov @ grad_clipped)
        diffusion = jnp.sqrt(2.0 * eps) * (params.scale @ noise)

    x_prop = state.x + drift + diffusion
    log_prob_prop, grad_prop = jax.value_and_grad(log_prob)(x_prop)

    # MH correction for non-symmetric proposal
    grad_prop_clipped = _clip_gradient(grad_prop, params.grad_clip)
    dx = x_prop - state.x

    if params.scale.ndim == 1:
        cov_diag = params.scale ** 2
        # Forward: x -> x_prop
        dx_fwd = dx - eps * (cov_diag * grad_clipped)
        # Backward: x_prop -> x
        dx_bwd = -dx - eps * (cov_diag * grad_prop_clipped)
        q_fwd = jnp.sum(params.cov_inv * dx_fwd ** 2)
        q_bwd = jnp.sum(params.cov_inv * dx_bwd ** 2)
    else:
        cov = params.scale @ params.scale.T
        dx_fwd = dx - eps * (cov @ grad_clipped)
        dx_bwd = -dx - eps * (cov @ grad_prop_clipped)
        q_fwd = dx_fwd @ params.cov_inv @ dx_fwd
        q_bwd = dx_bwd @ params.cov_inv @ dx_bwd

    log_q_ratio = (q_fwd - q_bwd) / (4.0 * eps)
    log_ratio = log_prob_prop - state.log_prob + log_q_ratio
    accept_prob = jnp.exp(jnp.minimum(0.0, log_ratio))

    u = jr.uniform(key_accept)
    is_accept = u < accept_prob

    # Update state
    x_new = jnp.where(is_accept, x_prop, state.x)
    log_prob_new = jnp.where(is_accept, log_prob_prop, state.log_prob)
    grad_new = jnp.where(is_accept, grad_prop, state.grad)

    # Diagnostics
    sq_jump = accept_prob * jnp.sum((x_prop - state.x) ** 2)

    return (
        State(x=x_new, log_prob=log_prob_new, grad=grad_new),
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
    grad_clip: float = jnp.inf,
) -> MCMCOutput:
    """
    Run MALA and return samples (convenience wrapper).

    Parameters
    ----------
    log_prob: Target log-density function (must be differentiable).
    x_init: Initial position, shape (dim,).
    key: JAX PRNG key.
    n_samples: Number of MCMC steps.
    step_size: Proposal step size.
    cov: Proposal covariance. Defaults to identity.
    grad_clip: Maximum gradient L2 norm. Default inf (no clipping).

    Returns
    -------
    MCMCOutput with states and stats trajectories.

    Example
    -------
    >>> output = mala.sample(log_prob, x0, key, 1000, step_size=0.01)
    >>> samples = output.states.x  # shape (1000, dim)
    """
    x_init = jnp.asarray(x_init)

    if cov is None:
        cov = jnp.ones(x_init.shape[-1])

    state = init_state(log_prob, x_init)
    params = make_params(step_size, cov, grad_clip)

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
    Adapt MALA parameters (step_size and optionally covariance).

    Parameters
    ----------
    log_prob: Target log-density function.
    xs: Sample positions for covariance estimation and short chains, shape (n, dim).
    params: Current MALA parameters.
    key: JAX PRNG key.
    adapt_cov: If True, estimate covariance from xs and update params.
    n_iters: Max iterations for step_size adaptation.
    n_steps: MCMC steps per iteration.
    target_accept_low: Lower acceptance target.
    target_accept_high: Upper acceptance target.

    Returns
    -------
    Adapted Params with updated step_size (and scale/cov_inv if adapt_cov=True).
    """
    xs = jnp.asarray(xs)

    # Optionally update covariance from samples
    if adapt_cov:
        diag = params.scale.ndim == 1
        cov = empirical_cov(xs, diag=diag)
        params = make_params(params.step_size, cov, params.grad_clip)

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

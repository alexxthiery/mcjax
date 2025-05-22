import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple, Callable, Optional
# from dataclasses import dataclass
from flax import struct

# from mcjax.proba.density import LogDensity
from .markov import MarkovKernel


# ==================================
# Random Walk Metropolis-Hastings
# with Gaussian proposal distribution
# ==================================
@struct.dataclass
class RwmState:
    x: jnp.ndarray
    logdensity: jnp.ndarray

@struct.dataclass
class RwmStats:
    sq_jump: jnp.ndarray
    is_accept: jnp.ndarray
    accept_MH: jnp.ndarray


@struct.dataclass
class RwmStatsSummary:
    acceptance_rate: jnp.ndarray
    n_accepted: jnp.ndarray
    sq_jump: jnp.ndarray


class Rwm(MarkovKernel):
    """Random Walk Metropolis-Hastings kernel with Gaussian proposals."""

    def __init__(
        self,
        *,
        logdensity: Callable[[jnp.ndarray], jnp.ndarray],
        step_size: float,
        cov: Optional[jnp.ndarray] = None,
    ):
        """
        Args:
            logdensity: Function to compute the log density of the target distribution.
            step_size: Step size for the proposal distribution.
            cov: Covariance matrix for the proposal distribution. If None, an identity matrix is used.
        """
        self.logdensity = logdensity
        self.step_size = step_size
        self.cov = cov  # defer validation to init_state
        self.L = None   # Cholesky will be computed in init_state

    def init_state(self, x_init: jnp.ndarray) -> RwmState:
        """Initializes the RWM kernel state and prepares the proposal distribution."""
        dim = x_init.shape[-1]

        if self.cov is None:
            self.cov = jnp.eye(dim)
        elif self.cov.ndim == 1:
            assert self.cov.shape[0] == dim, "Covariance dimension mismatch"
            self.cov = jnp.diag(self.cov)
        else:
            assert self.cov.shape == (dim, dim), "Covariance dimension mismatch"

        self.L = jnp.linalg.cholesky(self.cov)

        logdensity = self.logdensity(x_init)
        return RwmState(x=x_init, logdensity=logdensity)

    def step(self, state: RwmState, key: jax.Array) -> Tuple[RwmState, RwmStats]:
        """Perform a single step of the RWM kernel."""
        x = state.x
        logtarget_current = state.logdensity

        key, key_prop, key_accept = jr.split(key, 3)
        noise = jr.normal(key_prop, shape=x.shape)
        x_prop = x + self.step_size * (self.L @ noise)

        logtarget_proposal = self.logdensity(x_prop)

        log_ratio = logtarget_proposal - logtarget_current
        accept_MH = jnp.exp(jnp.minimum(0.0, log_ratio))
        u = jr.uniform(key_accept)

        is_accept = u < accept_MH
        x_new = jnp.where(is_accept, x_prop, x)
        logdensity_new = jnp.where(is_accept, logtarget_proposal, logtarget_current)

        new_state = RwmState(x=x_new, logdensity=logdensity_new)
        stats = RwmStats(
            sq_jump=accept_MH * jnp.linalg.norm(x_prop - x) ** 2,
            is_accept=is_accept,
            accept_MH=accept_MH,
        )
        return new_state, stats

    def summarize_stats_traj(self, stats_traj: RwmStats) -> RwmStatsSummary:
        return RwmStatsSummary(
            acceptance_rate=jnp.mean(stats_traj.is_accept),
            n_accepted=jnp.sum(stats_traj.is_accept),
            sq_jump=jnp.mean(stats_traj.sq_jump),
        )

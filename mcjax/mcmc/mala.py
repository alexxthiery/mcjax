import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct
from typing import Callable, Optional, Tuple

from mcjax.proba.gaussian import IsotropicGauss
from mcjax.proba.density import LogDensity
from .markov import MarkovKernel

# ========================
# Metropolis Adjusted Langevin Algorithm
# ========================
@struct.dataclass
class MalaState:
    x: jnp.ndarray
    logdensity: jnp.ndarray
    grad: jnp.ndarray


@struct.dataclass
class MalaStats:
    sq_jump: jnp.ndarray
    is_accept: jnp.ndarray
    accept_MH: jnp.ndarray


@struct.dataclass
class MalaStatsSummary:
    acceptance_rate: jnp.ndarray
    n_accepted: jnp.ndarray
    sq_jump: jnp.ndarray


class Mala(MarkovKernel):
    """Metropolis Adjusted Langevin Algorithm (MALA) kernel."""

    def __init__(
        self,
        *,
        logdensity: Callable[[jnp.ndarray], jnp.ndarray],
        grad_logdensity: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        step_size: float,
        cov: Optional[jnp.ndarray] = None,
    ):
        """
        Args:
            log_density_fn: Callable computing log-density of the target.
            step_size: MALA step size (epsilon).
            grad_logdensity: Optional callable for gradient of log-density.
            cov: Optional covariance matrix for the proposal distribution.
        """
        # define a value_and_grad function
        if grad_logdensity is None:
            self.value_and_grad = jax.value_and_grad(logdensity)
        else:
            def value_and_grad(x):
                return logdensity(x), grad_logdensity(x)
            self.value_and_grad = value_and_grad
        self.logdensity = logdensity
        self.grad_logdensity = grad_logdensity
        self.step_size = step_size
        self.cov = cov          # user-supplied
        self.L = None           # Cholesky of cov
        self.cov_inv = None     # Inverse of cov

    def init_state(self, x_init: jnp.ndarray) -> MalaState:
        """Initialize the MALA state from a starting point."""
        dim = x_init.shape[-1]

        if self.cov is None:
            self.cov = jnp.eye(dim)
        elif self.cov.ndim == 1:
            assert self.cov.shape[0] == dim, "Covariance dimension mismatch"
            self.cov = jnp.diag(self.cov)
        else:
            assert self.cov.shape == (dim, dim), "Covariance dimension mismatch"

        self.L = jnp.linalg.cholesky(self.cov)
        self.cov_inv = jnp.linalg.inv(self.cov)
        
        logdensity, grad = self.value_and_grad(x_init)
        return MalaState(x=x_init, logdensity=logdensity, grad=grad)

    def step(self, state: MalaState, key: jax.Array) -> Tuple[MalaState, MalaStats]:
        """Perform one MALA step."""
        x = state.x
        logtarget_current = state.logdensity
        grad_logp_x = state.grad
        eps = self.step_size


        key, key_prop, key_accept = jr.split(key, 3)
        noise = jr.normal(key_prop, shape=x.shape)

        # Proposal
        drift = eps * self.cov @ grad_logp_x
        diffusion = jnp.sqrt(2 * eps) * (self.L @ noise)
        x_prop = x + drift + diffusion

        # Compute log-density and gradient at the proposal
        logtarget_proposal, grad_logp_prop = self.value_and_grad(x_prop)

        # MH correction
        dx_forward = x_prop - x - eps * self.cov @ grad_logp_x
        dx_backward = x - x_prop - eps * self.cov @ grad_logp_prop

        q_fwd = dx_forward @ self.cov_inv @ dx_forward
        q_bwd = dx_backward @ self.cov_inv @ dx_backward
        log_q_ratio = (q_fwd - q_bwd) / (4 * eps)

        log_accept_ratio = logtarget_proposal - logtarget_current + log_q_ratio
        accept_MH = jnp.exp(jnp.minimum(0.0, log_accept_ratio))

        u = jr.uniform(key_accept)
        is_accept = u < accept_MH

        x_new = jnp.where(is_accept, x_prop, x)
        grad_new = jnp.where(is_accept, grad_logp_prop, grad_logp_x)
        logdensity_new = jnp.where(is_accept, logtarget_proposal, logtarget_current)

        new_state = MalaState(
                            x=x_new,
                            logdensity=logdensity_new,
                            grad=grad_new,
                        )

        stats = MalaStats(
            sq_jump=accept_MH * jnp.linalg.norm(x_prop - x) ** 2,
            is_accept=is_accept,
            accept_MH=accept_MH,
        )

        return new_state, stats

    def summarize_stats_traj(self, stats_traj: MalaStats) -> MalaStatsSummary:
        return MalaStatsSummary(
            acceptance_rate=jnp.mean(stats_traj.is_accept),
            n_accepted=jnp.sum(stats_traj.is_accept),
            sq_jump=jnp.mean(stats_traj.sq_jump),
        )
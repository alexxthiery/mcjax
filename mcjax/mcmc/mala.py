import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct
from typing import Callable, Optional, Tuple
from .core import MarkovKernel


# ========================
# Metropolis Adjusted Langevin Algorithm
# ========================
@struct.dataclass
class MalaState:
    x: jnp.ndarray
    logdensity: jnp.ndarray
    grad: jnp.ndarray


@struct.dataclass
class MalaParams:
    step_size: float
    cov: jnp.ndarray
    L: jnp.ndarray
    cov_inv: jnp.ndarray


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


def summarize_stats_traj(stats_traj: MalaStats) -> MalaStatsSummary:
    return MalaStatsSummary(
        acceptance_rate=jnp.mean(stats_traj.is_accept),
        n_accepted=jnp.sum(stats_traj.is_accept),
        sq_jump=jnp.mean(stats_traj.sq_jump),
    )


def mala_kernel(
    logdensity: Callable[[jnp.ndarray], jnp.ndarray],
    step_size: float,
    grad_logdensity: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    cov: Optional[jnp.ndarray] = None,
) -> MarkovKernel:
    """
    Construct the MALA kernel with optional user-supplied gradient and covariance matrix.
    Returns a functional MarkovKernel object with init, step, summarize.
    """

    value_and_grad = (
        jax.value_and_grad(logdensity)
        if grad_logdensity is None
        else lambda x: (logdensity(x), grad_logdensity(x))
    )

    def init(x0: jnp.ndarray) -> Tuple[MalaState, MalaParams]:
        dim = x0.shape[-1]
        cov_ = jnp.eye(dim) if cov is None else (
            jnp.diag(cov) if cov.ndim == 1 else cov
        )
        L = jnp.linalg.cholesky(cov_)
        cov_inv = jnp.linalg.inv(cov_)

        logdensity0, grad0 = value_and_grad(x0)
        state = MalaState(x=x0, logdensity=logdensity0, grad=grad0)
        params = MalaParams(step_size=step_size, cov=cov_, L=L, cov_inv=cov_inv)
        return state, params

    def step(
        key: jax.Array,
        state: MalaState,
        params: MalaParams
    ) -> Tuple[MalaState, MalaStats]:
        x = state.x
        logtarget_current = state.logdensity
        grad_x = state.grad
        eps = params.step_size

        key, key_prop, key_accept = jr.split(key, 3)
        noise = jr.normal(key_prop, shape=x.shape)

        # Proposal step
        drift = eps * (params.cov @ grad_x)
        diffusion = jnp.sqrt(2 * eps) * (params.L @ noise)
        x_prop = x + drift + diffusion

        logtarget_prop, grad_prop = value_and_grad(x_prop)

        # MH correction term
        dx_fwd = x_prop - x - eps * (params.cov @ grad_x)
        dx_bwd = x - x_prop - eps * (params.cov @ grad_prop)

        q_fwd = dx_fwd @ params.cov_inv @ dx_fwd
        q_bwd = dx_bwd @ params.cov_inv @ dx_bwd
        log_q_ratio = (q_fwd - q_bwd) / (4 * eps)

        log_accept_ratio = logtarget_prop - logtarget_current + log_q_ratio
        accept_MH = jnp.exp(jnp.minimum(0.0, log_accept_ratio))
        u = jr.uniform(key_accept)
        is_accept = u < accept_MH

        x_new = jnp.where(is_accept, x_prop, x)
        grad_new = jnp.where(is_accept, grad_prop, grad_x)
        logp_new = jnp.where(is_accept, logtarget_prop, logtarget_current)

        new_state = MalaState(x=x_new, logdensity=logp_new, grad=grad_new)
        stats = MalaStats(
            sq_jump=accept_MH * jnp.linalg.norm(x_prop - x) ** 2,
            is_accept=is_accept,
            accept_MH=accept_MH,
        )
        return new_state, stats

    return MarkovKernel(
        init=init,
        step=step,
        summarize=summarize_stats_traj
    )
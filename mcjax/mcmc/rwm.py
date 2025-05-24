from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable, Optional, Tuple
from .core import MarkovKernel


@struct.dataclass
class RwmState:
    x: jnp.ndarray
    logdensity: jnp.ndarray


@struct.dataclass
class RwmParams:
    step_size: float
    L: jnp.ndarray  # Cholesky of proposal covariance


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


def summarize_stats_traj(stats_traj: RwmStats) -> RwmStatsSummary:
    return RwmStatsSummary(
        acceptance_rate=jnp.mean(stats_traj.is_accept),
        n_accepted=jnp.sum(stats_traj.is_accept),
        sq_jump=jnp.mean(stats_traj.sq_jump),
    )


def rwm_kernel(
    logdensity: Callable[[jnp.ndarray], jnp.ndarray],
    step_size: float,
    cov: Optional[jnp.ndarray] = None
) -> MarkovKernel:

    def init(x0: jnp.ndarray) -> Tuple[RwmState, RwmParams]:
        dim = x0.shape[-1]
        cov_ = jnp.eye(dim) if cov is None else (
            jnp.diag(cov) if cov.ndim == 1 else cov
        )
        L = jnp.linalg.cholesky(cov_)
        return RwmState(x=x0, logdensity=logdensity(x0)), \
            RwmParams(step_size=step_size, L=L)

    def step(
        key: jax.Array,
        state: RwmState,
        params: RwmParams
    ) -> Tuple[RwmState, RwmStats]:
        key, key_prop, key_accept = jr.split(key, 3)
        noise = jr.normal(key_prop, shape=state.x.shape)
        x_prop = state.x + params.step_size * (params.L @ noise)
        logdensity_prop = logdensity(x_prop)

        log_ratio = logdensity_prop - state.logdensity
        accept_MH = jnp.exp(jnp.minimum(0.0, log_ratio))
        u = jr.uniform(key_accept)
        is_accept = u < accept_MH

        x_new = jnp.where(is_accept, x_prop, state.x)
        logp_new = jnp.where(is_accept, logdensity_prop, state.logdensity)

        new_state = RwmState(x_new, logp_new)
        stats = RwmStats(
            sq_jump=accept_MH * jnp.linalg.norm(x_prop - state.x) ** 2,
            is_accept=is_accept,
            accept_MH=accept_MH,
        )
        return new_state, stats

    return MarkovKernel(
            init=init,
            step=step,
            summarize=summarize_stats_traj)

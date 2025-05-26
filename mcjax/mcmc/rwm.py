from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable, Optional, Tuple
from .core import MarkovKernel


@struct.dataclass
class RwmState:
    x: jnp.ndarray
    log_prob: jnp.ndarray


@struct.dataclass
class RwmParams:
    """
    Parameters for the Random Walk Metropolis (RWM) kernel.

    Attributes:
        step_size: 
            The global step size applied to the proposal.
            This scales the entire proposal distribution.
        
        scale:
            The per-step noise transformation applied to standard normal noise.
            Its meaning depends on `cov_type`:
            
            - If `cov_type == "diag"`:
                `scale` is a 1D array of shape (D,) containing the square roots 
                of the diagonal elements of the covariance matrix. The proposal is:
                    x_prop = x + step_size * (scale * noise)
                where `noise ~ N(0, I)`. In applications, `cov` is often chosen as 
                the marginal covariances obtained empirically from samples.
                
            
            - If `cov_type == "full"`:
                `scale` is a lower-triangular Cholesky factor `L` of the full 
                covariance matrix. The proposal is:
                    x_prop = x + step_size * (L @ noise)

        cov_type:
            A string that indicates the structure of the covariance matrix used 
            for proposals. Must be one of:
                - "diag" : diagonal covariance matrix (scale is a vector)
                - "full" : full covariance matrix (scale is a Cholesky matrix)
    """
    step_size: float
    scale: jnp.ndarray  # Either Cholesky (full) or sqrt diag
    cov_type: str       # "full" or "diag"


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


def _forward(method_name):
    """Helper function to forward method calls to the base kernel."""
    def wrapper(self, *args, **kwargs):
        return getattr(self.base, method_name)(*args, **kwargs)
    return wrapper


@struct.dataclass
class RWMKernel:
    """
    Random Walk Metropolis (RWM) kernel implementation.

    Attributes:
        log_prob: Log of the unnormalized target density.
        cov_type: Covariance type: "diag" or "full"
        base: The underlying MarkovKernel with the algorithm logic.
    """
    log_prob: Callable[[jnp.ndarray], jnp.ndarray] = struct.field(pytree_node=False)
    cov_type: str = struct.field(pytree_node=False)
    base: MarkovKernel = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        log_prob: Callable[[jnp.ndarray], jnp.ndarray],
        cov_type: str = "diag"
    ) -> "RWMKernel":
        assert cov_type in ("diag", "full"), "cov_type must be 'diag' or 'full'"

        def init_state_fn(x0: jnp.ndarray) -> RwmState:
            return RwmState(x=x0, log_prob=log_prob(x0))

        def init_params(x0: jnp.ndarray, step_size: float, cov: Optional[jnp.ndarray] = None) -> RwmParams:
            dim = x0.shape[-1]

            if cov is None:
                if cov_type == "diag":
                    cov = jnp.ones(dim)
                elif cov_type == "full":
                    cov = jnp.eye(dim)
                else:
                    raise ValueError(f"Unsupported cov_type: {cov_type}")

            cov = jnp.asarray(cov)

            if cov_type == "diag":
                assert cov.ndim == 1, "Diagonal covariance must be 1D"
                scale = jnp.sqrt(cov)
            elif cov_type == "full":
                assert cov.ndim == 2 and cov.shape == (dim, dim), "Full covariance must be (D, D)"
                scale = jnp.linalg.cholesky(cov)
            else:
                raise ValueError(f"Unsupported cov_type: {cov_type}")

            return RwmParams(step_size=step_size, scale=scale, cov_type=cov_type)

        def compute_proposal_delta(noise: jnp.ndarray, params: RwmParams) -> jnp.ndarray:
            if params.cov_type == "diag":
                return params.step_size * (params.scale * noise)
            elif params.cov_type == "full":
                return params.step_size * (params.scale @ noise)
            else:
                raise ValueError(f"Invalid cov_type in params: {params.cov_type}")

        def step(key: jax.Array, state: RwmState, params: RwmParams) -> Tuple[RwmState, RwmStats]:
            key_prop, key_accept = jr.split(key)
            noise = jr.normal(key_prop, shape=state.x.shape)
            delta = compute_proposal_delta(noise, params)
            x_prop = state.x + delta
            logp_prop = log_prob(x_prop)

            log_ratio = logp_prop - state.log_prob
            accept_prob = jnp.exp(jnp.minimum(0.0, log_ratio))
            u = jr.uniform(key_accept)
            is_accept = u < accept_prob

            x_new = jnp.where(is_accept, x_prop, state.x)
            logp_new = jnp.where(is_accept, logp_prop, state.log_prob)

            return (
                RwmState(x=x_new, log_prob=logp_new),
                RwmStats(
                    sq_jump=accept_prob * jnp.linalg.norm(x_prop - state.x) ** 2,
                    is_accept=is_accept,
                    accept_MH=accept_prob,
                )
            )

        base = MarkovKernel(
            init_state_fn=init_state_fn,
            init_params=init_params,
            step=step,
            summarize=summarize_stats_traj
        )

        return cls(log_prob=log_prob, cov_type=cov_type, base=base)

    # Forward methods from base kernel
    init_state = _forward("init_state_fn")
    init_params = _forward("init_params")
    step = _forward("step")
    run_mcmc = _forward("run_mcmc")

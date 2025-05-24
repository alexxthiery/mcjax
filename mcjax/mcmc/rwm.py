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


def rwm_kernel(
    logdensity: Callable[[jnp.ndarray], jnp.ndarray],
    step_size: float,
    cov: Optional[jnp.ndarray] = None,
    cov_type: Optional[str] = None
) -> MarkovKernel:
    """
    Constructs a Random Walk Metropolis (RWM) kernel using a Gaussian proposal distribution.
    The proposal distribution is defined as:

        x_prop = x + step_size * proposal_noise

    where `proposal_noise` is drawn from a zero-mean Gaussian with covariance structure 
    specified by `cov` and `cov_type`.

    Args:
        logdensity: 
            A callable `logdensity(x)` that returns the log of the unnormalized 
            target density at `x`. Must be JAX-compatible.
        step_size: 
            A scalar float that scales the proposal noise globally. 
            This is typically tuned to achieve a desired acceptance rate.
        cov: 
            Optional specification of the proposal covariance structure. Can be:
                - A 1D array of shape (D,), interpreted as a diagonal covariance
                - A 2D array of shape (D, D), interpreted as a full covariance matrix
                - If None, a default identity matrix is used
        cov_type: 
            Optional string specifying how to interpret `cov`. Must be one of:
                - "diag": diagonal covariance (1D vector)
                - "full": full covariance (2D matrix)
                - If not provided, the type is inferred from the shape of `cov`
                - If neither `cov` nor `cov_type` is provided, defaults to "diag" with identity

    Returns:
        MarkovKernel:
            A MarkovKernel instance that implements the RWM algorithm.
    """
    def init(
            x0: jnp.ndarray,
            cov=cov,
            cov_type=cov_type,
            ) -> Tuple[RwmState, RwmParams]:
        dim = x0.shape[-1]

        if cov is None:
            if cov_type is None or cov_type == "diag":
                cov_type_ = "diag"
                scale = jnp.ones(dim)
            elif cov_type == "full":
                cov_type_ = "full"
                scale = jnp.eye(dim)
            else:
                raise ValueError(f"Unsupported cov_type: {cov_type}")
        else:
            cov = jnp.asarray(cov)
            if cov_type is None:
                if cov.ndim == 1:
                    cov_type_ = "diag"
                    scale = jnp.sqrt(cov)
                elif cov.ndim == 2:
                    assert cov.shape == (dim, dim), "Full covariance shape mismatch"
                    cov_type_ = "full"
                    scale = jnp.linalg.cholesky(cov)
                else:
                    raise ValueError("cov must be 1D or 2D")
            else:
                assert cov_type in {"diag", "full"}, "cov_type must be 'diag' or 'full'"
                if cov_type == "diag":
                    assert cov.ndim == 1, "cov_type 'diag' requires a 1D vector"
                    cov_type_ = "diag"
                    scale = jnp.sqrt(cov)
                elif cov_type == "full":
                    assert cov.ndim == 2 and cov.shape == (dim, dim), \
                        "cov_type 'full' requires a (dim, dim) matrix"
                    cov_type_ = "full"
                    scale = jnp.linalg.cholesky(cov)

        return (
            RwmState(x=x0, logdensity=logdensity(x0)),
            RwmParams(step_size=step_size, scale=scale, cov_type=cov_type_)
        )

    def compute_proposal_delta(noise: jnp.ndarray, params: RwmParams) -> jnp.ndarray:
        if params.cov_type == "diag":
            return params.step_size * (params.scale * noise)
        elif params.cov_type == "full":
            return params.step_size * (params.scale @ noise)
        else:
            raise ValueError(f"Unsupported cov_type in params: {params.cov_type}")

    def step(key: jax.Array, state: RwmState, params: RwmParams) -> Tuple[RwmState, RwmStats]:
        key, key_prop, key_accept = jr.split(key, 3)
        noise = jr.normal(key_prop, shape=state.x.shape)
        delta = compute_proposal_delta(noise, params)
        x_prop = state.x + delta
        logdensity_prop = logdensity(x_prop)

        log_ratio = logdensity_prop - state.logdensity
        accept_MH = jnp.exp(jnp.minimum(0.0, log_ratio))
        u = jr.uniform(key_accept)
        is_accept = u < accept_MH

        x_new = jnp.where(is_accept, x_prop, state.x)
        logp_new = jnp.where(is_accept, logdensity_prop, state.logdensity)

        return (
            RwmState(x=x_new, logdensity=logp_new),
            RwmStats(
                sq_jump=accept_MH * jnp.linalg.norm(x_prop - state.x) ** 2,
                is_accept=is_accept,
                accept_MH=accept_MH,
            ),
        )

    return MarkovKernel(init=init, step=step, summarize=summarize_stats_traj)
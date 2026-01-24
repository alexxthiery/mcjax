from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from .core import MarkovKernel
from .adaptation import adapt_step_size


@struct.dataclass
class RwmState:
    x: jnp.ndarray
    log_prob: jnp.ndarray


@struct.dataclass
class RwmParams:
    """
    Parameters for the Random Walk Metropolis (RWM) kernel.

    Attributes
    ----------
    step_size:
        Global step size applied to the proposal. This scales the
        entire proposal distribution.

    scale:
        Per-step noise transform applied to standard normal noise.
        Its interpretation depends on the covariance structure that
        was used when constructing the kernel:

        - Diagonal covariance:
            `scale` is a 1D array of shape (D,) containing the square
            roots of the diagonal entries of the covariance matrix.
            The proposal is:
                x_prop = x + step_size * (scale * noise)
            where noise ~ N(0, I).

        - Full covariance:
            `scale` is a lower-triangular Cholesky factor L of the
            full covariance matrix. The proposal is:
                x_prop = x + step_size * (L @ noise)
    """
    step_size: float
    scale: jnp.ndarray  # sqrt(diag(cov)) or Cholesky factor of cov


@struct.dataclass
class RwmStats:
    sq_jump: jnp.ndarray      # acceptance-weighted squared jump distance
    is_accept: jnp.ndarray    # boolean accept indicator
    accept_MH: jnp.ndarray    # MH acceptance probability


@struct.dataclass
class RwmStatsSummary:
    acceptance_rate: jnp.ndarray
    n_accepted: jnp.ndarray
    sq_jump: jnp.ndarray
    traj_length: jnp.ndarray


def summarize_stats_traj(stats_traj: RwmStats) -> RwmStatsSummary:
    """
    Summarize a trajectory of RWMStats along the leading axis (time).

    If stats_traj is batched over chains as well, this aggregates over
    all chains and time; for per-chain summaries, write a different
    summarizer that reduces over time only.
    """
    return RwmStatsSummary(
        acceptance_rate=jnp.mean(stats_traj.is_accept),
        n_accepted=jnp.sum(stats_traj.is_accept),
        sq_jump=jnp.mean(stats_traj.sq_jump),
        traj_length=stats_traj.is_accept.shape[0],
    )


def _empirical_cov(xs: jnp.ndarray, eps: float = 0.0) -> jnp.ndarray:
    """
    Empirical covariance of xs with optional diagonal jitter.

    xs: array of shape (n_samples, dim)
    """
    xs = jnp.asarray(xs)
    mean = jnp.mean(xs, axis=0, keepdims=True)
    xc = xs - mean
    n = xs.shape[0]
    cov = (xc.T @ xc) / n
    if eps > 0.0:
        d = xs.shape[1]
        cov = cov + eps * jnp.eye(d, dtype=xs.dtype)
    return cov


def create_rwm_kernel(dim: int, cov_type: str = "diag") -> MarkovKernel:
    """
    Construct a Random Walk Metropolis kernel as a MarkovKernel.

    Parameters
    ----------
    dim:
        Dimension of the state space.
    cov_type:
        Structure of the proposal covariance:
            - "diag": diagonal covariance, scale is sqrt(diag(cov)).
            - "full": full covariance, scale is Cholesky factor.

    Returns
    -------
    MarkovKernel
        A kernel implementing RWM with the standard interface.
    """
    assert cov_type in ("diag", "full"), "cov_type must be 'diag' or 'full'"

    def init_state_fn(
        log_prob: Callable[[jnp.ndarray], jnp.ndarray],
        x0: jnp.ndarray,
    ) -> RwmState:
        return RwmState(x=x0, log_prob=log_prob(x0))

    def init_params(step_size: float, cov: Optional[jnp.ndarray] = None) -> RwmParams:
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

        return RwmParams(step_size=step_size, scale=scale)

    def compute_proposal_delta(
        noise: jnp.ndarray,
        params: RwmParams,
    ) -> jnp.ndarray:
        if cov_type == "diag":
            # scale: (D,), noise: (D,)
            return params.step_size * (params.scale * noise)
        elif cov_type == "full":
            # scale: (D, D), noise: (D,)
            return params.step_size * (params.scale @ noise)
        else:
            raise ValueError(f"Unsupported cov_type: {cov_type}")

    def step(
        key: jax.Array,
        log_prob: Callable[[jnp.ndarray], jnp.ndarray],
        state: RwmState,
        params: RwmParams,
    ) -> Tuple[RwmState, RwmStats]:
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

        # compute squared jump distance weighted by acceptance
        sq_jump = jnp.linalg.norm(delta) ** 2 * accept_prob

        return (
            RwmState(x=x_new, log_prob=logp_new),
            RwmStats(
                sq_jump=sq_jump,
                is_accept=is_accept,
                accept_MH=accept_prob,
            ),
        )

    return MarkovKernel(
        init_state_fn=init_state_fn,
        init_params=init_params,
        step=step,
        summarize=summarize_stats_traj,
    )


# =========================================
# ADAPTATION HELPERS FOR RWM
# ========================================
def rwm_update_step_size(params: RwmParams, step_size: float) -> RwmParams:
    """
    Functional helper to update the step size of RWM parameters.
    """
    return params.replace(step_size=step_size)


def _rwm_reduce_acceptance(
    summary: RwmStatsSummary,
    acceptance_quantile: float,
) -> jnp.ndarray:
    """
    Convert an RWM summary into a scalar acceptance diagnostic.

    If `summary.acceptance_rate` is batched over chains, we take a lower
    quantile across chains to be conservative. If it is scalar, return
    it directly.
    
    Parameters
    ----------
    summary: RwmStatsSummary
        Summary statistics from an RWM run.
    acceptance_quantile: float
        Quantile to use when reducing acceptance rates.
    """
    acc = summary.acceptance_rate
    if jnp.ndim(acc) == 0:
        return acc
    return jnp.quantile(acc, acceptance_quantile)


def rwm_adapt(
    kernel: MarkovKernel,
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    xs: jnp.ndarray,
    params: RwmParams,
    key: jax.Array,
    *,
    n_steps: int = 10,
    short_mcmc_steps: int = 5,
    acceptance_threshold_up: float = 0.8,
    acceptance_threshold_down: float = 0.2,
    acceptance_quantile: float = 0.1,
) -> RwmParams:
    """
    RWM-specific wrapper around `adapt_step_size_smc`.

    This function:
      1. Estimates a proposal covariance from the cloud of samples `xs`.
      2. Rebuilds RWM parameters using that covariance (keeping step_size).
      3. Calls the generic SMC step-size adaptation routine.

    The covariance structure (diag/full) is inferred from the shape
    of `params.scale`:
        - scale.ndim == 1 -> diagonal
        - scale.ndim == 2 -> full
    
    Description
    -----------
    Description of the adaptation procedure: the step size is iteratively
    adjusted using short MCMC runs so that the acceptance rate is within
    the specified thresholds. The covariance is adapted only once at the start
    by estimating from `xs`.
        
    Parameters
    ----------
    kernel:
        RWM MarkovKernel.
    log_prob:
        Log-density function of the target distribution.
    xs:
        Approximate samples from the target distribution, shape (n_samples, dim).
    params:
        Initial RWM parameters.
    key:
        JAX PRNG key.
    n_steps:
        Maximum number of step size adaptation iterations.
    short_mcmc_steps:
        Number of MCMC steps per adaptation iteration.
    acceptance_threshold_up:
        Upper threshold to increase step size.
    acceptance_threshold_down:
        Lower threshold to decrease step size.
    acceptance_quantile:
        Quantile of acceptance rates to use for adaptation.
    """
    # test if params is of type RwmParams
    if not isinstance(params, RwmParams):
        raise ValueError("params must be of type RwmParams")

    xs = jnp.asarray(xs)
    if xs.ndim != 2:
        raise ValueError("xs must have shape (n_samples, dim)")

    dim_x = xs.shape[1]
    # Sanity check: last dimension of scale must match state dimension.
    if params.scale.shape[-1] != dim_x:
        raise ValueError(
            f"Inconsistent dimensions: xs dim {dim_x}, "
            f"params.scale last dim {params.scale.shape[-1]}"
        )

    # eps: small jitter for covariance estimation
    eps = 1e-5
    
    if params.scale.ndim == 1:
        # Diagonal covariance: estimate marginal variances.
        stds = jnp.clip(jnp.std(xs, axis=0), a_min=eps)
        cov = stds ** 2
    elif params.scale.ndim == 2:
        # Full covariance.
        cov = _empirical_cov(xs, eps=eps)
    else:
        raise ValueError(
            f"Unsupported scale shape for RWM: ndim={params.scale.ndim}"
        )

    # Rebuild params with the new covariance but same step size.
    params0 = kernel.init_params(step_size=params.step_size, cov=cov)

    # Closure for acceptance reduction with the chosen quantile.
    def reduce_acceptance(summary):
        return _rwm_reduce_acceptance(summary, acceptance_quantile)

    # Delegate to the generic adaptation routine, providing trait functions.
    return adapt_step_size(
        kernel=kernel,
        params=params0,
        log_prob=log_prob,
        xs=xs,
        key=key,
        max_steps=n_steps,
        short_mcmc_steps=short_mcmc_steps,
        acceptance_threshold_up=acceptance_threshold_up,
        acceptance_threshold_down=acceptance_threshold_down,
        get_step_size=lambda p: p.step_size,
        set_step_size=rwm_update_step_size,
        reduce_acceptance=reduce_acceptance,
    )
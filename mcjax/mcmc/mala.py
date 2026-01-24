from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from .core import MarkovKernel
from .adaptation import adapt_step_size


# ========================
# Metropolis Adjusted Langevin Algorithm
# ========================
@struct.dataclass
class MalaState:
    x: jnp.ndarray
    log_prob: jnp.ndarray
    grad: jnp.ndarray


@struct.dataclass
class MalaParams:
    """
    Parameters for the MALA kernel.

    Attributes
    ----------
    step_size:
        Global step size used in both drift and proposal noise.

    scale:
        Per-step noise transform applied to standard normal noise,
        analogous to RWM:

        - Diagonal covariance:
            1D array of shape (D,) with sqrt of diagonal covariance.
            Proposal:
                x_prop = x + drift + sqrt(2 * step_size) * (scale * noise)

        - Full covariance:
            Lower-triangular Cholesky factor L of full covariance.
            Proposal:
                x_prop = x + drift + sqrt(2 * step_size) * (L @ noise)

    cov_inv:
        Inverse covariance matrix (used in MH correction):
            - If diag: 1 / diag(cov).
            - If full: full matrix inverse.

    grad_clip_norm:
        L2 norm used to clip gradients:
            grad <- grad * min(1, grad_clip_norm / ||grad||).
    """
    step_size: float
    scale: jnp.ndarray
    cov_inv: jnp.ndarray
    grad_clip_norm: float


@struct.dataclass
class MalaStats:
    sq_jump: jnp.ndarray      # acceptance-weighted squared jump distance
    is_accept: jnp.ndarray    # boolean accept indicator
    accept_MH: jnp.ndarray    # MH acceptance probability


@struct.dataclass
class MalaStatsSummary:
    acceptance_rate: jnp.ndarray
    n_accepted: jnp.ndarray
    sq_jump: jnp.ndarray
    traj_length: jnp.ndarray


def summarize_stats_traj(stats_traj: MalaStats) -> MalaStatsSummary:
    """
    Summarize a trajectory of MalaStats along the leading axis (time).

    If stats_traj is batched over chains as well, this aggregates over
    all chains and time; for per-chain summaries, write a different
    summarizer that reduces over time only.
    """
    return MalaStatsSummary(
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


def create_mala_kernel(dim: int, cov_type: str = "diag") -> MarkovKernel:
    """
    Construct a Metropolis Adjusted Langevin Algorithm (MALA) kernel
    as a MarkovKernel, following the same design pattern as RWM.

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
        A kernel implementing MALA with the standard interface.
    """
    assert cov_type in ("diag", "full"), "cov_type must be 'diag' or 'full'"

    def init_state_fn(
        log_prob: Callable[[jnp.ndarray], jnp.ndarray],
        x0: jnp.ndarray,
    ) -> MalaState:
        value_and_grad = jax.value_and_grad(log_prob)
        logp, grad = value_and_grad(x0)
        return MalaState(x=x0, log_prob=logp, grad=grad)

    def init_params(
        step_size: float,
        cov: Optional[jnp.ndarray] = None,
        grad_clip_norm: float = jnp.inf,
    ) -> MalaParams:
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
            cov_inv = 1.0 / cov
        elif cov_type == "full":
            assert cov.ndim == 2 and cov.shape == (dim, dim), "Full covariance must be (D, D)"
            scale = jnp.linalg.cholesky(cov)
            cov_inv = jnp.linalg.inv(cov)
        else:
            raise ValueError(f"Unsupported cov_type: {cov_type}")

        return MalaParams(
            step_size=step_size,
            scale=scale,
            cov_inv=cov_inv,
            grad_clip_norm=grad_clip_norm,
        )

    def clip_grad(params: MalaParams, grad: jnp.ndarray) -> jnp.ndarray:
        """
        Clip gradient to have L2 norm at most params.grad_clip_norm.
        """
        norm = jnp.linalg.norm(grad)

        def do_clip(_):
            return grad * (params.grad_clip_norm / (norm + 1e-16))

        def no_clip(_):
            return grad

        return jax.lax.cond(
            norm > params.grad_clip_norm,
            do_clip,
            no_clip,
            operand=None,
        )

    def compute_drift_and_diffusion(
        grad: jnp.ndarray,
        noise: jnp.ndarray,
        params: MalaParams,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        grad_clip = clip_grad(params, grad)
        eps = params.step_size

        if cov_type == "diag":
            # cov = diag(scale^2)
            drift = eps * (params.scale ** 2 * grad_clip)
            diffusion = jnp.sqrt(2.0 * eps) * (params.scale * noise)
        elif cov_type == "full":
            cov = params.scale @ params.scale.T
            drift = eps * (cov @ grad_clip)
            diffusion = jnp.sqrt(2.0 * eps) * (params.scale @ noise)
        else:
            raise ValueError(f"Unsupported cov_type: {cov_type}")

        return drift, diffusion

    def compute_log_accept_ratio(
        x: jnp.ndarray,
        x_prop: jnp.ndarray,
        grad_x: jnp.ndarray,
        grad_prop: jnp.ndarray,
        log_p_x: jnp.ndarray,
        log_p_prop: jnp.ndarray,
        params: MalaParams,
    ) -> jnp.ndarray:
        """
        Compute log Metropolis–Hastings acceptance ratio for MALA.
        """
        eps = params.step_size
        dx = x_prop - x

        grad_x_clip = clip_grad(params, grad_x)
        grad_prop_clip = clip_grad(params, grad_prop)

        if cov_type == "diag":
            # cov = diag(scale^2), cov_inv is 1 / diag(cov)
            cov_diag = params.scale ** 2
            dx_fwd = dx - eps * (cov_diag * grad_x_clip)
            dx_bwd = -dx - eps * (cov_diag * grad_prop_clip)

            q_fwd = jnp.sum(params.cov_inv * dx_fwd ** 2)
            q_bwd = jnp.sum(params.cov_inv * dx_bwd ** 2)

        elif cov_type == "full":
            cov = params.scale @ params.scale.T
            cov_inv = params.cov_inv

            dx_fwd = dx - eps * (cov @ grad_x_clip)
            dx_bwd = -dx - eps * (cov @ grad_prop_clip)

            q_fwd = dx_fwd @ cov_inv @ dx_fwd
            q_bwd = dx_bwd @ cov_inv @ dx_bwd

        else:
            raise ValueError(f"Unsupported cov_type: {cov_type}")

        log_q_ratio = (q_fwd - q_bwd) / (4.0 * eps)
        return log_p_prop - log_p_x + log_q_ratio

    def step(
        key: jax.Array,
        log_prob: Callable[[jnp.ndarray], jnp.ndarray],
        state: MalaState,
        params: MalaParams,
    ) -> Tuple[MalaState, MalaStats]:
        value_and_grad = jax.value_and_grad(log_prob)

        x = state.x
        grad_x = state.grad
        log_p_x = state.log_prob

        key_prop, key_accept = jr.split(key)
        noise = jr.normal(key_prop, shape=x.shape)

        drift, diffusion = compute_drift_and_diffusion(grad_x, noise, params)
        x_prop = x + drift + diffusion
        log_p_prop, grad_prop = value_and_grad(x_prop)

        log_ratio = compute_log_accept_ratio(
            x=x,
            x_prop=x_prop,
            grad_x=grad_x,
            grad_prop=grad_prop,
            log_p_x=log_p_x,
            log_p_prop=log_p_prop,
            params=params,
        )

        accept_MH = jnp.exp(jnp.minimum(0.0, log_ratio))
        u = jr.uniform(key_accept)
        is_accept = u < accept_MH

        x_new = jnp.where(is_accept, x_prop, x)
        grad_new = jnp.where(is_accept, grad_prop, grad_x)
        logp_new = jnp.where(is_accept, log_p_prop, log_p_x)

        sq_jump = accept_MH * (jnp.linalg.norm(x_prop - x) ** 2)

        return (
            MalaState(x=x_new, log_prob=logp_new, grad=grad_new),
            MalaStats(
                sq_jump=sq_jump,
                is_accept=is_accept,
                accept_MH=accept_MH,
            ),
        )

    return MarkovKernel(
        init_state_fn=init_state_fn,
        init_params=init_params,
        step=step,
        summarize=summarize_stats_traj,
    )


# =========================================
# ADAPTATION HELPERS FOR MALA
# =========================================
def mala_update_step_size(params: MalaParams, step_size: float) -> MalaParams:
    """
    Functional helper to update the step size of MALA parameters.
    """
    return params.replace(step_size=step_size)

def _mala_reduce_acceptance(
    summary: MalaStatsSummary,
    acceptance_quantile: float,
) -> jnp.ndarray:
    """
    Convert a MALA summary into a scalar acceptance diagnostic.

    If `summary.acceptance_rate` is batched over chains, we take a lower
    quantile across chains to be conservative. If it is scalar, return
    it directly.
    """
    acc = summary.acceptance_rate
    if jnp.ndim(acc) == 0:
        return acc
    return jnp.quantile(acc, acceptance_quantile)


def mala_adapt(
    kernel: MarkovKernel,
    log_prob: Callable[[jnp.ndarray], jnp.ndarray],
    xs: jnp.ndarray,
    params: MalaParams,
    key: jax.Array,
    *,
    n_steps: int = 10,
    short_mcmc_steps: int = 5,
    acceptance_threshold_up: float = 0.8,
    acceptance_threshold_down: float = 0.2,
    acceptance_quantile: float = 0.1,
) -> MalaParams:
    """
    MALA-specific wrapper around `adapt_step_size_smc`.

    This function:
      * Estimates a proposal covariance from `xs` (diag or full).
      * Rebuilds MALA parameters using that covariance (keeping step_size).
      * Calls the generic SMC step-size adaptation routine.

    The covariance structure is inferred from the shape of `params.scale`:
        - scale.ndim == 1 -> diagonal
        - scale.ndim == 2 -> full

    Parameters
    ----------
    kernel:
        MALA MarkovKernel.
    log_prob:
        Log-density function of the target distribution.
    xs:
        Approximate samples from the target distribution, shape (n_samples, dim).
    params:
        Initial MALA parameters.
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
    # test if params is of type MalaParams
    if not isinstance(params, MalaParams):
        raise ValueError("params must be of type MalaParams")

    xs = jnp.asarray(xs)
    if xs.ndim != 2:
        raise ValueError("xs must have shape (n_samples, dim)")

    dim_x = xs.shape[1]
    if params.scale.shape[-1] != dim_x:
        raise ValueError(
            f"Inconsistent dimensions: xs dim {dim_x}, "
            f"params.scale last dim {params.scale.shape[-1]}"
        )

    # Estimate covariance from xs.
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
            f"Unsupported scale shape for MALA: ndim={params.scale.ndim}"
        )

    # Rebuild params with the new covariance but same step size and clip norm.
    params0 = kernel.init_params(
        step_size=params.step_size,
        cov=cov,
        grad_clip_norm=params.grad_clip_norm,
    )

    # Closure for acceptance reduction with the chosen quantile.
    def reduce_acceptance(summary: MalaStatsSummary) -> jnp.ndarray:
        return _mala_reduce_acceptance(summary, acceptance_quantile)

    # Delegate to the generic adaptation routine.
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
        set_step_size=mala_update_step_size,
        reduce_acceptance=reduce_acceptance,
    )
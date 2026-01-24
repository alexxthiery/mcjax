import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct
from typing import Callable, Optional, Tuple
from .core import MarkovKernel
from .adaptation import adapt_step_size_smc


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

    Attributes:
        step_size:
            Global step size used in both drift and proposal noise.

        scale:
            - If cov_type == "diag": sqrt of the diagonal covariance (1D array)
            - If cov_type == "full": Cholesky factor L of full covariance (2D)

        cov_inv:
            - Inverse of the covariance matrix:
                - If diag: 1 / diag
                - If full: inverse of full matrix

        cov_type:
            Either "diag" or "full" — determines how drift and correction terms are computed.
    """
    step_size: float
    scale: jnp.ndarray
    cov_inv: jnp.ndarray
    grad_clip_norm: float


@struct.dataclass
class MalaStats:
    sq_jump: jnp.ndarray
    is_accept: jnp.ndarray
    accept_MH: jnp.ndarray


@struct.dataclass
class MalaStatsSummary:
    acceptance_rate: jnp.ndarray
    n_accepted: jnp.ndarray
    traj_length: jnp.ndarray
    sq_jump: jnp.ndarray


def summarize_stats_traj(stats_traj: MalaStats) -> MalaStatsSummary:
    return MalaStatsSummary(
        acceptance_rate=jnp.mean(stats_traj.is_accept),
        n_accepted=jnp.sum(stats_traj.is_accept),
        traj_length=stats_traj.is_accept.shape[0],
        sq_jump=jnp.mean(stats_traj.sq_jump),
    )


def _forward(method_name):
    def wrapper(self, *args, **kwargs):
        return getattr(self.base, method_name)(*args, **kwargs)
    return wrapper


@struct.dataclass
class MALAKernel:
    """
    Metropolis Adjusted Langevin Algorithm (MALA) kernel.
    """
    dim: int = struct.field(pytree_node=False)
    cov_type: str = struct.field(pytree_node=False)
    base: MarkovKernel = struct.field(pytree_node=False)

    # Forwarding interface
    init_state = _forward("init_state_fn")
    init_params = _forward("init_params")
    step = _forward("step")
    run_mcmc = _forward("run_mcmc")
    run_mcmc_batch = _forward("run_mcmc_batch")

    @classmethod
    def create(
        cls,
        dim: int,
        cov_type: str = "diag"
    ) -> "MALAKernel":
        assert cov_type in {"diag", "full"}, "cov_type must be 'diag' or 'full'"

        def init_state_fn(
                log_prob: Callable[[jnp.ndarray], jnp.ndarray],
                x0: jnp.ndarray) -> MalaState:
            value_and_grad = jax.value_and_grad(log_prob)
            logp, grad = value_and_grad(x0)
            return MalaState(x=x0, log_prob=logp, grad=grad)

        def init_params(
                step_size: float,
                cov: Optional[jnp.ndarray] = None,
                grad_clip_norm: Optional[float] = jnp.inf,
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

        def clip_grad(params, grad):
            grad = jax.lax.cond(
                jnp.linalg.norm(grad) > params.grad_clip_norm,
                lambda: grad * (params.grad_clip_norm / jnp.linalg.norm(grad)),
                lambda: grad,
            )
            return grad

        def compute_drift_and_diffusion(grad, noise, params):
            grad_clip = clip_grad(params, grad)
            eps = params.step_size
            if cov_type == "diag":
                # drift = eps * (params.scale ** 2 * grad)
                drift = eps * (params.scale ** 2 * grad_clip)
                diffusion = jnp.sqrt(2 * eps) * (params.scale * noise)
            elif cov_type == "full":
                cov = params.scale @ params.scale.T
                # drift = eps * (cov @ grad)
                drift = eps * (cov @ grad_clip)
                diffusion = jnp.sqrt(2 * eps) * (params.scale @ noise)
            else:
                raise ValueError(f"Unsupported cov_type: {cov_type}")
            return drift, diffusion

        def compute_log_accept_ratio(x, x_prop, grad_x, grad_prop, log_p_x, log_p_prop, params):
            eps = params.step_size
            dx = x_prop - x
            
            grad_x_clip = clip_grad(params, grad_x)
            grad_prop_clip = clip_grad(params, grad_prop)

            if cov_type == "diag":
                # dx_fwd = dx - eps * (params.scale ** 2 * grad_x)
                # dx_bwd = -dx - eps * (params.scale ** 2 * grad_prop)
                dx_fwd = dx - eps * (params.scale ** 2 * grad_x_clip)
                dx_bwd = -dx - eps * (params.scale ** 2 * grad_prop_clip)
                
                q_fwd = jnp.sum(params.cov_inv * dx_fwd ** 2)
                q_bwd = jnp.sum(params.cov_inv * dx_bwd ** 2)
            elif cov_type == "full":
                cov_inv = params.cov_inv
                cov = params.scale @ params.scale.T
                # dx_fwd = dx - eps * (cov @ grad_x)
                # dx_bwd = -dx - eps * (cov @ grad_prop)
                dx_fwd = dx - eps * (cov @ grad_x_clip)
                dx_bwd = -dx - eps * (cov @ grad_prop_clip)
                
                q_fwd = dx_fwd @ cov_inv @ dx_fwd
                q_bwd = dx_bwd @ cov_inv @ dx_bwd
            else:
                raise ValueError(f"Unsupported cov_type: {cov_type}")

            log_q_ratio = (q_fwd - q_bwd) / (4 * eps)
            return log_p_prop - log_p_x + log_q_ratio

        def step(
                key: jax.Array,
                log_prob: Callable[[jnp.ndarray], jnp.ndarray],
                state: MalaState,
                params: MalaParams
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
                params=params
            )

            accept_MH = jnp.exp(jnp.minimum(0.0, log_ratio))
            u = jr.uniform(key_accept)
            is_accept = u < accept_MH

            x_new = jnp.where(is_accept, x_prop, x)
            grad_new = jnp.where(is_accept, grad_prop, grad_x)
            logp_new = jnp.where(is_accept, log_p_prop, log_p_x)

            return (
                MalaState(x=x_new, log_prob=logp_new, grad=grad_new),
                MalaStats(
                    sq_jump=accept_MH * jnp.linalg.norm(x_prop - x) ** 2,
                    is_accept=is_accept,
                    accept_MH=accept_MH,
                )
            )

        base = MarkovKernel(
            init_state_fn=init_state_fn,
            init_params=init_params,
            step=step,
            summarize=summarize_stats_traj
        )

        return cls(dim=dim, cov_type=cov_type, base=base)

    def adapt_smc(
        self,
        log_prob: Callable[[jnp.ndarray], jnp.ndarray],
        xs: jnp.ndarray,
        params: MalaParams,
        key: jax.Array,
        n_steps: int = 10,
    ) -> MalaParams:
        # update the norm of the gradient clipping
        grads = jax.vmap(
            lambda x: jax.grad(log_prob)(x),
            in_axes=0,
            out_axes=0,
        )(xs)
        grad_norms = jnp.linalg.norm(grads, axis=1)
        # compute the 95th percentile of the gradient norms
        grad_clip_norm_quantile = jnp.percentile(grad_norms, 95.0)
        params = self.update_grad_clip_norm(
                            params=params,
                            grad_clip_norm=grad_clip_norm_quantile)
        
        if self.cov_type == "diag":
            # first, adapt the scale based on the empirical std of xs
            eps = 1e-5
            stds = jnp.clip(jnp.std(xs, axis=0), a_min=eps)
            params_new = self.init_params(
                                step_size=params.step_size,
                                cov=stds**2,
                                grad_clip_norm=params.grad_clip_norm,)
            params_adapted = adapt_step_size_smc(
                        kernel=self,
                        params=params_new,
                        log_prob=log_prob,
                        xs=xs,
                        key=key,
                        max_steps=n_steps,
                        )
        elif self.cov_type == "full":
            # first, adapt the scale based on the empirical std of xs
            eps = 1e-5
            cov = jnp.cov(xs, rowvar=False)
            # add a small value to the diagonal to ensure numerical stability
            cov = cov + eps * jnp.eye(self.dim)
            params_new = self.init_params(
                                step_size=params.step_size,
                                cov=cov,
                                grad_clip_norm=params.grad_clip_norm,)
            params_adapted = adapt_step_size_smc(
                                kernel=self,
                                params=params_new,
                                log_prob=log_prob,
                                xs=xs,
                                key=key,
                                max_steps=n_steps,
                                )
        return params_adapted

    def update_step_size(self, params: MalaParams, step_size: float) -> MalaParams:
        """
        Update the step size in the RWM parameters.
        """
        return params.replace(step_size=step_size)

    def update_grad_clip_norm(self, params: MalaParams, grad_clip_norm: float) -> MalaParams:
        """
        Update the gradient clipping norm in the MALA parameters.
        """
        return params.replace(grad_clip_norm=grad_clip_norm)

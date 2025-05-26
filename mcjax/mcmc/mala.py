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
    cov_type: str  # "diag" or "full"


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


def _forward(method_name):
    def wrapper(self, *args, **kwargs):
        return getattr(self.base, method_name)(*args, **kwargs)
    return wrapper


@struct.dataclass
class MALAKernel:
    """
    Metropolis Adjusted Langevin Algorithm (MALA) kernel.
    """
    log_prob: Callable[[jnp.ndarray], jnp.ndarray] = struct.field(pytree_node=False)
    cov_type: str = struct.field(pytree_node=False)
    base: MarkovKernel = struct.field(pytree_node=False)

    # Forwarding interface
    init_state = _forward("init_state_fn")
    init_params = _forward("init_params")
    step = _forward("step")
    run_mcmc = _forward("run_mcmc")

    @classmethod
    def create(
        cls,
        log_prob: Callable[[jnp.ndarray], jnp.ndarray],
        grad_log_prob: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        cov_type: str = "diag"
    ) -> "MALAKernel":
        assert cov_type in {"diag", "full"}, "cov_type must be 'diag' or 'full'"

        value_and_grad = (
            jax.value_and_grad(log_prob)
            if grad_log_prob is None
            else lambda x: (log_prob(x), grad_log_prob(x))
        )

        def init_state_fn(x0: jnp.ndarray) -> MalaState:
            logp, grad = value_and_grad(x0)
            return MalaState(x=x0, log_prob=logp, grad=grad)

        def init_params(x0: jnp.ndarray, step_size: float, cov: Optional[jnp.ndarray] = None) -> MalaParams:
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
                cov_inv = 1.0 / cov
            elif cov_type == "full":
                assert cov.ndim == 2 and cov.shape == (dim, dim), "Full covariance must be (D, D)"
                scale = jnp.linalg.cholesky(cov)
                cov_inv = jnp.linalg.inv(cov)
            else:
                raise ValueError(f"Unsupported cov_type: {cov_type}")

            return MalaParams(step_size=step_size, scale=scale, cov_inv=cov_inv, cov_type=cov_type)

        def compute_drift_and_diffusion(grad, noise, params):
            eps = params.step_size
            if params.cov_type == "diag":
                drift = eps * (params.scale ** 2 * grad)
                diffusion = jnp.sqrt(2 * eps) * (params.scale * noise)
            elif params.cov_type == "full":
                cov = params.scale @ params.scale.T
                drift = eps * (cov @ grad)
                diffusion = jnp.sqrt(2 * eps) * (params.scale @ noise)
            else:
                raise ValueError(f"Unsupported cov_type: {params.cov_type}")
            return drift, diffusion

        def compute_log_accept_ratio(x, x_prop, grad_x, grad_prop, log_p_x, log_p_prop, params):
            eps = params.step_size
            dx = x_prop - x

            if params.cov_type == "diag":
                dx_fwd = dx - eps * (params.scale ** 2 * grad_x)
                dx_bwd = -dx - eps * (params.scale ** 2 * grad_prop)
                q_fwd = jnp.sum(params.cov_inv * dx_fwd ** 2)
                q_bwd = jnp.sum(params.cov_inv * dx_bwd ** 2)
            elif params.cov_type == "full":
                cov_inv = params.cov_inv
                cov = params.scale @ params.scale.T
                dx_fwd = dx - eps * (cov @ grad_x)
                dx_bwd = -dx - eps * (cov @ grad_prop)
                q_fwd = dx_fwd @ cov_inv @ dx_fwd
                q_bwd = dx_bwd @ cov_inv @ dx_bwd
            else:
                raise ValueError(f"Unsupported cov_type: {params.cov_type}")

            log_q_ratio = (q_fwd - q_bwd) / (4 * eps)
            return log_p_prop - log_p_x + log_q_ratio

        def step(key, state, params):
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

        return cls(log_prob=log_prob, cov_type=cov_type, base=base)



# def mala_kernel(
#     log_prob: Callable[[jnp.ndarray], jnp.ndarray],
#     step_size: float,
#     grad_log_prob: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     cov: Optional[jnp.ndarray] = None,
#     cov_type: Optional[str] = None
# ) -> MarkovKernel:
#     """
#     Construct the MALA kernel with optional user-supplied gradient and covariance matrix.
#     Returns a functional MarkovKernel object with init, step, summarize.
#     """

#     value_and_grad = (
#         jax.value_and_grad(log_prob)
#         if grad_log_prob is None
#         else lambda x: (log_prob(x), grad_log_prob(x))
#     )

#     def init(
#             x0: jnp.ndarray,
#             cov=cov,
#             cov_type=cov_type,
#             ) -> Tuple[MalaState, MalaParams]:
#         dim = x0.shape[-1]

#         if cov is None:
#             if cov_type is None or cov_type == "diag":
#                 cov_type_ = "diag"
#                 scale = jnp.ones(dim)
#             elif cov_type == "full":
#                 cov_type_ = "full"
#                 scale = jnp.eye(dim)
#             else:
#                 raise ValueError(f"Unsupported cov_type: {cov_type}")
#         else:
#             cov = jnp.asarray(cov)
#             if cov_type is None:
#                 if cov.ndim == 1:
#                     cov_type_ = "diag"
#                     scale = jnp.sqrt(cov)
#                 elif cov.ndim == 2:
#                     assert cov.shape == (dim, dim), "Full covariance shape mismatch"
#                     cov_type_ = "full"
#                     scale = jnp.linalg.cholesky(cov)
#                 else:
#                     raise ValueError("cov must be 1D or 2D")
#             else:
#                 assert cov_type in {"diag", "full"}, "cov_type must be 'diag' or 'full'"
#                 if cov_type == "diag":
#                     assert cov.ndim == 1, "cov_type 'diag' requires a 1D vector"
#                     cov_type_ = "diag"
#                     scale = jnp.sqrt(cov)
#                 elif cov_type == "full":
#                     assert cov.ndim == 2 and cov.shape == (dim, dim), \
#                         "cov_type 'full' requires a (dim, dim) matrix"
#                     cov_type_ = "full"
#                     scale = jnp.linalg.cholesky(cov)

#         if cov_type_ == "diag":
#             cov_inv = 1.0 / (scale ** 2)
#         else:  # full
#             cov = scale @ scale.T
#             cov_inv = jnp.linalg.inv(cov)

#         logp, grad = value_and_grad(x0)
#         return (
#             MalaState(x=x0, log_prob=logp, grad=grad),
#             MalaParams(step_size=step_size, scale=scale, cov_inv=cov_inv, cov_type=cov_type_)
#         )

#     def compute_drift_and_diffusion(
#         grad: jnp.ndarray,
#         noise: jnp.ndarray,
#         params: MalaParams
#     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         eps = params.step_size
#         if params.cov_type == "diag":
#             drift = eps * (params.scale ** 2 * grad)
#             diffusion = jnp.sqrt(2 * eps) * (params.scale * noise)
#         elif params.cov_type == "full":
#             cov = params.scale @ params.scale.T
#             drift = eps * (cov @ grad)
#             diffusion = jnp.sqrt(2 * eps) * (params.scale @ noise)
#         else:
#             raise ValueError(f"Unsupported cov_type: {params.cov_type}")
#         return drift, diffusion

#     def compute_log_accept_ratio(
#         x: jnp.ndarray,
#         x_prop: jnp.ndarray,
#         grad_x: jnp.ndarray,
#         grad_prop: jnp.ndarray,
#         log_p_x: jnp.ndarray,
#         log_p_prop: jnp.ndarray,
#         params: MalaParams
#     ) -> jnp.ndarray:
#         eps = params.step_size
#         dx = x_prop - x

#         if params.cov_type == "diag":
#             dx_fwd = dx - eps * (params.scale ** 2 * grad_x)
#             dx_bwd = -dx - eps * (params.scale ** 2 * grad_prop)
#             q_fwd = jnp.sum(params.cov_inv * dx_fwd ** 2)
#             q_bwd = jnp.sum(params.cov_inv * dx_bwd ** 2)
#         elif params.cov_type == "full":
#             cov_inv = params.cov_inv
#             cov = params.scale @ params.scale.T
#             dx_fwd = dx - eps * (cov @ grad_x)
#             dx_bwd = -dx - eps * (cov @ grad_prop)
#             q_fwd = dx_fwd @ cov_inv @ dx_fwd
#             q_bwd = dx_bwd @ cov_inv @ dx_bwd
#         else:
#             raise ValueError(f"Unsupported cov_type: {params.cov_type}")

#         log_q_ratio = (q_fwd - q_bwd) / (4 * eps)
#         return log_p_prop - log_p_x + log_q_ratio

#     def step(
#         key: jax.Array,
#         state: MalaState,
#         params: MalaParams
#     ) -> Tuple[MalaState, MalaStats]:
#         x = state.x
#         grad_x = state.grad
#         log_p_x = state.log_prob

#         key, key_prop, key_accept = jr.split(key, 3)
#         noise = jr.normal(key_prop, shape=x.shape)

#         drift, diffusion = compute_drift_and_diffusion(grad_x, noise, params)
#         x_prop = x + drift + diffusion
#         log_p_prop, grad_prop = value_and_grad(x_prop)

#         log_ratio = compute_log_accept_ratio(
#             x=x,
#             x_prop=x_prop,
#             grad_x=grad_x,
#             grad_prop=grad_prop,
#             log_p_x=log_p_x,
#             log_p_prop=log_p_prop,
#             params=params
#         )

#         accept_MH = jnp.exp(jnp.minimum(0.0, log_ratio))
#         u = jr.uniform(key_accept)
#         is_accept = u < accept_MH

#         x_new = jnp.where(is_accept, x_prop, x)
#         grad_new = jnp.where(is_accept, grad_prop, grad_x)
#         logp_new = jnp.where(is_accept, log_p_prop, log_p_x)

#         return (
#             MalaState(x=x_new, log_prob=logp_new, grad=grad_new),
#             MalaStats(
#                 sq_jump=accept_MH * jnp.linalg.norm(x_prop - x) ** 2,
#                 is_accept=is_accept,
#                 accept_MH=accept_MH,
#             ),
#         )

#     return MarkovKernel(
#         init=init,
#         step=step,
#         summarize=summarize_stats_traj
#     )

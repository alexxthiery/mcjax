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
# class MalaState(TypedDict):
#     """ State storing the current state of the MALA kernel 
#     x: current point
#     logdensity: pdf of the proposal function Z \propto q(x,.) = N(x-\epsilon \grad(V(x)), 2\epsilon I)
#     """
#     x: jnp.ndarray
#     logdensity: jnp.ndarray

# class MalaStats(TypedDict):
#     """ Stores the statistics of MALA at each step
#     is_accept: whether the drawn z is accepted or not
#     accept_MH: the threshold "a" in Metropolis-Hastings
#     """
#     is_accept: jnp.ndarray
#     accept_MH: jnp.ndarray

# class Mala(MarkovKernel):

#     def __init__(self,
#                  *,
#                  logdensity:LogDensity,
#                  epsilon):
#         self.logdensity = logdensity
#         self._dim = logdensity.dim
#         self.epsilon = epsilon
        

#     def init_state(
#         self,
#         x_init: jnp.ndarray,     # initial point
#         ) -> MalaState:
#         """ Initialize the state of the RWM kernel """
#         # check the dimension of the initial point
#         assert x_init.shape == (self._dim,), "Invalid initial point"
        
#         state = MalaState(x=x_init, logdensity=self.logdensity(x_init))
#         return state
    
#     def step(self,
#              state:MalaState,
#              key
#              ) -> Tuple[MalaState,MalaStats]:
#         """
#         A single step of MALA sampling
#         """
#         # unpack the state and density function
#         x = state['x']
#         logtarget_current = state['logdensity']
        
#         # create a proposal
#         key, key_ = jr.split(key)
#         # \log(f(x)) \propto -V(x) 

#         x_prop = x + self.epsilon*self.logdensity.grad(x) + jr.normal(key_, (self._dim,)) *jnp.sqrt(2*self.epsilon)
#         logtarget_proposal = self.logdensity(x_prop)
        
#         # accept or reject
#         key, key_ = jr.split(key)
#         u = jr.uniform(key_)
#         log_f_ratio = logtarget_proposal - logtarget_current
#         log_q_ratio = (jnp.linalg.norm(x_prop - x - self.epsilon*self.logdensity.grad(x))**2 - \
#                        jnp.linalg.norm(x - x_prop - self.epsilon*self.logdensity.grad(x_prop))**2)\
#                         /(4*self.epsilon)
#         accept_MH = jnp.exp(jnp.minimum(0., log_f_ratio+log_q_ratio))
        
        
#         # metropolis-hastings acceptance
#         is_accept = u < accept_MH
#         x_new = jnp.where(is_accept, x_prop, x)
#         logdensity_new = jnp.where(
#                                 is_accept,
#                                 logtarget_proposal,
#                                 logtarget_current)
        
#         # create the new state
#         state_new = MalaState(x=x_new, logdensity=logdensity_new)
        
#         # store the statistics
#         statistics = MalaStats(
#                         is_accept=is_accept,
#                         accept_MH=accept_MH)
#         return state_new, statistics
    
#     def summarize_stats_traj(
#             self,
#             stats_traj: MalaStats,
#             ) -> Dict:
#         """ Summarize the statistics of the RWM trajectory """
#         acceptance_rate = jnp.mean(stats_traj['accept_MH'])
#         n_accepted = jnp.sum(stats_traj['is_accept'])
#         stats_summary = {
#             'acceptance_rate': acceptance_rate,
#             'n_accepted': n_accepted
#         }
#         return stats_summary

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
        # grad_logdensity: Callable[[jnp.ndarray], jnp.ndarray],
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
        # if grad_logdensity is None:
        #     grad_logdensity = jax.grad(logdensity)
        if grad_logdensity is None:
            self.value_and_grad = jax.value_and_grad(logdensity)
        else:
            def value_and_grad(x):
                return logdensity(x), grad_logdensity(x)
            self.value_and_grad = value_and_grad
        self.logdensity = logdensity
        self.grad_logdensity = grad_logdensity
        self.step_size = step_size
        self.cov = cov  # user-supplied
        self.L = None  # Cholesky of cov
        self.cov_inv = None  # Inverse of cov

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

        # grad_logp_prop = self.grad_logdensity(x_prop)
        # logtarget_proposal = self.log_density_fn(x_prop)
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
        #grad_new = jnp.where(is_accept[:, None], grad_logp_prop, grad_logp_x)
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
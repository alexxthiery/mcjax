import jax
import jax.numpy as jnp
import jax.random as jr
from typing import TypedDict, Tuple, Dict
from mcjax.proba.density import LogDensity
from .markov import MarkovKernel

from dataclasses import dataclass
from flax import struct

# ==================================
# Random Walk Metropolis-Hastings
# with Gaussian proposal distribution
# ==================================

@struct.dataclass
class RwmState:
    """ State storing the current state of the RWM kernel """
    x: jnp.ndarray
    logdensity: jnp.ndarray

@struct.dataclass
class RwmStats:
    """ Stores the statistics of RWM steps """
    sq_jump: jnp.ndarray
    is_accept: jnp.ndarray
    accept_MH: jnp.ndarray
    step_size: float
    acc_rate: float


class Rwm(MarkovKernel):
    """ Random Walk Metropolis-Hastings kernel
    with Gaussian proposal distribution
    
    Usage:
    -------
    logtarget = LogDensity(...)
    rwm = Rwm(logtarget=logtarget, step_size=0.1)
    
    # initialize the state
    state_init = rwm.init_state(x_init)    
    
    # run MCMC
    mcmc_output = rwm.run_mcmc(
                        state_init=state_init,
                        n_iter=1000,
                        key=key)
    """
    def __init__(
                self,
                *,
                logtarget: LogDensity,  # logtarget distribution
                step_size: float,       # step size
                cov=None,               # covariance matrix
                ):
        self.logtarget = logtarget
        self.step_size = step_size
        self.dim = logtarget.dim
        
        # create covariance matrix of the proposals
        if cov is None:
            self.cov = jnp.eye(self.dim)
        elif jnp.ndim(cov) == 1:
            assert len(cov) == self.dim, "Invalid covariance matrix"
            self.cov = jnp.diag(cov)
        elif jnp.ndim(cov) == 2:
            assert cov.shape == (self.dim, self.dim), "Invalid covariance matrix"
            self.cov = cov
        else:
            raise ValueError("Invalid covariance matrix")
        
    def init_state(
            self,
            x_init: jnp.ndarray,     # initial point
            ) -> RwmState:
        """ Initialize the state of the RWM kernel """
        # check the dimension of the initial point
        assert x_init.shape == (self.dim,), "Invalid initial point"
        
        state = RwmState(x=x_init, logdensity=self.logtarget(x_init))
        return state
    
    def step(
            self,
            state: RwmState,    # current state
            key: jax.Array,     # random key
            step_size: float    # step size
            ) -> Tuple[RwmState, RwmStats]:
        """ Perform a single step of the RWM kernel """                
        # unpack the state and key
        x = state.x
        logtarget_current = state.logdensity
        
        # create a proposal
        key, key_ = jr.split(key)
        x_prop = x + jr.normal(key_, (x.shape)) * step_size
        logtarget_proposal = self.logtarget.batch(x_prop)
        
        # accept or reject for a batch of samples
        log_ratio = logtarget_proposal - logtarget_current
        accept_MH = jnp.exp(jnp.minimum(0., log_ratio))

        # square jump distance statistics (for a batch of samples)
        sq_jump = (x_prop - x)**2
        
        # metropolis-hastings acceptance (for a batch of samples)
        key, key_ = jr.split(key)
        u = jr.uniform(key_, shape=(x.shape[0],)) 
        is_accept = u < accept_MH
        is_accept = is_accept[:, None]
        x_new = jnp.where(is_accept, x_prop, x)
        
        logdensity_new = self.logtarget.batch(x_new)
        # create the new state
        state_new = RwmState(x=x_new, logdensity=logdensity_new)
        
        acc_rate = jnp.mean(is_accept)
        # store the statistics
        statistics = RwmStats(
                        sq_jump=sq_jump,
                        is_accept=is_accept,
                        accept_MH=accept_MH,
                        step_size=step_size,
                        acc_rate=acc_rate)
        return state_new, statistics
    
    def adaptive_step(self, state, key, max_iter=5):
        '''
        Take a step with adaptive step size: reiterate until the acceptance rate is within [0.2,0.5]
        '''
        key, key_ = jr.split(key)
        state, stats = self.step(state, key_, self.step_size)

        def cond_fun(carry):
            state, iter, key, stats = carry
            acc = stats.acc_rate
            return jnp.logical_and(~((acc >= 0.2) & (acc <= 0.5)), iter < max_iter)

        def body_fun(carry):
            state, iter, key, stats = carry
            step_size = stats.step_size
            key, key_ = jr.split(key)
            state_new, stats_new = self.step(state, key_, step_size)
            acc_rate = stats_new.acc_rate
            eta = 0.5; acc_target= 0.234
            new_step_size = jnp.exp(jnp.log(step_size) + eta * (acc_rate - acc_target))
            stats_new = RwmStats(sq_jump=stats_new.sq_jump, \
                                is_accept=stats_new.is_accept, \
                                accept_MH=stats_new.accept_MH, \
                                step_size=new_step_size, \
                                acc_rate=acc_rate)
            return (state_new, iter+1, key, stats_new)
        
        carry = (state, 0, key, stats)
        state, _, _, stats = jax.lax.while_loop(cond_fun, body_fun, carry)
        return state, stats
    
    def summarize_stats_traj(
            self,
            stats_traj: RwmStats,
            ) -> Dict:
        """ Summarize the statistics of the RWM trajectory """
        acceptance_rate = jnp.mean(stats_traj['accept_MH'])
        n_accepted = jnp.sum(stats_traj['is_accept'])
        sq_jump = jnp.mean(stats_traj['sq_jump'])
        stats_summary = {
            'acceptance_rate': acceptance_rate,
            'n_accepted': n_accepted,
            'sq_jump': sq_jump,
        }
        return stats_summary

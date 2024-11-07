import jax
import jax.numpy as jnp
import jax.random as jr
from typing import TypedDict, Tuple, Dict
from mcjax.proba.density import LogDensity
from .markov import MarkovKernel


# ==================================
# Random Walk Metropolis-Hastings
# with Gaussian proposal distribution
# ==================================
class RwmState(TypedDict):
    """ State storing the current state of the RWM kernel """
    x: jnp.ndarray
    logdensity: jnp.ndarray


class RwmStats(TypedDict):
    """ Stores the statistics of RWM steps """
    sq_jump: jnp.ndarray
    is_accept: jnp.ndarray
    accept_MH: jnp.ndarray


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
            ) -> Tuple[RwmState, RwmStats]:
        """ Perform a single step of the RWM kernel """                
        # unpack the state and key
        x = state['x']
        logtarget_current = state['logdensity']
        
        # create a proposal
        key, key_ = jr.split(key)
        x_prop = x + jr.normal(key_, (self.dim,)) * self.step_size
        logtarget_proposal = self.logtarget(x_prop)
        
        # accept or reject
        key, key_ = jr.split(key)
        u = jr.uniform(key_)
        log_ratio = logtarget_proposal - logtarget_current
        accept_MH = jnp.exp(jnp.minimum(0., log_ratio))
        
        # square jump distance statistics
        sq_jump = accept_MH * jnp.linalg.norm(x_prop - x)**2
        
        # metropolis-hastings acceptance
        is_accept = u < accept_MH
        x_new = jnp.where(is_accept, x_prop, x)
        logdensity_new = jnp.where(
                                is_accept,
                                logtarget_proposal,
                                logtarget_current)
        
        # create the new state
        state_new = RwmState(x=x_new, logdensity=logdensity_new)
        
        # store the statistics
        statistics = RwmStats(
                        sq_jump=sq_jump,
                        is_accept=is_accept,
                        accept_MH=accept_MH)
        return state_new, statistics
    
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

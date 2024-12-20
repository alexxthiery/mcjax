import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple, TypedDict, Dict

from mcjax.proba.gaussian import IsotropicGauss
from mcjax.proba.density import LogDensity
from .markov import MarkovKernel

# ========================
# Metropolis Adjusted Langevin Algorithm
# ========================
class MalaState(TypedDict):
    """ State storing the current state of the MALA kernel 
    x: current point
    logdensity: pdf of the proposal function Z \propto q(x,.) = N(x-\epsilon \grad(V(x)), 2\epsilon I)
    """
    x: jnp.ndarray
    logdensity: jnp.ndarray

class MalaStats(TypedDict):
    """ Stores the statistics of MALA at each step
    is_accept: whether the drawn z is accepted or not
    accept_MH: the threshold "a" in Metropolis-Hastings
    """
    is_accept: jnp.ndarray
    accept_MH: jnp.ndarray

class Mala(MarkovKernel):

    def __init__(self,
                 *,
                 logdensity:LogDensity,
                 epsilon):
        self.logdensity = logdensity
        self._dim = logdensity.dim
        self.epsilon = epsilon
        

    def init_state(
        self,
        x_init: jnp.ndarray,     # initial point
        ) -> MalaState:
        """ Initialize the state of the RWM kernel """
        # check the dimension of the initial point
        assert x_init.shape == (self._dim,), "Invalid initial point"
        
        state = MalaState(x=x_init, logdensity=self.logdensity(x_init))
        return state
    
    def step(self,
             state:MalaState,
             key
             ) -> Tuple[MalaState,MalaStats]:
        """
        A single step of MALA sampling
        """
        # unpack the state and density function
        x = state['x']
        logtarget_current = state['logdensity']
        
        # create a proposal
        key, key_ = jr.split(key)
        # \log(f(x)) \propto -V(x) 

        x_prop = x + self.epsilon*self.logdensity.grad_batch(x) + jr.normal(key_, (x.shape)) *jnp.sqrt(2*self.epsilon)
        logtarget_proposal = self.logdensity.batch(x_prop)
        
        # accept or reject
        key, key_ = jr.split(key)
        u = jr.uniform(key_, shape=(x.shape[0],)) 
        log_f_ratio = logtarget_proposal - logtarget_current
        log_q_ratio = (jnp.linalg.norm(x_prop - x - self.epsilon*self.logdensity.grad(x))**2 - \
                       jnp.linalg.norm(x - x_prop - self.epsilon*self.logdensity.grad(x_prop))**2)\
                        /(4*self.epsilon)
        accept_MH = jnp.exp(jnp.minimum(0., log_f_ratio+log_q_ratio))
        is_accept = u < accept_MH
        is_accept = is_accept[:, None]
        x_new = jnp.where(is_accept, x_prop, x)

        logdensity_new = jnp.where(
                                is_accept,
                                logtarget_proposal,
                                logtarget_current)
        
        # create the new state
        state_new = MalaState(x=x_new, logdensity=logdensity_new)
        
        # store the statistics
        statistics = MalaStats(
                        is_accept=is_accept,
                        accept_MH=accept_MH)
        return state_new, statistics
    
    def summarize_stats_traj(
            self,
            stats_traj: MalaStats,
            ) -> Dict:
        """ Summarize the statistics of the RWM trajectory """
        acceptance_rate = jnp.mean(stats_traj['accept_MH'])
        n_accepted = jnp.sum(stats_traj['is_accept'])
        stats_summary = {
            'acceptance_rate': acceptance_rate,
            'n_accepted': n_accepted
        }
        return stats_summary
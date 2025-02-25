import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple, TypedDict, Dict

from mcjax.proba.gaussian import IsotropicGauss
from mcjax.proba.density import LogDensity
from .markov import MarkovKernel

from dataclasses import dataclass
from flax import struct

# ========================
# Metropolis Adjusted Langevin Algorithm
# ========================
@struct.dataclass   
class MalaState:
    """ State storing the current state of the MALA kernel 
    x: current point
    logdensity: pdf of the proposal function Z \propto q(x,.) = N(x-\epsilon \grad(V(x)), 2\epsilon I)
    """
    x: jnp.ndarray
    logdensity: jnp.ndarray

@struct.dataclass  
class MalaStats:
    """ Stores the statistics of MALA at each step
    is_accept: whether the drawn z is accepted or not
    accept_MH: the threshold "a" in Metropolis-Hastings
    """
    is_accept: jnp.ndarray
    accept_MH: jnp.ndarray
    step_size: float
    acc_rate: float
    

class Mala(MarkovKernel):

    def __init__(self,
                 *,
                 logtarget:LogDensity,
                 step_size: float
                 ):
        self.logtarget = logtarget
        self._dim = logtarget.dim
        self.step_size = step_size
        

    def init_state(
        self,
        x_init: jnp.ndarray,     # initial point
        ) -> MalaState:
        """ Initialize the state of the RWM kernel """
        # check the dimension of the initial point
        assert x_init.shape == (self._dim,), "Invalid initial point"
        
        state = MalaState(x=x_init, logdensity=self.logtarget(x_init))
        return state
    
    def step(self,args):
        """
        A single step of MALA sampling
        """
        state, key, step_size, _ = args 
        # unpack the state and density function
        x = state.x
        logtarget_current = state.logdensity

        # create a proposal
        key, key_ = jr.split(key)
        # \log(f(x)) \propto -V(x) 

        x_prop = x + step_size*self.logtarget.grad_batch(x) + jr.normal(key_, (x.shape)) *jnp.sqrt(2*self.step_size)
        logtarget_proposal = self.logtarget.batch(x_prop)
        
        # accept or reject
        key, key_ = jr.split(key)
        u = jr.uniform(key_, shape=(x.shape[0],)) 
        log_f_ratio = logtarget_proposal - logtarget_current
        log_q_ratio = (jnp.linalg.norm(x_prop - x - step_size*self.logtarget.grad_batch(x), axis=1)**2 - \
                       jnp.linalg.norm(x - x_prop - step_size*self.logtarget.grad_batch(x_prop),axis=1)**2)\
                        /(4*step_size)

        accept_MH = jnp.exp(jnp.minimum(0., log_f_ratio+log_q_ratio))
        is_accept = u < accept_MH
        logdensity_new = jnp.where(
                                is_accept,
                                logtarget_proposal,
                                logtarget_current)
        
        is_accept = is_accept[:, None]
        x_new = jnp.where(is_accept, x_prop, x)

        
        # create the new state
        state_new = MalaState(x=x_new, logdensity=logdensity_new)
    
        
        # # update the step size
        acc_rate = jnp.mean(is_accept)

        # store the statistics
        statistics = MalaStats(
                        is_accept=is_accept,
                        accept_MH=accept_MH,
                        step_size = step_size,
                        acc_rate=acc_rate)
    

        return state_new, statistics
    
    def adaptive_step(self, args):
        '''
        Take a step with adaptive step size: reiterate until the acceptance rate is within [0.2,0.5]
        '''
        state, key, _, max_iter = args
        key, key_ = jr.split(key)
        args = (state, key_, self.step_size,0)
        state, stats = self.step(args)

        def cond_fun(carry):
            state, iter, key, stats = carry
            acc = stats.acc_rate
            return jnp.logical_and(~((acc >= 0.4) & (acc <= 0.7)), iter < max_iter)

        def body_fun(carry):
            state, iter, key, stats = carry
            step_size = stats.step_size
            key, key_ = jr.split(key)
            args = (state, key_, step_size,0)
            state_new, stats_new = self.step(args)
            acc_rate = stats_new.acc_rate
            eta = 0.5; acc_target= 0.574
            new_step_size = jnp.exp(jnp.log(step_size) + eta * (acc_rate - acc_target))
            stats_new = MalaStats(is_accept=stats_new.is_accept, \
                                accept_MH=stats_new.accept_MH, \
                                step_size=new_step_size, \
                                acc_rate=acc_rate)
            return (state_new, iter+1, key, stats_new)
        
        carry = (state, 0, key, stats)
        state, _, _, stats = jax.lax.while_loop(cond_fun, body_fun, carry)
        return state, stats




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
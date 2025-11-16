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
    step_size: jnp.ndarray
    acc_rate: jnp.ndarray
    

class Mala(MarkovKernel):

    def __init__(self,
                 *,
                 logtarget:LogDensity,
                 step_size: float,
                 mass_inv: jnp.ndarray = None
                 ):
        self.logtarget = logtarget
        self._dim = logtarget.dim
        self.step_size = step_size
        # check mass_inv: mass_inv is either None, a vector (diag) or a full matrix (precision = M^{-1})
        assert mass_inv is None or (mass_inv.ndim == 1 and mass_inv.shape[0] == self._dim) \
            or (mass_inv.ndim == 2 and mass_inv.shape == (self._dim, self._dim)), "Invalid mass_inv shape"
        self.mass_inv = mass_inv
        
    def _build_precond(self, step_size):
        if self.mass_inv is None: # Identity mass matrix
            Minv_mat = jnp.eye(self._dim)
            M_mat = jnp.eye(self._dim)
        else:
            if self.mass_inv.ndim == 1:
                # self.mass_inv is a vector of diagonal precision entries
                Minv_mat = jnp.diag(self.mass_inv)          # precision matrix
                M_mat = jnp.diag(1.0 / self.mass_inv)       # mass matrix
            else:
                # full precision matrix provided
                Minv_mat = self.mass_inv                    # precision matrix
                M_mat = jnp.linalg.inv(Minv_mat)            # mass matrix

        # proposal covariance Sigma = 2 * step_size * Minv_mat
        Sigma = 2.0 * step_size * Minv_mat
        # cholesky for sampling: L @ z ~ N(0, Sigma)
        L = jnp.linalg.cholesky(Sigma + 1e-8 * jnp.eye(self._dim))
        return Minv_mat, M_mat, Sigma, L
        

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
        state, key, step_size,_ = args 
        step_size = jnp.array(step_size, dtype=jnp.float32)
        # unpack the state and density function
        x = state.x # x: (num_particles, (dim))
        logtarget_current = state.logdensity

        # Setup Mass Matrix and Preconditioning Components
        Minv_mat, M_mat, Sigma, L = self._build_precond(step_size)
        
        
        # create a proposal
        key, key_ = jr.split(key)

        grads = self.logtarget.grad_batch(x)          
        drift = step_size * jax.vmap(lambda g: Minv_mat @ g)(grads)

        z = jr.normal(key_, shape=x.shape)
        noise = jax.vmap(lambda z_i: L @ z_i)(z)

        x_prop = x + drift + noise
        logtarget_proposal = self.logtarget.batch(x_prop)

        # Acceptance Ratio Calculation
        mu_curr = x + step_size * jax.vmap(lambda g: Minv_mat @ g)(grads)
        grads_prop = self.logtarget.grad_batch(x_prop)
        mu_prop = x_prop + step_size * jax.vmap(lambda g: Minv_mat @ g)(grads_prop)

        v1 = x - mu_prop              
        v2 = x_prop - mu_curr

        quad_fn = lambda v: jnp.einsum('i,ij,j->', v, M_mat, v)
        dist_sq_1 = jax.vmap(quad_fn)(v1)
        dist_sq_2 = jax.vmap(quad_fn)(v2)

        log_q_ratio = (1.0 / (4.0 * step_size)) * (dist_sq_2 - dist_sq_1)

        
        # accept or reject
        key, key_ = jr.split(key)
        u = jr.uniform(key_, shape=(x.shape[0],)) 
        log_f_ratio = logtarget_proposal - logtarget_current


        accept_MH = jnp.exp(jnp.minimum(0., log_f_ratio+log_q_ratio))
        is_accept = u < accept_MH
        logdensity_new = jnp.where(
                                is_accept,
                                logtarget_proposal,
                                logtarget_current)
        
        is_accept_col = is_accept[:, None]
        x_new = jnp.where(is_accept_col, x_prop, x)

        
        # create the new state
        state_new = MalaState(x=x_new, logdensity=logdensity_new)
    
        
        # # update the step size
        acc_rate = jnp.mean(is_accept)

        # store the statistics
        statistics = MalaStats(
                        is_accept=is_accept_col,
                        accept_MH=accept_MH,
                        step_size = step_size,
                        acc_rate=acc_rate)
    

        return state_new, statistics
    
    def step_single(self, args):
        """
        Perform a single step for one sample
        """
        state, key, step_size = args   
        step_size = jnp.array(step_size, dtype=jnp.float32)       
        # unpack the state and density function
        x = state.x # x: (num_particles, (dim))
        logtarget_current = state.logdensity

        # Setup Mass Matrix and Preconditioning Components
        Minv_mat, M_mat, Sigma, L = self._build_precond(step_size)


        # create a proposal
        key, key_ = jr.split(key)
        grad = self.logtarget.grad(x)
        drift = step_size * (Minv_mat @ grad)

        z = jr.normal(key_, shape=x.shape)
        noise = L @ z

        x_prop = x + drift + noise
        logtarget_proposal = self.logtarget.logdensity(x_prop)

        # Acceptance Ratio Calculation
        mu_curr = x + step_size * (Minv_mat @ grad)
        grad_prop = self.logtarget.grad(x_prop)
        mu_prop = x_prop + step_size * (Minv_mat @ grad_prop)

        v1 = x - mu_prop
        v2 = x_prop - mu_curr

        dist_sq_1 = jnp.einsum('i,ij,j->', v1, M_mat, v1)
        dist_sq_2 = jnp.einsum('i,ij,j->', v2, M_mat, v2)

        log_q_ratio = (0.25 / step_size) * (dist_sq_2 - dist_sq_1)

        
        # accept or reject
        key, key_ = jr.split(key)
        u = jr.uniform(key_, shape=())
        log_f_ratio = logtarget_proposal - logtarget_current

        # jax.debug.print("log_f_ratio: {lf}, log_q_ratio: {lq}", lf=log_f_ratio, lq=log_q_ratio)

        accept_MH = jnp.exp(jnp.minimum(0., log_f_ratio+log_q_ratio))

        is_accept = u < accept_MH
        logdensity_new = jnp.where(
                                is_accept,
                                logtarget_proposal,
                                logtarget_current)
        
        x_new = jnp.where(is_accept, x_prop, x)

        
        # create the new state
        state_new = MalaState(x=x_new, logdensity=logdensity_new)
        acc_rate = is_accept.astype(jnp.float32)

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
        args = (state, key_, self.step_size,_)
        state, stats = self.step(args)

        def cond_fun(carry):
            state, iter, key, stats = carry
            acc = stats.acc_rate
            return jnp.logical_and(~((acc >= 0.4) & (acc <= 0.7)), iter < max_iter)

        def body_fun(carry):
            state, iter, key, stats = carry
            step_size = stats.step_size
            key, key_ = jr.split(key)
            args = (state, key_, step_size,_)
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
        acceptance_rate = jnp.mean(stats_traj.accept_MH)
        n_accepted = jnp.sum(stats_traj.is_accept)
        stats_summary = {
            'acceptance_rate': acceptance_rate,
            'n_accepted': n_accepted,
        }
        return stats_summary
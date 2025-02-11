import jax
import jax.numpy as jnp
import jax.random as jr
from typing import TypedDict, Tuple, Dict, NamedTuple
from mcjax.proba.density import LogDensity,LogDensityGeneral
from ..mcmc.markov import MarkovKernel
from mcjax.mcmc.rwm import Rwm, RwmState
from mcjax.mcmc.mala import Mala, MalaState
from mcjax.util.weights import ess_normalized_log_weight

import time
from dataclasses import dataclass

@dataclass
class SmcState:
    """ State storing the states of SMC """
    particles: jnp.ndarray  # particles in the state space
    weights: jnp.ndarray    # weights associated with each particle

class GeometricSMC():
    def __init__(
            self,
            *,
            log_gamma_0:LogDensity,
            log_gamma_T:LogDensity,
            coefs,
            step_size,
            num_substeps):
        
        self.coefs = coefs
        self.log_gamma_0 = log_gamma_0
        self.log_gamma_T = log_gamma_T
        self.step_size = step_size
        self.num_substeps = num_substeps # number of substeps for the MC kernel

    def step(
            self,
            t,
            mc_method: str,     # MC method: RWM or MALA
            state: jax.Array,    # current state
            key: jax.Array,     # random key
            ):
        ''' a single step of SMC: resampling, sample particles from (t-1) to t, compute weights'''
        ######## Compute weights with the ratio of densities: gamma_t(x_{t-1}) / gamma_{t-1}(x_{t-1})
        particles, weights = state
        updated_weights = self.compute_weights(t,particles)
        
        ############ Resampling 
        key, key_ = jr.split(key)
        new_particles = self.resample(particles, updated_weights, key_)
        
        ############ Sample particles from (t-1) to t using the RWM kernel
        mc_methods = {
            'RWM': self.random_walk_batch,
            'MALA': self.mala_batch
        }
        mc_function = mc_methods[mc_method]

        # array to store acc_rate and step_size
        acc_rate_arr = jnp.zeros(self.num_substeps)
        step_size_arr = jnp.zeros(self.num_substeps)

        # use fori_loop to perform multiple substeps
        carry = (new_particles, key, step_size_arr, acc_rate_arr)
        def body_fun(i, carry):
            particles, key, step_size_arr, acc_rate_arr = carry
            key, key_ = jr.split(key)
            new_particles, step_size, acc_rate = mc_function(t, particles, key_)
            step_size_arr = step_size_arr.at[i].set(step_size)
            acc_rate_arr = acc_rate_arr.at[i].set(acc_rate)
            return (new_particles, key, step_size_arr, acc_rate_arr)
        
        new_particles, _ , step_size_arr, acc_rate_arr = jax.lax.fori_loop(0, self.num_substeps, body_fun, carry)
        return (new_particles, updated_weights, step_size_arr, acc_rate_arr)
    
    def resample(self, particles: jnp.ndarray, weights: jnp.ndarray, key: jax.Array):
        """ Perform resampling step using the weights """
        # Normalize the weights
        normalized_weights = weights / jnp.sum(weights)

        # Resample particles based on their weights
        num_particles = particles.shape[0]
        key, key_ = jr.split(key)
        indices = jax.random.choice(key_, num_particles, shape=(num_particles,), p=normalized_weights)
        resampled_particles = particles[indices]
        # TODO: systematic resampling

        return resampled_particles
    
    # Random walk Metropolis kernel
    def random_walk_batch(self, t, particles: jnp.ndarray, key: jax.Array):
        logdensity = LogDensityGeneral(logdensity = lambda x: self.coefs[t] * self.log_gamma_T.logdensity(x) \
            + (1 - self.coefs[t]) * self.log_gamma_0.logdensity(x), dim=particles.shape[1])
        rwm = Rwm(logtarget=logdensity, step_size = self.step_size)
        state = RwmState(x=particles, logdensity=logdensity.batch(particles))
        new_particles,stats = rwm.adaptive_step(state, key)
        this_step_size = stats.step_size
        this_acc_rate = stats.acc_rate
        return new_particles.x, this_step_size, this_acc_rate

    # MALA kernel 
    def mala_batch(self, t, particles: jnp.ndarray, key: jax.Array):
        logdensity = LogDensityGeneral(logdensity = lambda x: self.coefs[t] * self.log_gamma_T.logdensity(x) \
            + (1 - self.coefs[t]) * self.log_gamma_0.logdensity(x), dim=particles.shape[1])
        mala = Mala(logtarget=logdensity, step_size = self.step_size)
        state = MalaState(x=particles, logdensity=logdensity.batch(particles))
        new_particles,stats = mala.adaptive_step(state, key)
        this_step_size = stats.step_size
        this_acc_rate = stats.acc_rate
        return new_particles.x, this_step_size, this_acc_rate

    def compute_weights(self, t, resampled_particle: jnp.ndarray):
        """ Compute the weights as the ratio of the target densities """
        # gamma_t = jnp.take(self.coefs, t)*self.log_gamma_T.batch(resampled_particle) \
        #     + (1-jnp.take(self.coefs, t))*self.log_gamma_0.batch(resampled_particle)
        gamma_t = self.coefs[t]*self.log_gamma_T.batch(resampled_particle) \
            + (1-self.coefs[t])*self.log_gamma_0.batch(resampled_particle)
        gamma_t_minus_1 = jnp.take(self.coefs, t-1)*self.log_gamma_T.batch(resampled_particle) \
            + (1-jnp.take(self.coefs, t-1))*self.log_gamma_0.batch(resampled_particle) 
        # weights = jnp.exp(gamma_t - gamma_t_minus_1) \
        #     * jnp.exp((jnp.take(self.coefs, t)-jnp.take(self.coefs, t-1))\
        #               *(self.log_gamma_T._log_Z-self.log_gamma_0._log_Z))
        weights = jnp.exp(gamma_t - gamma_t_minus_1) 
        return weights 


    def run(self,num_particles, key, mc_method):
        # Initialize the state at t=0 with particles sampled from the initial distribution        
        key, key_ = jr.split(key)
        initial_particles = self.log_gamma_0.sample(key_, num_particles)
        initial_weights = jnp.ones(num_particles) / num_particles  
        state = {"particles": initial_particles, "weights": initial_weights}
        # use fori_loop
        particles_arr = jnp.zeros((num_particles, self.log_gamma_0.dim, len(self.coefs)))
        particles_arr = particles_arr.at[:, :, 0].set(initial_particles)
        weights_arr = jnp.zeros((num_particles, len(self.coefs)))
        weights_arr = weights_arr.at[:, 0].set(initial_weights)
        step_size_arr = jnp.zeros((self.num_substeps, len(self.coefs)-1))
        acc_rate_arr = jnp.zeros((self.num_substeps, len(self.coefs)-1))

        carry = (particles_arr, weights_arr, key, step_size_arr, acc_rate_arr)
        def body_fun(t, carry):
            particles_arr, weights_arr, key, step_size_arr,acc_rate_arr = carry
            key, key_ = jr.split(key)
            state = self.step(t, mc_method, (particles_arr[:, :, t-1], weights_arr[:, t-1]), key)
            particles_arr = particles_arr.at[:, :, t].set(state[0])
            weights_arr = weights_arr.at[:, t].set(state[1])
            step_size_arr = step_size_arr.at[:, t-1].set(state[2])
            acc_rate_arr = acc_rate_arr.at[:, t-1].set(state[3])
            return (particles_arr, weights_arr, key, step_size_arr, acc_rate_arr)
        
        particles_arr, weights_arr, _, step_size_arr, acc_rate_arr = jax.lax.fori_loop(1, len(self.coefs), body_fun, carry)
        return particles_arr, weights_arr, step_size_arr, acc_rate_arr

    # Calculate adaptively the coefficient for the geometric SMC 
    def selfadaptive_run(self,num_particles, key, mc_method):
        key, key_ = jr.split(key)
        initial_particles = self.log_gamma_0.sample(key_, num_particles)
        initial_weights = jnp.ones(num_particles) / num_particles  
        state = {"particles": initial_particles, "weights": initial_weights}

        particles_arr = jnp.zeros((num_particles, self.log_gamma_0.dim, len(self.coefs)))
        particles_arr = particles_arr.at[:, :, 0].set(initial_particles)
        weights_arr = jnp.zeros((num_particles, len(self.coefs)))
        weights_arr = weights_arr.at[:, 0].set(initial_weights)
        step_size_arr = jnp.zeros((self.num_substeps, len(self.coefs)-1))
        acc_rate_arr = jnp.zeros((self.num_substeps, len(self.coefs)-1))

        carry = (self.coefs, particles_arr, weights_arr, key, step_size_arr, acc_rate_arr)
        def body_fun(t, carry):
            coefs, particles_arr, weights_arr, key, step_size_arr, acc_rate_arr = carry
            key, key_ = jr.split(key)
            state = self.step(t, mc_method, (particles_arr[:, :, t-1], weights_arr[:, t-1]), key)
            particles_arr = particles_arr.at[:, :, t].set(state[0])
            weights_arr = weights_arr.at[:, t].set(state[1])
            step_size_arr = step_size_arr.at[:, t-1].set(state[2])
            acc_rate_arr = acc_rate_arr.at[:, t-1].set(state[3])

            # update coefficients
            key, key_ = jr.split(key)
            new_coef = self.update_coef(t, coefs, state, key_, 0.5)
            coefs = coefs.at[t].set(new_coef)
            return (coefs, particles_arr, weights_arr, key, step_size_arr, acc_rate_arr)
        
        self.coefs, particles_arr, weights_arr, _, step_size_arr, acc_rate_arr = jax.lax.fori_loop(1, len(self.coefs), body_fun, carry)
        return particles_arr, weights_arr, step_size_arr, acc_rate_arr


    def update_coef(self, t, coefs, state, key, ess_normalized_target, tol = 1e-5):
        '''use target_ess_normalized to find the optimal coefficient by bissection
        If ESS_normalized(tmax*log_weights) >= ess_normalized_target, return tmax. '''
        particles, weights,_,_ = state
        coef_max = 1.0
        coef_min = jnp.take(coefs, t-1)
        coef_min_copy = coef_min

        # calculate the log_weights (log_weight) of: log_gamma_t(x_{t-1}) - log_gamma_0(x_{t-1})
        log_weights = self.log_gamma_T.batch(particles) - self.log_gamma_0.batch(particles)
        ess_tmax = ess_normalized_log_weight((coef_max-coef_min) * log_weights)
        
        def cond_fun(vars):
            coef_max, coef_min = vars
            return coef_max - coef_min > tol
        
        def body_fun(vars):
            coef_max, coef_min = vars
            coef = (coef_max + coef_min) / 2
            ess_new = ess_normalized_log_weight((coef- coef_min_copy) * log_weights)
            coef_max = jax.lax.cond(ess_new > ess_normalized_target, lambda _: coef_max, lambda _: coef, None)
            coef_min = jax.lax.cond(ess_new > ess_normalized_target, lambda _: coef, lambda _: coef_min, None)
            return coef_max, coef_min

        result = jax.lax.cond(ess_tmax >= ess_normalized_target, lambda _: (coef_max, coef_min),\
                               lambda _: jax.lax.while_loop(cond_fun, body_fun, (coef_max, coef_min)), None)
        return (result[0] + result[1]) / 2















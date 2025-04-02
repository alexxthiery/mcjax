import jax
import jax.numpy as jnp
import jax.random as jr
from typing import TypedDict, Tuple, Dict, NamedTuple
from mcjax.proba.density import LogDensity,LogDensityGeneral
from ..mcmc.markov import MarkovKernel
from mcjax.mcmc.rwm import Rwm, RwmState
from mcjax.mcmc.mala import Mala, MalaState
from mcjax.util.weights import ess_normalized_log_weight
from mcjax.util.resample import systematic_resample

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
            num_substeps,
            keep_particles = False):
        
        self.coefs = coefs
        self.log_gamma_0 = log_gamma_0
        self.log_gamma_T = log_gamma_T
        self.step_size = step_size
        self.num_substeps = num_substeps # number of substeps for the MC kernel
        self.keep_particles = keep_particles # if True, keep all particles in the SMC chain, otherwise only keep the last one

    def step(
            self,
            t,
            coefs,              # Temperature (ranging from [0,1])
            mc_method: str,     # MC method: RWM or MALA
            state: jax.Array,    # current state
            key: jax.Array,     # random key
            ):
        ''' a single step of SMC: resampling, sample particles from (t-1) to t, compute weights'''
        ######## Compute weights with the ratio of densities: gamma_t(x_{t-1}) / gamma_{t-1}(x_{t-1})
        particles, _ = state
        updated_log_weights = self.compute_weights(t,coefs,particles)
        
        ############ Resampling 
        key, key_ = jr.split(key)
        new_particles = self.resample(particles, updated_log_weights, key_)
        
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
        carry = (new_particles, key, self.step_size, step_size_arr, acc_rate_arr)
        def body_fun(i, carry):
            particles, key, step_size, step_size_arr, acc_rate_arr = carry
            key, key_ = jr.split(key)

            new_particles, step_size, acc_rate = mc_function(t, coefs, particles, key_, step_size, if_adjust_step_size = (i==0))
            step_size_arr = step_size_arr.at[i].set(step_size)
            acc_rate_arr = acc_rate_arr.at[i].set(acc_rate)
            return (new_particles, key, step_size, step_size_arr, acc_rate_arr)
        
        new_particles, _ ,_, step_size_arr, acc_rate_arr = jax.lax.fori_loop(0, self.num_substeps, body_fun, carry)
        return (new_particles, updated_log_weights, step_size_arr, acc_rate_arr)
    
    def resample(self, particles: jnp.ndarray, log_weights: jnp.ndarray, key: jax.Array):
        """ Perform resampling step using the weights """
        # Normalize the weights
        normalized_log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)

        # Resample particles based their weights by systematic resampling
        key, key_ = jr.split(key)
        indices = systematic_resample(key_, jnp.exp(normalized_log_weights))
        resampled_particles = particles[indices]

        return resampled_particles
    
    # Random walk Metropolis kernel
    def random_walk_batch(self, t, coefs, particles, key, step_size, if_adjust_step_size):
        logdensity = LogDensityGeneral(logdensity = lambda x: coefs[t] * self.log_gamma_T.logdensity(x) \
            + (1 - coefs[t]) * self.log_gamma_0.logdensity(x), dim=particles.shape[1])
        rwm = Rwm(logtarget=logdensity, step_size=step_size)
        state = RwmState(x=particles, logdensity=logdensity.batch(particles))
        max_iter = 5
        args = (state, key, step_size, max_iter)

        # Adaptive step size only at the first temperature
        new_particles, stats = jax.lax.cond(
        if_adjust_step_size,
        rwm.adaptive_step,
        rwm.step,
        operand=args
        )
        # self.step_size = stats.step_size # update the step size
        return new_particles.x, stats.step_size, stats.acc_rate

    # MALA kernel 
    def mala_batch(self, t, coefs, particles, key, step_size, if_adjust_step_size):
        logdensity = LogDensityGeneral(logdensity = lambda x: coefs[t] * self.log_gamma_T.logdensity(x) \
            + (1 - coefs[t]) * self.log_gamma_0.logdensity(x), dim=particles.shape[1])
        # Pilot RWM chain to estimate per-coordinate variances at π_t
        # pilot_steps = 20
        # rwm = Rwm(logtarget=logdensity, step_size=step_size)
        # state_rwm = RwmState(x=particles, logdensity=logdensity.batch(particles))
        # key_pilot = key
        # for _ in range(pilot_steps):
        #     state_rwm, _ = rwm.step((state_rwm, key_pilot, step_size, 0))
        #     key_pilot, _ = jr.split(key_pilot)
        # pilot_samples = state_rwm.x
        # var_pilot = jnp.var(pilot_samples, axis=0) + 1e-8
        # mass_inv = 1.0 / var_pilot    
        mass_inv = None

        # Create MALA with that frozen mass matrix
        mala = Mala(
            logtarget=logdensity,
            step_size=step_size,
            mass_inv=mass_inv
        )
        state = MalaState(x=particles, logdensity=logdensity.batch(particles))
        max_iter = 5
        args = (state, key, step_size, max_iter)

        # Adaptive step size only at the first temperature
        new_particles, stats = jax.lax.cond(
        if_adjust_step_size,
        mala.adaptive_step,
        mala.step,
        operand=args
        )
        # self.step_size = stats.step_size # update the step size
        return new_particles.x, stats.step_size, stats.acc_rate

    def compute_weights(self, t, coefs, resampled_particle: jnp.ndarray):
        """ Compute the log_weights as the ratio of the target densities """
        gamma_t = coefs[t]*self.log_gamma_T.batch(resampled_particle) \
            + (1-coefs[t])*self.log_gamma_0.batch(resampled_particle)
        gamma_t_minus_1 = coefs[t-1]*self.log_gamma_T.batch(resampled_particle) \
            + (1-coefs[t-1])*self.log_gamma_0.batch(resampled_particle) 
        log_weights = gamma_t - gamma_t_minus_1
        return log_weights 


    def run(self,num_particles, key, mc_method):
        # Initialize the state at t=0 with particles sampled from the initial distribution        
        key, key_ = jr.split(key)
        initial_particles = self.log_gamma_0.sample(key_, num_particles)
        initial_weights = jnp.zeros(num_particles)
        state = {"particles": initial_particles, "weights": initial_weights}
        # use fori_loop
        if self.keep_particles:
            particles_arr = jnp.zeros((num_particles, self.log_gamma_0.dim, len(self.coefs)))
            particles_arr = particles_arr.at[:, :, 0].set(initial_particles)
        else:
            particles_arr = initial_particles
        log_weights_arr = jnp.zeros((num_particles, len(self.coefs)))
        log_weights_arr = log_weights_arr.at[:, 0].set(initial_weights)
        step_size_arr = jnp.zeros((self.num_substeps, len(self.coefs)-1))
        acc_rate_arr = jnp.zeros((self.num_substeps, len(self.coefs)-1))

        carry = (particles_arr, log_weights_arr, key, step_size_arr, acc_rate_arr)
        def body_fun(t, carry):
            particles_arr, log_weights_arr, key, step_size_arr,acc_rate_arr = carry
            key, key_ = jr.split(key)
            prev_particles = (particles_arr[:, :, t-1]
                      if self.keep_particles
                      else particles_arr)
            state = self.step(t, self.coefs, mc_method, (prev_particles, log_weights_arr[:, t-1]), key)
            if self.keep_particles:
                particles_arr = particles_arr.at[:, :, t].set(state[0])
            else:
                particles_arr = state[0]
            log_weights_arr = log_weights_arr.at[:, t].set(state[1])
            step_size_arr = step_size_arr.at[:, t-1].set(state[2])
            acc_rate_arr = acc_rate_arr.at[:, t-1].set(state[3])
            return (particles_arr, log_weights_arr, key, step_size_arr, acc_rate_arr)
        
        particles_arr, log_weights_arr, _, step_size_arr, acc_rate_arr = jax.lax.fori_loop(1, len(self.coefs), body_fun, carry)
        return particles_arr, log_weights_arr, step_size_arr, acc_rate_arr

    # Calculate adaptively the coefficient for the geometric SMC 
    def selfadaptive_run(self,num_particles, key, mc_method, max_steps=100):
        tol = 1e-5
        key, key_ = jr.split(key)
        initial_particles = self.log_gamma_0.sample(key_, num_particles)
        initial_weights = jnp.zeros(num_particles)
        
        dim = self.log_gamma_0.dim
        num_substeps = self.num_substeps

        # Preallocate arrays for up to max_steps iterations.
        if self.keep_particles:
            particles_arr = jnp.zeros((num_particles, self.log_gamma_0.dim, len(self.coefs)))
            particles_arr = particles_arr.at[:, :, 0].set(initial_particles)
        else:
            particles_arr = initial_particles
        log_weights_arr  = jnp.zeros((num_particles, max_steps+1))
        step_size_arr = jnp.zeros((num_substeps, max_steps))
        acc_rate_arr  = jnp.zeros((num_substeps, max_steps))
        coefs         = jnp.zeros((max_steps+1,))
        
        # Set initial values.
        log_weights_arr   = log_weights_arr.at[:, 0].set(initial_weights)
        coefs         = coefs.at[0].set(0.0)

        # Carry holds: (current iteration t, coefs, particles_arr, weights_arr, key, step_size_arr, acc_rate_arr)
        carry = (1, coefs, particles_arr, log_weights_arr, key_, step_size_arr, acc_rate_arr)

        def cond_fun(carry):
            t, coefs, *_ = carry
            # Continue while t < max_steps and the last computed coef is below threshold.
            return (t < max_steps) & (coefs[t - 1] < 1.0)

        def body_fun(carry):
            t, coefs, particles_arr, log_weights_arr, key, step_size_arr, acc_rate_arr = carry
            key, key_ = jr.split(key)
            # Retrieve previous step's particles and weights.
            prev_particles = (particles_arr[:, :, t-1]
                      if self.keep_particles
                      else particles_arr)
            prev_log_weights   = log_weights_arr[:, t - 1] # no need for log-weights, step_size and acc_rate in update_coef
            step_size = step_size_arr[:, t - 1] 
            acc_rate = acc_rate_arr[:, t - 1]   
            state = (prev_particles, prev_log_weights, step_size, acc_rate)
            
            # update coefs
            # If t equals max_steps-1, force the new coefficient to be 1.0.
            key, key_ = jr.split(key)
            new_coef = jax.lax.cond(
                t == max_steps - 1,
                lambda _: 1.0,
                lambda _: self.update_coef(t, coefs, state, key_, 0.6),
                operand=None
            )
            coefs = coefs.at[t].set(new_coef)

            # Compute new state.
            state = self.step(t, coefs, mc_method, (prev_particles, prev_log_weights), key)
            new_particles, new_log_weights, step_size, acc_rate = state

            # Update preallocated arrays at the current iteration index.
            if self.keep_particles:
                particles_arr = particles_arr.at[:, :, t].set(new_particles)
            else:
                particles_arr = new_particles
            log_weights_arr   = log_weights_arr.at[:, t].set(new_log_weights)
            step_size_arr = step_size_arr.at[:, t - 1].set(step_size)
            acc_rate_arr  = acc_rate_arr.at[:, t - 1].set(acc_rate)
            return (t + 1, coefs, particles_arr, log_weights_arr, key, step_size_arr, acc_rate_arr)
        
        # use loop_while
        carry = jax.lax.while_loop(cond_fun, body_fun, carry)
        t, coefs, particles_arr, log_weights_arr, key, step_size_arr, acc_rate_arr = carry

        return t, particles_arr, log_weights_arr, step_size_arr, acc_rate_arr, coefs



    def update_coef(self, t, coefs, state, key, ess_normalized_target, tol = 1e-5):
        '''use target_ess_normalized to find the optimal coefficient by bissection
        If ESS_normalized(tmax*log_weights) >= ess_normalized_target, return tmax. '''
        particles, *_ = state
        coef_max = 1.0
        coef_min = coefs[t-1]
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
        return result[0]















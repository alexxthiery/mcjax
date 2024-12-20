import jax
import jax.numpy as jnp
import jax.random as jr
from typing import TypedDict, Tuple, Dict, NamedTuple
from mcjax.proba.density import LogDensity,LogDensityGeneral
from ..mcmc.markov import MarkovKernel
from mcjax.mcmc.rwm import Rwm, RwmState
from mcjax.mcmc.mala import Mala, MalaState
from mcjax.util.weights import ess_normalized_log_weight


class SmcState(TypedDict):
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
        
        self.coefs = list(coefs)
        self.log_gamma_0 = log_gamma_0
        self.log_gamma_T = log_gamma_T
        self.step_size = step_size
        self.num_substeps = num_substeps # number of substeps for the MC kernel

    def step(
            self,
            t,
            mc_method: str,     # MC method: RWM or MALA
            state: SmcState,    # current state
            key: jax.Array,     # random key
            ):
        ''' a single step of SMC: resampling, sample particles from (t-1) to t, compute weights'''
        ######## Compute weights with the ratio of densities: gamma_t(x_{t-1}) / gamma_{t-1}(x_{t-1})
        updated_weights = self.compute_weights_batch(t,state["particles"])
        
        ############ Resampling 
        key, key_ = jr.split(key)
        new_particles = self.resample(state["particles"], updated_weights, key_)
        
        ############ Sample particles from (t-1) to t using the RWM kernel
        mc_methods = {
            'RWM': self.random_walk_batch,
            'MALA': self.mala_batch
        }
        # if mc_method not in mc_methods:
        #     raise ValueError('MC method not supported')
        mc_function = mc_methods[mc_method]
        for i in range(self.num_substeps):
            key, key_ = jr.split(key)
            new_particles = self.random_walk_batch(t, jnp.array(new_particles), key_, )

        return SmcState(particles=new_particles, weights=updated_weights)
    
    def resample(self, particles: jnp.ndarray, weights: jnp.ndarray, key: jax.Array):
        """ Perform resampling step using the weights """
        # Normalize the weights
        normalized_weights = weights / jnp.sum(weights)

        # Resample particles based on their weights
        num_particles = particles.shape[0]
        key, key_ = jr.split(key)
        indices = jax.random.choice(key_, num_particles, shape=(num_particles,), p=normalized_weights)
        resampled_particles = particles[indices]
        
        return resampled_particles
    
    # Random walk Metropolis kernel
    # def random_walk(self, t, particle: jnp.ndarray, key: jax.Array):
    #     """ Perform RWM for each particle """
    #     new_particle = particle + jr.normal(key, particle.shape) * self.step_size
    #     gamma_t_current = self.coefs[t] * self.log_gamma_T.logdensity(particle) \
    #         + (1 - self.coefs[t]) * self.log_gamma_0.logdensity(particle)
    #     gamma_t_proposal = self.coefs[t] * self.log_gamma_T.logdensity(new_particle) \
    #         + (1 - self.coefs[t]) * self.log_gamma_0.logdensity(new_particle)
        
    #     # accept or reject
    #     key, key_ = jr.split(key)
    #     u = jr.uniform(key_)
    #     log_ratio = gamma_t_proposal - gamma_t_current
    #     accept_MH = jnp.exp(jnp.minimum(0., log_ratio))
    #     accept = u < accept_MH
    #     new_particle = jnp.where(accept, new_particle, particle)
    #     return new_particle
    
    def random_walk_batch(self, t, particles: jnp.ndarray, key: jax.Array):
        logdensity = LogDensityGeneral(logdensity = lambda x: self.coefs[t] * self.log_gamma_T.logdensity(x) \
            + (1 - self.coefs[t]) * self.log_gamma_0.logdensity(x), dim=particles.shape[1])
        rwm = Rwm(logtarget=logdensity, step_size = self.step_size)
        state = RwmState(x=particles, logdensity=logdensity.batch(particles))
        new_particles,_ = rwm.step(state, key)
        return new_particles['x']

    # MALA kernel ()
    # def mala(self, t, particle: jnp.ndarray, key: jax.Array):
    #     """ Perform MALA for each particle """
    #     grad_gamma_t = self.coefs[t] * self.log_gamma_T.grad(particle) \
    #         + (1 - self.coefs[t]) * self.log_gamma_0.grad(particle)
    #     noise = jr.normal(key, particle.shape)
    #     new_particle = particle + self.step_size * grad_gamma_t + jnp.sqrt(2 * self.step_size) * noise
    #     gamma_t_current = self.coefs[t] * self.log_gamma_T.logdensity(particle) \
    #         + (1 - self.coefs[t]) * self.log_gamma_0.logdensity(particle)
    #     gamma_t_proposal = self.coefs[t] * self.log_gamma_T.logdensity(new_particle) \
    #         + (1 - self.coefs[t]) * self.log_gamma_0.logdensity(new_particle)
        
    #     # accept or reject
    #     key, key_ = jr.split(key)
    #     u = jr.uniform(key_)
    #     log_ratio = gamma_t_proposal - gamma_t_current
    #     accept_MH = jnp.exp(jnp.minimum(0., log_ratio))
    #     accept = u < accept_MH
    #     new_particle = jnp.where(accept, new_particle, particle)
    #     return new_particle

    def mala_batch(self, t, particles: jnp.ndarray, key: jax.Array):
        logdensity = LogDensityGeneral(logdensity = lambda x: self.coefs[t] * self.log_gamma_T.logdensity(x) \
            + (1 - self.coefs[t]) * self.log_gamma_0.logdensity(x), dim=particles.shape[1])
        mala = Mala(logtarget=logdensity, step_size = self.step_size)
        state = MalaState(x=particles, logdensity=logdensity.batch(particles))
        new_particles,_ = mala.step(state, key)
        return new_particles['x']

    def compute_weights(self, t, resampled_particle: jnp.ndarray):
        """ Compute the weights as the ratio of the target densities """
        gamma_t = self.coefs[t]*self.log_gamma_T.logdensity(resampled_particle) \
            + (1-self.coefs[t])*self.log_gamma_0.logdensity(resampled_particle)
        gamma_t_minus_1 = self.coefs[t-1]*self.log_gamma_T.logdensity(resampled_particle) \
            + (1-self.coefs[t-1])*self.log_gamma_0.logdensity(resampled_particle) 
        weights = jnp.exp(gamma_t - gamma_t_minus_1)
        return weights

    def compute_weights_batch(self,t, resampled_particles:jnp.ndarray):
        vectorized_weights = jax.vmap(self.compute_weights,in_axes=(None,0))
        weights = vectorized_weights(t,resampled_particles)
        return weights

    def run(self,num_particles, key, mc_method):
        # Initialize the state at t=0 with particles sampled from the initial distribution        
        key, key_ = jr.split(key)
        initial_particles = self.log_gamma_0.sample(key_, num_particles)
        initial_weights = jnp.ones(num_particles) / num_particles  
        state = SmcState(particles=initial_particles, weights=initial_weights)
        states = [state]  
        for t in range(1, len(self.coefs)):
            state = self.step(t, mc_method, state, key)
            states.append(state)
        return states

    # Calculate adaptively the coefficient for the geometric SMC 
    def selfadaptive_run(self,num_particles, key, mc_method):
        key, key_ = jr.split(key)
        initial_particles = self.log_gamma_0.sample(key_, num_particles)
        initial_weights = jnp.ones(num_particles) / num_particles  
        state = SmcState(particles=initial_particles, weights=initial_weights)
        states = [state]  
        for t in range(1, len(self.coefs)):
            key, key_ = jr.split(key)
            state = self.step(t, mc_method,state, key)
            states.append(state)
            # Update the coefficients
            key, key_ = jr.split(key)
            self.coefs[t] = self.update_coef(t, state, key, ess_normalized_target=0.7)
        return states


    def update_coef(self, t, state, key, ess_normalized_target, tol = 1e-5):
        # use target_ess_normalized to find the optimal coefficient by bissection
        # If ESS_normalized(tmax*log_weights) >= ess_normalized_target, return tmax.
        coef_max = 1.0
        coef_min = self.coefs[t-1]

        # calculate the log_weights (log_weight) of: log_gamma_t(x_{t-1}) - log_gamma_0(x_{t-1})
        log_weights = self.log_gamma_T.batch(state["particles"]) - self.log_gamma_0.batch(state["particles"])
        ess_tmax = ess_normalized_log_weight((coef_max-coef_min) * log_weights)
        if ess_tmax >= ess_normalized_target:
            return coef_max
        while coef_max - coef_min > tol:
            coef = (coef_max + coef_min) / 2
            ess_new = ess_normalized_log_weight(coef * log_weights)
            if ess_new > ess_normalized_target:
                coef_max = coef
            else:
                coef_min = coef
        return (coef_max + coef_min) / 2

    # calculate the normalizing constant estimator for a single run
    def single_run(self, num_particles, key, mc_method):
        states = self.run(num_particles, key, mc_method)
        Z = jnp.sum(jnp.array([jnp.log(jnp.mean(state["weights"])) for state in states]))
        return Z

    # calculate the variance of the normalizing constant estimator (in log)
    def compute_variance(self, key, num_particles, mc_method,N=100):
        '''
        Variance for N times of the normalizing constant estimator
        Z = \Pi_{t=1}^T 1/N \sum_{n=1}^N w_t(x_{t-1}^n) 
        '''
        # Vectorize the computation over N keys
        keys = jr.split(key, N)
        mult_run = jax.vmap(self.single_run, in_axes=(None,0,None))
        mult_run = jax.jit(mult_run,static_argnums=(0,2))
        Z_arr = mult_run(num_particles,keys,mc_method)

        # Compute the variance of log(Z)
        Z_var = jnp.var(Z_arr)
        return Z_var













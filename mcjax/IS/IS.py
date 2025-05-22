import jax
import jax.numpy as jnp
import jax.random as jr
from mcjax.proba.density import LogDensity
from typing import Callable
from jax.scipy.special import logsumexp

""" Importance Sampling """
class IS:
    def __init__(self,
                 *,
                 logtarget:LogDensity,
                 ):
        self.logtarget = logtarget
        self.dim = logtarget.dim

    def run(self,
            *,
            key: jnp.ndarray,               # random key
            n_samples: int,                 # number of iterations
            proposal: LogDensity,           # proposal distribution
            test: Callable,
            verbose: bool = False
            ):
        
        # sample from proposal density
        key, key_ = jr.split(key)
        x = proposal.sample(key=key_, n_samples=n_samples)

        proposal_density = proposal.batch(x)
        target_density = self.logtarget.batch(x)
        log_omega = target_density - proposal_density

        estimator = jnp.dot(jnp.exp(log_omega),(test(x).squeeze()))/x.shape[0]


        denominator = jnp.sum(jnp.exp(log_omega))
        log_omega_normalized = log_omega - jnp.log(denominator)
        ess = 1/jnp.sum(jnp.exp(2*log_omega_normalized))

        dict_output = {"estimator":estimator,
                       "ess":ess}

        return dict_output


class AIS:
    def __init__(self,
                 *,
                 logtarget:LogDensity,
                 ):
        self.logtarget = logtarget
        self.dim = logtarget.dim

    def run(self,
            *,
            key: jnp.ndarray,               # random key
            n_samples: int,                 # number of iterations
            proposal: LogDensity,           # proposal distribution
            test: Callable,
            verbose: bool = False
            ):
        
        # sample from proposal density
        key, key_ = jr.split(key)
        x = proposal.sample(key=key_, n_samples=n_samples)

        proposal_density = proposal.batch(x)
        target_density = self.logtarget.batch(x)
        log_omega = target_density - proposal_density

        log_omega_normalized = log_omega - logsumexp(log_omega)
        weights = jnp.exp(log_omega_normalized)

        estimator = jnp.dot(jnp.exp(log_omega_normalized),(test(x).squeeze()))

        ess = 1/jnp.sum(jnp.exp(2*log_omega_normalized))

        dict_output = {"estimator":estimator,
                       "ess":ess}

        return dict_output
import jax
import jax.numpy as jnp
import jax.random as jr

from mcjax.dist.dist import Dist


class RWM:
    """ Random Walk Metropolis-Hastings 
    Use Gaussian proposal distribution
    """
    def __init__(
                self,
                *,
                target: Dist,   # target distribution
                step_size,      # step size
                cov=None,       # covariance matrix
                ):
        self.target = target
        self.step_size = step_size
        self.dim = target.dim
        # covariance matrix of the proposal distribution
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

    def sample(
            self,
            key,        # random key
            n_samples,  # number of samples
            x_init,     # initial state
            ):
        """ sample from the target distribution using RWM """

        # implement a single step of RWM
        def rwm_step(carry, _):
            key, x = carry
            key, key_ = jr.split(key)
            x_prop = x + jr.normal(key_, (self.dim,)) * self.step_size
            log_target_current = self.target.logpdf(x)
            log_target_proposal = self.target.logpdf(x_prop)
            log_ratio = log_target_proposal - log_target_current
            # accept or reject
            key, key_ = jr.split(key)
            u = jr.uniform(key_)
            accept_MH = jnp.exp(jnp.minimum(0., log_ratio))
            # square jump distance statistics
            sq_jump = accept_MH * jnp.linalg.norm(x_prop - x)**2
            is_accept = u < accept_MH
            x = jnp.where(is_accept, x_prop, x)
            log_target = jnp.where(is_accept, log_target_proposal, log_target_current)
            output_dict = {'x': x,
                           'log_target': log_target,
                           'is_accept': is_accept,
                           'accept_MH': accept_MH,
                           'sq_jump': sq_jump,
                           }
            return (key, x), output_dict
        
        # run RWM
        key, key_ = jr.split(key)
        _, mcmc_trajectory = jax.lax.scan(rwm_step,
                                          (key_, x_init),
                                          length=n_samples)
        # compute statistics
        mcmc_trajectory['acceptance_rate'] = jnp.mean(mcmc_trajectory['accept_MH'])
        mcmc_trajectory['n_accepted'] = jnp.sum(mcmc_trajectory['is_accept'])
        mcmc_trajectory['sq_jump'] = jnp.mean(mcmc_trajectory['sq_jump'])
        return mcmc_trajectory
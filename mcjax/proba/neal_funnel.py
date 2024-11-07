from mcjax.proba.density import LogDensity
import jax.numpy as jnp
import jax.random as jr


# ==================================
# Neal's Funnel Distribution
# x ~ N(0, sigma_x^2), y ~ N(0, variance=exp(x))
# with sigma_x = 3
# ==================================
class NealFunnel(LogDensity):
    """ Neal's Funnel Distribution
    x ~ N(0, sigma_x^2), y ~ N(0, exp(x/2))
    with sigma_x = 3
    """
    def __init__(
                self,
                *,
                sigma_x=3.,    # noise standard deviation
                ):
        self.sigma_x = sigma_x
        self._dim = 2

    # define the logpdf
    def logdensity(self, x):
        x0, x1 = x[0], x[1]
        std = jnp.exp(x0/2.)
        return -0.5*(x0/self.sigma_x)**2 - 0.5*(x1/std)**2 - 0.5*jnp.log(std**2)

    def sample(self, key, n_samples):
        # samples x0_s
        key, key_ = jr.split(key)
        x0_s = self.sigma_x * jr.normal(key_, (n_samples,))
        # samples x1_s
        key, key_ = jr.split(key)
        stds = jnp.exp(x0_s/2.)
        x1_s = stds * jr.normal(key_, (n_samples,))
        return jnp.stack([x0_s, x1_s], axis=-1)
    

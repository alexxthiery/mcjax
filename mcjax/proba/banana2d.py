from .density import LogDensity
import jax.numpy as jnp
import jax.random as jr


# ==================================
# Banana 2D Distribution
# ==================================
class Banana2D(LogDensity):
    """ Banana 2D Distribution:
     target(x,y) \propto exp{ -0.5*( (y-x^2)^2/noise_std^2 + (x-1)^2 ) }
    remark: with noise_std = 0.1, the -logpdf is the Rosenbrock function
    """
    def __init__(
                self,
                *,
                noise_std=0.1,    # noise standard deviation
                ):
        self.noise_std = noise_std
        self._dim = 2

    # define the logpdf
    def logdensity(self, x):
        # assert the dimension
        assert x.shape == (self.dim,), "Invalid dimension"
        x0, x1 = x[0], x[1]
        return -0.5*((x0 - 1.)**2 + (x1 - x0**2)**2 / self.noise_std**2)

    def sample(self, key, n_samples):
        # samples x0_s
        key, key_ = jr.split(key)
        x0_s = 1. + jr.normal(key_, (n_samples,))
        # samples x1_s
        key, key_ = jr.split(key)
        x1_s = x0_s**2 + self.noise_std * jr.normal(key_, (n_samples,))
        return jnp.stack([x0_s, x1_s], axis=-1)
    
    def log_Z(self):
        """ log partition function """
        return 0.5*jnp.log(2*jnp.pi) + 0.5*jnp.log(2*jnp.pi*self.noise_std**2)

    

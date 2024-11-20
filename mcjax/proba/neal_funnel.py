from .density import LogDensity
import jax.numpy as jnp
import jax.random as jr


# ==================================
# D-dimensional Neal's Funnel Distribution
# x0 ~ N(0, sigma_x^2),
# xi ~ N(0, variance=exp(x0)) are independent for i=1,...,D-1
# By default:
#   sigma_x = 3
#   D = 2
# ==================================
class NealFunnel(LogDensity):
    """ Neal's Funnel Distribution
    x0 ~ N(0, sigma_x^2),
    xi ~ N(0, variance=exp(x0)) are independent for i=1,...,D-1
    By default:
    sigma_x = 3
    D = 2
    """
    def __init__(
                self,
                *,
                sigma_x=3.,    # noise standard deviation
                dim=2,         # dimension
                ):
        self.sigma_x = sigma_x
        self._dim = dim

    # define the logpdf
    def logdensity(self, x):
        # assert the dimension
        assert x.shape == (self.dim,), "Invalid dimension"
        x0, x1 = x[0], x[1:]
        std = jnp.exp(x0/2.)
        return -0.5*(x0/self.sigma_x)**2 - 0.5*jnp.sum(x1/std)**2 - 0.5*(self.dim-1)*jnp.log(std**2)

    def sample(self, key, n_samples):
        # samples x0_s
        key, key_ = jr.split(key)
        x0_s = self.sigma_x * jr.normal(key_, (n_samples, 1))
        # samples x1_s
        key, key_ = jr.split(key)
        stds = jnp.exp(x0_s/2.)
        x1_s = stds * jr.normal(key_, (n_samples, self.dim-1))
        return jnp.concatenate([x0_s, x1_s], axis=1)
    

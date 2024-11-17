from .density import LogDensity
import jax
import jax.numpy as jnp
import mcjax.util.psd as psd


# ==================================
# Isotropic Gaussian Distribution
# The covariance matrix is a scalar multiple of the identity matrix
# ==================================
class Exponential(LogDensity):
    """ Exponential Distribution:
        f(x) = \gamma*exp(-\gamma*(x - \theta))
    """
    def __init__(
                self,
                *,
                theta,         
                gamma,    
                ):

        self.theta = theta
        self.gamma = gamma
        self._dim = len(theta)

    def logdensity(self, x):
        assert jnp.all(x >= self.theta), "Zero Density"
        return jnp.log(self.gamma)*self._dim - self.gamma*(jnp.sum(x) - jnp.sum(self.theta))

    
    def batch(
            self,
            x_batch,   # (B, D): B batch size, D dimension
            ):
        assert jnp.all(x_batch >= self.theta), "Zero Density"
        return jnp.log(self.gamma) * self._dim - self.gamma*(jnp.sum(x_batch,axis=-1) - jnp.sum(self.theta))
        

    
    def grad(self, x):
        assert jnp.all(x >= self.theta), "Zero Density"
        return -jnp.log(self.gamma)*2*self._dim - self.gamma*(jnp.sum(x) - jnp.sum(self.theta))
    
    def grad_batch(self,
                   x_batch,   # (B, D): B batch size, D dimension
                   ):
        assert jnp.all(x_batch >= self.theta), "Zero Density"
        return -jnp.log(self.gamma)*2*self._dim - self.gamma*(jnp.sum(x_batch,axis=-1) - jnp.sum(self.theta))
    
    def sample(self, key, n_samples):

        u = jax.random.uniform(key, shape=(n_samples, self._dim))
        # Apply inverse CDF
        samples = self.theta - (1.0 / self.gamma) * jnp.log(1 - u)
        return samples


    
    def log_Z(self):
        """ log partition function """
        return self._dim*jnp.log(self.gamma)

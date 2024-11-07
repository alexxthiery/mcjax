from .density import LogDensity
import jax
import jax.numpy as jnp


# ==================================
# Isotropic Gaussian Distribution
# The covariance matrix is a scalar multiple of the identity matrix
# ==================================
class IsotropicGauss(LogDensity):
    """ Isotropic Gaussian Distribution:
    The covariance matrix is a scalar multiple of the identity matrix
     mu: mean vector
     log_var: log variance (scalar)
    """
    def __init__(
                self,
                *,
                mu,         # mean vector
                log_var,    # log variance
                ):
        # make sure that sigma is a scalar
        assert jnp.isscalar(log_var)
        self.mu = mu
        self.log_var = log_var
        self.sigma = jnp.exp(0.5*self.log_var)
        self._dim = len(mu)

    def logdensity(self, x):
        return -0.5 * jnp.sum(jnp.square((x - self.mu[None, :]) / self.sigma))
    
    def batch(
            self,
            x_batch,   # (B, D): B batch size, D dimension
            ):
        return -0.5 * jnp.sum(jnp.square((x_batch - self.mu[None, :]) / self.sigma), axis=-1)
    
    def grad(self, x):
        return -(x - self.mu) / self.sigma**2
    
    def grad_batch(self,
                   x_batch,   # (B, D): B batch size, D dimension
                   ):
        return -(x_batch - self.mu[None, :]) / self.sigma**2
    
    def sample(self, key, n_samples):
        return jax.random.normal(key, (n_samples, self.dim)) * self.sigma + self.mu[None, :]
    
    def log_Z(self):
        """ log partition function """
        return -0.5 * self.dim * jnp.log(2 * jnp.pi) - self.dim*jnp.log(self.sigma)


# ==================================
# Diagonal Gaussian Distribution
# The covariance matrix is a diagonal matrix
# ==================================
class DiagGauss(LogDensity):
    """ Gaussian Distribution with Diagonal Covariance Matrix:
    The covariance matrix is a scalar multiple of the identity matrix
     mu: mean vector
     log_var: vector with marginal log variance
    """
    def __init__(
                self,
                mu,         # mean vector
                log_var     # vector of marginal log variance
                ):
        # make sure that sigma is a vector
        assert jnp.ndim(log_var) == 1
        self.mu = mu
        self.log_var = log_var
        self.sigma = jnp.exp(0.5*self.log_var)  # vector of marginal standard deviation
        self._dim = len(mu)

    def logdensity(self, x):
        return -0.5 * jnp.sum(jnp.square((x - self.mu) / self.sigma))
    
    def batch(
            self,
            x_batch,   # (B, D): B batch size, D dimension
            ):
        z_batch = (x_batch - self.mu[None, :]) / self.sigma[None, :]
        return -0.5 * jnp.sum(jnp.square(z_batch), axis=-1)
    
    def grad(self, x):
        return -(x - self.mu) / self.sigma**2
    
    def grad_batch(
                    self,
                    x_batch,   # (B, D): B batch size, D dimension
                    ):
        return -(x_batch - self.mu[None, :]) / self.sigma[None, :]**2
    
    def sample(self, key, n_samples):
        return jax.random.normal(key, (n_samples, self.dim)) * self.sigma[None,:] + self.mu[None, :]
    
    def log_Z(self):
        """ log partition function """
        return -0.5 * self.dim * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(self.sigma))
    

# ==================================
# General Gaussian Distribution
# The covariance matrix is a general matrix
# ==================================
class Gauss(LogDensity):
    """ Gaussian Distribution with general Covariance Matrix:
    The covariance matrix is a scalar multiple of the identity matrix
     mu: mean vector
     cov: covariance matrix
    """
    def __init__(
                self,
                mu,         # mean vector
                cov         # covariance matrix
                ):
        # make sure that cov is a square matrix
        assert jnp.ndim(cov) == 2
        assert cov.shape[0] == cov.shape[1]
        assert cov.shape[0] == len(mu)
        
        self.mu = mu
        self.cov = cov
        self._dim = len(mu)
        
        # compute cholesky decomposition of the covariance matrix
        self.L = jnp.linalg.cholesky(cov)
        # inverse of covariance matrix
        self.inv_cov = jnp.linalg.inv(cov)

    def logdensity(self, x):
        return -0.5 * jnp.dot((x - self.mu), self.inv_cov @ (x - self.mu))

    def batch(
            self,
            x_batch,   # (B, D): B batch size, D dimension
            ):
        x_centred = x_batch - self.mu[None, :]
        return -0.5*jnp.sum(x_centred * (x_centred @ self.inv_cov.T), axis=-1)
    
    def grad(self, x):
        return -self.inv_cov @ (x - self.mu)
    
    def grad_batch(
                self,
                x_batch,   # (B, D): B batch size, D dimension
                ):
        x_centred = x_batch - self.mu[None, :]
        return -x_centred @ self.inv_cov.T
    
    def sample(self, key, n_samples):
        return jax.random.normal(key, (n_samples, self.dim)) @ self.L.T + self.mu[None, :]
    
    def log_Z(self):
        """ log partition function """
        return -0.5 * self.dim * jnp.log(2 * jnp.pi) - 0.5*jnp.linalg.slogdet(self.cov)[1]


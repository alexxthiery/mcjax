from .density import LogDensity
import jax
import jax.numpy as jnp
import mcjax.util.psd as psd


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
        logdet = self.dim*self.log_var
        self._log_Z = 0.5 * self.dim * jnp.log(2 * jnp.pi) + 0.5*logdet

    def logdensity(self, x):
        return -0.5 * jnp.sum(jnp.square((x - self.mu[None, :]) / self.sigma)) - self._log_Z
    
    def batch(
            self,
            x_batch,   # (B, D): B batch size, D dimension
            ):
        return -0.5 * jnp.sum(jnp.square((x_batch - self.mu[None, :]) / self.sigma), axis=-1) - self._log_Z
    
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
    The covariance matrix is diagonal
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
        logdet = jnp.sum(self.log_var)
        self._log_Z = 0.5 * self.dim * jnp.log(2 * jnp.pi) + 0.5*logdet
        
    def logdensity(self, x):
        return -0.5 * jnp.sum(jnp.square((x - self.mu) / self.sigma)) - self._log_Z
    
    def batch(
            self,
            x_batch,   # (B, D): B batch size, D dimension
            ):
        z_batch = (x_batch - self.mu[None, :]) / self.sigma[None, :]
        return -0.5 * jnp.sum(jnp.square(z_batch), axis=-1) - self._log_Z
    
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
                mu: jnp.ndarray,                # mean vector
                cov: jnp.ndarray,               # covariance matrix
                cov_inv: jnp.ndarray = None,    # inverse of covariance matrix
                chol: jnp.ndarray = None,       # cholesky decomposition of covariance matrix
                ):
        # make sure that cov is a square matrix
        assert jnp.ndim(cov) == 2
        assert cov.shape[0] == cov.shape[1]
        assert cov.shape[0] == len(mu)
        
        # set the dimension
        self._dim = len(mu)
        
        self.update_mu(mu)
        self.update_cov(cov=cov, cov_inv=cov_inv, chol=chol)

    def update_mu(self, mu):        
        self.mu = mu

    def update_cov(
                self,
                *,
                cov: jnp.ndarray,               # covariance matrix
                cov_inv: jnp.ndarray = None,    # inverse of covariance matrix
                chol: jnp.ndarray = None,       # cholesky decomposition of covariance matrix
                ):
        # compute cholesky decomposition of the covariance matrix
        self.chol = jax.lax.cond(
                        chol is None,
                        lambda _: jnp.linalg.cholesky(cov),
                        lambda _: chol if chol is not None else jnp.zeros_like(cov),
                        operand=None)

        # inverse of covariance matrix
        self.cov_inv = jax.lax.cond(
                        cov_inv is None,
                        lambda _: psd.invert_psd_matrix(A=cov, chol=self.chol),
                        lambda _: cov_inv if cov_inv is not None else jnp.zeros_like(cov),
                        operand=None,)

        # update normalization
        logdet = 2*jnp.sum(jnp.log(jnp.diag(self.chol)))
        self._log_Z = 0.5 * self.dim * jnp.log(2 * jnp.pi) + 0.5*logdet

    def update_params(
                    self,
                    mu: jnp.ndarray,                # mean vector
                    cov: jnp.ndarray,               # covariance matrix
                    cov_inv: jnp.ndarray = None,    # inverse of covariance matrix
                    ):
        self.update_mu(mu)
        self.update_cov(cov=cov, cov_inv=cov_inv)

    def logdensity(self, x):
        # assert the shape of x
        assert x.shape == (self.dim,), "Invalid input shape"
        x_centred = x - self.mu
        log_unormalized = -0.5 * jnp.sum(x_centred * (self.cov_inv @ x_centred))
        return log_unormalized - self._log_Z

    def batch(
            self,
            x_batch,   # (B, D): B batch size, D dimension
            ):
        # assert the shape of x_batch
        assert x_batch.ndim == 2, "Invalid input shape: x_batch.ndim must be 2"
        assert x_batch.shape[1] == self.dim, "Invalid input shape: x_batch.shape[1] must be equal to self.dim"
        x_centred = x_batch - self.mu[None, :]
        return -0.5*jnp.sum(x_centred * (x_centred @ self.cov_inv.T), axis=-1) - self._log_Z
    
    def grad(self, x):
        assert x.shape == (self.dim,), "Invalid input shape"
        return -self.cov_inv @ (x - self.mu)
    
    def grad_batch(
                self,
                x_batch,   # (B, D): B batch size, D dimension
                ):
        assert x_batch.ndim == 2, "Invalid input shape: x_batch.ndim must be 2"
        assert x_batch.shape[1] == self.dim, "Invalid input shape: x_batch.shape[1] must be equal to self.dim"        
        x_centred = x_batch - self.mu[None, :]
        return -x_centred @ self.cov_inv.T
    
    def sample(
            self,
            key: jnp.ndarray,       # random key
            n_samples: int,         # number of samples
            center: bool = False,   # whether to center the samples
            ):
        zs = jax.random.normal(key, (n_samples, self.dim))
        zs_mean = jnp.mean(zs, axis=0)
        zs = jax.lax.cond(
                    center,
                    lambda _: zs - zs_mean[None, :],
                    lambda _: zs,
                    operand=None)
        return zs @ self.chol.T + self.mu[None, :]


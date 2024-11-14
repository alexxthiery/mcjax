from .density import LogDensity
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jspecial


# ==================================
# General Gaussian Distribution
# The covariance matrix is a general matrix
# ==================================
class Student(LogDensity):
    """ Multivariate Student-t Distribution:
     mu: location vector
     cov: covariance matrix
    """
    def __init__(
                self,
                mu,         # mean vector
                cov,        # covariance matrix
                deg,        # degrees of freedom
                ):
        # make sure that cov is a square matrix
        assert jnp.ndim(cov) == 2
        assert cov.shape[0] == cov.shape[1]
        assert cov.shape[0] == len(mu)
        # assert degrees of freedom is larger than zero
        assert deg > 0, "Degrees of freedom must be larger than zero"
        
        self.mu = mu
        self.cov = cov
        self.deg = deg
        self._dim = len(mu)
        
        # compute cholesky decomposition of the covariance matrix
        self.L = jnp.linalg.cholesky(cov)
        # get logdet from the cholesky decompositsion
        self.cov_logdet = 2*jnp.sum(jnp.log(jnp.diag(self.L)))
        # log_normalization: log_density = (.. some function of x ..) - log_Z
        neg_log_Z = 0
        neg_log_Z = neg_log_Z + jspecial.gammaln(0.5*(self.deg + self._dim))
        neg_log_Z = neg_log_Z - jspecial.gammaln(0.5*self.deg)
        neg_log_Z = neg_log_Z- 0.5*self._dim*jnp.log(self.deg*jnp.pi) - 0.5*self.cov_logdet
        self._log_Z = -neg_log_Z
        
        # inverse of covariance matrix
        self.inv_cov = jnp.linalg.inv(cov)

    def logdensity(self, x):
        x_centred = x - self.mu
        quad_form = jnp.dot(x_centred, self.inv_cov @ x_centred)
        return -0.5*(self.deg + self._dim)*jnp.log(1 + quad_form/self.deg) - self._log_Z
    
    def batch(
            self,
            x_batch,   # (B, D): B batch size, D dimension
            ):
        x_centred = x_batch - self.mu[None, :]
        quad_form = jnp.sum(x_centred * (x_centred @ self.inv_cov.T), axis=-1)
        return -0.5*(self.deg + self._dim)*jnp.log(1 + quad_form/self.deg) - self._log_Z
    
    def sample(self, key, n_samples):
        # samples some standard normal random variables
        key, key_ = jr.split(key)
        z = jr.normal(key_, (n_samples, self.dim))
        # samples some chi-square random variables from Gaussian
        key, key_ = jr.split(key)
        chi2 = jr.chisquare(key_, df=self.deg, shape=(n_samples,))
        # samples some student-t random variables
        x = self.mu[None, :] + jnp.sqrt(self.deg/chi2)[:, None] * z @ self.L.T
        return x
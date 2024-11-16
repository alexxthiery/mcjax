from .density import LogDensity
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jspecial
import mcjax.util.psd as psd


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
                *,
                mu: jnp.ndarray,        # mean vector
                cov: jnp.ndarray,       # covariance matrix
                deg: int,               # degrees of freedom
                cov_inv=None,           # inverse of the covariance matrix
                chol=None,              # Cholesky decomposition of the covariance matrix
                ):
        # make sure that cov is a square matrix
        assert jnp.ndim(cov) == 2
        assert cov.shape[0] == cov.shape[1]
        assert cov.shape[0] == len(mu)
        # assert degrees of freedom is larger than zero
        assert deg > 0, "Degrees of freedom must be larger than zero"
        self.deg = deg
        self._dim = len(mu)
        
        # location
        self.mu = None
        self.update_mu(mu)
        
        # covariance structure
        self.cov = None
        self.chol = None
        self.cov_inv = None
        self.cov_logdet = None
        self._log_Z = None
        self.update_cov(cov=cov, cov_inv=cov_inv, chol=chol)
                
    def update_deg(self, deg):
        self.deg = deg
        self.update_normalization()

    def update_normalization(self):
        # log_normalization: log_density = (.. some function of x ..) - log_Z
        neg_log_Z = 0
        neg_log_Z = neg_log_Z + jspecial.gammaln(0.5*(self.deg + self._dim))
        neg_log_Z = neg_log_Z - jspecial.gammaln(0.5*self.deg)
        neg_log_Z = neg_log_Z - 0.5*self._dim*jnp.log(self.deg*jnp.pi) - 0.5*self.cov_logdet
        self._log_Z = -neg_log_Z
                
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
        # get logdet from the cholesky decompositsion
        self.cov_logdet = 2*jnp.sum(jnp.log(jnp.diag(self.chol)))
        
        # inverse of covariance matrix
        self.cov_inv = jax.lax.cond(
                        cov_inv is None,
                        lambda _: psd.invert_psd_matrix(A=cov, chol=self.chol),
                        lambda _: cov_inv if cov_inv is not None else jnp.zeros_like(cov),
                        operand=None)

        # update normalization
        self.update_normalization()

    def update_params(
                    self,
                    *,
                    mu: jnp.ndarray,                # mean vector
                    cov: jnp.ndarray,               # covariance matrix
                    deg: float,                     # degrees of freedom
                    cov_inv: jnp.ndarray = None,    # inverse of covariance matrix
                    chol: jnp.ndarray = None,       # cholesky decomposition of covariance matrix
                    ):
        self.update_mu(mu)
        self.update_cov(cov=cov, cov_inv=cov_inv, chol=chol)
        self.update_deg(deg)

    def logdensity(self, x):
        x_centred = x - self.mu
        quad_form = jnp.dot(x_centred, self.cov_inv @ x_centred)
        return -0.5*(self.deg + self._dim)*jnp.log(1 + quad_form/self.deg) - self._log_Z
    
    def batch(
            self,
            x_batch,   # (B, D): B batch size, D dimension
            ):
        x_centred = x_batch - self.mu[None, :]
        quad_form = jnp.sum(x_centred * (x_centred @ self.cov_inv.T), axis=-1)
        return -0.5*(self.deg + self._dim)*jnp.log(1 + quad_form/self.deg) - self._log_Z
    
    def sample(
            self,
            key: jnp.ndarray,       # random key
            n_samples: int,         # number of samples
            center: bool = False,    # center the samples
            ):
        # samples some standard normal random variables
        key, key_ = jr.split(key)
        zs = jr.normal(key_, (n_samples, self.dim))
        zs_mean = jnp.mean(zs, axis=0)
        # center the samples
        zs = jax.lax.cond(
                center,
                lambda _: zs - zs_mean[None, :],
                lambda _: zs,
                operand=None)
        # samples some chi-square random variables from Gaussian
        key, key_ = jr.split(key)
        chi2 = jr.chisquare(key_, df=self.deg, shape=(n_samples,))
        # samples some student-t random variables
        x = self.mu[None, :] + jnp.sqrt(self.deg/chi2)[:, None] * zs @ self.chol.T
        return x
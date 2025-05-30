from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln
from jax.scipy.linalg import solve_triangular
from typing import Optional, Callable
from .distribution import DistributionLike, generic_neg_elbo


#########################################
# Diagonal Student-t variational family
#########################################
@struct.dataclass
class StudentDiagParams:
    mu: jnp.ndarray
    log_std: jnp.ndarray
    df: jnp.ndarray  # degrees of freedom (scalar array)

@struct.dataclass
class StudentDiag:
    """
    Student-t distribution with diagonal scale matrix:
        q(x) = t_deg(μ, diag(exp(log_std)^2))
        where df is degrees of freedom, mu is mean, log_std is log scale.
    """
    dim: int

    @classmethod
    def create(cls, *, dim: int) -> "StudentDiag":
        return cls(dim=dim)

    def init_params(
        self,
        mu: Optional[jnp.ndarray] = None,
        log_std: Optional[jnp.ndarray] = None,
        df: float = 3.0,
    ) -> StudentDiagParams:
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        mu = mu if mu is not None else jnp.zeros(self.dim)
        log_std = log_std if log_std is not None else jnp.zeros(self.dim)

        if mu.shape != (self.dim,):
            raise ValueError(f"mu must be shape ({self.dim},), got {mu.shape}")
        if log_std.shape != (self.dim,):
            raise ValueError(f"log_std must be shape ({self.dim},), got {log_std.shape}")

        return StudentDiagParams(mu=mu, log_std=log_std, df=jnp.asarray(df))

    def sample(self, params: StudentDiagParams, key: jax.Array, n_samples: int) -> jnp.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        key_z, key_u = jr.split(key)
        z = jr.normal(key_z, shape=(n_samples, self.dim))
        u = jr.gamma(key_u, params.df / 2.0, shape=(n_samples,)) * 2.0 / params.df

        std = jnp.exp(params.log_std)
        scale = std / jnp.sqrt(u)[:, None]
        return params.mu + z * scale

    def log_prob(self, params: StudentDiagParams, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 1 or x.shape[0] != self.dim:
            raise ValueError(f"x must be 1D with shape ({self.dim},), got {x.shape}")

        std = jnp.exp(params.log_std)
        delta = (x - params.mu) / std
        quad = jnp.sum(delta**2)
        log_det = 2*jnp.sum(params.log_std)
        dim = self.dim
        df = params.df

        norm_const = gammaln((df + dim) / 2) - gammaln(df / 2) - 0.5 * dim * jnp.log(df * jnp.pi) - 0.5 * log_det
        exponent = -0.5 * (df + dim) * jnp.log1p(quad / df)
        return norm_const + exponent

    def log_normalization(self, params: StudentDiagParams) -> jnp.ndarray:
        # Already normalized; handled in log_prob
        return 0.

    def postprocess(self, params: StudentDiagParams) -> dict:
        return {
            "mu": jnp.asarray(params.mu),
            "std": jnp.exp(jnp.asarray(params.log_std)),
            "df": params.df,
        }

    def neg_elbo(
        self,
        params: StudentDiagParams,
        xs: jnp.ndarray,
        logtarget: Callable[[jnp.ndarray], jnp.ndarray],
        stop_gradient_entropy: bool = True,
        key: Optional[jax.Array] = None,
        n_samples: Optional[int] = None,
    ) -> jnp.ndarray:
        return generic_neg_elbo(
            dist=self,
            params=params,
            xs=xs,
            logtarget=logtarget,
            stop_gradient_entropy=stop_gradient_entropy,
            key=key,
            n_samples=n_samples,
        )


# StudentDiag follows the DistributionLike protocol
dist: DistributionLike = StudentDiag.create(dim=1)


###############################################
# Full-covariance Student-t variational family
###############################################
@struct.dataclass
class StudentFullCovParams:
    mu: jnp.ndarray
    log_diag: jnp.ndarray
    cov_chol_lower: jnp.ndarray
    df: jnp.ndarray  # degrees of freedom

@struct.dataclass
class StudentFullCov:
    """
    Student-t distribution with full covariance via Cholesky factor:
        q(x) = t_deg(μ, Σ), Σ = L @ L.T
        where L = diag(exp(log_diag)) + tril(cov_chol_lower, k=-1)
    """
    dim: int

    @classmethod
    def create(cls, *, dim: int) -> "StudentFullCov":
        return cls(dim=dim)

    def init_params(
        self,
        mu: Optional[jnp.ndarray] = None,
        cov: Optional[jnp.ndarray] = None,
        log_diag: Optional[jnp.ndarray] = None,
        df: float = 3.0,
    ) -> StudentFullCovParams:
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")

        mu = mu if mu is not None else jnp.zeros(self.dim)
        if cov is not None:
            L = jnp.linalg.cholesky(cov)
            log_diag = jnp.log(jnp.diag(L))
            cov_chol_lower = L - jnp.diag(jnp.diag(L))
        else:
            log_diag = log_diag if log_diag is not None else jnp.zeros(self.dim)
            cov_chol_lower = jnp.zeros((self.dim, self.dim))

        if mu.shape != (self.dim,):
            raise ValueError(f"mu must be shape ({self.dim},), got {mu.shape}")
        if log_diag.shape != (self.dim,):
            raise ValueError(f"log_diag must be shape ({self.dim},), got {log_diag.shape}")
        if cov_chol_lower.shape != (self.dim, self.dim):
            raise ValueError(f"cov_chol_lower must be shape ({self.dim},{self.dim}), got {cov_chol_lower.shape}")

        return StudentFullCovParams(
            mu=mu,
            log_diag=log_diag,
            cov_chol_lower=cov_chol_lower,
            df=jnp.asarray(df),
        )

    def _construct_cholesky(self, params: StudentFullCovParams) -> jnp.ndarray:
        diag = jnp.exp(params.log_diag)
        return jnp.diag(diag) + jnp.tril(params.cov_chol_lower, k=-1)

    def sample(self, params: StudentFullCovParams, key: jax.Array, n_samples: int) -> jnp.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        key_z, key_u = jr.split(key)
        z = jr.normal(key_z, shape=(n_samples, self.dim))
        u = jr.gamma(key_u, params.df / 2.0, shape=(n_samples,)) * 2.0 / params.df

        L = self._construct_cholesky(params)
        x = z @ L.T
        return params.mu + x / jnp.sqrt(u)[:, None]

    def log_prob(self, params: StudentFullCovParams, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 1 or x.shape[0] != self.dim:
            raise ValueError(f"x must be 1D with shape ({self.dim},), got {x.shape}")

        L = self._construct_cholesky(params)
        xs_centered = x - params.mu
        y = solve_triangular(L, xs_centered, lower=True)
        quad = jnp.sum(y**2)
        log_det = 2.0 * jnp.sum(params.log_diag)
        dim = self.dim
        df = params.df

        norm_const = gammaln((df + dim)/2) - gammaln(df/2) - 0.5 * dim * jnp.log(df * jnp.pi) - 0.5*log_det
        exponent = -0.5 * (df + dim) * jnp.log1p(quad / df)
        return norm_const + exponent

    def log_normalization(self, params: StudentFullCovParams) -> jnp.ndarray:
        return 0.

    def postprocess(self, params: StudentFullCovParams) -> dict:
        L = self._construct_cholesky(params)
        cov = L @ L.T
        return {
            "mu": params.mu,
            "cov_chol": L,
            "cov": cov,
            "df": params.df,
        }

    def neg_elbo(
        self,
        params: StudentFullCovParams,
        xs: jnp.ndarray,
        logtarget: Callable[[jnp.ndarray], jnp.ndarray],
        stop_gradient_entropy: bool = True,
        key: Optional[jax.Array] = None,
        n_samples: Optional[int] = None,
    ) -> jnp.ndarray:
        return generic_neg_elbo(
            dist=self,
            params=params,
            xs=xs,
            logtarget=logtarget,
            stop_gradient_entropy=stop_gradient_entropy,
            key=key,
            n_samples=n_samples,
        )


# StudentFullCov follows the DistributionLike protocol
dist_full_cov: DistributionLike = StudentFullCov.create(dim=1)

from typing import Optional, Callable
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct
from .var_family import VarFamily


##############################################################
# Diagonal Gaussian variational family
#   q(x) = N(mu, diag(exp(log_std)^2))
#   where mu is the mean vector and log_std is the log of the standard deviation
#   of the diagonal Gaussian distribution.
#   The covariance matrix is a diagonal matrix
#   with diagonal elements exp(log_std)^2.
#   The log-density is computed using the formula:
#   log q(x) = -0.5 * sum((x - mu)^2 / exp(log_std)^2) - 0.5 * D * log(2 * pi) - sum(log_std)
################################################################
@struct.dataclass
class DiagGaussianParams:
    mu: jnp.ndarray
    log_std: jnp.ndarray


class DiagGaussian(VarFamily):
    """
    Mean-field diagonal Gaussian variational family:
        q(x) = N(mu, diag(exp(log_std)^2))
    """
    def __init__(
        self,
        *,
        dim: int,
        mu_init: Optional[jnp.ndarray] = None,
        log_std_init: Optional[jnp.ndarray] = None,
    ):
        self.dim = dim

        if mu_init is None:
            mu_init = jnp.zeros(dim)
        else:
            assert mu_init.shape == (dim,), "mu_init must be shape (dim,)"

        if log_std_init is None:
            log_std_init = jnp.zeros(dim)
        else:
            assert log_std_init.shape == (dim,), "log_std_init must be shape (dim,)"

        self.mu_init = mu_init
        self.log_std_init = log_std_init

    def init_params(
            self,
            key: Optional[jax.Array] = None, # not used, kept for consistency
            ) -> DiagGaussianParams:
        return DiagGaussianParams(mu=self.mu_init, log_std=self.log_std_init)

    def sample(
        self,
        params: DiagGaussianParams,
        key: jax.Array,
        n_samples: int
    ) -> jnp.ndarray:
        eps = jr.normal(key, shape=(n_samples, self.dim))
        std = jnp.exp(params.log_std)
        return params.mu[None, :] + eps * std[None, :]

    def logdensity(
            self,
            params: DiagGaussianParams,
            xs: jnp.ndarray) -> jnp.ndarray:
        mu = params.mu
        log_std = params.log_std
        std = jnp.exp(log_std)

        log_det = jnp.sum(log_std)  # log |diag(std^2)| = sum log(std)
        normed = (xs - mu[None, :]) / std[None, :]
        log_prob = -0.5 * jnp.sum(normed**2, axis=-1)
        log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)
        log_prob -= log_det
        return log_prob

    def postprocess(self, params: DiagGaussianParams) -> dict:
        return {
            "mu": params.mu,
            "std": jnp.exp(params.log_std),
        }


###############################################################
# Full-covariance Gaussian variational family
#   q(x) = N(mu, Sigma), where Sigma = L @ L.T
#   L = diag(exp(log_diag)) + tril(cov_chol_lower, k=-1)
#   The covariance matrix is a full matrix
#   with Cholesky decomposition L @ L.T.
#   The log-density is computed using the formula:
#   log q(x) = -0.5 * (x - mu)^T @ L⁻¹ @ L⁻¹.T @ (x - mu) - 0.5 * log |L| - 0.5 * D * log(2 * pi)
#   where L is the Cholesky decomposition of the covariance matrix.
#################################################################
@struct.dataclass
class FullCovGaussianParams:
    mu: jnp.ndarray
    log_diag: jnp.ndarray
    cov_chol_lower: jnp.ndarray  # strictly lower-triangular (zeros on diag)


class FullCovGaussian(VarFamily):
    """
    Full-covariance Gaussian:
        q(x) = N(mu, Sigma), where Sigma = L @ L.T
        L = diag(exp(log_diag)) + tril(cov_chol_lower, k=-1)
    """
    def __init__(
        self,
        *,
        dim: int,
        mu_init: Optional[jnp.ndarray] = None,
        cov_init: Optional[jnp.ndarray] = None,
        log_diag_init: Optional[jnp.ndarray] = None,
        cov_chol_init: Optional[jnp.ndarray] = None,
    ):
        self.dim = dim

        if mu_init is None:
            mu_init = jnp.zeros(dim)
        else:
            assert mu_init.shape == (dim,), "mu_init must be shape (dim,)"

        if cov_init is not None:
            if log_diag_init is not None or cov_chol_init is not None:
                raise ValueError("Specify either cov_init or (log_diag_init, cov_chol_init), not both.")
            assert cov_init.shape == (dim, dim), "cov_init must be shape (dim, dim)"
            L = jnp.linalg.cholesky(cov_init)
            log_diag_init = jnp.log(jnp.diag(L))
            cov_chol_init = L - jnp.diag(jnp.diag(L))
        else:
            if log_diag_init is None:
                log_diag_init = jnp.zeros(dim)
            else:
                assert log_diag_init.shape == (dim,), "log_diag_init must be shape (dim,)"
            if cov_chol_init is None:
                cov_chol_init = jnp.zeros((dim, dim))
            else:
                assert cov_chol_init.shape == (dim, dim), "cov_chol_init must be shape (dim, dim)"

        self.mu_init = mu_init
        self.log_diag_init = log_diag_init
        self.cov_chol_init = cov_chol_init

    def init_params(
            self,
            key: Optional[jax.Array] = None,  # not used, kept for consistency
            ) -> FullCovGaussianParams:
        mu = self.mu_init
        log_diag = self.log_diag_init
        cov_chol_lower = self.cov_chol_init
        return FullCovGaussianParams(
                    mu=mu,
                    log_diag=log_diag,
                    cov_chol_lower=cov_chol_lower)

    def _construct_cholesky(
            self,
            params: FullCovGaussianParams) -> jnp.ndarray:
        diag = jnp.exp(params.log_diag)
        L = jnp.diag(diag) + jnp.tril(params.cov_chol_lower, k=-1)
        return L

    def sample(
            self, params: FullCovGaussianParams,
            key: jax.Array, n_samples: int,
            ) -> jnp.ndarray:
        z = jr.normal(key, shape=(n_samples, self.dim))
        L = self._construct_cholesky(params)
        return params.mu[None, :] + z @ L.T

    def logdensity(
            self,
            params: FullCovGaussianParams,
            xs: jnp.ndarray,
            ) -> jnp.ndarray:
        mu = params.mu
        L = self._construct_cholesky(params)
        xs_centered = xs - mu[None, :]  # (n_samples, dim)

        # Solve L y = (x - mu)^T → y = L⁻¹(x - mu)^T
        y = jax.scipy.linalg.solve_triangular(L, xs_centered.T, lower=True).T  # (n_samples, dim)
        log_det = jnp.sum(params.log_diag)  # log|L| = sum(log_diag)
        quad = jnp.sum(y**2, axis=-1)  # Mahalanobis term

        return -0.5 * quad - log_det - 0.5 * self.dim * jnp.log(2 * jnp.pi)

    def postprocess(self, params: FullCovGaussianParams) -> dict:
        mu = params.mu
        L = self._construct_cholesky(params)
        cov = L @ L.T
        return {
            "mu": mu,
            "cov_chol": L,
            "cov": cov,
        }


################################################################
# Mixture of diagonal Gaussians variational family
#   q(x) = sum_k softmax(log_weights)[k] * N(mu_k, diag(exp(log_stds_k)^2))
#   where mu_k is the mean vector and log_stds_k is the log of the standard deviation
#   of the k-th diagonal Gaussian distribution.
#   The covariance matrix is a diagonal matrix
#   with diagonal elements exp(log_stds_k)^2.
#   The log-density is computed using the formula:
#   log q(x) = log(sum_k softmax(log_weights)[k] * N(mu_k, diag(exp(log_stds_k)^2)))
###############################################################
@struct.dataclass
class MixtureDiagGaussianParams:
    mus: jnp.ndarray          # shape (K, D)
    log_stds: jnp.ndarray     # shape (K, D)
    log_weights: jnp.ndarray  # shape (K,)


class MixtureDiagGaussian(VarFamily):
    """
    Mixture of diagonal Gaussians variational family.

    This approximates a distribution q(x) as:
        q(x) = sum_k softmax(log_weights)[k] * N(mu_k, diag(exp(log_stds_k)^2))

    Initialization Strategy:
    -------------------------
    - Component means `mu_k` are initialized from a Gaussian centered at `mu_init`
      with spread determined by `log_std_init`.
    - Components are spread out using a scaled average nearest-neighbor distance
      to reduce initial mode collapse.
    - Each component gets its own log-std vector, initially scaled from `log_std_init`.
    - Mixture weights are initialized uniformly in log-space.

    Parameters:
    -----------
    dim : int
        Dimensionality of the latent space.
    num_components : int
        Number of mixture components K (must be ≥ 2).
    mu_init : Optional[jnp.ndarray], shape (dim,)
        Global initialization center for component means (default: zeros).
    log_std_init : Optional[jnp.ndarray], shape (dim,)
        Global initialization for component log standard deviations (default: zeros).
    """

    def __init__(
        self,
        *,
        dim: int,
        num_components: int,
        mu_init: Optional[jnp.ndarray] = None,
        log_std_init: Optional[jnp.ndarray] = None,
    ):
        assert num_components >= 2, "num_components must be >= 2"
        self.dim = dim
        self.K = num_components

        if mu_init is None:
            mu_init = jnp.zeros(dim)
        else:
            assert mu_init.shape == (dim,), "mu_init must have shape (dim,)"

        if log_std_init is None:
            log_std_init = jnp.zeros(dim)
        else:
            assert log_std_init.shape == (dim,), "log_std_init must have shape (dim,)"

        self.mu_init = mu_init
        self.log_std_init = log_std_init

    def init_params(self, key: jax.Array) -> MixtureDiagGaussianParams:
        key, key_z = jr.split(key)

        # Sample component means: base + spread
        zs = jr.normal(key_z, shape=(self.K, self.dim))  # (K, D)
        base_std = jnp.exp(self.log_std_init)
        mus = base_std[None, :] * zs + self.mu_init[None, :]  # shape (K, D)

        # Compute pairwise distances to determine spread
        pdist_sq = jnp.sum((zs[:, None, :] - zs[None, :, :]) ** 2, axis=-1)
        pdist = jnp.sqrt(pdist_sq)
        pdist = pdist + jnp.eye(self.K) * jnp.max(pdist)  # mask diagonal
        nearest_dists = jnp.min(pdist, axis=1)
        avg_dist = jnp.mean(nearest_dists)

        log_stds = jnp.tile(self.log_std_init, (self.K, 1)) + jnp.log(avg_dist / 2.)
        log_weights = jnp.log(jnp.ones(self.K) / self.K)

        return MixtureDiagGaussianParams(
                    mus=mus, log_stds=log_stds,
                    log_weights=log_weights)

    def sample(
            self,
            params: MixtureDiagGaussianParams,
            key: jax.Array,
            n_samples: int,
            ) -> jnp.ndarray:
        K, D = params.mus.shape

        # Split PRNG keys for each sample
        keys = jr.split(key, n_samples)
        log_weights = jax.nn.log_softmax(params.log_weights)
        stds = jnp.exp(params.log_stds)

        def sample_one(key_i):
            key_idx, key_eps = jr.split(key_i)
            k = jr.categorical(key_idx, log_weights)
            eps = jr.normal(key_eps, shape=(D,))
            mu_k = params.mus[k]
            std_k = stds[k]
            return mu_k + eps * std_k

        return jax.vmap(sample_one)(keys)  # shape (n_samples, D)

    def logdensity(
            self,
            params: MixtureDiagGaussianParams,
            xs: jnp.ndarray,
            ) -> jnp.ndarray:
        K, D = params.mus.shape
        log_weights = jax.nn.log_softmax(params.log_weights)  # (K,)

        def logdensity_k(mu_k, log_std_k):
            std_k = jnp.exp(log_std_k)
            x_centered = (xs - mu_k[None, :]) / std_k[None, :]
            log_det = jnp.sum(log_std_k)
            logp = -0.5 * jnp.sum(x_centered**2, axis=-1)
            logp -= 0.5 * D * jnp.log(2 * jnp.pi)
            logp -= log_det
            return logp  # shape (N,)

        logdensities = jax.vmap(logdensity_k, in_axes=(0, 0))(params.mus, params.log_stds)  # (K, N)
        return jax.nn.logsumexp(logdensities + log_weights[:, None], axis=0)  # (N,)

    def postprocess(self, params: MixtureDiagGaussianParams) -> dict:
        return {
            "mu": params.mus,
            "std": jnp.exp(params.log_stds),
            "weights": jax.nn.softmax(params.log_weights),
        }

    def neg_elbo(
        self,
        *,
        params: MixtureDiagGaussianParams,
        xs: Optional[jnp.ndarray] = None,  # not used in this override
        logtarget_batch: Callable[[jnp.ndarray], jnp.ndarray],
        stop_gradient_entropy: bool = True,
        key: jax.Array,
        n_samples: int,
    ) -> jnp.ndarray:
        """
        Override neg_elbo for Gaussian mixture with per-component estimation.

        Computes:
            KL[q || p] ≈ sum_k alpha_k [ E_{q_k}[log q(x)] - E_{q_k}[log p(x)] ]
        """
        K, D = params.mus.shape
        log_weights = jax.nn.log_softmax(params.log_weights)
        alphas = jnp.exp(log_weights)

        def one_component_term(k, key):
            mu_k = params.mus[k]
            log_std_k = params.log_stds[k]
            std_k = jnp.exp(log_std_k)

            # Sample from component q_k
            eps = jr.normal(key, shape=(n_samples, D))
            xs_k = mu_k + eps * std_k  # shape (N, D)

            # Entropy term
            logdensity_k = self.logdensity_batch(params if not stop_gradient_entropy else jax.lax.stop_gradient(params), xs_k)
            entropy_k = -jnp.mean(logdensity_k)

            # Cross-entropy term
            logp_k = jnp.mean(logtarget_batch(xs_k))

            return alphas[k] * (-entropy_k - logp_k)  # this is KL_k

        keys = jr.split(key, K)
        kl_terms = jax.vmap(one_component_term, in_axes=(0, 0))(jnp.arange(K), keys)
        return jnp.sum(kl_terms)

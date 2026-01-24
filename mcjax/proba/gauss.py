from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct
from jax.scipy.linalg import solve_triangular

from .distribution import DistributionLike
from .mixture import MixtureSameFamily, MixtureSameFamilyParams


def _forward(method_name: str):
    """
    Helper to forward method calls to an underlying `base` distribution.

    Used by mixture wrappers to expose the same API as `MixtureSameFamily`
    without writing boilerplate.
    """
    def wrapper(self, *args, **kwargs):
        return getattr(self.base, method_name)(*args, **kwargs)
    return wrapper


#######################################
# Diagonal Gaussian
#######################################
@struct.dataclass
class GaussianDiagParams:
    mu: jnp.ndarray        # shape: (D,) or (K, D) for mixtures
    log_std: jnp.ndarray   # shape: (D,) or (K, D)


@struct.dataclass
class GaussianDiag:
    """
    Gaussian distribution with diagonal covariance:
        q(x) = N(mu, diag(exp(log_std)^2))

    where:
        - mu:      mean vector in R^D
        - log_std: log standard deviation vector in R^D

    This follows the `DistributionLike` protocol.
    """
    dim: int

    @classmethod
    def create(cls, *, dim: int) -> "GaussianDiag":
        return cls(dim=dim)

    def init_params(
        self,
        mu: Optional[jnp.ndarray] = None,
        log_std: Optional[jnp.ndarray] = None,
    ) -> GaussianDiagParams:
        """
        Initialize parameters for the diagonal Gaussian distribution.

        Parameters
        ----------
        mu : jnp.ndarray, optional
            Mean vector, shape (dim,). If None, defaults to zeros.
        log_std : jnp.ndarray, optional
            Log standard deviation vector, shape (dim,). If None, defaults to zeros.

        Returns
        -------
        GaussianDiagParams
            Initialized parameters.
        """
        if mu is not None and mu.shape != (self.dim,):
            raise ValueError(f"mu must have shape ({self.dim},), got {mu.shape}")
        if log_std is not None and log_std.shape != (self.dim,):
            raise ValueError(f"log_std must have shape ({self.dim},), got {log_std.shape}")

        mu = mu if mu is not None else jnp.zeros(self.dim)
        log_std = log_std if log_std is not None else jnp.zeros(self.dim)
        return GaussianDiagParams(mu=mu, log_std=log_std)

    def sample(
        self,
        params: GaussianDiagParams,
        key: jax.Array,
        n_samples: int,
    ) -> jnp.ndarray:
        """
        Sample from N(mu, diag(exp(log_std)^2)).

        Parameters
        ----------
        params : GaussianDiagParams
            Distribution parameters.
        key : jax.Array
            PRNG key.
        n_samples : int
            Number of samples.

        Returns
        -------
        jnp.ndarray
            Samples of shape (n_samples, dim).
        """
        eps = jr.normal(key, shape=(n_samples, self.dim))
        std = jnp.exp(params.log_std)
        return params.mu + eps * std

    def log_prob(self, params: GaussianDiagParams, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log-density log q(x; params) for a single x.

        Parameters
        ----------
        params : GaussianDiagParams
        x : jnp.ndarray
            Point in R^dim, shape (dim,).

        Returns
        -------
        jnp.ndarray
            Scalar log-density.
        """
        assert x.ndim == 1 and x.shape[0] == self.dim, \
            f"x must be a 1D array of shape ({self.dim},), got {x.shape}"

        std = jnp.exp(params.log_std)
        normed = (x - params.mu) / std
        log_det_cov = 2 * jnp.sum(params.log_std)
        log_prob = -0.5 * jnp.sum(normed**2)
        log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)
        log_prob -= 0.5 * log_det_cov
        return log_prob

    def log_normalization(self, params: GaussianDiagParams) -> jnp.ndarray:
        """
        Log normalization constant of q(x; params).

        For a properly parameterized Gaussian, this is zero, since we define
        `log_prob` as the full normalized log-density.
        """
        return jnp.array(0.0)

    def postprocess(self, params: GaussianDiagParams) -> dict:
        """
        Transform internal parameters into user-facing outputs.

        Returns
        -------
        dict
            Dictionary with keys:
                - "mu":  mean vector
                - "std": standard deviation vector
        """
        return {
            "mu": jnp.asarray(params.mu),
            "std": jnp.exp(jnp.asarray(params.log_std)),
        }


# For static protocol checking only; not used at runtime.
_dist_gauss_diag: DistributionLike = GaussianDiag.create(dim=1)


################################################
# Full-covariance Gaussian
################################################
@struct.dataclass
class GaussianFullCovParams:
    mu: jnp.ndarray           # shape: (D,) or (K, D)
    log_diag: jnp.ndarray     # shape: (D,) or (K, D)
    cov_chol_lower: jnp.ndarray  # shape: (D, D) or (K, D, D)
    # L = diag(exp(log_diag)) + tril(cov_chol_lower, k=-1) is the Cholesky factor


@struct.dataclass
class GaussianFullCov:
    """
    Full-covariance Gaussian distribution:

        q(x) = N(mu, Sigma), with Sigma = L @ L.T

    where:
        L = diag(exp(log_diag)) + tril(cov_chol_lower, k=-1).

    This structure allows initializing from a covariance matrix or from
    unconstrained Cholesky parameters; it is JAX-friendly and works both
    for single-component and batched parameters (e.g. in mixtures).
    """
    dim: int

    @classmethod
    def create(cls, dim: int) -> "GaussianFullCov":
        return cls(dim=dim)

    def init_params(
        self,
        mu: Optional[jnp.ndarray] = None,
        cov: Optional[jnp.ndarray] = None,
        log_diag: Optional[jnp.ndarray] = None,
    ) -> GaussianFullCovParams:
        """
        Initialize parameters for the full-covariance Gaussian.

        Parameters
        ----------
        mu : jnp.ndarray, optional
            Mean vector, shape (dim,). If None, defaults to zeros.
        cov : jnp.ndarray, optional
            Covariance matrix, shape (dim, dim). If provided, overrides `log_diag`.
        log_diag : jnp.ndarray, optional
            Log of the diagonal entries of the Cholesky factor. Shape (dim,).
            Used if `cov` is None.

        Returns
        -------
        GaussianFullCovParams
            Initialized parameters.
        """
        mu = mu if mu is not None else jnp.zeros(self.dim)

        if cov is not None:
            L = jnp.linalg.cholesky(cov)
            log_diag = jnp.log(jnp.diag(L))
            cov_chol_lower = L - jnp.diag(jnp.diag(L))
        else:
            log_diag = log_diag if log_diag is not None else jnp.zeros(self.dim)
            cov_chol_lower = jnp.zeros((self.dim, self.dim))

        # shape checks
        if mu.ndim != 1 or mu.shape[0] != self.dim:
            raise ValueError(f"mu must be a 1D array of shape ({self.dim},), got {mu.shape}")
        if log_diag.ndim != 1 or log_diag.shape[0] != self.dim:
            raise ValueError(f"log_diag must be a 1D array of shape ({self.dim},), got {log_diag.shape}")
        if cov_chol_lower.ndim != 2 or cov_chol_lower.shape != (self.dim, self.dim):
            raise ValueError(
                f"cov_chol_lower must be a 2D array of shape ({self.dim}, {self.dim}), "
                f"got {cov_chol_lower.shape}"
            )

        return GaussianFullCovParams(
            mu=mu,
            log_diag=log_diag,
            cov_chol_lower=cov_chol_lower,
        )

    def _construct_cholesky(self, params: GaussianFullCovParams) -> jnp.ndarray:
        """
        Construct the Cholesky factor L from the parameters.

        Supports both:
            - unbatched params: log_diag (D,), cov_chol_lower (D, D) -> L (D, D)
            - batched params:  log_diag (K, D), cov_chol_lower (K, D, D) -> L (K, D, D)
        """
        diag = jnp.exp(params.log_diag)

        if diag.ndim == 1:
            diag_mat = jnp.diag(diag)
        elif diag.ndim == 2:
            # diag: (K, D) -> diag_mat: (K, D, D)
            diag_mat = jax.vmap(jnp.diag)(diag)
        else:
            raise ValueError(
                f"log_diag must have shape (D,) or (K, D), got shape {diag.shape}"
            )

        cov_lower = jnp.tril(params.cov_chol_lower, k=-1)
        L = diag_mat + cov_lower
        return L

    def sample(
        self,
        params: GaussianFullCovParams,
        key: jax.Array,
        n_samples: int,
    ) -> jnp.ndarray:
        """
        Sample from N(mu, Σ) where Σ = L @ L.T.

        Parameters
        ----------
        params : GaussianFullCovParams
        key : jax.Array
        n_samples : int

        Returns
        -------
        jnp.ndarray
            Samples of shape (n_samples, dim).
        """
        z = jr.normal(key, shape=(n_samples, self.dim))
        L = self._construct_cholesky(params)   # (D, D)
        return params.mu + z @ L.T

    def log_prob(
        self,
        params: GaussianFullCovParams,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute log-density log q(x; params) for a single x.

        Parameters
        ----------
        params : GaussianFullCovParams
        x : jnp.ndarray
            Point in R^dim, shape (dim,).

        Returns
        -------
        jnp.ndarray
            Scalar log-density.
        """
        assert x.ndim == 1 and x.shape[0] == self.dim, \
            f"x must be a 1D array of shape ({self.dim},), got {x.shape}"

        mu = params.mu
        L = self._construct_cholesky(params)   # (D, D)
        xs_centered = x - mu
        y = solve_triangular(L, xs_centered, lower=True)
        quad = jnp.sum(y**2)
        log_det_cov = 2 * jnp.sum(params.log_diag)

        return -0.5 * quad - 0.5 * log_det_cov - 0.5 * self.dim * jnp.log(2 * jnp.pi)

    def log_normalization(self, params: GaussianFullCovParams) -> jnp.ndarray:
        """
        Log normalization constant of q(x; params).

        For a properly parameterized Gaussian, this is zero, since `log_prob`
        is the full normalized log-density.
        """
        return jnp.array(0.0)

    def postprocess(self, params: GaussianFullCovParams) -> dict:
        """
        Transform parameters into user-facing outputs.

        Supports both single-component and batched parameters.

        Returns
        -------
        dict
            Dictionary with keys:
                - "mu":       mean(s), shape (D,) or (K, D)
                - "cov_chol": Cholesky factor(s) L, shape (D, D) or (K, D, D)
                - "cov":      covariance(s) Sigma = L @ L.T, shape (D, D) or (K, D, D)
        """
        mu = params.mu
        L = self._construct_cholesky(params)        # (D, D) or (K, D, D)
        cov = jnp.matmul(L, jnp.swapaxes(L, -1, -2))
        return {
            "mu": mu,
            "cov_chol": L,
            "cov": cov,
        }


# For static protocol checking only; not used at runtime.
_dist_gauss_full: DistributionLike = GaussianFullCov.create(dim=1)


################################################
# Mixture of Diagonal Gaussians
################################################
@struct.dataclass
class GaussianDiagMixture:
    """
    Thin wrapper around `MixtureSameFamily` using `GaussianDiag`
    as the component distribution.

    Represents:
        q(x) = sum_k softmax(log_weights)[k] * N(mu_k, diag(exp(log_std_k)^2))
    """
    dim: int
    num_components: int
    base: MixtureSameFamily

    # Forward core methods to `base`
    sample = _forward("sample")
    log_prob = _forward("log_prob")
    log_prob_batch = _forward("log_prob_batch")
    log_normalization = _forward("log_normalization")
    postprocess = _forward("postprocess")

    @classmethod
    def create(
        cls,
        dim: int,
        num_components: int,
    ) -> "GaussianDiagMixture":
        """
        Factory to construct a mixture-of-diagonal-Gaussians model.

        Parameters
        ----------
        dim : int
            Dimensionality of each Gaussian component.
        num_components : int
            Number of mixture components K.

        Returns
        -------
        GaussianDiagMixture
            Mixture model. Parameters are initialized via `init_params`.
        """
        assert num_components >= 2, "num_components must be >= 2"

        base_dist = GaussianDiag.create(dim=dim)
        base = MixtureSameFamily.create(base_dist=base_dist)

        return cls(
            dim=dim,
            num_components=num_components,
            base=base,
        )

    def init_params(
        self,
        key: jax.Array,
        mu: Optional[jnp.ndarray] = None,
        log_std: Optional[jnp.ndarray] = None,
    ) -> MixtureSameFamilyParams:
        """
        Initialize parameters for the GaussianDiagMixture.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        mu : jnp.ndarray, optional
            Global center for component means, shape (dim,). Defaults to zeros.
        log_std : jnp.ndarray, optional
            Base log standard deviation, shape (dim,). Defaults to zeros.

        Returns
        -------
        MixtureSameFamilyParams
            Initialized mixture parameters.
        """
        mu_init = mu if mu is not None else jnp.zeros(self.dim)
        log_std_init = log_std if log_std is not None else jnp.zeros(self.dim)

        if mu_init.ndim != 1 or mu_init.shape[0] != self.dim:
            raise ValueError(f"mu must be a 1D array of shape ({self.dim},), got {mu_init.shape}")
        if log_std_init.ndim != 1 or log_std_init.shape[0] != self.dim:
            raise ValueError(f"log_std must be a 1D array of shape ({self.dim},), got {log_std_init.shape}")

        key, key_z = jr.split(key)

        # Random offsets for component means
        base_std = jnp.exp(log_std_init)
        zs = jr.normal(key_z, shape=(self.num_components, self.dim))   # (K, D)
        mus = zs * base_std[None, :] + mu_init[None, :]                # (K, D)

        # Replicate base log_std per component
        log_stds = jnp.tile(log_std_init[None, :], (self.num_components, 1))  # (K, D)

        # Uniform mixture weights
        log_weights = jnp.full((self.num_components,), -jnp.log(self.num_components))

        # Pack component params as a batched GaussianDiagParams
        component_params = GaussianDiagParams(
            mu=mus,
            log_std=log_stds,
        )

        params = self.base.init_params(
            component_params=component_params,
            log_weights=log_weights,
        )

        return params


################################################
# Mixture of Full-Covariance Gaussians
################################################
@struct.dataclass
class GaussianFullMixture:
    """
    Thin wrapper around `MixtureSameFamily` using `GaussianFullCov`
    as the component distribution.

    Each component has its own mean, and starts with an identical covariance
    structure, which can then be learned independently.
    """
    dim: int
    num_components: int
    base: MixtureSameFamily

    # Forward core methods to `base`
    sample = _forward("sample")
    log_prob = _forward("log_prob")
    log_prob_batch = _forward("log_prob_batch")
    log_normalization = _forward("log_normalization")
    postprocess = _forward("postprocess")

    @classmethod
    def create(
        cls,
        dim: int,
        num_components: int,
    ) -> "GaussianFullMixture":
        """
        Factory to construct a mixture-of-full-covariance-Gaussians model.

        Parameters
        ----------
        dim : int
            Dimensionality of each Gaussian component.
        num_components : int
            Number of mixture components K.

        Returns
        -------
        GaussianFullMixture
            Mixture model. Parameters are initialized via `init_params`.
        """
        assert num_components >= 2, "num_components must be >= 2"

        base_dist = GaussianFullCov.create(dim=dim)
        base = MixtureSameFamily.create(base_dist=base_dist)

        mixture = cls(
            dim=dim,
            num_components=num_components,
            base=base,
        )

        return mixture

    def init_params(
        self,
        key: jax.Array,
        mu: Optional[jnp.ndarray] = None,
        cov: Optional[jnp.ndarray] = None,
        log_diag: Optional[jnp.ndarray] = None,
    ) -> MixtureSameFamilyParams:
        """
        Initialize parameters for the GaussianFullMixture.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        mu : jnp.ndarray, optional
            Global mean for sampling component means. Shape (dim,). Defaults to zeros.
        cov : jnp.ndarray, optional
            Global covariance matrix, shape (dim, dim). If provided, overrides `log_diag`.
        log_diag : jnp.ndarray, optional
            Log diagonal of the Cholesky factor, shape (dim,). Used if `cov` is None.

        Returns
        -------
        MixtureSameFamilyParams
            Initialized mixture parameters.
        """
        D = self.dim
        K = self.num_components

        if mu is not None:
            if mu.ndim != 1 or mu.shape[0] != D:
                raise ValueError(f"mu must be a 1D array of shape ({D},), got {mu.shape}")
        if cov is not None:
            if cov.ndim != 2 or cov.shape != (D, D):
                raise ValueError(f"cov must be a 2D array of shape ({D}, {D}), got {cov.shape}")
        if log_diag is not None:
            if log_diag.ndim != 1 or log_diag.shape[0] != D:
                raise ValueError(f"log_diag must be a 1D array of shape ({D},), got {log_diag.shape}")

        mu_init = mu if mu is not None else jnp.zeros(D)

        # Cholesky parameters for the global covariance
        if cov is not None:
            L = jnp.linalg.cholesky(cov)
            log_diag_global = jnp.log(jnp.diag(L))
            cov_chol_lower_global = L - jnp.diag(jnp.diag(L))
        else:
            log_diag_global = log_diag if log_diag is not None else jnp.zeros(D)
            cov_chol_lower_global = jnp.zeros((D, D))

        # Base sampler for component means: N(mu_init, Σ_global)
        def sample_from_base(subkey: jax.Array) -> jnp.ndarray:
            eps = jr.normal(subkey, shape=(D,))
            L = jnp.diag(jnp.exp(log_diag_global)) + jnp.tril(cov_chol_lower_global, k=-1)
            return mu_init + L @ eps

        keys = jr.split(key, K)
        mus = jax.vmap(sample_from_base)(keys)  # shape: (K, D)

        # Replicate covariance parameters per component
        log_diags = jnp.tile(log_diag_global[None, :], (K, 1))                 # (K, D)
        cov_chol_lowers = jnp.tile(cov_chol_lower_global[None, :, :], (K, 1, 1))  # (K, D, D)

        # Uniform mixture weights
        log_weights = jnp.full((K,), -jnp.log(K))

        # Pack component parameters (batched) for GaussianFullCov
        component_params = GaussianFullCovParams(
            mu=mus,
            log_diag=log_diags,
            cov_chol_lower=cov_chol_lowers,
        )

        params = self.base.init_params(
            component_params=component_params,
            log_weights=log_weights,
        )

        return params
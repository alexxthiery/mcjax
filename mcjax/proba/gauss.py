from typing import Optional, Callable, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct
from jax.scipy.linalg import solve_triangular
#from .var_family import VarFamily
from .distribution import Distribution
from .mixture import MixtureSameFamily, MixtureSameFamilyParams


def _forward(method_name):
    """ helper function to forward method calls to the base distribution """
    def wrapper(self, *args, **kwargs):
        return getattr(self.base, method_name)(*args, **kwargs)
    return wrapper


#######################################
# Diagonal Gaussian variational family
#######################################
@struct.dataclass
class GaussianDiagParams:
    mu: jnp.ndarray
    log_std: jnp.ndarray


@struct.dataclass
class GaussianDiag(Distribution):
    dim: int

    @classmethod
    def create(
        cls,
        *,
        dim: Optional[int] = None,
        mu_init: Optional[jnp.ndarray] = None,
        log_std_init: Optional[jnp.ndarray] = None,
    ) -> tuple["GaussianDiag", GaussianDiagParams]:
        """
        Factory method to create a GaussianDiag instance and its initialized parameters.

        Parameters
        ----------
        dim : int, optional
            Dimensionality of the Gaussian. Can be inferred from mu_init if not provided.
        mu_init : jnp.ndarray, optional
            Initial mean vector. If not provided, defaults to zeros.
        log_std_init : jnp.ndarray, optional
            Initial log standard deviation. If not provided, defaults to zeros.

        Returns
        -------
        model : GaussianDiag
            The variational family object.
        params : GaussianDiagParams
            Initialized parameters for the model.

        Raises
        ------
        ValueError
            If neither dim nor mu_init/log_std_init are provided.
        """
        if mu_init is not None:
            inferred_dim = mu_init.shape[0]
        elif log_std_init is not None:
            inferred_dim = log_std_init.shape[0]
        elif dim is not None:
            inferred_dim = dim
        else:
            raise ValueError("Must specify at least one of dim, mu_init, or log_std_init.")

        mu = mu_init if mu_init is not None else jnp.zeros(inferred_dim)
        log_std = log_std_init if log_std_init is not None else jnp.zeros(inferred_dim)

        return cls(dim=inferred_dim), GaussianDiagParams(mu=mu, log_std=log_std)

    def sample(self, params: GaussianDiagParams, key: jax.Array, n_samples: int) -> jnp.ndarray:
        eps = jr.normal(key, shape=(n_samples, self.dim))
        std = jnp.exp(params.log_std)
        return params.mu + eps * std

    def log_prob(self, params: GaussianDiagParams, xs: jnp.ndarray) -> jnp.ndarray:
        std = jnp.exp(params.log_std)
        normed = (xs - params.mu) / std
        log_det_cov = 2 * jnp.sum(params.log_std)
        log_prob = -0.5 * jnp.sum(normed**2, axis=-1)
        log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)
        log_prob -= 0.5 * log_det_cov
        return log_prob

    def log_normalization(self) -> jnp.ndarray:
        # the distribution is already normalized
        return 1.

    def postprocess(self, params: GaussianDiagParams) -> GaussianDiagParams:
        return {"mu": params.mu, "std": jnp.exp(params.log_std)}

################################################
# Full-covariance Gaussian variational family
################################################
@struct.dataclass
class GaussianFullCovParams:
    mu: jnp.ndarray
    log_diag: jnp.ndarray
    cov_chol_lower: jnp.ndarray  # strictly lower triangular (zeros on diag)
    # L = exp(log_diag)+cov_chol_lower is the Cholesky factor of the covariance matrix


@struct.dataclass
class GaussianFullCov(Distribution):
    """
    Full-covariance Gaussian variational family:
        q(x) = N(mu, Σ), with Σ = L @ L.T
        where L = diag(exp(log_diag)) + tril(cov_chol_lower, k=-1)

    Designed to be JAX-compatible and support flexible initialization.
    """
    dim: int

    @classmethod
    def create(
        cls,
        *,
        dim: Optional[int] = None,
        mu_init: Optional[jnp.ndarray] = None,
        cov_init: Optional[jnp.ndarray] = None,
        log_diag_init: Optional[jnp.ndarray] = None,
    ) -> tuple["GaussianFullCov", GaussianFullCovParams]:
        # infer the dimension from provided initializations
        if mu_init is not None:
            inferred_dim = mu_init.shape[0]
        elif log_diag_init is not None:
            inferred_dim = log_diag_init.shape[0]
        elif cov_init is not None:
            inferred_dim = cov_init.shape[0]
        elif dim is not None:
            inferred_dim = dim
        else:
            raise ValueError("Must provide at least one of dim, mu_init, log_diag_init, cov_init.")

        # initialize mu
        mu = mu_init if mu_init is not None else jnp.zeros(inferred_dim)

        # initialize log_diag and cov_chol_lower
        if cov_init is not None:
            L = jnp.linalg.cholesky(cov_init)
            log_diag = jnp.log(jnp.diag(L))
            cov_chol_lower = L - jnp.diag(jnp.diag(L))
        else:
            log_diag = log_diag_init if log_diag_init is not None else jnp.zeros(inferred_dim)
            cov_chol_lower = jnp.zeros((inferred_dim, inferred_dim))

        # create the parameters
        params = GaussianFullCovParams(
            mu=mu,
            log_diag=log_diag,
            cov_chol_lower=cov_chol_lower,
        )
        return cls(dim=inferred_dim), params

    def _construct_cholesky(self, params: GaussianFullCovParams) -> jnp.ndarray:
        """ Construct the Cholesky factor L from the parameters. """
        diag = jnp.exp(params.log_diag)
        L = jnp.diag(diag) + jnp.tril(params.cov_chol_lower, k=-1)
        return L

    def sample(self, params: GaussianFullCovParams, key: jax.Array, n_samples: int) -> jnp.ndarray:
        z = jr.normal(key, shape=(n_samples, self.dim))
        L = self._construct_cholesky(params)
        return params.mu + z @ L.T

    def log_prob(self, params: GaussianFullCovParams, xs: jnp.ndarray) -> jnp.ndarray:
        mu = params.mu
        L = self._construct_cholesky(params)
        xs_centered = xs - mu
        y = solve_triangular(L, xs_centered.T, lower=True).T
        log_det_cov = 2*jnp.sum(params.log_diag)
        quad = jnp.sum(y**2, axis=-1)
        return -0.5 * quad - 0.5 * log_det_cov - 0.5 * self.dim * jnp.log(2 * jnp.pi)

    def log_normalization(self) -> jnp.ndarray:
        # the distribution is already normalized
        return 1.

    def postprocess(self, params: GaussianFullCovParams) -> dict:
        mu = params.mu
        L = self._construct_cholesky(params)  # shape: (K, D, D) or (D, D)
        cov = jnp.matmul(L, jnp.swapaxes(L, -1, -2))  # works with or without batch
        return {
            "mu": mu,
            "cov_chol": L,
            "cov": cov,
        }

################################################
# Mixture of Diagonal Gaussians
################################################
@struct.dataclass
class GaussianDiagMixture:
    """
    Thin wrapper around MixtureSameFamily using GaussianDiag as the component distribution.

    Represents:
        q(x) = sum_k softmax(log_weights)[k] * N(mu_k, diag(exp(log_std_k)^2))
    """
    dim: int
    num_components: int
    base: MixtureSameFamily

    # forward methods to base distribution
    sample = _forward("sample")
    log_prob = _forward("log_prob")
    og_prob_batch = _forward("log_prob_batch")
    log_normalization = _forward("log_normalization")
    postprocess = _forward("postprocess")
    neg_elbo = _forward("neg_elbo")
    
    @classmethod
    def create(
        cls,
        *,
        dim: int,
        num_components: int,
        mu_init: Optional[jnp.ndarray] = None,
        log_std_init: Optional[jnp.ndarray] = None,
        key: jax.Array,
    ) -> Tuple["GaussianDiagMixture", MixtureSameFamilyParams]:
        """
        Factory to initialize a mixture of diagonal Gaussians with spread-out components.

        Parameters
        ----------
        dim : int
            Dimensionality of each Gaussian component.
        num_components : int
            Number of components K.
        mu_init : jnp.ndarray, shape (D,), optional
            Center for the means.
        log_std_init : jnp.ndarray, shape (D,), optional
            Initial log standard deviation vector.
        key : jax.Array
            PRNG key.

        Returns
        -------
        model : GaussianDiagMixture
        params : MixtureSameFamilyParams
        """
        assert num_components >= 2, "num_components must be >= 2"
        mu_init = mu_init if mu_init is not None else jnp.zeros(dim)
        log_std_init = log_std_init if log_std_init is not None else jnp.zeros(dim)

        key, key_z = jr.split(key)

        # Create random base for means
        zs = jr.normal(key_z, shape=(num_components, dim))  # shape (K, D)
        base_std = jnp.exp(log_std_init)
        mus = zs * base_std[None, :] + mu_init[None, :]      # shape: (K, D)

        # Spread components to avoid mode collapse
        pdist_sq = jnp.sum((zs[:, None, :] - zs[None, :, :]) ** 2, axis=-1)
        pdist = jnp.sqrt(pdist_sq + jnp.eye(num_components) * 1e6)
        avg_nn_dist = jnp.mean(jnp.min(pdist, axis=1))

        log_stds = jnp.tile(log_std_init, (num_components, 1)) + jnp.log(avg_nn_dist / 2.)
        log_weights = jnp.full((num_components,), -jnp.log(num_components))

        # Pack into GaussianDiagParams with batch shape
        component_params = struct.dataclass(GaussianDiagParams)(
            mu=mus,
            log_std=log_stds
        )

        # Instantiate mixture model
        base, params = MixtureSameFamily.create(
            component_cls=GaussianDiag,
            component_kwargs=dict(dim=dim),
            component_params=component_params,
            log_weights=log_weights,
        )

        mixture = cls(
            dim=dim,
            num_components=num_components,
            base=base
        )
        return mixture, params


################################################
# Mixture of Full Covariance Gaussians
################################################
@struct.dataclass
class GaussianFullMixture:
    """
    Thin wrapper around MixtureSameFamily using GaussianFullCov as the component distribution.

    Each component has its own mean, and starts with an identical covariance structure,
    which is learnable independently during training.
    """
    dim: int
    num_components: int
    base: MixtureSameFamily

    # forward methods to base distribution
    sample = _forward("sample")
    log_prob = _forward("log_prob")
    og_prob_batch = _forward("log_prob_batch")
    log_normalization = _forward("log_normalization")
    postprocess = _forward("postprocess")
    neg_elbo = _forward("neg_elbo")

    @classmethod
    def create(
        cls,
        *,
        dim: Optional[int] = None,
        num_components: int,
        mu_init: Optional[jnp.ndarray] = None,
        cov_init: Optional[jnp.ndarray] = None,
        log_diag_init: Optional[jnp.ndarray] = None,
        key: jax.Array,
    ) -> Tuple["GaussianFullMixture", MixtureSameFamilyParams]:
        """
        Initialize a mixture of full-covariance Gaussians.

        Each component starts with its own mean (sampled from a base Gaussian),
        and identical covariance structure (which becomes learnable per component).

        Parameters
        ----------
        dim : int, optional
            Dimensionality of the Gaussian. Inferred if not given.
        num_components : int
            Number of mixture components.
        mu_init : jnp.ndarray, shape (D,), optional
            Initial global mean.
        cov_init : jnp.ndarray, shape (D, D), optional
            Initial full covariance (overrides log_diag/cov_chol).
        log_diag_init : jnp.ndarray, shape (D,), optional
            Initial diagonal of the Cholesky factor.
        key : jax.Array
            PRNG key.

        Returns
        -------
        model : GaussianFullMixture
        params : MixtureSameFamilyParams
        """
        assert num_components >= 2, "num_components must be >= 2"

        # Infer dimensionality
        if mu_init is not None:
            D = mu_init.shape[0]
        elif log_diag_init is not None:
            D = log_diag_init.shape[0]
        elif cov_init is not None:
            D = cov_init.shape[0]
        elif dim is not None:
            D = dim
        else:
            raise ValueError("Must provide dim or one of the init tensors.")

        K = num_components

        # Default mu and covariance init
        mu_init = mu_init if mu_init is not None else jnp.zeros(D)

        if cov_init is not None:
            L = jnp.linalg.cholesky(cov_init)
            log_diag = jnp.log(jnp.diag(L))
            cov_chol_lower = L - jnp.diag(jnp.diag(L))
        else:
            log_diag = log_diag_init if log_diag_init is not None else jnp.zeros(D)
            cov_chol_lower = jnp.zeros((D, D))

        # Construct full Cholesky matrix: L = diag(exp(log_diag)) + tril(cov_chol_lower, k=-1)
        def sample_from_base(key):
            eps = jr.normal(key, shape=(D,))
            L = jnp.diag(jnp.exp(log_diag)) + jnp.tril(cov_chol_lower, k=-1)
            return mu_init + L @ eps

        # Sample K means from base distribution
        keys = jr.split(key, K)
        mus = jax.vmap(sample_from_base)(keys)  # shape: (K, D)

        # Replicate covariances per component
        log_diags = jnp.tile(log_diag[None, :], (K, 1))            # (K, D)
        cov_chol_lowers = jnp.tile(cov_chol_lower[None, :, :], (K, 1, 1))  # (K, D, D)

        # Uniform mixture weights
        log_weights = jnp.full((K,), -jnp.log(K))

        # Pack parameters
        component_params = GaussianFullCovParams(
            mu=mus,
            log_diag=log_diags,
            cov_chol_lower=cov_chol_lowers
        )

        base, params = MixtureSameFamily.create(
            component_cls=GaussianFullCov,
            component_kwargs={"dim": D},
            component_params=component_params,
            log_weights=log_weights
        )
        
        mixture = cls(
            dim=D,
            num_components=K,
            base=base
        )

        return mixture, params

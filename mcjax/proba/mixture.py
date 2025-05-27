from flax import struct
import jax
import jax.numpy as jnp
from typing import Any, Optional, Callable
from .distribution import DistributionLike


@struct.dataclass
class MixtureSameFamilyParams:
    component_params: Any
    log_weights: jnp.ndarray


@struct.dataclass
class MixtureSameFamily:
    """
    Mixture distribution where all components belong to the same distribution class.

    This class enables defining and working with mixtures of distributions that share the same
    parameter structure and functional interface. It is designed to be JAX-friendly and can be
    used seamlessly with JIT, vmap, grad, and optimization tools like Optax.

    Example:
    --------
    To create a mixture of diagonal Gaussians with 5 components in D=3:

        mixture = MixtureSameFamily.create(
            base_dist=DiagGaussian(dim=3),)
        parameters = mixture.init_params(
            component_params=DiagGaussianParams(
                mu=jnp.zeros((5, 3)),
                log_std=jnp.zeros((5, 3))
            ),
            log_weights=jnp.zeros(5)
        )

    Attributes:
    -----------
    component_cls : Type[Distribution]
        The distribution class shared by all components (e.g., DiagGaussian).
        Must implement the standard Distribution interface.
    component_kwargs : dict[str, Any]
        Arguments used to instantiate component_cls when needed (e.g., {"dim": 3}).
    """
    dim: int
    base_dist: DistributionLike

    @classmethod
    def create(
        cls,
        *,
        base_dist: DistributionLike,
    ) -> "MixtureSameFamily":
        """
        Factory method to construct a mixture of same-family components.

        Parameters:
        -----------
        base_dist : DistributionLike
            The base distribution shared by all components (e.g., DiagGaussian).

        Returns:
        --------
        mixture : MixtureSameFamily
            An instance representing the mixture model.
        """
        dim = base_dist.dim
        return cls(
            dim=dim,
            base_dist=base_dist,
        )

    def init_params(
        self,
        *,
        component_params: Any,
        log_weights: jnp.ndarray,
    ) -> MixtureSameFamilyParams:
        """
        Factory method to construct a mixture of same-family components.

        Parameters:
        -----------
        component_params : Any
            A batched PyTree of parameters for all K components, structured according to
            the component distribution (e.g., DiagGaussianParams with shape [K, D]).
        log_weights : jnp.ndarray
            Unnormalized log-weights of the mixture components, shape (K,).

        Returns:
        --------
        params : MixtureSameFamilyParams
            A dataclass containing the initialized parameters of the mixture.
        """

        if log_weights.ndim != 1:
            raise ValueError("log_weights must be a 1D array of shape (n_components,)")

        params = MixtureSameFamilyParams(
                    component_params=component_params,
                    log_weights=log_weights,
                    )
        return params

    def sample(self, *, params: MixtureSameFamilyParams, key: jax.Array, n_samples: int) -> jnp.ndarray:
        key_cat, key_sample = jax.random.split(key)

        # Sample component indices: (n_samples,)
        indices = jax.random.categorical(key_cat, logits=params.log_weights, shape=(n_samples,))

        # Generate one key per sample
        keys = jax.random.split(key_sample, n_samples)
        dist = self.base_dist

        def sample_one(i, key_i):
            k = indices[i]
            comp_params = jax.tree_map(lambda x: x[k], params.component_params)
            return dist.sample(comp_params, key_i, 1)[0]

        return jax.vmap(sample_one, in_axes=(0, 0))(jnp.arange(n_samples), keys)

    def _log_prob_single(self, params: MixtureSameFamilyParams, x: jnp.ndarray) -> jnp.ndarray:
        dist = self.base_dist
        log_weights = jax.nn.log_softmax(params.log_weights)
        log_probs = jax.vmap(lambda p: dist.log_prob(p, x))(params.component_params)  # shape: (K,)
        return jax.nn.logsumexp(log_probs + log_weights)

    def log_prob(self, params: MixtureSameFamilyParams, x: jnp.ndarray) -> jnp.ndarray:
        return self._log_prob_single(params=params, x=x)

    def log_prob_batch(self, params: MixtureSameFamilyParams, xs: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self._log_prob_single, in_axes=(None, 0))(params, xs)

    def log_normalization(self, params: MixtureSameFamilyParams) -> jnp.ndarray:
        # the distribution is already normalized
        return 1.

    def postprocess(self, params: MixtureSameFamilyParams) -> dict:
        dist = self.base_dist
        K = params.log_weights.shape[0]

        # Convert batched struct into list of K individual structs
        def get_kth_params(k):
            return jax.tree_map(lambda x: x[k], params.component_params)

        processed_components = [dist.postprocess(get_kth_params(k)) for k in range(K)]

        return {
            "components": processed_components,
            "weights": jax.nn.softmax(params.log_weights),
        }

    def neg_elbo(
        self,
        *,
        params: MixtureSameFamilyParams,
        xs: Optional[jnp.ndarray],
        logtarget: Callable[[jnp.ndarray], jnp.ndarray],
        stop_gradient_entropy: bool,
        key: jax.Array,
        n_samples: int,
    ) -> jnp.ndarray:
        logtarget_batch = jax.vmap(logtarget)
        log_weights = jax.nn.log_softmax(params.log_weights)
        alphas = jnp.exp(log_weights)
        K = log_weights.shape[0]
        keys = jax.random.split(key, K)
        dist = self.base_dist

        def get_kth_params(component_params, k):
            return jax.tree_map(lambda x: x[k], component_params)

        def one_term(k, key_k):
            comp_params_k = get_kth_params(params.component_params, k)
            xs_k = dist.sample(comp_params_k, key_k, n_samples)
            params_q = params if not stop_gradient_entropy else jax.lax.stop_gradient(params)
            log_q_k = self.log_prob_batch(params=params_q, xs=xs_k)
            entropy_k = -jnp.mean(log_q_k)
            log_p_k = jnp.mean(logtarget_batch(xs_k))
            return alphas[k] * (-entropy_k - log_p_k)

        return jnp.sum(jax.vmap(one_term, in_axes=(0, 0))(jnp.arange(K), keys))

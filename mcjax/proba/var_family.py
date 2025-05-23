from typing import Any, Callable, Optional
import abc
import jax
import jax.numpy as jnp


class VarFamily(abc.ABC):
    """
    Abstract interface for variational distributions q(x; phi).
    
    Subclasses must implement:
    - init_params: initialize variational parameters phi
    - sample: generate samples from q(x; phi)
    - logdensity: compute log-density log q(x; phi)
    - postprocess: convert internal params into user-facing form
    """

    @abc.abstractmethod
    def init_params(self, key: jax.Array) -> Any:
        """
        Initialize the variational parameters phi.
        Should return a PyTree (e.g. a flax dataclass or dict).
        """
        pass

    @abc.abstractmethod
    def sample(self, params: Any, key: jax.Array, n_samples: int) -> jnp.ndarray:
        """
        Generate samples x₁, ..., xₙ ~ q(x; phi) using reparameterization.
        
        Args:
            params: variational parameters phi
            key: PRNG key
            n_samples: number of samples to generate
        
        Returns:
            Array of shape (n_samples, dim)
        """
        pass

    @abc.abstractmethod
    def logdensity(self, params: Any, xs: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log q(x; phi) for a batch of xs.
        
        Args:
            params: variational parameters
            xs: samples, shape (n_samples, dim)
        
        Returns:
            logdensity values, shape (n_samples,)
        """
        pass
    
    def logdensity_batch(self, params: Any, xs: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log q(x; phi) for a batch of xs.
        
        Args:
            params: variational parameters
            xs: samples, shape (n_samples, dim)
        
        Returns:
            logdensity values, shape (n_samples,)
        """
        return jax.vmap(self.logdensity, in_axes=(None, 0))(params, xs)

    @abc.abstractmethod
    def postprocess(self, params: Any) -> dict:
        """
        Convert internal parameter representation to user-friendly output.
        For example, convert log-std to std, or return covariance matrix.
        """
        pass

    def neg_elbo(
        self,
        *,
        params: Any,
        xs: jnp.ndarray,
        logtarget_batch: Callable[[jnp.ndarray], jnp.ndarray],
        stop_gradient_entropy: bool = True,
        key: Optional[jax.Array] = None,  # Not used in this method, but kept for consistency
        n_samples: Optional[int] = 0,  # Not used in this method, but kept for consistency
    ) -> jnp.ndarray:
        """
        Estimate the negative Evidence Lower Bound (ELBO) using Monte Carlo sampling.

        This returns an unbiased estimate of:
            negElbo = E_q[log q(x)] - E_q[logdensity(x)]
        where logdensity is only known up to an unkown additive constant.

        The expectation is approximated using a fixed set of samples `xs ~ q(x; phi)`.

        Parameters
        ----------
        params : Any
            Parameters of the variational approximation q(x; phi).
        xs : jnp.ndarray
            Samples drawn from q(x; phi), shape (n_samples, dim).
        logtarget_batch : Callable
            Function computing log target(x) over a batch of points.
            Must accept shape (n_samples, dim) and return (n_samples,).
            Note: logdensity is only known up to an unkown additive constant

        stop_gradient_entropy : bool, default=True
            Whether to stop the gradient flow through log q(x) (i.e. entropy term).
            Recommended: True for the Sticking the Landing estimator.

        Returns
        -------
        neg_elbo : jnp.ndarray
            Scalar estimate of -ELBO, equivalent to KL[q || p] (up to a constant).
        """
        # sticking the landing?
        params_to_use = params
        if stop_gradient_entropy:
            # use the params for the log density
            # but stop gradient flow through them
            params_to_use = jax.lax.stop_gradient(params)
        log_qs = self.logdensity_batch(params_to_use, xs)
        entropy = -jnp.mean(log_qs)
        logtargets = jnp.mean(logtarget_batch(xs))
        return -entropy - logtargets

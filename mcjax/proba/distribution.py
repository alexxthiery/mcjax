from flax import struct
import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional


@struct.dataclass
class Distribution:
    """
    Abstract base class for Probability Density Functions (PDFs) q(x; phi), 
    for parameter phi and samples x.

    Subclasses should implement 
    key methods like `init_params`, `sample`, `log_prob`, and `postprocess`.

    Methods:
    - init_params(key): Initialize parameters phi of the distribution.
    - sample(params, key, n_samples): Draw samples x₁,...,xₙ ~ q(x; phi).
    - log_prob(params, xs): Evaluate log q(x; phi) for a batch of inputs.
    - log_prob_batch(params, xs): Vectorized wrapper over `log_prob`.
    - postprocess(params): Transform internal representation to user-facing output.
    - neg_elbo(...): Estimate the negative ELBO (KL[q || p] up to a constant).
    """

    def init_params(self, key: jax.Array) -> Any:
        raise NotImplementedError("init_params must be implemented in subclass.")

    def sample(self, params: Any, key: jax.Array, n_samples: int) -> jnp.ndarray:
        raise NotImplementedError("sample must be implemented in subclass.")

    def log_prob(self, params: Any, xs: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("log_prob must be implemented in subclass.")

    def log_normalization(self) -> jnp.ndarray:
        """
        Return the log normalization constant of the distribution.
        This is typically used for distributions where the normalization constant is known.
        If the normalization constant is not known, this will raise NotImplementedError.
        """
        raise NotImplementedError("log_normalization must be implemented in subclass.")

    def log_prob_batch(self, params: Any, xs: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.log_prob, in_axes=(None, 0))(params, xs)

    def postprocess(self, params: Any) -> Any:
        raise NotImplementedError("postprocess must be implemented in subclass.")

    def neg_elbo(
        self,
        *,
        params: Any,
        xs: jnp.ndarray,
        logtarget: Callable[[jnp.ndarray], jnp.ndarray],
        stop_gradient_entropy: bool = True,
        key: Optional[jax.Array] = None,    # not used, kept for consistency
        n_samples: Optional[int] = 0,       # not used, kept for consistency
    ) -> jnp.ndarray:
        """
        Estimate the negative Evidence Lower Bound (negative ELBO) for variational inference.

        Computes:
            KL[q(x; phi) || p(x)] ≈ -E_q[log p(x)] + E_q[log q(x; phi)]
        
        where:
            - q(x; phi) is the variational distribution defined by `self` and `params`.
            - p(x) is the unnormalized target log-density given by `logtarget`.

        Parameters:
        ----------
        params : Any
            Parameters of the variational distribution.
        xs : jnp.ndarray
            Samples drawn from q(x; phi), shape (N, ...) where N is the number of samples.
        logtarget : Callable[[jnp.ndarray], jnp.ndarray]
            Function returning log p(x) for input x. Must support batch input via `vmap`.
        stop_gradient_entropy : bool, default=True
            If True, prevents gradient flow through the entropy estimate (log q(x)).
            This is useful when optimizing ELBO w.r.t. phi but not including entropy gradients.
        key : Optional[jax.Array]
            Not used in this implementation; included for interface consistency.
        n_samples : Optional[int]
            Not used in this implementation; included for interface consistency.

        Returns:
        -------
        neg_elbo : jnp.ndarray
            Scalar negative ELBO estimate (KL[q || p] up to constant).
        """
        logtarget_batch = jax.vmap(logtarget, in_axes=(0,))
        params_to_use = jax.lax.stop_gradient(params) if stop_gradient_entropy else params
        log_qs = self.log_prob_batch(params_to_use, xs)
        entropy = -jnp.mean(log_qs)
        logtargets = jnp.mean(logtarget_batch(xs))
        return -entropy - logtargets

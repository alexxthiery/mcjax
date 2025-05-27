import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional
from typing import Protocol, runtime_checkable


@runtime_checkable
class DistributionLike(Protocol):
    """
    Protocol for Distribution-like objects.

    Required:
    - name (str)
    - dim (int)
    - log_prob(params, xs)

    Optional:
    - sample(params, key, n_samples)
    - postprocess(params)
    - log_normalization(params)
    - neg_elbo(...)
    """
    dim: int

    def log_prob(self, *, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log probability density log q(x; params)."""
        ...

    def sample(self, *, params: Any, key: jax.Array, n_samples: int) -> jnp.ndarray:
        """Draw n_samples from q(x; params) using PRNG key.
        If not implemented, raise NotImplementedError.
        """
        ...

    def postprocess(self, *, params: Any) -> Any:
        """Transform internal parameters into user-facing outputs."""
        ...

    def log_normalization(self, *, params: Any) -> jnp.ndarray:
        """Return log normalization constant of q(x; params), if known.
        If not known, raise NotImplementedError.
        """
        ...

    def neg_elbo(
        self,
        *,
        params: Any,
        xs: jnp.ndarray,
        log_target: Callable[[jnp.ndarray], jnp.ndarray],
        stop_gradient_entropy: bool = True,
        key: Optional[jax.Array] = None,
        n_samples: Optional[int] = 0,
    ) -> jnp.ndarray:
        """
        Estimate negative ELBO = E_q[log q(x; params)] - E_q[log p(x)].

        Parameters
        ----------
        params : Any
            Parameters of the variational distribution.
        xs : jnp.ndarray
            Samples drawn from q(x; params).
        logtarget : Callable
            Function computing log p(x).
        stop_gradient_entropy : bool
            If True, stop gradients through entropy term.
        key : Optional[jax.Array]
            Unused; for interface consistency.
        n_samples : Optional[int]
            Unused; for interface consistency.

        Returns
        -------
        jnp.ndarray
            Scalar negative ELBO estimate.
        """
        ...


def generic_neg_elbo(
    dist: DistributionLike,
    *,
    params: Any,
    xs: jnp.ndarray,
    logtarget: Callable[[jnp.ndarray], jnp.ndarray],
    stop_gradient_entropy: bool = True,
    key: Optional[jax.Array] = None,    # not used, kept for consistency
    n_samples: Optional[int] = 0,       # not used, kept for consistency
) -> jnp.ndarray:
    log_q_batch = jax.vmap(dist.log_prob, in_axes=(None, 0))
    log_p_batch = jax.vmap(logtarget)

    q_params = jax.lax.stop_gradient(params) if stop_gradient_entropy else params
    entropy = -jnp.mean(log_q_batch(q_params, xs))
    expected_log_p = jnp.mean(log_p_batch(xs))
    return -expected_log_p - entropy

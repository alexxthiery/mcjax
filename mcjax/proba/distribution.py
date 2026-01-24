"""
distribution.py

Lightweight interface for "distribution-like" objects and a generic ELBO helper.

Design
------
- A `DistributionLike` instance represents a *family* q(x; params).
- Structural info (e.g. `dim`) lives on the distribution object.
- Tunable parameters live in a separate `params` PyTree.
- The core API is:
    - dist.dim: int
    - dist.log_prob(params, x): scalar log-density at a single x
    - dist.sample(params, key, n_samples): optional

On top of that, we provide:
    - `generic_neg_elbo`:
        A reusable implementation of the negative ELBO:
            E_q[log q(x; params)] - E_q[log p(x)]
      given:
        - a DistributionLike `dist`
        - its parameters `params`
        - samples xs ~ q(x; params)
        - a target log-density `log_target`.

This is deliberately minimal: enough to plug into your SMC/MCMC code and basic
variational experiments, without over-abstracting.
"""

from typing import Any, Callable, Optional, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import struct


@runtime_checkable
class DistributionLike(Protocol):
    """
    Protocol for distribution-like objects.

    Minimal contract:
        - `dim`: int, the dimension of x.
        - `log_prob(params, x)`: scalar log-density for a single point x.

    Optional (but commonly useful) methods:
        - `sample(params, key, n_samples)`: draw samples from q(x; params).
        - `postprocess(params)`: map internal params to user-facing outputs.
        - `log_normalization(params)`: log Z, if known analytically.
        - `neg_elbo(...)`: distribution-specific ELBO estimator.

    This is intentionally light; the code relies only on `dim` and `log_prob`
    for `generic_neg_elbo`. Everything else is sugar.
    """

    dim: int

    # Core method: one-point log-density.
    def log_prob(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log q(x; params) for a single point x.

        Parameters
        ----------
        params : Any
            Distribution parameters (PyTree).
        x : jnp.ndarray
            Point in R^dim. Expected shape: (dim,).

        Returns
        -------
        jnp.ndarray
            Scalar log-density.
        """
        ...

    # =====================
    # OPTIONAL METHODS
    # =====================
    def sample(
        self, params: Any, key: jax.Array, n_samples: int
    ) -> jnp.ndarray:
        """
        Draw n_samples ~ q(x; params) using the given PRNG key.

        Expected output shape: (n_samples, dim).

        If not implemented, raise NotImplementedError.
        """
        ...

    def postprocess(self, params: Any) -> Any:
        """
        Transform internal parameters into user-facing outputs.

        This is a convenience hook, e.g. to convert unconstrained parameters
        into constrained ones (stddevs, correlation matrices, etc.).
        """
        ...

    def log_normalization(self, params: Any) -> jnp.ndarray:
        """
        Return log Z for q(x; params) if known analytically.

        If not known or not needed, raise NotImplementedError.
        """
        ...

    def neg_elbo(
        self,
        params: Any,
        xs: jnp.ndarray,
        log_target: Callable[[jnp.ndarray], jnp.ndarray],
        stop_gradient_entropy: bool = True,
        key: Optional[jax.Array] = None,
        n_samples: Optional[int] = 0,
    ) -> jnp.ndarray:
        """
        Estimate negative ELBO = E_q[log q(x; params)] - E_q[log p(x)].

        This is the standard variational objective:

            -ELBO(params) = E_q[log q(x; params)] - E_q[log p(x)]

        A typical implementation delegates to `generic_neg_elbo`.

        Parameters
        ----------
        params : Any
            Variational parameters of q.
        xs : jnp.ndarray
            Samples from q(x; params), shape (n_samples, dim).
        log_target : Callable[[jnp.ndarray], jnp.ndarray]
            Function computing log p(x); takes x of shape (dim,) and returns scalar.
        stop_gradient_entropy : bool
            If True, stop gradients through the entropy term E_q[log q].
        key : Optional[jax.Array]
            Unused placeholder for interface consistency.
        n_samples : Optional[int]
            Unused placeholder for interface consistency.

        Returns
        -------
        jnp.ndarray
            Scalar negative ELBO estimate.
        """
        ...


def generic_neg_elbo(
    dist: DistributionLike,
    params: Any,
    xs: jnp.ndarray,
    log_target: Callable[[jnp.ndarray], jnp.ndarray],
    stop_gradient_entropy: bool = True,
    key: Optional[jax.Array] = None,    # kept for interface symmetry; unused
    n_samples: Optional[int] = 0,       # kept for interface symmetry; unused
) -> jnp.ndarray:
    """
    Generic negative ELBO estimator for a DistributionLike.

    Computes:

        -ELBO(params)
        = E_q[log q(x; params)] - E_q[log p(x)]
        ≈ mean_i [log q(x_i; params) - log p(x_i)]

    given:
        - a distribution `dist` (with .dim and .log_prob),
        - parameters `params`,
        - samples `xs` ~ q(x; params),
        - a target log-density `log_target`.

    Parameters
    ----------
    dist : DistributionLike
        Distribution object providing `.dim` and `.log_prob(params, x)`.
    params : Any
        Parameters of the variational distribution q(x; params).
    xs : jnp.ndarray
        Samples from q(x; params), shape (n_samples, dist.dim).
    log_target : Callable[[jnp.ndarray], jnp.ndarray]
        Function computing log p(x) for a single x of shape (dist.dim,).
    stop_gradient_entropy : bool
        If True, stop gradients through E_q[log q], treating entropy as a
        constant w.r.t. params. If False, backpropagate through both terms.
    key : Optional[jax.Array]
        Unused; present for API consistency with method signatures.
    n_samples : Optional[int]
        Unused; present for API consistency.

    Returns
    -------
    jnp.ndarray
        Scalar negative ELBO estimate.

    Notes
    -----
    - This assumes xs has shape (n_samples, dist.dim) and represent
      samples from q(x; params) (reparameterized or otherwise).
    - If you want a fully reparameterized gradient, set
      `stop_gradient_entropy=False` and ensure `xs` are reparameterized.
    """
    # Basic shape check: xs is a batch of points in R^dim.
    if xs.ndim != 2 or xs.shape[1] != dist.dim:
        raise ValueError(
            f"Expected xs to have shape (n_samples, {dist.dim}), got {xs.shape}"
        )

    # Vectorized log q and log p over the sample dimension.
    # log_q_batch(params, xs[i]) = dist.log_prob(params, xs[i])
    log_q_batch = jax.vmap(dist.log_prob, in_axes=(None, 0))
    log_p_batch = jax.vmap(log_target)

    # Optionally block gradients through the entropy term.
    q_params = jax.lax.stop_gradient(params) if stop_gradient_entropy else params

    log_q_vals = log_q_batch(q_params, xs)   # shape (n_samples,)
    log_p_vals = log_p_batch(xs)            # shape (n_samples,)

    entropy = -jnp.mean(log_q_vals)
    expected_log_p = jnp.mean(log_p_vals)

    elbo = expected_log_p + entropy
    return -elbo



# =====================
# Helper to create a DistributionLike from log_prob function
# =====================

@struct.dataclass
class EmptyParams:
    pass
        
def make_distibution(
    log_prob_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    dim: int,
) -> DistributionLike:
    """ Create a DistributionLike from a log_prob function and dimension. """
    

    @struct.dataclass
    class CustomDistribution(DistributionLike):
        dim: int

        def log_prob(self,
                x: jnp.ndarray,
                params: EmptyParams,
                ) -> jnp.ndarray:
            return log_prob_fn(params, x)
        
        def log_prob_only(self,
                x: jnp.ndarray) -> jnp.ndarray:
            return log_prob_fn(EmptyParams(), x)
        
    # create an instance
    params = EmptyParams()
    dist = CustomDistribution(dim=dim)
    return dist, params
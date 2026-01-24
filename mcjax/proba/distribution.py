"""
distribution.py

Lightweight interface for "distribution-like" objects.

Design
------
- A `DistributionLike` instance represents a *family* q(x; params).
- Structural info (e.g. `dim`) lives on the distribution object.
- Tunable parameters live in a separate `params` PyTree.
- The core API is:
    - dist.dim: int
    - dist.log_prob(params, x): scalar log-density at a single x
    - dist.sample(params, key, n_samples): optional

This is deliberately minimal: enough to plug into your SMC/MCMC code
without over-abstracting.
"""

from typing import Any, Protocol, runtime_checkable

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

    This is intentionally light; the code relies only on `dim` and `log_prob`.
    Everything else is sugar.
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
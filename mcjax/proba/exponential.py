from typing import Optional, Callable
import jax
import jax.numpy as jnp
from flax import struct
from jax import random as jr
from .distribution import DistributionLike, generic_neg_elbo


@struct.dataclass
class ExponentialParams:
    log_rate: jnp.ndarray  # natural parameter; rate = exp(log_rate)


@struct.dataclass
class Exponential:
    """
    Exponential distribution:
        q(x) = rate * exp(-rate * x),  x >= 0
        where rate = exp(log_rate)
    
    Uses log_rate as the parameter for better numerical stability and optimization.

    Follows the DistributionLike protocol.
    """
    dim: int

    @classmethod
    def create(cls, *, dim: int) -> "Exponential":
        return cls(dim=dim)

    def init_params(
        self,
        log_rate: Optional[jnp.ndarray] = None,
    ) -> ExponentialParams:
        """
        Initialize parameters for the exponential distribution.

        Parameters
        ----------
        log_rate : jnp.ndarray, optional
            Log of the rate parameter (shape: [dim]). Defaults to zeros.

        Returns
        -------
        ExponentialParams
        """
        if log_rate is not None and log_rate.shape != (self.dim,):
            raise ValueError(f"log_rate must have shape ({self.dim},), got {log_rate.shape}")

        log_rate = log_rate if log_rate is not None else jnp.zeros(self.dim)
        return ExponentialParams(log_rate=log_rate)

    def sample(self, params: ExponentialParams, key: jax.Array, n_samples: int) -> jnp.ndarray:
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
        
        if params.log_rate.shape != (self.dim,):
            raise ValueError(f"log_rate must have shape ({self.dim},), got {params.log_rate.shape}")

        key = jr.split(key, self.dim)
        samples = jnp.stack([
            jr.exponential(k, shape=(n_samples,)) / r for k, r in zip(key, rate)
        ], axis=1)
        return samples  # shape: (n_samples, dim)

    def log_prob(self, params: ExponentialParams, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 1 or x.shape[0] != self.dim:
            raise ValueError(f"x must be a 1D array of shape ({self.dim},), got {x.shape}")
        if jnp.any(x < 0):
            return -jnp.inf  # Exponential is defined only for x >= 0

        rate = jnp.exp(jnp.clip(params.log_rate, -30.0, 30.0))
        log_prob = jnp.log(rate) - rate * x
        return jnp.sum(log_prob)

    def log_normalization(
        self,
        params: ExponentialParams,
    ) -> jnp.ndarray:
        # Already normalized
        return 0.

    def postprocess(self, params: ExponentialParams) -> dict:
        """Transform parameters into user-facing outputs."""
        return {
            "rate": jnp.exp(jnp.asarray(params.log_rate))
        }

    def neg_elbo(
        self,
        params: ExponentialParams,
        xs: jnp.ndarray,
        logtarget: Callable[[jnp.ndarray], jnp.ndarray],
        stop_gradient_entropy: bool = True,
        key: Optional[jax.Array] = None,
        n_samples: Optional[int] = None,
    ) -> jnp.ndarray:
        if key is not None:
            raise ValueError("key argument must be None in this method")
        if n_samples not in (None, 0):
            raise ValueError("n_samples must be None in this method")

        return generic_neg_elbo(
            dist=self,
            params=params,
            xs=xs,
            logtarget=logtarget,
            stop_gradient_entropy=stop_gradient_entropy,
            key=key,
            n_samples=n_samples,
        )


# Exponental following DistributionLike protocol
dist: DistributionLike = Exponential.create(dim=1)

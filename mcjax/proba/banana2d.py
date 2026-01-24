# banana_2d.py

"""
Banana-shaped 2D target distribution.

Definition
----------
Unnormalized density:

    target(x, y) ∝ exp{
        -0.5 * ( (x - 1)^2 + (y - x^2)^2 / noise_std^2 )
    }

With noise_std = 0.1, the negative log-density (up to a constant) is the
Rosenbrock function, which is a standard "banana-shaped" test problem.

This module implements the distribution in a form compatible with
`DistributionLike`, and provides:

    - `Banana2D.create()`: construct a distribution object
    - `Banana2D.init_params()`: construct parameter PyTree
    - `Banana2D.log_prob(params, x)`: log-density at a single point x
    - `Banana2D.sample(params, key, n_samples)`: draw samples
"""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from .distribution import DistributionLike


@struct.dataclass
class Banana2DParams:
    """
    Parameters for the Banana 2D distribution.

    Attributes
    ----------
    noise_std : float
        Standard deviation of the vertical noise term.
        Smaller values make the banana narrower and more challenging.
    """
    noise_std: float


@struct.dataclass
class Banana2D:
    """
    Banana-shaped 2D distribution.

    The (unnormalized) log-density is:

        log target(x, y)
        = -0.5 * ( (x - 1)^2 + (y - x^2)^2 / noise_std^2 ) + const

    We optionally include the Gaussian normalizing constants so that
    `log_prob` is a proper log-density (up to floating-point error).
    """
    dim: int

    @classmethod
    def create(cls) -> "Banana2D":
        """
        Construct a 2D Banana distribution object.

        Returns
        -------
        Banana2D
            Distribution object with dim=2.
        """
        dim = 2
        return cls(dim=dim)

    def init_params(self, noise_std: float = 0.1) -> Banana2DParams:
        """
        Initialize parameters for the Banana 2D distribution.

        Parameters
        ----------
        noise_std : float, default=0.1
            Standard deviation of the vertical noise term. Must be positive.

        Returns
        -------
        Banana2DParams
        """
        assert noise_std > 0.0, "noise_std must be positive"
        return Banana2DParams(noise_std=noise_std)

    def log_prob(self, params: Banana2DParams, x: jnp.ndarray) -> jnp.ndarray:
        """
        Log-density log q(x; params) at a single point x.

        Parameters
        ----------
        params : Banana2DParams
            Distribution parameters.
        x : jnp.ndarray
            Point in R^2. Expected shape: (2,).

        Returns
        -------
        jnp.ndarray
            Scalar log-density.
        """
        assert x.ndim == 1 and x.shape[0] == self.dim, (
            f"Expected x to have shape ({self.dim},), got {x.shape}"
        )

        x0, x1 = x[0], x[1]
        noise_std = params.noise_std

        # Quadratic "banana" energy
        quad = (x0 - 1.0) ** 2 + (x1 - x0 ** 2) ** 2 / (noise_std ** 2)

        # Include Gaussian normalizing constants so this is a proper log-density.
        # One standard normal term in x0, one in (x1 | x0) with std=noise_std.
        log_norm = -0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * jnp.log(
            2.0 * jnp.pi * noise_std ** 2
        )

        return -0.5 * quad + log_norm

    def log_prob_only(self, params: Banana2DParams) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Return a function `log_prob_x(x)` that closes over `params`.

        Useful for passing the target log-density into MCMC/SMC routines.
        """
        return lambda x: self.log_prob(params=params, x=x)

    def sample(
        self,
        params: Banana2DParams,
        key: jax.Array,
        n_samples: int,
    ) -> jnp.ndarray:
        """
        Draw samples from the Banana 2D distribution.

        Sampling scheme:
            x0 ~ N(1, 1)
            x1 | x0 ~ N(x0^2, noise_std^2)

        Parameters
        ----------
        params : Banana2DParams
            Distribution parameters.
        key : jax.Array
            PRNG key.
        n_samples : int
            Number of samples.

        Returns
        -------
        jnp.ndarray
            Samples of shape (n_samples, 2).
        """
        noise_std = params.noise_std

        key, key_x0 = jr.split(key)
        x0_s = 1.0 + jr.normal(key_x0, (n_samples,))

        key, key_x1 = jr.split(key)
        x1_s = x0_s ** 2 + noise_std * jr.normal(key_x1, (n_samples,))

        return jnp.stack([x0_s, x1_s], axis=-1)

    def log_normalization(self, params: Banana2DParams) -> jnp.ndarray:
        """
        Log normalizing constant of the Banana distribution.

        For the way we defined `log_prob`, the distribution is already
        normalized, so the log-normalization is 0.

        Returns
        -------
        jnp.ndarray
            Scalar 0.0.
        """
        return jnp.array(0.0)

    def postprocess(self, params: Banana2DParams) -> dict:
        """
        Transform internal parameters into user-facing outputs.

        For this simple distribution, parameters are already interpretable.
        """
        return {"noise_std": params.noise_std}


# Hint for static checkers: Banana2D conforms to DistributionLike.
_dist_banana2d: DistributionLike = Banana2D.create()
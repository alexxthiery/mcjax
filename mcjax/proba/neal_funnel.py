# neal_funnel.py

"""
Neal's funnel distribution (D-dimensional).

Definition
----------
Let x = (x0, x1, ..., x_{D-1}) with:

    x0 ~ N(0, sigma_x^2)
    x_i | x0 ~ N(0, exp(x0))   for i = 1,...,D-1

This induces a strongly non-Gaussian, funnel-shaped target distribution
that is widely used as a stress test for MCMC algorithms.

This module implements Neal's funnel in a form compatible with `DistributionLike`,
and provides:

    - `NealFunnel.create(dim)`: construct a funnel of given dimension
    - `NealFunnel.init_params(sigma_x)`: construct parameter PyTree
    - `NealFunnel.log_prob(params, x)`: log-density at a single point x
    - `NealFunnel.sample(params, key, n_samples)`: draw samples
    - `NealFunnel.neg_elbo(...)`: convenience wrapper around `generic_neg_elbo`
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from .distribution import DistributionLike, generic_neg_elbo


@struct.dataclass
class NealFunnelParams:
    """
    Parameters for Neal's funnel distribution.

    Attributes
    ----------
    sigma_x : float
        Standard deviation of the top-level Gaussian on x0.
    """
    sigma_x: float


@struct.dataclass
class NealFunnel:
    """
    Neal's funnel distribution in D dimensions.

    Generative definition:
        x0 ~ N(0, sigma_x^2)
        x_i | x0 ~ N(0, exp(x0))   for i = 1,...,D-1

    The resulting log-density is:

        log p(x)
        = log N(x0 | 0, sigma_x^2)
          + sum_{i=1}^{D-1} log N(x_i | 0, exp(x0))

    This leads to a highly non-isotropic density and is a standard MCMC benchmark.
    """
    dim: int

    @classmethod
    def create(cls, dim: int = 2) -> "NealFunnel":
        """
        Construct a Neal's funnel distribution with given dimension.

        Parameters
        ----------
        dim : int, default=2
            Dimension of the space. Must be >= 2 for the "funnel" structure.

        Returns
        -------
        NealFunnel
        """
        assert dim >= 2, "NealFunnel requires dim >= 2"
        return cls(dim=dim)

    def init_params(self, sigma_x: float = 3.0) -> NealFunnelParams:
        """
        Initialize parameters for Neal's funnel.

        Parameters
        ----------
        sigma_x : float, default=3.0
            Standard deviation of x0. Must be positive.

        Returns
        -------
        NealFunnelParams
        """
        assert sigma_x > 0.0, "sigma_x must be positive"
        return NealFunnelParams(sigma_x=sigma_x)

    def log_prob(self, params: NealFunnelParams, x: jnp.ndarray) -> jnp.ndarray:
        """
        Log-density log q(x; params) at a single point x.

        Parameters
        ----------
        params : NealFunnelParams
            Distribution parameters.
        x : jnp.ndarray
            Point in R^dim. Expected shape: (dim,).

        Returns
        -------
        jnp.ndarray
            Scalar log-density.
        """
        assert x.ndim == 1 and x.shape[0] == self.dim, (
            f"Expected x to have shape ({self.dim},), got {x.shape}"
        )

        sigma_x = params.sigma_x
        dim = self.dim

        x0 = x[0]
        x_rest = x[1:]  # shape (dim-1,)

        # Conditional standard deviation for x_i | x0
        std_cond = jnp.exp(x0 / 2.0)

        # Log N(x0 | 0, sigma_x^2)
        log_p_x0 = -0.5 * (x0 / sigma_x) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi * sigma_x ** 2)

        # Log N(x_i | 0, exp(x0)) for i >= 1.
        # exp(x0) is the variance, so std_cond^2 = exp(x0).
        quad_rest = jnp.sum((x_rest / std_cond) ** 2)
        log_p_rest = (
            -0.5 * quad_rest
            - 0.5 * (dim - 1) * jnp.log(2.0 * jnp.pi * std_cond ** 2)
        )

        return log_p_x0 + log_p_rest

    def log_prob_only(self, params: NealFunnelParams) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Return a function `log_prob_x(x)` that closes over `params`.

        Useful for passing the target log-density into MCMC/SMC routines.
        """
        return lambda x: self.log_prob(params=params, x=x)

    def sample(
        self,
        params: NealFunnelParams,
        key: jax.Array,
        n_samples: int,
    ) -> jnp.ndarray:
        """
        Draw samples from Neal's funnel.

        Sampling scheme:
            x0 ~ N(0, sigma_x^2)
            x_i | x0 ~ N(0, exp(x0))  for i = 1,...,dim-1

        Parameters
        ----------
        params : NealFunnelParams
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
        sigma_x = params.sigma_x
        dim = self.dim

        # Sample x0
        key, key_x0 = jr.split(key)
        x0_s = sigma_x * jr.normal(key_x0, (n_samples, 1))  # shape (n_samples, 1)

        # Conditional std for x_rest
        std_cond = jnp.exp(x0_s / 2.0)  # shape (n_samples, 1)

        # Sample x_rest | x0
        key, key_rest = jr.split(key)
        x_rest_s = std_cond * jr.normal(key_rest, (n_samples, dim - 1))

        # Concatenate to shape (n_samples, dim)
        return jnp.concatenate([x0_s, x_rest_s], axis=1)

    def log_normalization(self, params: NealFunnelParams) -> jnp.ndarray:
        """
        Log normalizing constant.

        By construction this is a normalized density, so the log-normalization
        is zero.

        Returns
        -------
        jnp.ndarray
            Scalar 0.0.
        """
        return jnp.array(0.0)

    def postprocess(self, params: NealFunnelParams) -> dict:
        """
        Transform internal parameters into user-facing outputs.

        For Neal's funnel, there is nothing to transform.
        """
        return {"sigma_x": params.sigma_x}

    def neg_elbo(
        self,
        params: NealFunnelParams,
        xs: jnp.ndarray,
        log_target: Callable[[jnp.ndarray], jnp.ndarray],
        stop_gradient_entropy: bool = True,
        key: Optional[jax.Array] = None,
        n_samples: Optional[int] = 0,
    ) -> jnp.ndarray:
        """
        Convenience wrapper around `generic_neg_elbo` for this distribution.

        Parameters
        ----------
        params : NealFunnelParams
            Variational parameters of q(x; params).
        xs : jnp.ndarray
            Samples from q(x; params), shape (n_samples, dim).
        log_target : Callable[[jnp.ndarray], jnp.ndarray]
            Target log-density log p(x).
        stop_gradient_entropy : bool
            If True, stop gradients through E_q[log q].
        key : Optional[jax.Array]
            Unused; for interface symmetry.
        n_samples : Optional[int]
            Unused; for interface symmetry.

        Returns
        -------
        jnp.ndarray
            Scalar negative ELBO estimate.
        """
        return generic_neg_elbo(
            dist=self,
            params=params,
            xs=xs,
            log_target=log_target,
            stop_gradient_entropy=stop_gradient_entropy,
            key=key,
            n_samples=n_samples,
        )


# Hint for static checkers: NealFunnel conforms to DistributionLike.
_dist_neal_funnel: DistributionLike = NealFunnel.create(dim=2)
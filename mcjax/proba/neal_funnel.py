import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Optional, Callable
from flax import struct
from .distribution import generic_neg_elbo, DistributionLike

# ==================================
# D-dimensional Neal's Funnel Distribution
# x0 ~ N(0, sigma_x^2),
# xi ~ N(0, variance=exp(x0)) are independent for i=1,...,D-1
# By default:
#   sigma_x = 3
#   D = 2
# ==================================


@struct.dataclass
class NealFunnelParams:
    """ Parameters for Neal's Funnel distribution """
    sigma_x: float


@struct.dataclass
class NealFunnel:
    """ Neal's Funnel Distribution
    x0 ~ N(0, sigma_x^2),
    xi ~ N(0, variance=exp(x0)) are independent for i=1,...,D-1
    By default:
    sigma_x = 3
    D = 2
    """
    dim: int 
    
    @classmethod
    def create(cls, dim: int = 2) -> "NealFunnel":
        """ Create a Neal's Funnel distribution with given dimension """
        return cls(dim=dim)
    
    def init_params(self, sigma_x: float = 3.) -> "NealFunnel":
        """ Initialize parameters for Neal's Funnel distribution """
        assert sigma_x > 0, "sigma_x must be positive"
        return NealFunnelParams(sigma_x=sigma_x)

    def log_prob(self, *, params: NealFunnelParams, x: jnp.ndarray) -> jnp.ndarray:
        """ Compute log probability density log q(x; params) """
        std_x = params.sigma_x
        dim = self.dim

        x0, x1 = x[0], x[1:]
        std = jnp.exp(x0/2.)
        out = -0.5 * (x0/std_x)**2 - 0.5 * jnp.sum(x1/std)**2
        out += -0.5 * (dim-1) * jnp.log(2 * jnp.pi * std**2)
        out += -0.5 * jnp.log(2 * jnp.pi * std_x**2)
        return out

    def log_prob_only(self, *, params: NealFunnelParams) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """ Return a function that computes the log probability density """
        return lambda x: self.log_prob(params=params, x=x)

    def sample(self, *, params: NealFunnelParams, key: jax.Array, n_samples: int) -> jnp.ndarray:
        # samples x0_s
        key, key_ = jr.split(key)
        std_x = params.sigma_x
        x0_s = std_x * jr.normal(key_, (n_samples, 1))
        # samples x1_s
        key, key_ = jr.split(key)
        stds = jnp.exp(x0_s/2.)
        x1_s = stds * jr.normal(key_, (n_samples, self.dim-1))
        return jnp.concatenate([x0_s, x1_s], axis=1)

    def log_normalization(self, *, params: NealFunnelParams) -> float:
        """ log partition function """
        # it is already normalized
        return 0.

    def postprocess(self, *, params: NealFunnelParams) -> dict:
        """Transform internal parameters into user-facing outputs."""
        return {
            "sigma_x": params.sigma_x,
        }

    def neg_elbo(
        self,
        *,
        params: NealFunnelParams,
        xs: jnp.ndarray,
        log_target: Callable[[jnp.ndarray], jnp.ndarray],
        stop_gradient_entropy: bool = True,
        key: Optional[jax.Array] = None,
        n_samples: Optional[int] = 0,
    ) -> jnp.ndarray:
        return generic_neg_elbo(
            params=params,
            xs=xs,
            log_target=log_target,
            stop_gradient_entropy=stop_gradient_entropy,
            key=key,
            n_samples=n_samples
        )


# NealFunnel follows the DistributionLike protocol
_dist_neal_funnel: DistributionLike = NealFunnel.create(dim=2)

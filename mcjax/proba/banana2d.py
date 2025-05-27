import jax.numpy as jnp
import jax.random as jr
from .distribution import generic_neg_elbo, DistributionLike
from typing import Optional, Callable
from flax import struct
import jax

# ==================================
# Banana 2D Distribution
# ==================================


@struct.dataclass
class Banana2DParams:
    """ Parameters for the Banana 2D distribution """
    noise_std: float


@struct.dataclass
class Banana2D:
    """ Banana 2D Distribution:

        target(x,y) \propto exp{ -0.5*( (y-x^2)^2/noise_std^2 + (x-1)^2 ) }

    Remark: with noise_std = 0.1, the -logpdf is the Rosenbrock function
    """
    dim: int

    @classmethod
    def create(cls) -> "Banana2D":
        dim = 2
        return cls(dim=dim)

    def init_params(self, noise_std: float = 0.1) -> Banana2DParams:
        assert noise_std > 0, "noise_std must be positive"
        params = Banana2DParams(noise_std=noise_std)
        return params

    def log_prob(self, *, params: Banana2DParams, x: jnp.ndarray) -> jnp.ndarray:
        x0, x1 = x[0], x[1]
        out = -0.5*((x0 - 1.)**2 + (x1 - x0**2)**2 / params.noise_std**2)
        out += -0.5*jnp.log(2*jnp.pi) - 0.5*jnp.log(2*jnp.pi*params.noise_std**2)
        return out
    
    def log_prob_only(self, *, params: Banana2DParams) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """ Return a function that computes the log probability density """
        return lambda x: self.log_prob(params=params, x=x)

    def sample(self, *, params: Banana2DParams, key: jax.Array, n_samples: int) -> jnp.ndarray:
        # samples x0_s
        key, key_ = jr.split(key)
        x0_s = 1. + jr.normal(key_, (n_samples,))
        # samples x1_s
        key, key_ = jr.split(key)
        x1_s = x0_s**2 + params.noise_std * jr.normal(key_, (n_samples,))
        return jnp.stack([x0_s, x1_s], axis=-1)

    def log_normalization(self, *, params: Banana2DParams) -> float:
        """ log partition function """
        # it is already normalized
        return 0.

    def postprocess(self, *, params: Banana2DParams) -> dict:
        """Transform internal parameters into user-facing outputs."""
        return {
            "noise_std": params.noise_std,
        }

    def neg_elbo(
        self,
        *,
        params: Banana2DParams,
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


# Banana2D follows the DistributionLike protocol
_dist_banana2d: DistributionLike = Banana2D.create()
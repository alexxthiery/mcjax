import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

from mcjax.proba.density import LogDensity


# ==================================
# ORNSTEIN–UHLENBECK process
# ==================================

class OU:
    """
    Discrete time Ornstein-Uhlenbeck forward process:
      y_{k+1} = sqrt(1 - alpha[k]) * y_k + sigma * sqrt(alpha[k]) * eps
    plus an injectable reverse-process sampler.
    """
    def __init__(self,
                 alpha: jnp.ndarray, # array of shape (K,), where alpha[k] = 1 - exp(-2 ∫β_s ds) over step k
                 sigma: float,
                 init_dist: LogDensity):

        self.alpha = alpha
        self.sigma = sigma
        self.sqrt_1m_alpha = jnp.sqrt(1.0 - alpha)
        self.sqrt_alpha = jnp.sqrt(alpha)
        self.init_dist = init_dist
        self.K = alpha.shape[0]

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self,
               key: jr.PRNGKey,
               N: int,
               k: int) -> jnp.ndarray:
        """
        Draw N samples at time index k of the forward OU chain,
        marginalizing out all intermediate epsilons in one shot.
        This uses the *exact* marginal:
          y_k = sqrt(prod_{j<k} (1-alpha[j])) * y_0
                + sigma * sqrt(1 - prod_{j<k} (1-alpha[j])) * eps
        
        For simplicity we just iterate one step at a time here.
        """
        def body(i, carry):
            y, key = carry
            key, key_ = jr.split(key)
            eps = jr.normal(key_, shape=(N, self.init_dist.dim))
            y = self.sqrt_1m_alpha[k] * y + self.sigma * self.sqrt_alpha[k] * eps
            return (y, key)

        # sample y_0
        key, key_ = jr.split(key)
        y0 = self.init_dist.sample(key_, N)  # shape (N, D,…)
        # run exactly k steps
        (y_k, _ ) = jax.lax.fori_loop(0, k, body, (y0, key))
        return y_k

    @partial(jax.jit, static_argnums=(0, 2, 4))
    def score_sample(self,
                       key: jr.PRNGKey,
                       N: int,
                       k: int,
                       score_fn,
                       params) -> jnp.ndarray:
        """
        One-step reverse sampler at step k given a score network.
        
        Reverse-SDE discretization:
          y_{k} = sqrt(1-alpha_k) * y_{k+1}
                  + 2sigma^2(1 - sqrt(1-alpha_k)) * score_fn(k, y_{k+1})
                  + sigma * sqrt(alpha_k) * eps
        """
        # first sample y_{k+1} from the forward marginal
        key, key_ = jr.split(key)
        y_next = self.sample(key_, N, k+1)

        # now step backwards
        key, key_ = jr.split(key)
        eps = jr.normal(key_, shape=y_next.shape)
        drift = 2 * (self.sigma**2) * (1 - self.sqrt_1m_alpha[k]) * score_fn(params, k, y_next)
        y_k = (
            self.sqrt_1m_alpha[k] * y_next
            + drift
            + self.sigma * self.sqrt_alpha[k] * eps
        )
        return y_k
    
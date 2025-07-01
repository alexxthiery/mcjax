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
            y = self.sqrt_1m_alpha[i] * y + self.sigma * self.sqrt_alpha[i] * eps
            return (y, key)

        # sample y_0
        key, key_ = jr.split(key)
        y0 = self.init_dist.sample(key_, N)  # shape (N, D,…)
        # run exactly k steps
        (y_k, _ ) = jax.lax.fori_loop(0, k, body, (y0, key))
        return y_k

    @partial(jax.jit, static_argnums=(0,4))
    def reverse_step(self, key, y_next, k, score_fn, params):
        # y_next is the actual next state from the reverse chain
        key, key_ = jr.split(key)
        eps = jr.normal(key_, shape=y_next.shape)
        score = score_fn(params, self.K - k - 1, y_next)
        drift = 2 * self.sigma**2 * (1 - self.sqrt_1m_alpha[self.K - k - 1]) * score
        y_k = (self.sqrt_1m_alpha[self.K - k - 1] * y_next
              + drift
              + self.sigma * self.sqrt_alpha[self.K - k - 1] * eps)
        return key, y_k,score

    @partial(jax.jit, static_argnums=(0, 3))
    def integrate_reverse(self, key: jr.PRNGKey, x1: jnp.ndarray,
                          score_fn, params) -> jnp.ndarray:
        """
        Starting from x1 (the prior sample at t=1), run the reverse SDE for K steps
        to obtain x0.  Returns a tensor of shape (N, D, …).
        """
        def body(i, carry):
            y, key = carry
            key, y = self.reverse_step(key, y, i, score_fn, params)
            return y, key

        # Initialize carry with (y_K = x1, PRNG key)
        (y0, _key) = jax.lax.fori_loop(0, self.K, body, (x1, key))
        return y0
    
    
    def log_marginal(self, x: jnp.ndarray, k: int) -> jnp.ndarray:
        """
        Compute the log marginal density of x at time index k for 
        standard gaussian inital dstribution
        """
        # compute total variance factor: v = σ² (1 - ∏ (1-α))
        prod_1m = jnp.prod(1.0 - self.alpha[:k])
        var = self.sigma**2 * (1.0 - prod_1m)
        D   = x.shape[-1]
        norm = -0.5 * (D * jnp.log(2*jnp.pi*var))
        quad = -0.5 * jnp.sum(x**2, axis=-1) / var
        return norm + quad

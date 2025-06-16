from abc import ABC, abstractmethod
import jax
import jax.random as jr
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import logsumexp

class BaseLoss(ABC):
    """Abstract interface for any training loss."""

    @abstractmethod
    def __call__(self, params, key, process, init_dist, target_dist, score_fn, batch_size, **kwargs):
        """
        Given:
          - params (PyTree of model weights)
          - key (PRNGKey)
          - process (e.g. OU instance)
          - init_dist (reference density)
          - target_dist (actual target density)
          - score_fn (callable(params, k, y) → score vector)
          - batch_size
          - any extra flags (e.g. add_score, etc.)
        Return:
          - scalar-loss (averaged over batch)
        """
        pass


class DDSLoss(BaseLoss):
    """
    Reverse KL / Log Variance losses for DDS (as in “dDS”).
    """
    def __init__(self, add_score: bool = False):
        self.add_score = add_score

    def __call__(self, params, key, process, init_dist, target_dist, score_fn, batch_size, **kwargs):
        K = process.K
        sigma = process.sigma

        # 1) sample y0 ~ init_dist
        key, sub = jr.split(key)
        y0 = init_dist.sample(sub, batch_size)

        def scan_step(carry, k):
            y_k, r_k, key = carry
            key, sub2 = jr.split(key)
            eps = jr.normal(sub2, shape=y_k.shape)

            idx = K - 1 - k
            alpha_Kmk = process.alpha[idx]
            sqrt1m = jnp.sqrt(1.0 - alpha_Kmk)
            lam = 1.0 - sqrt1m

            s = score_fn(params, idx, y_k)  
            # reverse‐OU update
            y_next = (sqrt1m * y_k
                      + 2.0 * (sigma**2) * lam * s
                      + sigma * jnp.sqrt(alpha_Kmk) * eps)

            # accumulate path‐integral term
            main_term = (2.0 * sigma**2) * (lam**2 / alpha_Kmk) * jnp.sum(s**2, axis=-1)
            if self.add_score:
                zero_exp_term = 2.0 * sigma * jnp.sqrt(lam**2 / alpha_Kmk) * jnp.sum(s * eps, axis=-1)
                r_next = r_k + main_term + zero_exp_term
            else:
                r_next = r_k + main_term

            return (y_next, r_next, key), None

        r0 = jnp.zeros(batch_size)
        (yK, rK, _), _ = jax.lax.scan(
            scan_step,
            (y0, r0, key),
            jnp.arange(K)
        )

        # now compute log‐ratio = log p_ref(yK) - log p_target(yK)
        log_ref = init_dist.batch(yK)
        log_targ = target_dist.batch(yK)
        loss = jnp.mean(rK + log_ref - log_targ)
        return loss

class IDEMLoss(BaseLoss):
    """
    Implements the Iterated Denoising Energy Matching (iDEM) inner-loop loss:
      L_DEM(x_t, t) = || S_K(x_t, t) - s_theta(x_t, t) ||^2,
    """
    def __init__(self, K: int, sigma_fn: callable, buffer, target_dist, score_fn):
        """
        Args:
          K: number of Monte Carlo samples used in estimating the score S_K.
        """
        self.K = K
        self.sigma_fn = sigma_fn  # (geometric) noise schedule
        self.buffer = buffer  # a Buffer instance to sample x0 from
        self.target_dist = target_dist
        self.score_fn = score_fn  # Store score_fn here

    @partial(jax.jit, static_argnums=(0,3))
    def __call__(self,
                params,
                key: jr.PRNGKey,
                batch_size: int):
        """
        Returns:
        loss: scalar, the average MSE between S_K(x_t, t) and s_theta(x_t, t).
        """
        # Draw a batch of x0 ∼ buffer
        x0,key = self.buffer.sample(key, batch_size)    # shape: (B, d, ...)

        # Sample t ∼ Uniform(0,1) for each x0 in the batch
        key,sub = jr.split(key)
        t = jr.uniform(sub, shape=(batch_size,), minval=0.0, maxval=1.0)  # → (B,)

        # Form x_t = x0 + sigma_t * eps, where σ_t = sigma_fn(t)
        sigma_t = self.sigma_fn(t)  # → (B,)
        # reshape so that σ_t can broadcast over (d, …) dimensions of x0
        sigma_t = sigma_t.reshape((batch_size,) + (1,) * (x0.ndim - 1))
        key, sub = jr.split(key)
        eps = jr.normal(sub, shape=x0.shape)
        x_t = x0 + sigma_t * eps    # → (B, d, …)

        # Define a function that, for one (x_t_single, t_single),
        #    draws K samples x0_i ∼ N(x_t_single, σ_t^2), computes log p and grad log p,
        #    and returns the weighted average of gradients
        def mc_estimate_single(x_t_single, t_single, key_single):
            sigma = self.sigma_fn(t_single)    

            # draw K independent x0_i ∼ N(x_t_single, σ² I)
            keys_MC = jr.split(key_single, self.K)
            x0_MC = jnp.stack([
                x_t_single + sigma * jr.normal(k, shape=x_t_single.shape)
                for k in keys_MC
            ], axis=0) 

            # evaluate log-density and score at each of the K samples:
            logp_MC = self.target_dist.batch(x0_MC)     
            grad_logp_MC = self.target_dist.grad_batch(x0_MC)   

            lse = logsumexp(logp_MC)               
            w_norm = jnp.exp(logp_MC - lse)         

            # compute weighted average of gradient vectors:
            expand_dims = (1,) * (grad_logp_MC.ndim - 1)
            w_shaped = w_norm.reshape((self.K,) + expand_dims)  # → (K, 1, 1, …)
            numerator = jnp.sum(w_shaped * grad_logp_MC, axis=0)  # → (d, …)

            return numerator

        keys_batch = jr.split(key, batch_size)  # → (B,) of PRNGKey

        # Vectorize mc_estimate_single over the batch dimension
        S_K_batch = jax.vmap(mc_estimate_single)(
            x_t,       # shape: (B, d, …)
            t,         # shape: (B,)
            keys_batch # shape: (B,)
        )  # → (B, d, …)

        s_pred = self.score_fn(params, t, x_t)  # → (B, d, …)

        # Compute per-example squared ‖S_K - s_pred‖² and average:
        sq_err = jnp.sum((S_K_batch - s_pred) ** 2,
                        axis=tuple(range(1, S_K_batch.ndim)))  # → (B,)
        loss = jnp.mean(sq_err)  # scalar

        return loss
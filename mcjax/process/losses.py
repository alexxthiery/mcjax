from abc import ABC, abstractmethod
import jax
import jax.random as jr
import jax.numpy as jnp

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

# You could similarly create classes for other path‐measure losses:
# class RKLoss(BaseLoss): ...
# class LogVarianceLoss(BaseLoss): ...
# class TrajectoryBalanceLoss(BaseLoss): ...
# class DetailedBalanceLoss(BaseLoss): ...

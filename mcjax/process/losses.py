from abc import ABC, abstractmethod
import jax
import jax.random as jr
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import logsumexp
from jax.scipy.stats.norm import logpdf

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

        # sample y0 ~ init_dist
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

        # Sample t ∼ Uniform(0,1) for all the x0 in the batch
        key,sub = jr.split(key)
        t = jr.uniform(sub, minval=0.0, maxval=1.0)  

        # Form x_t = x0 + sigma_t * eps, where σ_t = sigma_fn(t)
        sigma_t = self.sigma_fn(t) 

        # Get the batch of xt
        key, sub = jr.split(key)
        eps = jr.normal(sub, shape=x0.shape)
        x_t = x0 + sigma_t * eps


        def mc_estimate_single(x_t_single, t, key_single):
            # compute Sk for one single sampled x0
            sigma = self.sigma_fn(t)    

            # draw K independent x0_i ∼ N(x_t_single, σ² I)
            keys_MC = jr.split(key_single, self.K) # 10_000 is arbitrary, can be larger or smaller
            # create an array of x0_MC of shape (K, d, ...)
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

        # Vectorize mc_estimate_single over x_t and keys_batch:
        S_K_batch = jax.vmap(mc_estimate_single, in_axes=(0, None, 0), out_axes=0)(
            x_t, t, keys_batch)

        s_pred = self.score_fn(params, t, x_t) 

        # Compute per-example squared ‖S_K - s_pred‖² and average:
        sq_err = jnp.sum((S_K_batch - s_pred) ** 2,
                        axis=tuple(range(1, S_K_batch.ndim)))  
        loss = jnp.mean(sq_err)  # scalar

        return loss

class PISLoss(BaseLoss):

    def __init__(self, add_score: bool = False):
        self.add_score = add_score # NO NEED, just to align with other losses

    def __call__(self, params, key, process,init_dist, target_dist, score_fn, batch_size, **kwargs):
        # forward controlled SDE from x0 ~ ν
        self.delta_t = 1/process.K
        self.n_steps = process.K      # align with OU steps
        key, sub = jr.split(key)
        x = init_dist.sample(sub, batch_size)

        running_cost = jnp.zeros(batch_size) # first term in the loss function
        def body(carry, t):
            x, running, key = carry
            u = score_fn(params, t, x)
            running = running + 0.5*jnp.sum(u**2, axis=-1)*self.delta_t

            key, sub = jr.split(key)
            dW = jr.normal(sub, x.shape)*jnp.sqrt(self.delta_t)
            x  = x + u*self.delta_t + dW
            return (x, running, key), None

        times = jnp.arange(self.n_steps, dtype=jnp.float32)
        (xT, running, _), _ = jax.lax.scan(body, (x, running_cost, key), times)

        # terminal cost Ψ = log q_T(x_T) - log p(x_T) under pure Brownian motion
        T_total = 1.0  
        var_total = 1.0 + T_total  # Total variance = 1+T (pure Brownian motion)
        d = xT.shape[-1]
        log_qT = -0.5 * d * jnp.log(2 * jnp.pi * var_total) \
                - 0.5 * jnp.sum(xT**2, axis=-1) / var_total
        log_p  = target_dist.batch(xT)
        psi    = log_qT - log_p

        return jnp.mean(running + psi)

import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

class CMCDLoss(BaseLoss):
    def __init__(self, use_control_in_denominator: bool, add_score: bool = False):
        self.use_ctrl_den = use_control_in_denominator
        self.add_score = add_score

    def __call__(self, params, key, process, init_dist, target_dist, score_fn, batch_size, **kwargs):
        delta_t = 1 / process.K
        n_steps = process.K
        sigma2 = process.sigma**2
        
        # Sample initial state
        key, subkey = jr.split(key)
        x0 = init_dist.sample(subkey, batch_size)
        log_p0 = init_dist.batch(x0)  # π₀(x₀)
        
        # Forward pass to generate trajectory
        def forward_step(carry, t):
            x, key = carry
            key, subkey = jr.split(key)
            u = score_fn(params, t, x)
            gradp = target_dist.grad_batch(x)  # Time-dependent score
            
            # Forward transition
            noise = jr.normal(subkey, x.shape) * jnp.sqrt(2 * sigma2 * delta_t)
            x_next = x + (sigma2 * gradp + u) * delta_t + noise
            return (x_next, key), (x, x_next, u, gradp)
        
        times = jnp.arange(n_steps)
        (xK, _), forward_vals = jax.lax.scan(
            forward_step, (x0, key), times
        )
        x_t, x_tp1, u_t, gradp_t = forward_vals
        # Include initial state in trajectory
        x_traj = jnp.concatenate([x0[None], x_t])
        
        # Compute final control and gradp
        uK = score_fn(params, n_steps, xK)
        gradpK = target_dist.grad_batch(xK)
        
        # Build full trajectory arrays
        states = jnp.concatenate([x_traj, xK[None]], axis=0)  # [x0, x1, ..., xK]
        controls = jnp.concatenate([u_t, uK[None]], axis=0)  # [u0, u1, ..., uK]
        gradps = jnp.concatenate([gradp_t, gradpK[None]], axis=0)  # [gradp0, ..., gradpK]
        
        # Compute transition terms
        def compute_transition_term(i):
            x_cur = states[i]
            x_next = states[i+1]
            u_cur = controls[i]
            u_next = controls[i+1]
            gradp_cur = gradps[i]
            gradp_next = gradps[i+1]
            
            # Forward transition density
            mu_fwd = x_cur + (sigma2 * gradp_cur + u_cur) * delta_t
            log_pfwd = self._log_gauss(x_next, mu_fwd, 2*sigma2*delta_t)
            
            # Backward transition density
            factor = jax.lax.cond(
                self.use_ctrl_den,
                lambda: -1.0, # CMCD
                lambda: 0.0 # MCD
            )
            mu_bwd = x_next + (sigma2 * gradp_next + factor * u_next) * delta_t
            log_pbwd = self._log_gauss(x_cur, mu_bwd, 2*sigma2*delta_t)
            
            return log_pfwd - log_pbwd
        
        # Vectorize over time steps
        trans_terms = jax.vmap(compute_transition_term)(jnp.arange(n_steps))
        trans_sum = jnp.sum(trans_terms, axis=0)  # Sum over time
        
        # Endpoint terms
        log_pT = target_dist.batch(xK)  # π_T(x_T)
        log_ratio = log_p0 - log_pT + trans_sum
        
        return jnp.mean(-log_ratio)

    def _log_gauss(self, x, mu, var):
        """Log-density of isotropic Gaussian"""
        d = x.shape[-1]
        log_z = -0.5 * d * jnp.log(2 * jnp.pi * var)
        return log_z - 0.5 * jnp.sum((x - mu)**2, axis=-1) / var



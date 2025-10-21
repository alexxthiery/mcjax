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
    Reverse KL / Log Variance losses for DDS.
    """
    def __init__(self, add_score: bool = False):
        self.add_score = add_score

    def __call__(self, params, key, process, init_dist, target_dist, score_fn, batch_size, **kwargs):
        K = process.K
        sigma = process.sigma

        # sample y0 ~ init_dist
        key, sub = jr.split(key)
        y0 = init_dist.sample(sub, batch_size)*sigma  # scale by sigma

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

            ##########################################################
            # normalize the weights (lam**2 / alpha_Kmk) across t 
            # raw_w = (lam**2 / alpha_Kmk)
            # norm_w = raw_w / (jnp.mean(raw_w) + 1e-12)   # normalize across t 
            # main_term = (2.0 * sigma**2) * norm_w * jnp.sum(s**2, axis=-1) 

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

        # jax.debug.print("rK mean: {}, log_ref mean: {}, log_targ mean: {}", jnp.mean(rK), jnp.mean(log_ref), jnp.mean(log_targ))

        return loss

class IDEMLoss(BaseLoss):
    """
    Implements the Iterated Denoising Energy Matching (iDEM) inner-loop loss:
      L_DEM(x_t, t) = || S_K(x_t, t) - s_theta(x_t, t) ||^2,
    """
    def __init__(self, num_samples: int, sigma_fn: callable, target_dist, score_fn, total_step, add_drift, sample_t_weight):
        """
        Args:
          num_samples: number of Monte Carlo samples used in estimating the score S_K.
        """
        self.num_samples = num_samples
        self.sigma_fn = sigma_fn  # (geometric) noise schedule
        # self.buffer = buffer  # a Buffer instance to sample x0 from
        self.target_dist = target_dist
        self.score_fn = score_fn  # Store score_fn here
        self.total_step = total_step # total number of steps in the process
        self.add_drift = add_drift
        self.sample_t_weight = sample_t_weight

    def true_score(self, x_t, step, mu, comp_sigmas, weights):
        '''
        Exact score for 1d mixed-isotropic Gaussian mixture at time t.
        '''
        # jax.debug.print("x_t shape: {}, step shape: {}", x_t.shape, step.shape)
        x_t = jnp.squeeze(x_t)
        step = jnp.squeeze(step)
        # current noise level
        t = step / self.total_step  # normalize t to [0,1]
        sigma_max = self.sigma_fn(1)
        sigma_min = self.sigma_fn(0)
        int_sigma = jnp.sqrt(1/3*t**3*(sigma_max-sigma_min)**2 + sigma_min*(sigma_max-sigma_min)*t**2\
                             +sigma_min**2*t)

        if self.add_drift:
            # OU-style drifted process
            a = jnp.exp(-0.5 * int_sigma**2)   # decay factor
            v = a**2 * (comp_sigmas**2) + (1 - a**2)   # variance per component
            m = a * mu.flatten()               # mean per component

        else:
            #   v_i = σ_i^2 + σ_t^2
            v   = comp_sigmas**2 + int_sigma**2  
            m = mu.flatten()
        
        diffs = m - x_t        # (n_comp,)
        norm  = jnp.sqrt(2 * jnp.pi * v)
        exps  = jnp.exp(-0.5 * (diffs**2) / v) / norm

        # unnormalized component responsibilities
        pis = weights * exps              # (n_comp,)

        # numerator and denominator for score
        numer = jnp.sum(pis * (diffs / v))
        denom = jnp.sum(pis)
        # denom = jnp.clip(denom, 1e-10, jnp.inf)  # avoid division by zero

        return numer / denom

    def get_int_sigma(self, t):
        # calculate int_sigma = sqrt(∫₀ᵗ σ(s)²) ds for a given t
        sigma_max = self.sigma_fn(1)
        sigma_min = self.sigma_fn(0)
        int_sigma = jnp.sqrt(1/3*t**3*(sigma_max-sigma_min)**2 + sigma_min*(sigma_max-sigma_min)*t**2\
                             +sigma_min**2*t)
        return int_sigma

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self,
                params,
                key: jr.PRNGKey,
                buffer):
        """
        Returns:
        loss: scalar, the average MSE between S_K(x_t, t) and s_theta(x_t, t).
        """
        # Draw a batch of x0 ∼ buffer
        x0,key = buffer.sample(key, self.num_samples)    # shape: (B, d, ...)
        key,sub = jr.split(key)

        if self.sample_t_weight:
            # ---- weight \propto int_sigma^2/(1+int_sigma^2) ----
            steps = jnp.arange(1, self.total_step+1)  
            ts = steps / self.total_step
            int_sigmas = self.get_int_sigma(ts)
            weights = int_sigmas**2/(1+int_sigmas**2)
            weights = weights / jnp.sum(weights)

            # sample from categorical distribution
            step = jr.choice(sub, steps, shape=(self.num_samples,1), p=weights)

        else:
            # Sample t ∼ Uniform(0,1) for all the x0 in the batch
            step = jr.randint(sub, shape=(self.num_samples,), minval=1, maxval=self.total_step+1).reshape(-1,1)  # shape (B, 1)
        t = step / self.total_step

        # calculate int_sigma = sqrt(∫₀ᵗ σ(s)²) ds for all t in the batch
        # Here we use the default σ(t) = σ_max*t + σ_min*(1-t)
        int_sigma = self.get_int_sigma(t) 
        int_sigma = int_sigma.reshape((-1,) + (1,) * (x0.ndim - 1))  # broadcasting to x0's shape
        key, sub = jr.split(key)
        eps = jr.normal(sub, shape=x0.shape)
        if self.add_drift:
            # X_t = exp(-int_sigma**2/2)*X_0 + N(0, 1-exp(-int_sigma**2)) 
            x_t = jnp.exp(-int_sigma**2 / 2) * x0 + eps * jnp.sqrt(1 - jnp.exp(-int_sigma**2))

        else:
            x_t = x0 + int_sigma * eps

        def mc_estimate_single(x_t_single, t, key_single):
            int_sigma = self.get_int_sigma(t)
            eps_MC = jr.normal(key_single, shape=(self.num_samples,) + x_t_single.shape)
            if self.add_drift:
                x0_MC = jnp.exp(-int_sigma**2 / 2) * x_t_single[None, ...] + eps_MC * jnp.sqrt(1 - jnp.exp(-int_sigma**2))

            else:
                x0_MC  = x_t_single[None, ...] + int_sigma * eps_MC 

            # target log-prob
            logw = self.target_dist.batch(x0_MC)    

            # Normalize
            lse   = logsumexp(logw, axis=0)
            w_norm = jnp.exp(logw - lse)
            if self.add_drift:
                score_terms = (x0_MC * jnp.exp(-int_sigma**2 / 2) - x_t_single) / (1 - jnp.exp(-int_sigma**2))

            else:
                score_terms = (x0_MC - x_t_single) / (int_sigma**2)
            numerator   = jnp.sum(w_norm[:, None] * score_terms, axis=0)

            return numerator


        keys_batch = jr.split(key, self.num_samples) 


        # For debugging, use true_score instead of MC estimate
        mu = self.target_dist.mu
        comp_sigmas = jnp.exp(self.target_dist.log_var)**0.5
        weights = jnp.exp(self.target_dist.log_w)
        # S_K_batch_true = jax.vmap(self.true_score, in_axes=(0, 0, None, None, None), out_axes=0)(
        #     x_t, step, mu, comp_sigmas, weights)
        # jax.debug.print("S_K_batch shape: {}", S_K_batch.shape)
        S_K_batch = jax.vmap(mc_estimate_single, in_axes=(0, 0, 0), out_axes=0)(
            x_t, t, keys_batch)
        

        def score_fn_wrapper(params, step_scalar, x_t_single):
#            add batch dimension for x_t_single so it matches score_fn’s API
            x_t_single = x_t_single[None, ...]       # shape (1, d, …)
            out = self.score_fn(params, step_scalar, x_t_single)
            return out[0]          

        # Vectorize over (step, x_t)
        s_pred = jax.vmap(score_fn_wrapper, in_axes=(None, 0, 0))(params, step, x_t)


        # Compute per-example squared ‖S_K - s_pred‖² and average:
        sq_err = jnp.sum((S_K_batch - s_pred) ** 2,
                        axis=tuple(range(1, S_K_batch.ndim)))  
        loss = jnp.mean(sq_err)  # scalar

        
        ############################################################
        # # test the difference between MC estimate and true_score
        # diff_true_est = jnp.mean(jnp.abs(S_K_batch - S_K_batch_true))
        diff_true_est = jnp.array(0.0)  # dummy, not used
        ############################################################

        return loss, diff_true_est

class PISLoss(BaseLoss):

    def __init__(self, add_score: bool = False):
        self.add_score = add_score # NO NEED, just to align with other losses

    def __call__(self, params, key, process,init_dist, target_dist, score_fn, batch_size, **kwargs):
        # forward controlled SDE from x0 ~ ν
        self.delta_t = process.T /process.K
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
        var_total = 1.0 + process.T  # Total variance = 1+T (pure Brownian motion)
        d = xT.shape[-1]
        log_qT = -0.5 * d * jnp.log(2 * jnp.pi * var_total) \
                - 0.5 * jnp.sum(xT**2, axis=-1) / var_total
        log_p  = target_dist.batch(xT)
        psi    = log_qT - log_p

        return jnp.mean(running + psi)

class CMCDLoss(BaseLoss):
    """
    Discrete-time CMCD loss based on the paper
    'Transport Meets Variational Inference'. 

    The loss per sample is:
      L = log π0(x0) - log πT(xK) + sum_{k=0}^{K-1} [ log N(x_{k+1}; mu_fwd_k, var)
                                                       - log N(x_k; mu_bwd_k, var) ]
    where
      mu_fwd_k = x_k + (sigma^2 * grad ln π_{t_k} + factor * u_k) * Δt
      mu_bwd_k = x_{k+1} + (sigma^2 * grad ln π_{t_{k+1}} - u_{k+1}) * Δt
    and var = 2 * sigma^2 * Δt.

    """
    def __init__(self, use_control_in_denominator: bool, add_score: bool = False):
        self.use_ctrl_den = use_control_in_denominator
        self.add_score = add_score

    def __call__(self, params, key, process, init_dist, target_dist, score_fn, batch_size, **kwargs):
        n_steps = process.K
        delta_t = 1.0 / n_steps          
        sigma = process.sigma
        sigma2 = sigma**2
        var = 2.0 * sigma2 * delta_t     

        # sample initial states
        key, sub = jr.split(key)
        x0 = init_dist.sample(sub, batch_size)     
        log_p0 = init_dist.batch(x0)              

        # forward Euler-Maruyama scan 
        def forward_step(carry, t):
            x, key = carry
            key, sub = jr.split(key)
            t_norm = jnp.array(t, dtype=jnp.float32) / jnp.array(n_steps, dtype=jnp.float32)

            u = score_fn(params, t, x)  

            # geometric interpolation of score between init and target (\pi_t)
            gradp = (1.0 - t_norm) * init_dist.grad_batch(x) + t_norm * target_dist.grad_batch(x)

            # forward step
            noise = jr.normal(sub, x.shape) * jnp.sqrt(2.0 * sigma2 * delta_t)
            x_next = x + (sigma2 * gradp + u) * delta_t + noise

            out = (x, x_next, u, gradp)
            return (x_next, key), out

        times = jnp.arange(n_steps)  
        (xK, _), forward_vals = jax.lax.scan(forward_step, (x0, key), times)
        x_t, x_tp1, u_t, gradp_t = forward_vals  

        # compute final control and gradp at time T 
        uK = score_fn(params, n_steps, xK)            
        gradpK = target_dist.grad_batch(xK)          

        states = jnp.concatenate([x_t, xK[None, ...]], axis=0)     
        controls = jnp.concatenate([u_t, uK[None, ...]], axis=0)  
        gradps = jnp.concatenate([gradp_t, gradpK[None, ...]], axis=0)

        def _log_gauss(x, mu, variance):
            D = x.shape[-1]
            norm = -0.5 * D * jnp.log(2.0 * jnp.pi * variance)
            quad = -0.5 * jnp.sum((x - mu)**2, axis=-1) / variance
            return norm + quad  

        # compute transition log-ratio for each k = 0..K-1
        def per_step(k):
            # extract arrays for step k
            x_k = states[k]        
            x_kp1 = states[k+1]
            u_k = controls[k]
            u_kp1 = controls[k+1]
            grad_k = gradps[k]
            grad_kp1 = gradps[k+1]

            # factor controls inclusion in forward mean depending on the MCD/CMCD variant:
            # - CMCD (use_ctrl_den=True): include control in forward (factor=1) and denom uses -u_{k+1}
            # - MCD  (use_ctrl_den=False): set phi=0 in numerator (factor=0), denom still uses -u_{k+1}
            factor = 1.0 if self.use_ctrl_den else 0.0

            mu_fwd = x_k + (sigma2 * grad_k + factor * u_k) * delta_t
            log_pfwd = _log_gauss(x_kp1, mu_fwd, var)

            mu_bwd = x_kp1 + (sigma2 * grad_kp1 - u_kp1) * delta_t
            log_pbwd = _log_gauss(x_k, mu_bwd, var)

            return log_pfwd - log_pbwd  

        # vectorize over k
        ks = jnp.arange(n_steps)
        trans_terms = jax.vmap(per_step)(ks)  
        trans_sum = jnp.sum(trans_terms, axis=0) 
        # jax.debug.print("trans_terms: {t}", t=trans_terms)

        # endpoints
        log_pT = target_dist.batch(xK)    
        log_ratio = log_p0 - log_pT + trans_sum  
        # test if x, log_pT or trans_sum contain NaNs (jax.cond)
        # jax.debug.print("x = {x}", x=states[:,14,:])
        # jax.debug.print("gradpT = {g}", g=gradps[:,14,:])
        # jax.debug.print("size = {s}", s=states.shape)

        # jax.lax.cond(jnp.any(jnp.isnan(states)), lambda: jax.debug.print("x contains NaNs"), lambda: None)
        # jax.lax.cond(jnp.any(jnp.isnan(log_pT)), lambda: jax.debug.print("log_pT contains NaNs"), lambda: None)
        # jax.lax.cond(jnp.any(jnp.isnan(trans_sum)), lambda: jax.debug.print("trans_sum contains NaNs"), lambda: None)
        # inspect states (print one element in the batch)
        # jax.debug.print("x: {x}", x=states[:,0,:])
        # jax.debug.print("gradp_t: {g}", g=gradps[:,0,:])
        # jax.debug.print("u_t: {u}", u=controls[:,0,:])


        return jnp.mean(log_ratio)


class SupervisedScoreMatchingLoss(BaseLoss):
    '''
    For debugging purposes, this loss computes the supervised score matching loss(the true loss) for a mixture of Gaussians.
    It is used to verify the correctness of the score function.
    '''
    def __init__(self, mu, comp_sigmas, weights, ou, data_dim):
        self.mu = mu
        self.comp_sigmas = comp_sigmas
        self.weights = weights
        self.ou = ou
        self.data_dim = data_dim
        self.add_score = False  # Not used in this loss, just to align with other losses

        one_minus_alpha = 1.0 - self.ou.alpha
        cumprod = jnp.cumprod(one_minus_alpha)    
        self.a = jnp.concatenate([jnp.array([1.0]), cumprod[:-1]], axis=0)  

    def __call__(self, params, key, process, init_dist, target_dist, score_fn, batch_size, **kwargs):
        K = process.K
        key, sub = jr.split(key)
        x = jr.normal(sub, (batch_size, self.data_dim))
        key, sub = jr.split(key)
        t = jr.randint(sub, (batch_size,), minval=0, maxval=K)

        s_pred = score_fn(params, t, x)
        s_true = self.mixture_score(x, t)

        return jnp.mean(jnp.sum((s_pred - s_true) ** 2, axis=-1))
    
    def mixture_score(self, x, k):
        """
        Compute the score function in OU process for 1d (isotropic) mixed-gaussian initial distribution.
        """
        # x:   of shape (batch)
        # k:   integer time index
        # mu:  array (n_comp, 1)
        # comp_sigmas: array (n_comp,)  # component std devs
        # weights: array (n_comp,)
    
        a_k = jnp.take(self.a, k)

        sqrt_a = jnp.sqrt(a_k)[:, None, None]       # (batch, 1, 1)
        m_k = sqrt_a * self.mu[None, :, :]    # (batch, n_comp, data_dim)
        v_k = (
            a_k[:, None] * (self.comp_sigmas**2)[None, :]
            + (1 - a_k)[:, None] * (self.ou.sigma**2)
        )                                      # (batch, n_comp)

        # Expand x for components
        x_exp = x[:, None, :]                  # (batch, 1, data_dim)
        v_exp = v_k[:, :, None]                # (batch, n_comp, 1)
        w_exp = self.weights[None, :, None]    # (1, n_comp, 1)
    
        # Compute Gaussian PDFs: shape (batch, n_comp, data_dim)
        diffs = m_k - x_exp                    # (batch, n_comp, data_dim)
        normalizer = jnp.sqrt(2 * jnp.pi * v_exp)
        exps = jnp.exp(-0.5 * (diffs**2) / v_exp) / normalizer

        # Weighted mixture densities (per‑dim)
        pis = w_exp * exps                     # (batch, n_comp, data_dim)

        # Numerator: ∑_i [ w_i N_i(x) * (m_i - x) / v_i ]
        numer = jnp.sum(pis * (diffs / v_exp), axis=1)   # (batch, data_dim)
        # Denominator: ∑_i [ w_i N_i(x) ]
        denom = jnp.sum(pis, axis=1)                    # (batch, data_dim)
    
        return numer / denom                            # (batch, data_dim)

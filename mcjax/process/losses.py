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
    def __init__(self, 
                 add_score: bool = False, 
                 loss_type: str = 'kl',
                 sde_ctrl_noise: float = 0.0):
        """
        Args:
          add_score: bool, whether to add the zero-expectation score term.
          loss_type: str, 'kl' for reverse KL loss or 'lv' for log-variance loss.
          sde_ctrl_noise: float, stddev of Gaussian noise added to the
                          detached SDE control (path) for exploration.
                          Only used if loss_type='lv'.
        """
        self.add_score = add_score
        if loss_type not in ['kl', 'lv']:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'kl' or 'lv'.")
        self.loss_type = loss_type
        self.sde_ctrl_noise = sde_ctrl_noise

        if self.sde_ctrl_noise > 0.0 and self.loss_type == 'kl':
            print(
                "Warning: DDSLoss sde_ctrl_noise > 0.0 but loss_type is 'kl'. "
                "SDE noise is only applied for 'lv' loss and will be ignored."
            )

    def __call__(self, params, key, process, init_dist, target_dist, score_fn, batch_size, **kwargs):
        K = process.K
        sigma = process.sigma

        # sample y0 ~ init_dist
        key, sub = jr.split(key)
        y0 = init_dist.sample(sub, batch_size)

        def scan_step(carry, k):
            y_k, r_k, key = carry
            key_next, key_eps, key_noise = jr.split(key, 3)
            eps = jr.normal(key_eps, shape=y_k.shape)

            idx = K - 1 - k
            alpha_Kmk = process.alpha[idx]
            sqrt1m = jnp.sqrt(1.0 - alpha_Kmk)
            lam = 1.0 - sqrt1m

            s_attached = score_fn(params, idx, y_k)
            if self.loss_type == 'kl':
                s_path = s_attached
                s_loss = s_attached
            
            else: # self.loss_type == 'lv'
                s_detached = jax.lax.stop_gradient(s_attached)

                if self.sde_ctrl_noise > 0.0:
                    noise = jr.normal(key_noise, shape=s_detached.shape)
                    s_path = s_detached + self.sde_ctrl_noise * noise
                else:
                    s_path = s_detached
                
                s_loss = s_attached

            y_next = (sqrt1m * y_k
                      + 2.0 * (sigma**2) * lam * s_path
                      + sigma * jnp.sqrt(alpha_Kmk) * eps)

            main_term = (2.0 * sigma**2) * (lam**2 / alpha_Kmk) * jnp.sum(s_loss**2, axis=-1)

            if self.add_score:
                zero_exp_term = 2.0 * sigma * jnp.sqrt(lam**2 / alpha_Kmk) * jnp.sum(s_loss * eps, axis=-1)
                r_next = r_k + main_term + zero_exp_term
            else:
                r_next = r_k + main_term

            return (y_next, r_next, key_next), None

        r0 = jnp.zeros(batch_size)
        (yK, rK, _), _ = jax.lax.scan(
            scan_step,
            (y0, r0, key),
            jnp.arange(K)
        )

        # now compute log‐ratio = log p_ref(yK) - log p_target(yK)
        log_ref = init_dist.batch(yK)
        log_targ = target_dist.batch(yK)

        w_batch = rK + log_ref - log_targ

        if self.loss_type == 'kl':
            loss = jnp.mean(w_batch)
        else: # for self.loss_type == 'lv'
            loss = jnp.var(w_batch)

        
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
        self.target_dist = target_dist
        self.score_fn = score_fn 
        self.total_step = total_step # total number of steps in the process
        self.add_drift = add_drift
        self.sample_t_weight = sample_t_weight

    def true_score(self, x_t, step, mu, comp_sigmas, weights):
        '''
        Exact score for 1d mixed-isotropic Gaussian mixture at time t.
        '''
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
        
        diffs = m - x_t      
        norm  = jnp.sqrt(2 * jnp.pi * v)
        exps  = jnp.exp(-0.5 * (diffs**2) / v) / norm

        # unnormalized component responsibilities
        pis = weights * exps       

        # numerator and denominator for score
        numer = jnp.sum(pis * (diffs / v))
        denom = jnp.sum(pis)

        return numer / denom

    def get_int_sigma(self, t):
        # int_sigma = sqrt(\int_0^t σ(s)^2 ds) for a given t
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
        x0,key = buffer.sample(key, self.num_samples)
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
            step = jr.randint(sub, shape=(self.num_samples,), minval=1, maxval=self.total_step+1).reshape(-1,1)
        t = step / self.total_step

        # int_sigma = sqrt(\int_0^t σ(s)^2 ds) for all t in the batch
        # Here we use the default σ(t) = σ_max*t + σ_min*(1-t)
        int_sigma = self.get_int_sigma(t) 
        int_sigma = int_sigma.reshape((-1,) + (1,) * (x0.ndim - 1)) 
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
        weights = jnp.exp(self.target_dist.log_w)
        S_K_batch = jax.vmap(mc_estimate_single, in_axes=(0, 0, 0), out_axes=0)(
            x_t, t, keys_batch)
        

        def score_fn_wrapper(params, step_scalar, x_t_single):
            x_t_single = x_t_single[None, ...]
            out = self.score_fn(params, step_scalar, x_t_single)
            return out[0]          

        s_pred = jax.vmap(score_fn_wrapper, in_axes=(None, 0, 0))(params, step, x_t)


        # Compute per-example squared ‖S_K - s_pred‖^2 and average:
        sq_err = jnp.sum((S_K_batch - s_pred) ** 2,
                        axis=tuple(range(1, S_K_batch.ndim)))  
        loss = jnp.mean(sq_err) 

        
        ############################################################
        # # test the difference between MC estimate and true_score
        # diff_true_est = jnp.mean(jnp.abs(S_K_batch - S_K_batch_true))
        diff_true_est = jnp.array(0.0)  # dummy, not used
        ############################################################

        return loss, diff_true_est

class PISLoss(BaseLoss):

    def __init__(self, add_score: bool = False, loss_type: str = 'kl', sde_ctrl_noise: float = 0.0):
        self.add_score = add_score # NO NEED, just to align with other losses
        self.loss_type = loss_type
        self.sde_ctrl_noise = sde_ctrl_noise
        if self.sde_ctrl_noise > 0.0 and self.loss_type == 'kl':
            print("Warning: PISLoss sde_ctrl_noise > 0.0 but loss_type is 'kl'. Will be ignored.")

    def __call__(self, params, key, process,init_dist, target_dist, score_fn, batch_size, **kwargs):
        # forward controlled SDE from x0 ~ ν
        self.delta_t = process.T /process.K
        self.n_steps = process.K      # align with OU steps
        key, sub = jr.split(key)
        x = init_dist.sample(sub, batch_size)

        running_cost = jnp.zeros(batch_size) # first term in the loss function
        def body(carry, t):
            x, running, key = carry
            key_next, key_dw, key_noise = jr.split(key, 3)

            u_attached = score_fn(params, t, x)

            if self.loss_type == 'kl':
                u_path = u_attached
                u_loss = u_attached
            else: # for 'lv'
                u_detached = jax.lax.stop_gradient(u_attached)
                if self.sde_ctrl_noise > 0.0:
                    noise = jr.normal(key_noise, shape=u_detached.shape)
                    u_path = u_detached + self.sde_ctrl_noise * noise
                else:
                    u_path = u_detached
                u_loss = u_attached

            running = running + 0.5 * jnp.sum(u_loss**2, axis=-1) * self.delta_t

            dW = jr.normal(key_dw, x.shape) * jnp.sqrt(self.delta_t)
            x_next = x + u_path * self.delta_t + dW
            
            return (x_next, running, key_next), None


        times = jnp.arange(self.n_steps, dtype=jnp.float32)
        (xT, running, _), _ = jax.lax.scan(body, (x, running_cost, key), times)

        var_total = 1.0 + process.T  # Total variance = 1+T (pure Brownian motion)
        d = xT.shape[-1]
        log_qT = -0.5 * d * jnp.log(2 * jnp.pi * var_total) \
                - 0.5 * jnp.sum(xT**2, axis=-1) / var_total
        log_p  = target_dist.batch(xT)
        psi    = log_qT - log_p
        w_batch = running + psi

        if self.loss_type == 'kl':
            loss = jnp.mean(w_batch)
        else: # for self.loss_type == 'lv'
            w_mean_baseline = jax.lax.stop_gradient(jnp.mean(w_batch))
            loss = jnp.mean((w_batch - w_mean_baseline) ** 2)

        return loss

class CMCDLoss(BaseLoss):
    """
    Discrete-time CMCD/MCD loss
    """
    def __init__(self, use_control_in_denominator: bool, add_score: bool = False,\
                 loss_type: str = 'kl', sde_ctrl_noise: float = 0.0):
        self.use_ctrl_den = use_control_in_denominator
        self.add_score = add_score
        self.loss_type = loss_type
        self.sde_ctrl_noise = sde_ctrl_noise
        if self.sde_ctrl_noise > 0.0 and self.loss_type == 'kl':
            print("Warning: CMCDLoss sde_ctrl_noise > 0.0 but loss_type is 'kl'. Will be ignored.")

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
            key_next, key_noise_sde, key_noise_ctrl = jr.split(key, 3)
            t_norm = jnp.array(t, dtype=jnp.float32) / jnp.array(n_steps, dtype=jnp.float32)

            u_attached = score_fn(params, t, x)  
            gradp = (1.0 - t_norm) * init_dist.grad_batch(x) + t_norm * target_dist.grad_batch(x) 

            if self.loss_type == 'kl':
                u_path = u_attached
            else: # 'lv'
                u_detached = jax.lax.stop_gradient(u_attached)
                if self.sde_ctrl_noise > 0.0:
                    noise = jr.normal(key_noise_ctrl, shape=u_detached.shape)
                    u_path = u_detached + self.sde_ctrl_noise * noise
                else:
                    u_path = u_detached
            
            noise_sde = jr.normal(key_noise_sde, x.shape) * jnp.sqrt(2.0 * sigma2 * delta_t)
            x_next = x + (sigma2 * gradp + u_path) * delta_t + noise_sde 

            out = (x, x_next, u_attached, gradp) 
            return (x_next, key_next), out

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

        def per_step(k):
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

        # endpoints
        log_pT = target_dist.batch(xK)    
        w_batch = log_p0 - log_pT + trans_sum  


        if self.loss_type == 'kl':
            loss = jnp.mean(w_batch)
        else: # self.loss_type == 'lv'
            w_mean_baseline = jax.lax.stop_gradient(jnp.mean(w_batch))
            loss = jnp.mean((w_batch - w_mean_baseline) ** 2)
            
        return loss

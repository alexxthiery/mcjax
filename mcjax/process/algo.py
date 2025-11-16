from abc import ABC, abstractmethod
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import gaussian_kde
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation
from flax import struct
from jax.scipy.special import logsumexp

from models import MLPModel, ResBlockModel
from ou import OU
from mcjax.proba.neal_funnel import NealFunnel
from mcjax.proba.banana2d import Banana2D
from mcjax.proba.gaussian import IsotropicGauss, MixedIsotropicGauss, GMM40, GMMFixed
from mcjax.proba.doublewell import DoubleWell
from mcjax.proba.log_gauss_pines import LogGaussPines
from losses import DDSLoss, IDEMLoss, PISLoss, CMCDLoss
from trainer import Trainer, InnerTrainer
from mcjax.proba.sonar import BayesianLogisticTarget

class BaseAlgorithm(ABC):
    """
    Abstract template for any sampler (PIS, DDS, iDEM, MCD, CMCD, etc.).
    A concrete subclass must implement:
      - __init__(config)
      - make_score_fn()
      - make_loss()
      - train(rng_key)
      - sample(params, rng_key, num_samples)
    In addition, every subclass must set the following attributes in __init__:
      - self.ou          : the OU forward/reverse process
      - self.init_dist   : the reference (initial) distribution
      - self.target_dist : the target distribution
      - self.score_fn    : the score network function
      - self.params      : the network parameters (Flax/Haiku state)
      - self.cfg         : the config namespace/dict
      - self.data_dim    : 1 or 2 (used in visualize_samples)
    """
    def __init__(self, config):
        """
        config: argparse.Namespace (or dict) containing all needed hyperparameters.
        Subclass __init__ must at least define:
          self.cfg, self.ou, self.init_dist, self.target_dist,
          self.score_fn, self.params, self.state, self.data_dim
        """
        self.cfg = config

        # Build the target distribution
        if config.target_dist == 'gmm40':
            self.target_dist = GMM40(n_mixes=2, loc_scaling=5.0, weights=jnp.array([0.6, 0.4])) # test with 2 components for faster debugging
            self.data_dim = 2
        elif config.target_dist == 'gmmfixed':
            self.target_dist = GMMFixed()
            self.data_dim = 2
        elif config.target_dist == 'doublewell':
            self.target_dist = DoubleWell(dim=5, m=5, delta=4.0)
            self.data_dim = 5
        elif config.target_dist == 'doublewell2': # higher dimension
            self.target_dist = DoubleWell(dim=50, m=5, delta=2.0)
            self.data_dim = 50
        elif config.target_dist == '1d':
            mu = jnp.array([[-1.],[1.]])
            dist_sigma = jnp.array([0.5,0.6])
            log_var = jnp.log(dist_sigma**2)
            weights = jnp.array([0.6,0.4])
            self.target_dist = MixedIsotropicGauss(
                mu=mu, log_var=log_var, weights=weights
            )
            self.data_dim = 1
        elif config.target_dist == 'funnel':
            self.target_dist = NealFunnel(sigma_x=3.0, dim=2)
            self.data_dim = 2

        elif config.target_dist == 'banana2d':
            self.target_dist = Banana2D(noise_std=0.1)
            self.data_dim = 2
        
        elif config.target_dist == 'pines':
            self.target_dist = LogGaussPines(grid_dim=40, use_whitened=False)
            self.data_dim = 40 * 40  # 1600 dimensions
        
        elif config.target_dist == 'sonar':
            self.target_dist = BayesianLogisticTarget(prior_var=1.0)
            self.data_dim = self.target_dist.d  # 61 features + bias = 62

        else:
            raise ValueError(f"Unknown target_dist: {config.target_dist}")

        # Build the reference initial distribution
        self.init_dist = IsotropicGauss(
            mu=jnp.zeros(self.data_dim), log_var=2*jnp.log(config.sigma)
        )
        
        # create timesteps / beta / alpha schedule
        K = config.K
        T = config.T
        ts = jnp.arange(K, dtype=jnp.float32)
        ############################
        # Always set this to false: No need to use variable time steps 
        ############################
        if config.variable_ts:
            beta_start, beta_end = 1.0, 20.0
            beta = beta_start + (beta_end - beta_start) * (ts / (K - 1))
        else:
            beta = jnp.ones(K) * 1
        alpha = 1.0 - jnp.exp(-2.0 * beta * T / K)

        # make the OU process
        self.ou = OU(T = T, alpha=alpha, sigma=config.sigma, init_dist=self.init_dist)

        # Set a unified optimizer
        self.opt = optax.chain(
            optax.clip_by_global_norm(5.0),
            optax.adamw(learning_rate=config.lr, b1=0.9, b2=0.99, weight_decay=1e-4)
        )



    def make_score_fn(self):
        """
        Returns: score_fn(params, k, y) -> shape (batch, data_dim)
        Implements:
          nn1, nn2 = model.apply(params, y, t=k)
          log_mu = target_dist.batch(y)
          grad_log_mu = target_dist.grad_batch(y)
          according to condition_term: 'none'/'score'/'grad_score'
        """
        condition = self.cfg.condition_term
        target = self.target_dist

        def score_fn(params, k, y):

            # k: int index in [0, K-1], y: shape (batch, data_dim)
            batch_k = jnp.full((y.shape[0],), k, dtype=jnp.int32)
            nn1, nn2 = self.model.apply(params, y, batch_k)

            if condition == 'none':
                return nn1

            elif condition == 'score':
                logp = target.batch(y)  
                normed = logp / (jnp.std(logp, axis=0, keepdims=True) + 1e-5)
                return nn1 + nn2 * normed[:, None]  

            elif condition == 'grad_score':
                gradp = target.grad_batch(y)  
                normed = gradp / (jnp.std(gradp, axis=0, keepdims=True) + 1e-5)
                return nn1 + nn2 * normed

            else:
                raise ValueError(f"Unknown condition_term: {condition}")

        return jax.jit(score_fn)

    @abstractmethod
    def make_loss(self):
        """
        Returns a BaseLoss instance (e.g. DDSLoss, IDEMLoss, etc.).
        """
        pass

    @partial(jax.jit, static_argnums=(0,))
    def train(self, rng_key):
        """
        Runs the outer training loop using the generic Trainer.
        Returns (final_state, final_key, loss_history, logZ_vals, logZ_vars)
        """
        trainer = Trainer(
            algorithm    = self,
            process      = self.ou,                
            init_dist    = self.init_dist,
            target_dist  = self.target_dist,
            score_fn     = self.score_fn,     
            loss_obj     = self.loss_obj,
            state        = self.state,
            batch_size   = self.cfg.batch_size,
            num_steps    = self.cfg.num_steps,
            if_logZ      = self.cfg.if_logZ              
        )
        return trainer.run(rng_key)

    def sample(self, params, rng_key, num_samples: int):
        """
        Generate samples by running reverse OU chain.
        """
        @jax.jit
        def generate(params, key):
            key, sub = jr.split(key)
            yK = self.init_dist.sample(sub, num_samples)  # shape (num_samples, data_dim)
            def body(carry, k):
                y_next, key_ = carry
                key_, yk, score = self.ou.reverse_step(key_, y_next, k, self.score_fn, params)
                return (yk, key_), (yk,score)
            (y0, _), (seq,score_seq) = jax.lax.scan(
                body, (yK, key), jnp.arange(self.ou.K)
            )
            return seq,score_seq  
        return generate(params, rng_key)   

    @abstractmethod
    def estimate_logZ(self):
        """
        Estimate log partition function using reverse OU chain.
        Returns: logZ estimates of shape (num_samples,)
        """
        pass

    def visualize_samples(self, sample_seq):
        """
        Generic 1D / 2D visualization of the reverse chain. 
        sample_seq is expected to have shape (K, num_samples, data_dim).

        Subclasses must set:
         - self.data_dim ∈ {1,2}
         - self.init_dist, self.target_dist to sample enough points for KDE/contour
         - self.cfg.K, self.cfg.folder_path, etc.
        """
    
        if self.data_dim == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot initial‐ and target‐density reference lines
            xs = jnp.linspace(-7, 10, 1000)
            init_samples = self.init_dist.sample(jr.PRNGKey(0), 100000).flatten()
            targ_samples = self.target_dist.sample(jr.PRNGKey(1), 100000).flatten()

            initial_kde = gaussian_kde(init_samples)
            target_kde  = gaussian_kde(targ_samples)

            ax.plot(xs, initial_kde(xs), 'b--', lw=2, label='Init Dist')
            ax.plot(xs, target_kde(xs),  'g--', lw=2, label='Target Dist')

            # Precompute KDEs for each frame
            kde_x = jnp.linspace(-7, 10, 500)
            frame_densities = []
            for frame in range(self.cfg.K):
                curr = sample_seq[frame].flatten()
                kde = gaussian_kde(curr)
                frame_densities.append(kde(kde_x))

            line, = ax.plot([], [], 'r-', lw=2, label='Samples')
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(-7, 10)
            ax.set_ylim(0, 0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('density')
            ax.set_title('1D Density Evolution')
            ax.legend(loc='upper right')

            def animate(frame):
                line.set_data(kde_x, frame_densities[frame])
                time_text.set_text(f'Step: {frame}/{self.cfg.K}')
                return line, time_text

            ani = animation.FuncAnimation(
                fig=fig,
                func=animate,
                frames=self.cfg.K,
                interval=20,
                blit=True
            )
            writer = FFMpegWriter(fps=30, metadata=dict(artist='BaseAlgorithm'), bitrate=1800)
            fname = f'{self.cfg.folder_path}/{self.cfg.target_dist}/density_evolution_{self.cfg.algo}_{self.cfg.loss_type}.mp4'
            ani.save(fname, writer=writer)
            plt.close()

        elif self.data_dim == 2:
            key = jr.PRNGKey(42)
            pts = self.target_dist.sample(key, 100_000)
            pts = jax.device_get(pts)

            lower = np.percentile(pts, 0.5, axis=0)
            upper = np.percentile(pts, 99.5, axis=0)
            margin = 0.05 * (upper - lower)
            xmin, xmax = lower[0] - margin[0], upper[0] + margin[0]
            ymin, ymax = lower[1] - margin[1], upper[1] + margin[1]

            x = np.linspace(xmin, xmax, 200)
            y = np.linspace(ymin, ymax, 200)
            X, Y = np.meshgrid(x, y)
            grid = np.stack([X.ravel(), Y.ravel()], axis=1)
            grid = jnp.array(grid)


            Ztarg = self.target_dist.batch(grid).reshape(X.shape)
            Ztarg = np.exp(np.array(Ztarg))
            Ztarg_norm = (Ztarg - Ztarg.min()) / (Ztarg.max() - Ztarg.min())
            vmin, vmax = Ztarg_norm.min(), Ztarg_norm.max()

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            ax_left, ax_right = axes
            plt.tight_layout()

            # --- Right: target density (fixed contour) ---

            contour_right = ax_right.contourf(X, Y, Ztarg_norm, levels=30, cmap="viridis")
            ax_right.set_title("Target Density")
            ax_right.set_xlabel("x1")
            ax_right.set_ylabel("x2")

            # --- Left: evolving sample density  ---
            curr = sample_seq[0]
            H, xe, ye = np.histogram2d(curr[:, 1], curr[:, 0], bins=200, 
                                     range=[[ymin, ymax], [xmin, xmax]], density=True)
            H_norm = (H - H.min()) / (H.max() - H.min())
            ax_left.contourf(X, Y, H_norm, levels=30, cmap="viridis", vmin=vmin, vmax=vmax)
            ax_left.set_title("Evolving Sample Density")
            ax_left.set_xlabel("x1")
            ax_left.set_ylabel("x2")

            time_text = fig.text(0.45, 0.92, '', fontsize=12, ha='center')

            def animate(frame):
                # clear only the left axis
                ax_left.cla()

                # recompute histogram for this frame
                curr = sample_seq[frame]
                H, _, _ = np.histogram2d(curr[:, 1], curr[:, 0], bins=200,
                                         range=[[ymin, ymax], [xmin, xmax]], density=True)
                H_norm_frame = (H - H.min()) / (H.max() - H.min())

                # redraw contour on left axis
                ax_left.contourf(X, Y, H_norm_frame, levels=30, cmap="viridis", vmin=vmin, vmax=vmax)
                ax_left.set_title("Evolving Sample Density")
                ax_left.set_xlabel("x1")
                ax_left.set_ylabel("x2")

                time_text.set_text(f"Step: {frame}/{self.cfg.K}")

                return [time_text]
            
            ani = animation.FuncAnimation(
                fig=fig,
                func=animate,
                frames=self.cfg.K,
                interval=80,
                blit=False
            )

            writer = FFMpegWriter(fps=30, metadata=dict(artist='BaseAlgorithm'), bitrate=1800)
            fname = f"{self.cfg.folder_path}/{self.cfg.target_dist}/sample_movement_{self.cfg.algo}_{self.cfg.loss_type}.mp4"
            ani.save(fname, writer=writer)
            plt.close(fig)

        else:
            print(f"Unsupported data_dim: {self.data_dim}; Visualization only implemented for 1D and 2D data.")

class DDSAlgorithm(BaseAlgorithm):
    """
    Implements the DDS sampler (Denoising Diffusion Sampler)
    """

    def __init__(self, config):
        super().__init__(config)
        # build the network
        #    choose MLP or ResBlock based on config.model_type
        if config.network_name == 'mlp':
            self.model = MLPModel(dim=self.data_dim, T=config.K)
        elif config.network_name == 'resblock':
            self.model = ResBlockModel(dim=self.data_dim, T=config.K)
        else:
            raise ValueError(f"Unknown model_type: {config.network_name}")
        

        # initialize network params
        key = jr.PRNGKey(config.seed)
        key, sub = jr.split(key)
        dummy_x = jnp.zeros((config.batch_size, self.data_dim))
        dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.int32)
        self.params = self.model.init(sub, dummy_x, dummy_t)
        
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=self.params, tx=self.opt
        )

        # build score_fn
        self.score_fn = self.make_score_fn()

        # build loss object
        self.loss_obj = self.make_loss()

    def make_loss(self):
        return DDSLoss(add_score=self.cfg.add_score, loss_type=self.cfg.loss_type, sde_ctrl_noise=self.cfg.sde_ctrl_noise)

    @partial(jax.jit, static_argnums=(0, 3))
    def estimate_logZ(self, params, key, num_samples: int):
        """
        Generic reverse‐OU log‐Z estimator:
        Runs one Monte Carlo path from y_0 ~ init_dist, reverse‐OU to y_K,
        accumulates the quadratic term r_K, and returns 
          log Z = r_K + log p_ref(y_K) – log p_target(y_K)
        for each sample in the batch.

        Expects subclasses to have set:
          self.ou, self.init_dist, self.target_dist, self.score_fn
        """
        key, key_ = jr.split(key)
        # Draw y0 ~ init_dist
        y0 = self.init_dist.sample(key_, num_samples)

        # Define the per‐step reverse OU scan:
        def scan_step(carry, k):
            y_k, r_k, key = carry
            key, sub = jr.split(key)
            eps = jr.normal(sub, shape=y_k.shape)

            idx = self.ou.K - 1 - k
            alpha_Kmk = self.ou.alpha[idx]
            sqrt1m    = self.ou.sqrt_1m_alpha[idx]
            lam       = 1.0 - sqrt1m

            # network score at time index idx
            s = self.score_fn(params, idx, y_k)

            # reverse OU update
            y_next = (
                sqrt1m * y_k
                + 2.0 * (self.ou.sigma ** 2) * lam * s
                + self.ou.sigma * jnp.sqrt(alpha_Kmk) * eps
            )
            # accumulate r‐term
            r_next = r_k + (2.0 * self.ou.sigma ** 2) * ((lam ** 2) / alpha_Kmk) * jnp.sum(s ** 2, axis=-1)
            return (y_next, r_next, key), None

        # Initialize r_0 = 0
        init_carry = (y0, jnp.zeros(num_samples), key)
        (yK, rK, _), _ = jax.lax.scan(
            scan_step,
            init_carry,
            jnp.arange(self.ou.K)
        )

        log_ref  = self.init_dist.batch(yK)     # shape: (num_samples,)
        log_targ = self.target_dist.batch(yK)   # shape: (num_samples,)
        logZ     = -(rK + log_ref - log_targ)      # shape: (num_samples,)

        return logZ

    
class IDEMAlgorithm(BaseAlgorithm):
    """
    Implements the iDEM (Iterated Denoising Energy Matching).
    """

    @struct.dataclass
    class ReplayBuffer:
        data:     jnp.ndarray
        idx:      jnp.ndarray
        size:     jnp.ndarray
        max_size: int

        @classmethod
        def create(cls, max_size: int, data_dim: int):
            return cls(
                data=jnp.zeros((max_size, data_dim), dtype=jnp.float32),
                idx=jnp.array(0, dtype=jnp.int32),
                size=jnp.array(0, dtype=jnp.int32),
                max_size=max_size
            )

        def add(self, x: jnp.ndarray):
            batch = x.shape[0]
            indices = (self.idx + jnp.arange(batch)) % self.max_size
            new_data = self.data.at[indices].set(x)
            new_idx  = (indices[-1] + 1) % self.max_size
            new_size = jnp.minimum(self.size + batch, self.max_size)
            return type(self)(data=new_data,
                            idx=new_idx,
                            size=new_size,
                            max_size=self.max_size)


        @partial(jax.jit, static_argnames=('batch_size',))
        def sample(self, key: jr.PRNGKey, batch_size: int):
            """
            Pure functional sample: returns (samples, new_key).
            samples has shape (batch_size, data_dim).
            """
            key, sub = jr.split(key)
            # assume size > 0
            idxs = jr.randint(sub, (batch_size,), 0, self.size)
            return self.data[idxs], key

    def __init__(self, config):
        super().__init__(config) 

        # Override the initial distribution
        if config.add_drift:
            # always converge to N(0,I)
            self.init_dist = IsotropicGauss(
                mu=jnp.zeros(self.data_dim), log_var=0.0
            )
        else:
            self.init_dist = IsotropicGauss(
                mu=jnp.zeros(self.data_dim), log_var=2*jnp.log(config.sigma_max)
            )

        # Build the neural network (MLP or ResBlock)  
        if config.network_name == 'mlp':
            self.model = MLPModel(dim=self.data_dim, T=config.K)
        elif config.network_name == 'resblock':
            self.model = ResBlockModel(dim=self.data_dim, T=config.K)
        else:
            raise ValueError(f"Unknown network_name: {config.network_name}")

         
        # Initialize network parameters
        key = jr.PRNGKey(config.seed)
        key, sub = jr.split(key)
        dummy_x = jnp.zeros((config.batch_size, self.data_dim))
        dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.int32)
        initial_params = self.model.init(sub, dummy_x, dummy_t)

         
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=initial_params,
            tx=self.opt
        )

         
        # Build the score function (same conditioning logic as DDS)
        self.score_fn = self.make_score_fn()

         
        # Define linear σ(t) inside this class:
        #     σ(t) = σ_max*t + σ_min*(1-t)      for t in [0,1]
        sigma_min = config.sigma_min
        sigma_max = config.sigma_max

        def sigma_fn(t):
            return sigma_max * t + sigma_min * (1 - t)

        self.sigma_fn = sigma_fn

         
        # Create the replay buffer (size from config.buffer_size)
        self.buffer = IDEMAlgorithm.ReplayBuffer.create(max_size=config.buffer_size,
                                  data_dim=self.data_dim)

        ###################################
        # for debugging, fill the buffer with initial samples from target_dist
        if config.debug_fill_buffer:
            print("Debug: filling the buffer with initial samples from target_dist")
            key, sub = jr.split(key)
            init_samples = self.target_dist.sample(sub, config.buffer_size)
            self.buffer = self.buffer.add(init_samples)


         
        # Build the iDEM loss object
        self.loss_obj = self.make_loss()

    def make_loss(self):
        return IDEMLoss(num_samples=self.cfg.num_samples_for_sk, sigma_fn=self.sigma_fn,
                         target_dist=self.target_dist, score_fn=self.score_fn, total_step= self.cfg.K, 
                         add_drift=self.cfg.add_drift, sample_t_weight=self.cfg.sample_t_weight)

    # override train function
    @partial(jax.jit, static_argnums=(0,))
    def train(self, rng_key):
        inner_trainer = InnerTrainer(
        loss_obj=self.loss_obj,
        state=self.state,
        inner_iters=self.cfg.inner_iters,
        )


        def scan_body(carry, idx):
            key, state, buffer, logz_vals, logz_vars, all_losses, all_diff_true_ests, buffer_data, buffer_size = carry

            # sample & buffer update
            seq, _ = self.sample(state.params, key, self.cfg.num_samples_per_outer)
            new_x0s = seq[-1]
            # print("new_x0s shape: {}", new_x0s.shape)
            buffer = buffer.add(new_x0s)
            # store the buffer data
            buffer_data = buffer_data.at[idx].set(buffer.data)
            buffer_size = buffer_size.at[idx].set(buffer.size)

            # logZ
            def yes(c):
                key, lz, lv = c
                key, sub = jr.split(key)
                logz = self.estimate_logZ(state.params, sub, self.cfg.num_samples_per_outer)
                lz = lz.at[idx].set(jnp.mean(logz))
                lv = lv.at[idx].set(jnp.var(logz))
                return key, lz, lv
            key, logz_vals, logz_vars = jax.lax.cond(
                self.cfg.if_logZ, yes, lambda c: c, (key, logz_vals, logz_vars)
            )

            # inner training step
            state, key, losses, diff_true_ests = inner_trainer.run(key, buffer)
            # check the number of NaNs in losses
            num_nans = jnp.isnan(losses).sum()
            # jax.debug.print("Number of NaNs in losses at outer step {}: {}", idx, num_nans)
            all_losses = all_losses.at[idx].set(losses)
            all_diff_true_ests = all_diff_true_ests.at[idx].set(diff_true_ests)
            jax.debug.print("Outer step {}, loss = {}", idx, losses.mean())

            return (key, state, buffer, logz_vals, logz_vars, all_losses, all_diff_true_ests, buffer_data, buffer_size), None


        # re-initialize carry for main training loop
        init_carry = (
            rng_key,
            self.state,
            self.buffer,
            jnp.zeros((self.cfg.outer_iters,)), # logZ values
            jnp.zeros((self.cfg.outer_iters,)), # logZ variances
            jnp.zeros((self.cfg.outer_iters, self.cfg.inner_iters)), # all losses (outer_iters x inner_iters)
            jnp.zeros((self.cfg.outer_iters, self.cfg.inner_iters)), # all diff_true_ests (outer_iters x inner_iters)
            jnp.zeros((self.cfg.outer_iters, self.buffer.max_size, self.data_dim)), # buffer data (outer_iters x data_dim
            jnp.zeros((self.cfg.outer_iters,)) # buffer size
        )

        # run the scan over indices 0..outer_iters-1
        print("Starting main training loop for {} outer steps...".format(self.cfg.outer_iters))
        (key, state, buffer, logz_vals, logz_vars, all_losses, all_diff_true_ests, buffer_data, buffer_size), _ = \
                    jax.lax.scan(scan_body, init_carry, jnp.arange(self.cfg.outer_iters))

        # write back buffer and state
        self.buffer = buffer
        self.state = state

        flat_losses = all_losses.reshape(-1)
        flat_diff_true_ests = all_diff_true_ests.reshape(-1)
        return state, key, flat_losses, flat_diff_true_ests, logz_vals, logz_vars, buffer_data, buffer_size

    def sample(self, params, rng_key, num_samples: int):
        """
        Generate samples by running an annealed-Langevin reverse pass
        through the single-shot VE corruption x_t = x0 + sigma(t)*eps
        and 1-order Euler‐Maruyama discretization with K steps.
        """
        @partial(jax.jit, static_argnums=(2))
        def generate(params, key, num_samples):
            key, sub = jr.split(key)
            g0 = self.sigma_fn(0.0)
            g1 = self.sigma_fn(1.0)
            dt = 1.0 / self.cfg.K

            sigma_T = jnp.sqrt(g0**2 + g0*(g1-g0) + (g1-g0)**2/3.0)
            if self.cfg.add_drift:
                xT = jr.normal(sub, (num_samples, self.data_dim))
            
            else:
                xT = jr.normal(sub, (num_samples, self.data_dim)) * sigma_T

            #reverse step-indices
            Ks = jnp.arange(self.cfg.K, 0, -1)

            def body(carry, k):
                x_next, key = carry

                t_k   = k / self.cfg.K
                g_k   = self.sigma_fn(t_k)       

                # integrate g^2 over [t_{k-1}, t_k] by trapezoid:
                delta_sigma = jnp.sqrt(g_k**2 * dt)

                # score at (x_k, t_k)
                u = self.score_fn(params, k, x_next)   # shape (N, data_dim)

                # noise with sqrt(delta_sigma2)
                key, sub = jr.split(key)
                noise = jr.normal(sub, x_next.shape) * delta_sigma

                # reverse-time Euler step (note the minus sign)
                if self.cfg.add_drift:
                    x_prev = x_next + delta_sigma**2*(u + x_next/2) + noise
                else:
                    x_prev = x_next + delta_sigma**2 * u + noise

                return (x_prev, key), (x_prev, u)

            # run the reverse chain
            (x0, _), (seq, score_seq) = jax.lax.scan(
                body,
                (xT, key),
                Ks
            )
            # attach xT at the beginning of the sequence
            seq = jnp.concatenate([xT[None, ...], seq], axis=0)
            return seq, score_seq

        return generate(params, rng_key, num_samples)
    
    def mixture_score(self, y, t, mu, comp_sigmas, weights):
        """
        Exact score for 1d mixed-isotropic Gaussian mixture at time t.
            y:           array (batch,) or (batch,1)
            t:           integar time index in [0, K-1]
            mu:          array (n_comp, 1)
            comp_sigmas: array (n_comp,)    # sigma_i of each mixture component
            weights:     array (n_comp,)    # mixture weights w_i
            Returns:
            score:       array (batch,)    
        """

        t = t/ self.cfg.K  # normalize t to [0,1]
        sigma_max = self.sigma_fn(1)
        sigma_min = self.sigma_fn(0)
        int_sigma = jnp.sqrt(1/3*t**3*(sigma_max-sigma_min)**2 + sigma_min*(sigma_max-sigma_min)*t**2\
                             +sigma_min**2*t)
        

        #   v_i = σ_i^2 + σ_t^2
        v   = comp_sigmas**2 + int_sigma**2  
        diffs = mu[:, None, :] - y[None, :, :]   

        # component pdfs: N(y | m_i, v_i) 
        v_e  = v[:, None, None]                             
        norm = jnp.sqrt(2 * jnp.pi * v_e)                  
        exps = jnp.exp(-0.5 * (diffs**2) / v_e) / norm       
        pis  = weights[:, None, None] * exps                

        # weighted average of (m_i - y)/v_i
        numer = jnp.sum(pis * (diffs / v_e), axis=0)            # (batch,1)
        denom = jnp.sum(pis, axis=0)                           # (batch,1)

        return (numer / denom).reshape(-1)                     # (batch,)

    def sample_xT_from_mixture(self, key, num_samples, mu, comp_sigmas, weights, delta_T):
        '''
        Sample x_T from the true p1
        '''
        n_comp = mu.shape[0]
        key, sub1, sub2 = jr.split(key, 3)
        # choose components
        comp_ids = jr.choice(sub1, n_comp, shape=(num_samples,), p=weights)
        # component means and variances
        means = mu[comp_ids].squeeze()
        vars  = comp_sigmas[comp_ids]**2 + delta_T
        stds  = jnp.sqrt(vars)
        # sample
        eps = jr.normal(sub2, (num_samples,))
        xT  = means + stds * eps
        return xT.reshape(-1,1)


    def true_score(self, x_t, step, mu, comp_sigmas, weights):
        '''
        Exact score for 1d mixed-isotropic Gaussian mixture at time t.
        '''
        # current noise level
        t = step / self.cfg.K  # normalize t to [0,1]
        sigma_max = self.sigma_fn(1)
        sigma_min = self.sigma_fn(0)
        int_sigma = jnp.sqrt(1/3*t**3*(sigma_max-sigma_min)**2 + sigma_min*(sigma_max-sigma_min)*t**2\
                            +sigma_min**2*t) # constant * t

        #   v_i = σ_i^2 + σ_t^2
        v   = comp_sigmas**2 + int_sigma**2  
        diffs = mu.flatten() - x_t        # (n_comp,)
        norm  = jnp.sqrt(2 * jnp.pi * v)
        exps  = jnp.exp(-0.5 * (diffs**2) / v) / norm

        # unnormalized component responsibilities
        pis = weights * exps              # (n_comp,)

        # numerator and denominator for score
        numer = jnp.sum(pis * (diffs / v))
        denom = jnp.sum(pis)
        denom = jnp.clip(denom, 1e-10, jnp.inf)

        #################################
        # test which value is nan
        # jax.debug.print("pis: {}", pis)
        # jax.debug.print("diffs: {}", diffs)
        # jax.debug.print("exps: {}", exps)

        return numer / denom

    # run the backward diffusion with the true score
    def sample_backward_true(self, rng_key, num_samples):
        '''
        run the backward diffusion with the true score:
        x_{k-1} = x_k - delta_sigma^2 * true_score(x_k, k) + noise
        '''
        @partial(jax.jit, static_argnums=(1,))
        def generate(key, num_samples):
            key, sub = jr.split(key)
            g0 = self.sigma_fn(0.0)
            g1 = self.sigma_fn(1.0)
            dt = 1.0 / self.cfg.K

            sigma_T = jnp.sqrt(g0**2 + g0*(g1-g0) + (g1-g0)**2/3.0)
            xT = jr.normal(sub, (num_samples, self.data_dim)) * sigma_T
            # xT = self.sample_xT_from_mixture(sub, num_samples, 
            #     self.target_dist.mu, 
            #     jnp.sqrt(jnp.exp(self.target_dist.log_var)), 
            #     jnp.exp(self.target_dist.log_w), 
            #     sigma_T**2)  

            #reverse step-indices
            Ks = jnp.arange(self.cfg.K, 0, -1)

            def body(carry, k):
                x_next, key = carry
                t_k   = k / self.cfg.K
                g_k   = self.sigma_fn(t_k) # constant

                delta_sigma = jnp.sqrt(g_k**2 * dt)

                # score at (x_k, t_k)
                true_score = jax.vmap(self.true_score, in_axes=(0, None, None, None, None))
                u = true_score(x_next.squeeze(-1), k, self.target_dist.mu, self.target_dist.sigma, jnp.exp(self.target_dist.log_w))
                u = u.reshape(-1,1)

                key, sub = jr.split(key)
                noise = jr.normal(sub, x_next.shape) * delta_sigma

                # reverse-time Euler step
                x_prev = x_next + delta_sigma**2 * u + noise

                return (x_prev, key), (x_prev, u)

            # run the reverse chain
            (x0, _), (seq, score_seq) = jax.lax.scan(
                body,
                (xT, key),
                Ks
            )
            # attach xT at the beginning of the sequence
            seq = jnp.concatenate([xT[None, ...], seq], axis=0)
            return seq, score_seq

        return generate(rng_key, num_samples)
    
    def sample_forward(self, rng_key, num_samples):
        '''
        Run the forward OU process(dxt = \sigma * dW) from the mixed‐Gaussian (self.target_dist)
        toward the stationary Gaussian (self.init_dist).
        Returns:
          sample_seq: jnp.ndarray of shape (K, num_samples, data_dim)
        '''
        dt = self.cfg.T / self.cfg.K

        key, sub = jr.split(rng_key)
        x0 = self.target_dist.sample(sub, num_samples)
        seq = [x0]
        x_curr = x0

        for k in range(self.cfg.K):
            key, sub = jr.split(key)

            # diffuse x_curr by dxt = \sigma(t) * dW
            sigma_t = jnp.sqrt(self.sigma_fn(k * dt) * dt)
            x_next = x_curr + jr.normal(sub, x_curr.shape) * sigma_t
            seq.append(x_next)
            x_curr = x_next

        sample_seq = jnp.stack(seq[:-1], axis=0)  # shape (K, num_samples, data_dim)
        return sample_seq

    def visualize_forward(self, rng_key, num_samples):
        """
        propagate forward from mixed Gaussian -> approx standard Gaussian
        call visualize_samples to animate that path,then swap init/target
        """
        # get the forward trajectory
        sample_seq = self.sample_forward(rng_key, num_samples)

        # swap init_dist <-> target_dist
        # Here we take sigma as constant
        orig_init   = self.init_dist
        orig_target = self.target_dist
        self.init_dist  = orig_target
        self.target_dist = IsotropicGauss(mu = jnp.zeros(1), log_var=jnp.log(self.cfg.T)+2*jnp.log(self.cfg.sigma_max))

        # visualize
        self.visualize_samples(sample_seq,figname="forward")

        # restore
        self.init_dist   = orig_init
        self.target_dist = orig_target


class PISAlgorithm(BaseAlgorithm):
    def __init__(self, config):
        super().__init__(config)
        if config.network_name == 'mlp':
            self.model = MLPModel(dim=self.data_dim, T=config.K)
        elif config.network_name == 'resblock':
            self.model = ResBlockModel(dim=self.data_dim, T=config.K)
        else:
            raise ValueError(f"Unknown model_type: {config.network_name}")

        # Initialize params & optimizer state
        key = jr.PRNGKey(config.seed)
        key, sub = jr.split(key)
        dummy_x = jnp.zeros((config.batch_size, self.data_dim))
        dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.float32)
        self.params = self.model.init(sub, dummy_x, dummy_t)
        
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=self.params, tx=self.opt
        )

        # Control function and loss
        self.score_fn = self.make_score_fn()
        self.loss_obj    = self.make_loss()

    def make_loss(self):
        return PISLoss(add_score=self.cfg.add_score, loss_type=self.cfg.loss_type, sde_ctrl_noise=self.cfg.sde_ctrl_noise)
    
    
    @partial(jax.jit, static_argnums=(0, 3))
    def sample(self, params, rng_key, num_samples: int):
        """Forward SDE simulation with learned control"""
        T_total = self.cfg.T
        delta_t = T_total / self.cfg.K
        
        key, sub = jr.split(rng_key)
        x0 = self.init_dist.sample(sub, num_samples) 
        
        def body(carry, k):
            x_curr, key = carry
            key, sub = jr.split(key)
            
            # Get control at current state and time
            t_batch = jnp.full((num_samples,), k, dtype=jnp.float32)
            u = self.score_fn(params, t_batch, x_curr)  
            
            # Euler-Maruyama step
            noise = jr.normal(sub, shape=x_curr.shape) * jnp.sqrt(delta_t)
            x_next = x_curr + u * delta_t + noise
            
            return (x_next, key), (x_next,u)
        
        init_carry = (x0, key)
        (x_final, _), (seq,score_seq) = jax.lax.scan(
            body,
            init_carry,
            jnp.arange(self.cfg.K) 
        )
        
        # Include initial state in sequence
        full_seq = jnp.concatenate([x0[None, ...], seq], axis=0)
        return full_seq, score_seq

    @partial(jax.jit, static_argnums=(0, 3))
    def estimate_logZ(self, params, key, num_samples: int):
        """
        Estimate log normalizing constant using Girsanov's theorem
        for the controlled forward SDE.
        """
        T_total = self.cfg.T  # Total simulation time
        delta_t = T_total / self.cfg.K
        
        key, sub = jr.split(key)
        x0 = self.init_dist.sample(sub, num_samples) 
        
        # Initialize stochastic integral and running cost
        stochastic_integral = jnp.zeros(num_samples)
        running_cost = jnp.zeros(num_samples)
        
        def body(carry, k):
            x_curr, stoch_int, run_cost, key = carry
            key, sub = jr.split(key)
            
            # Get control at current state and time
            t_batch = jnp.full((num_samples,), k, dtype=jnp.float32)
            u = self.score_fn(params, t_batch, x_curr) 
            
            # Generate Brownian increment
            dW = jr.normal(sub, shape=x_curr.shape) * jnp.sqrt(delta_t)
            
            # Update state
            x_next = x_curr + u * delta_t + dW
            
            # Update stochastic integral
            stoch_int_update = jnp.sum(u * dW, axis=-1)
            new_stoch_int = stoch_int + stoch_int_update
            
            run_cost_update = 0.5 * jnp.sum(u**2, axis=-1) * delta_t
            new_run_cost = run_cost + run_cost_update
            
            return (x_next, new_stoch_int, new_run_cost, key), None
        
        init_carry = (x0, stochastic_integral, running_cost, key)
        (xT, stoch_int_final, run_cost_final, _), _ = jax.lax.scan(
            body,
            init_carry,
            jnp.arange(self.cfg.K)
        )
        
        # Compute terminal cost: log p_target(xT)
        log_p_target = self.target_dist.batch(xT)
        
        # Compute log reference density (initial distribution)
        log_ref = self.init_dist.batch(x0)
        logZ = -stoch_int_final - run_cost_final + log_p_target - log_ref
        
        return logZ

class ControlledMonteCarloDiffusion(BaseAlgorithm):
    """
    Implements both MCD (use_control_in_denominator=False) and
    CMCD (use_control_in_denominator=True) 
    """
    def __init__(self, config):
        super().__init__(config)
        if config.network_name == 'mlp':
            self.model = MLPModel(dim=self.data_dim, T=config.K)
        elif config.network_name == 'resblock':
            self.model = ResBlockModel(dim=self.data_dim, T=config.K)
        else:
            raise ValueError(f"Unknown model_type: {config.network_name}")

        key = jr.PRNGKey(config.seed)
        dummy_x = jnp.zeros((config.batch_size, self.data_dim))
        dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.float32)
        self.params = self.model.init(key, dummy_x, dummy_t)

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=self.opt
        )

        self.score_fn = self.make_score_fn()
        self.use_control_in_denominator = config.use_control_in_denominator # True for CMCD, False for MCD
        self.loss_obj = self.make_loss()


    def make_loss(self):
        return CMCDLoss(
            use_control_in_denominator = self.use_control_in_denominator,
            add_score=self.cfg.add_score,
            loss_type=self.cfg.loss_type,
            sde_ctrl_noise=self.cfg.sde_ctrl_noise
        )
    
    def sample(self, params, rng_key, num_samples):
        """
        Euler-Maruyama sampler returning the full trajectory for CMCD/MCD.
        Output shape: (K+1, num_samples, dim), including initial state.
        """
        @jax.jit
        def gen(key):
            key, sub = jr.split(key)
            x0 = self.init_dist.sample(sub, num_samples) 
            delta_t = 1.0 / self.cfg.K

            def body(carry, i):
                x, key = carry
                u = self.score_fn(params, i, x)
                alpha = i / self.cfg.K
                gradp = (1 - alpha) * self.init_dist.grad_batch(x) \
                    + alpha  * self.target_dist.grad_batch(x)
                drift = self.ou.sigma**2 * gradp + u
                key, sub = jr.split(key)
                noise = jr.normal(sub, x.shape) * jnp.sqrt(2 * self.ou.sigma**2 * delta_t)
                x_new = x + drift * delta_t + noise
                return (x_new, key), (x_new,u)

            steps = jnp.arange(self.ou.K)
            (final, _), (seq,score_seq) = jax.lax.scan(body, (x0, key), steps)
            full_seq = jnp.concatenate([x0[None, ...], seq], axis=0)
            return full_seq,score_seq

        return gen(rng_key)


    @partial(jax.jit, static_argnums=(0, 3))
    def estimate_logZ(self, params, key, num_samples: int):
        """
        Importance sampling estimator of logZ.
        logZ ≈ logmeanexp(log_ratio) with log_ratio defined as in eq. (24).
        """
        K = self.ou.K
        delta_t = 1.0 / K
        sigma2 = self.ou.sigma**2
        var = 2.0 * sigma2 * delta_t

        key, sub = jr.split(key)
        x0 = self.init_dist.sample(sub, num_samples)
        log_ratio = -self.init_dist.batch(x0)  # -log π0(x0)

        def body(carry, t):
            x, lr, key = carry
            key, sub = jr.split(key)

            # continuous/normalized time
            t_norm = t.astype(jnp.float32) / jnp.float32(K)

            # current control and score
            u = self.score_fn(params, t, x)
            gradp = (1.0 - t_norm) * self.init_dist.grad_batch(x) \
                    + t_norm * self.target_dist.grad_batch(x)

            # forward step
            mu_fwd = x + (sigma2 * gradp + u) * delta_t
            noise = jr.normal(sub, x.shape) * jnp.sqrt(var)
            x_next = mu_fwd + noise

            # next control/score for backward mean
            t_next = (t+1).astype(jnp.float32) / jnp.float32(K)
            u_next = self.score_fn(params, t+1, x_next)
            gradp_next = (1.0 - t_next) * self.init_dist.grad_batch(x_next) \
                        + t_next * self.target_dist.grad_batch(x_next)

            mu_bwd = x_next + (sigma2 * gradp_next - u_next) * delta_t

            # log forward/backward densities
            def log_gauss(x, mu):
                D = x.shape[-1]
                norm = -0.5 * D * jnp.log(2 * jnp.pi * var)
                quad = -0.5 * jnp.sum((x - mu)**2, axis=-1) / var
                return norm + quad

            log_pfwd = log_gauss(x_next, mu_fwd)
            log_pbwd = log_gauss(x, mu_bwd)

            lr = lr + (log_pbwd - log_pfwd)
            return (x_next, lr, key), None

        steps = jnp.arange(K)
        (xT, log_ratio, _), _ = jax.lax.scan(body, (x0, log_ratio, key), steps)

        # endpoint correction
        log_pT = self.target_dist.batch(xT)
        log_ratio = log_ratio + log_pT

        return log_ratio

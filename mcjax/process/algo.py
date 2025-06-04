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
from jax.tree_util import tree_map

from models import MLPModel, ResBlockModel
from ou import OU
from mcjax.proba.gaussian import IsotropicGauss, MixedIsotropicGauss, GMM40
from losses import DDSLoss, IDEMLoss
from trainer import Trainer, InnerTrainer

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
    @abstractmethod
    def __init__(self, config):
        """
        config: argparse.Namespace (or dict) containing all needed hyperparameters.
        Subclass __init__ must at least define:
          self.cfg, self.ou, self.init_dist, self.target_dist,
          self.score_fn, self.params, self.state, self.data_dim
        """
        pass

    @abstractmethod
    def make_score_fn(self):
        """
        Returns a Python function: score_fn(params, k, y) → shape (batch, data_dim).
        """
        pass

    @abstractmethod
    def make_loss(self):
        """
        Returns a BaseLoss instance (e.g. DDSLoss, IDEMLoss, etc.).
        """
        pass

    @abstractmethod
    def train(self, rng_key):
        """
        Runs the training loop, returns final params, train‐loss history, maybe logZ stats.
        """
        pass

    def sample(self, params, rng_key, num_samples: int):
        """
        Generate samples by running reverse OU chain.
        """
        @jax.jit
        def generate(params, key):
            key, sub = jr.split(key)
            yK = self.init_dist.sample(sub, num_samples)
            def body(carry, k):
                y_next, key_ = carry
                key_, yk = self.ou.reverse_step(key_, y_next, k, self.score_fn, params)
                return (yk, key_), yk
            (y0, _), seq = jax.lax.scan(
                body, (yK, key), jnp.arange(self.ou.K)
            )
            return seq  # shape (K, num_samples, dim)
        return generate(params, rng_key)   

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
        logZ     = rK + log_ref - log_targ      # shape: (num_samples,)

        return logZ

    def visualize_samples(self, sample_seq):
        """
        Generic 1D / 2D visualization of the reverse chain. 
        sample_seq is expected to have shape (K, num_samples, data_dim).

        Subclasses must set:
         - self.data_dim ∈ {1,2}
         - self.init_dist, self.target_dist to sample enough points for KDE/contour
         - self.cfg.K, self.cfg.results_dir, etc.
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
            fname = f'{self.cfg.results_dir}/density_evolution.mp4'
            ani.save(fname, writer=writer)
            plt.close()

        elif self.data_dim == 2:
            fig, ax = plt.subplots(figsize=(10, 10))
            x = jnp.linspace(-45, 45, 200)
            y = jnp.linspace(-45, 45, 200)
            X, Y = jnp.meshgrid(x, y)
            pts = jnp.stack([X.ravel(), Y.ravel()], axis=1)

            Ztarg = self.target_dist.batch(pts).reshape(X.shape)
            contour = ax.contourf(X, Y, jnp.exp(Ztarg), levels=10)
            fig.colorbar(contour, ax=ax)

            scatter = ax.scatter([], [], c='red', s=10, alpha=0.6, label='Samples')
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(-45, 45)
            ax.set_ylim(-45, 45)
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title('2D Sample Movement')

            def animate(frame):
                curr = sample_seq[frame]
                scatter.set_offsets(curr)
                time_text.set_text(f'Step: {frame}/{self.cfg.K}')
                return scatter, time_text

            ani = FuncAnimation(
                fig=fig,
                func=animate,
                frames=self.cfg.K,
                interval=20,
                blit=True
            )
            writer = FFMpegWriter(fps=30, metadata=dict(artist='BaseAlgorithm'), bitrate=1800)
            fname = f'{self.cfg.results_dir}/sample_movement.mp4'
            ani.save(fname, writer=writer)
            plt.close()

        else:
            raise ValueError(f"Unsupported data_dim: {self.data_dim}")

class DDSAlgorithm(BaseAlgorithm):
    """
    Implements the DDS sampler (Denoising Diffusion Sampler)
    """

    def __init__(self, config):
        # config is an argparse.Namespace in the main script
        self.cfg = config

        # build the target distribution
        if config.target_dist == 'gmm40':
            self.target_dist = GMM40()
            self.data_dim = 2
        elif config.target_dist == '1d':
            mu = jnp.array([[-2.],[0.],[2.]])
            dist_sigma = jnp.array([0.3, 0.3, 0.3])
            log_var = jnp.log(dist_sigma**2)
            weights = jnp.array([0.3, 0.4, 0.3])
            self.target_dist = MixedIsotropicGauss(mu=mu, log_var=log_var, weights=weights)
            self.data_dim = 1


        # build the reference init distribution
        self.init_dist   = IsotropicGauss(mu=jnp.zeros(self.data_dim), log_var=0.0)

        # create timesteps / beta / alpha schedule
        K = config.K
        ts = jnp.arange(K, dtype=jnp.float32)
        if config.variable_ts:
            beta_start, beta_end = 0.1, 20.0
            beta = beta_start + (beta_end - beta_start) * (ts / (K - 1))
        else:
            beta = jnp.ones(K) * 0.5
        alpha = 1.0 - jnp.exp(-2.0 * beta / K)

        # make the OU process
        self.ou = OU(alpha=alpha, sigma=config.sigma, init_dist=self.init_dist)

        # build the network
        #    choose MLP or ResBlock based on config.model_type
        if config.network_name == 'mlp':
            self.model = MLPModel(dim=self.data_dim, T=K)
        elif config.network_name == 'resblock':
            self.model = ResBlockModel(dim=self.data_dim, T=K)
        else:
            raise ValueError(f"Unknown model_type: {config.network_name}")

        # initialize network params
        key = jr.PRNGKey(config.seed)
        key, sub = jr.split(key)
        dummy_x = jnp.zeros((config.batch_size, self.data_dim))
        dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.int32)
        self.params = self.model.init(sub, dummy_x, dummy_t)

        # optimizer (Adam)
        self.opt = optax.adam(config.lr)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=self.params, tx=self.opt
        )

        # build score_fn
        self.score_fn = self.make_score_fn()

        # build loss object
        self.loss_obj = self.make_loss()

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
                logp = target.batch(y)  # shape (batch,)
                normed = logp / (jnp.std(logp, axis=0, keepdims=True) + 1e-5)
                return nn1 + nn2 * normed[:, None]  

            elif condition == 'grad_score':
                gradp = target.grad_batch(y)  # shape (batch, dim)
                normed = gradp / (jnp.std(gradp, axis=0, keepdims=True) + 1e-5)
                return nn1 + nn2 * normed

            else:
                raise ValueError(f"Unknown condition_term: {condition}")

        return jax.jit(score_fn)

    def make_loss(self):
        return DDSLoss(add_score=self.cfg.add_score)

    def train(self, rng_key):
        """
        Run the training loop (jax.lax.scan).
        Returns (final_params, losses, maybe logZs).
        """
        trainer = Trainer(
            algorithm=self,
            process=self.ou,
            init_dist=self.init_dist,
            target_dist=self.target_dist,
            score_fn=self.score_fn,
            loss_obj=self.loss_obj,
            state=self.state,
            batch_size=self.cfg.batch_size,
            num_steps=self.cfg.num_steps,
            if_logZ=self.cfg.if_logZ
        )
        return trainer.run(rng_key)

class IDEMAlgorithm(BaseAlgorithm):
    """
    Implements the iDEM sampler (Iterated Denoising Energy Matching).
    """

    class ReplayBuffer:
        def __init__(self, max_size: int, data_dim: int):
            self.max_size = max_size
            self.data = np.zeros((max_size, data_dim), dtype=np.float32)
            self.idx = 0
            self.size = 0

        def add(self, x: jnp.ndarray):
            x_np = np.asarray(x)  # to NumPy
            n = x_np.shape[0]
            for i in range(n):
                self.data[self.idx] = x_np[i]
                self.idx = (self.idx + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)

        def sample(self, key: jax.random.PRNGKey, batch_size: int):
            key, sub = jr.split(key)
            if self.size == 0:
                raise ValueError("ReplayBuffer is empty—cannot sample.")
            idxs = jr.randint(sub, (batch_size,), 0, self.size)  # uses self.size as a Python int
            buf_jax = jnp.asarray(self.data)
            selected = buf_jax[idxs]
            return selected, key

    def __init__(self, config):
        # ----------------------------------------------------------------------
        # 1) Save the config namespace/dict
        # ----------------------------------------------------------------------
        self.cfg = config

        # ----------------------------------------------------------------------
        # 2) Build the target distribution
        # ----------------------------------------------------------------------
        if config.target_dist == 'gmm40':
            self.target_dist = GMM40()
            self.data_dim = 2
        elif config.target_dist == '1d':
            mu = jnp.array([[-2.], [0.], [2.]])
            dist_sigma = jnp.array([0.3, 0.3, 0.3])
            log_var = jnp.log(dist_sigma**2)
            weights = jnp.array([0.3, 0.4, 0.3])
            self.target_dist = MixedIsotropicGauss(
                mu=mu, log_var=log_var, weights=weights
            )
            self.data_dim = 1
        else:
            raise ValueError(f"Unknown target_dist: {config.target_dist}")

        # ----------------------------------------------------------------------
        # 3) Build the reference initial distribution
        # ----------------------------------------------------------------------
        self.init_dist = IsotropicGauss(
            mu=jnp.zeros(self.data_dim), log_var=0.0
        )

        # ----------------------------------------------------------------------
        # 4) Create discrete timesteps / beta / alpha schedule for OU
        # ----------------------------------------------------------------------
        K = config.K
        ts = jnp.arange(K, dtype=jnp.float32)
        if config.variable_ts:
            beta_start, beta_end = 0.1, 20.0
            beta = beta_start + (beta_end - beta_start) * (ts / (K - 1))
        else:
            beta = jnp.ones(K, dtype=jnp.float32) * 0.5
        alpha = 1.0 - jnp.exp(-2.0 * beta / K)

        # ----------------------------------------------------------------------
        # 5) Instantiate the OU forward/reverse process
        # ----------------------------------------------------------------------
        self.ou = OU(alpha=alpha, sigma=config.sigma, init_dist=self.init_dist)

        # ----------------------------------------------------------------------
        # 6) Build the neural network (MLP or ResBlock)
        # ----------------------------------------------------------------------
        if config.network_name == 'mlp':
            self.model = MLPModel(dim=self.data_dim, T=K)
        elif config.network_name == 'resblock':
            self.model = ResBlockModel(dim=self.data_dim, T=K)
        else:
            raise ValueError(f"Unknown network_name: {config.network_name}")

        # ----------------------------------------------------------------------
        # 7) Initialize network parameters
        # ----------------------------------------------------------------------
        key = jr.PRNGKey(config.seed)
        key, sub = jr.split(key)
        dummy_x = jnp.zeros((config.batch_size, self.data_dim))
        dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.int32)
        initial_params = self.model.init(sub, dummy_x, dummy_t)

        # ----------------------------------------------------------------------
        # 8) Set up optimizer (Adam) and Flax train state
        # ----------------------------------------------------------------------
        self.opt = optax.adam(config.lr)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=initial_params,
            tx=self.opt
        )

        # ----------------------------------------------------------------------
        # 9) Build the score function (same conditioning logic as DDS)
        # ----------------------------------------------------------------------
        self.score_fn = self.make_score_fn()

        # ----------------------------------------------------------------------
        # 10) Define geometric σ(t) inside this class:
        #     σ(t) = σ_min * (σ_max/σ_min)^t      for t ∈ [0,1]
        # ----------------------------------------------------------------------
        sigma_min = 0.1
        sigma_max = 5.0

        def sigma_fn(t):
            # t can be a scalar or an array in [0,1]
            ratio = sigma_max / sigma_min
            return sigma_min * (ratio ** t)

        self.sigma_fn = sigma_fn

        # ----------------------------------------------------------------------
        # 11) Create the replay buffer (size from config.buffer_size)
        # ----------------------------------------------------------------------
        self.buffer = IDEMAlgorithm.ReplayBuffer(
            max_size=config.buffer_size,
            data_dim=self.data_dim
        )

        # ----------------------------------------------------------------------
        # 12) Build the iDEM loss object
        # ----------------------------------------------------------------------
        self.loss_obj = self.make_loss()

    def make_score_fn(self):
        """
        Returns: score_fn(params, k, y) -> shape (batch, data_dim)
        Implements the same conditioning options as in DDSAlgorithm:
          - 'none'
          - 'score'
          - 'grad_score'
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
                # use log p(y) as conditioning
                logp = target.batch(y)  # shape (batch,)
                normed = logp / (jnp.std(logp, axis=0, keepdims=True) + 1e-5)
                return nn1 + nn2 * normed[:, None]

            elif condition == 'grad_score':
                # use ∇ log p(y) as conditioning
                gradp = target.grad_batch(y)  # shape (batch, data_dim)
                normed = gradp / (jnp.std(gradp, axis=0, keepdims=True) + 1e-5)
                return nn1 + nn2 * normed

            else:
                raise ValueError(f"Unknown condition_term: {condition}")

        return jax.jit(score_fn)

    def make_loss(self):
        """
        Instantiate the IDEMLoss with the appropriate number of Monte Carlo samples
        and the noise schedule σ(t).
        """
        return IDEMLoss(K=1000, sigma_fn=self.sigma_fn,buffer=self.buffer,\
                         target_dist=self.target_dist, score_fn=self.score_fn)

    def train(self, rng_key):
        """
        The outer loop of iDEM:
          for each of outer_iters:
            1) sample new x₀ via reverse SDE (integrate_reverse)
            2) buffer.add(new_x₀)
            3) run inner_iters gradient steps (via InnerTrainer)

        Returns (final_params, final_key, all_loss_values).
        """
        key = rng_key
        all_losses = []
        logz_vals = jnp.zeros((self.cfg.outer_iters,))
        logz_vars = jnp.zeros_like(logz_vals)


        for outer in range(self.cfg.outer_iters):
            # ─── Outer: generate new x₀’s and add to buffer ─────────────────
            key, sub = jr.split(key)
            seq = self.sample(self.state.params, sub, self.cfg.num_samples_per_outer)
            new_x0s = seq[-1]  # take the last step of the reverse chain
            # new_x0s: shape (num_samples_per_outer, data_dim)
            self.buffer.add(new_x0s)

            # ─── (Optional) Estimate logZ here if you want, using self.estimate_logZ(...) ─
            if self.cfg.if_logZ:
                key, sub_z = jr.split(key)
                logz = self.estimate_logZ(self.state.params, sub_z, num_samples=self.cfg.num_samples_per_outer)
                # store logZ means/vars
                logz_vals = logz_vals.at[outer].set(jnp.mean(logz))
                logz_vars = logz_vars.at[outer].set(jnp.var(logz))

            # ─── Inner: run `inner_iters` gradient steps via InnerTrainer ─────────────────
            inner_trainer = InnerTrainer(
                loss_obj=self.loss_obj,
                state=self.state,
                batch_size=self.cfg.batch_size,
                inner_iters=self.cfg.inner_iters
            )
            self.state, key, losses = inner_trainer.run(key)
            all_losses.append(losses)

            # Optional print of the last inner loss for debugging
            last_loss = losses[-1]
            jax.debug.print(
                "[Outer {}/{}] last_inner_loss = {:.5f}",
                outer+1, self.cfg.outer_iters, last_loss
            )

        # Concatenate all inner losses across outer iterations
        all_losses = jnp.concatenate(all_losses, axis=0)  # shape: (outer_iters * inner_iters,)

        return self.state.params, key, all_losses, logz_vals, logz_vars
        
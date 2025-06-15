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

            ani = animation.FuncAnimation(
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
        # self.opt = optax.adam(config.lr)
        # add gradient clipping
        optax.chain(
            optax.clip(20.0),
            optax.adamw(config.lr)
        )
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
        self.cfg = config

        # Build the target distribution
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

        # Build the reference initial distribution
        self.init_dist = IsotropicGauss(
            mu=jnp.zeros(self.data_dim), log_var=0.0
        )


        # Create discrete timesteps / beta / alpha schedule for OU
        K = config.K
        ts = jnp.arange(K, dtype=jnp.float32)
        if config.variable_ts:
            beta_start, beta_end = 0.1, 20.0
            beta = beta_start + (beta_end - beta_start) * (ts / (K - 1))
        else:
            beta = jnp.ones(K, dtype=jnp.float32) * 0.5
        alpha = 1.0 - jnp.exp(-2.0 * beta / K)

        # Instantiate the OU forward/reverse process
        self.ou = OU(alpha=alpha, sigma=config.sigma, init_dist=self.init_dist)

         
        # Build the neural network (MLP or ResBlock)  
        if config.network_name == 'mlp':
            self.model = MLPModel(dim=self.data_dim, T=K)
        elif config.network_name == 'resblock':
            self.model = ResBlockModel(dim=self.data_dim, T=K)
        else:
            raise ValueError(f"Unknown network_name: {config.network_name}")

         
        # Initialize network parameters
        key = jr.PRNGKey(config.seed)
        key, sub = jr.split(key)
        dummy_x = jnp.zeros((config.batch_size, self.data_dim))
        dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.int32)
        initial_params = self.model.init(sub, dummy_x, dummy_t)

         
        # Set up optimizer (Adam) and Flax train state
        self.opt = optax.adam(config.lr)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=initial_params,
            tx=self.opt
        )

         
        # Build the score function (same conditioning logic as DDS)
        self.score_fn = self.make_score_fn()

         
        # Define geometric σ(t) inside this class:
        #     σ(t) = σ_min * (σ_max/σ_min)^t      for t ∈ [0,1]
        sigma_min = 1e-2
        sigma_max = 3.0

        def sigma_fn(t):
            ratio = sigma_max / sigma_min
            return sigma_min * (ratio ** t)

        self.sigma_fn = sigma_fn

         
        # Create the replay buffer (size from config.buffer_size)
        self.buffer = IDEMAlgorithm.ReplayBuffer.create(max_size=config.buffer_size,
                                  data_dim=self.data_dim)


         
        # Build the iDEM loss object
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
            t_continuous = k * (self.ou.K - 1)  # Scale to [0, K-1]
            batch_t = jnp.full((y.shape[0],), t_continuous, dtype=jnp.float32)
            nn1, nn2 = self.model.apply(params, y, batch_t)

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
        return IDEMLoss(K=200, sigma_fn=self.sigma_fn,buffer=self.buffer,\
                         target_dist=self.target_dist, score_fn=self.score_fn)

    @partial(jax.jit, static_argnums=(0))
    def train(self, rng_key):
        """
        JIT-compatible version using jax.lax.scan for outer loop.
        Returns (final_params, key, all_losses, logz_vals, logz_vars)
        """
        # Initialize carry for scan: 
        # (rng_key, train_state, buffer, logz_vals, logz_vars)
        # init_carry = (
        #     rng_key,
        #     self.state,
        #     self.buffer,
        #     jnp.zeros((self.cfg.outer_iters,)),
        #     jnp.zeros((self.cfg.outer_iters,))
        # )
        
        # def outer_step(carry, outer_idx):
        #     jax.debug.print("Starting outer loop {} ", outer_idx)
        #     key, state, buffer, logz_vals, logz_vars = carry
            
        #     # Sample new x₀'s and update buffer
        #     key, subkey = jax.random.split(key)
        #     seq = self.sample(state.params, subkey, self.cfg.num_samples_per_outer)
        #     new_x0s = seq[-1]
        #     # test: print new_x0s
        #     # jax.debug.print("New x0s: {}", new_x0s)
        #     buffer = buffer.add(new_x0s)  
            
        #     def yes_branch(inputs):
        #         key, logz_vals, logz_vars = inputs
        #         key, sub = jr.split(key)
        #         logz = self.estimate_logZ(state.params, sub, self.cfg.num_samples_per_outer)
        #         logz_vals = logz_vals.at[outer_idx].set(jnp.mean(logz))
        #         logz_vars = logz_vars.at[outer_idx].set(jnp.var(logz))
        #         return key, logz_vals, logz_vars

        #     def no_branch(inputs):
        #         return inputs

        #     key, logz_vals, logz_vars = jax.lax.cond(
        #         self.cfg.if_logZ,
        #         yes_branch,
        #         no_branch,
        #         operand=(key, logz_vals, logz_vars)
        #     )

            
        #     # Run inner training
        #     inner_trainer = InnerTrainer(
        #         loss_obj=self.loss_obj,
        #         state=state,
        #         batch_size=self.cfg.batch_size,
        #         inner_iters=self.cfg.inner_iters,
        #     )
        #     state, key, losses = inner_trainer.run(key)
            
        #     new_carry = (key, state, buffer, logz_vals, logz_vars)
        #     output = losses  # Shape: (inner_iters,)
        #     return new_carry, output

        # # Execute scan over outer iterations
        # final_carry, all_losses = jax.lax.scan(
        #     outer_step,
        #     init_carry,
        #     jnp.arange(self.cfg.outer_iters)
        # )

    #################################
    # Rewrite with for loop

        init_carry = (
            rng_key,
            self.state,
            self.buffer,
            jnp.zeros((self.cfg.outer_iters,)),
            jnp.zeros((self.cfg.outer_iters,))
        )
        for outer_idx in range(self.cfg.outer_iters):
            jax.debug.print("Starting outer loop {}", outer_idx)
            key, state, buffer, logz_vals, logz_vars = init_carry
            
            # Sample new x₀'s and update buffer
            seq = self.sample(self.state.params, key, self.cfg.num_samples_per_outer)
            new_x0s = seq[-1]
            self.buffer = self.buffer.add(new_x0s)
            print(f"Buffer size after adding new samples: {self.buffer.size}")
            def yes_branch(inputs):
                key, logz_vals, logz_vars = inputs
                key, sub = jr.split(key)
                logz = self.estimate_logZ(state.params, sub, self.cfg.num_samples_per_outer)
                logz_vals = logz_vals.at[outer_idx].set(jnp.mean(logz))
                logz_vars = logz_vars.at[outer_idx].set(jnp.var(logz))
                return key, logz_vals, logz_vars

            def no_branch(inputs):
                return inputs

            key, logz_vals, logz_vars = jax.lax.cond(
                self.cfg.if_logZ,
                yes_branch,
                no_branch,
                operand=(key, logz_vals, logz_vars)
            )

            
            # Run inner training
            inner_trainer = InnerTrainer(
                loss_obj=self.loss_obj,
                state=state,
                batch_size=self.cfg.batch_size,
                inner_iters=self.cfg.inner_iters,
            )
            state, key, losses = inner_trainer.run(key)

            # print data in buffer
            if outer_idx % 10 == 0:
                data = jax.device_get(self.buffer.data[: int(self.buffer.size) ])
                size = jax.device_get(self.buffer.size).astype(int)
                print(size)
                data = data[:size]  # Only take the filled part of the buffer
                plt.figure(figsize=(10, 6))
                plt.hist(data, bins='auto', density=True, alpha=0.5, label='Buffer Samples')
                # Plot target distribution
                # x = jnp.linspace(-7, 7, 1000)
                # target_samples = self.target_dist.sample(jr.PRNGKey(1), 100000).flatten()
                # target_kde = gaussian_kde(target_samples)
                # plt.plot(x, target_kde(x), 'g--', lw=2, label='Target Dist')
                plt.title('Replay Buffer Samples')
                plt.xlabel('x')
                plt.ylabel('Density')
                plt.savefig(f"{self.cfg.results_dir}/buffer_samples_{outer_idx}.png")
                plt.close()
        
        final_carry = (key, state, self.buffer, logz_vals, logz_vars)
        all_losses = losses
        # Unpack results
        key, state, buffer, logz_vals, logz_vars = final_carry
        self.state = state
        self.buffer = buffer
     
        # Flatten losses: (outer_iters, inner_iters) -> (outer_iters*inner_iters,)
        all_losses_flat = all_losses.reshape(-1)
        
        return state, key, all_losses_flat, logz_vals, logz_vars
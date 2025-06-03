from abc import ABC, abstractmethod
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

from models import MLPModel, ResBlockModel
from ou import OU
from mcjax.proba.gaussian import IsotropicGauss, MixedIsotropicGauss, GMM40
from losses import DDSLoss
from trainer import Trainer

class BaseAlgorithm(ABC):
    """
    Abstract template for any sampler (PIS, DDS, iDEM, MCD, CMCD, etc.).
    A concrete subclass must implement:
      - make_score_fn(): returns a callable(params, k, y) → score vector
      - make_loss(): returns a BaseLoss instance
      - sample(): after training, generate samples from p_target
      - maybe some additional methods for logging / metrics
    """

    @abstractmethod
    def __init__(self, config):
        """
        config: argparse.Namespace or simple dict with all hyperparameters
        """
        pass

    @abstractmethod
    def make_score_fn(self):
        """
        Returns a Python function: score_fn(params, k, y) → shape (batch, dim).
        """
        pass

    @abstractmethod
    def make_loss(self):
        """
        Returns a `BaseLoss` instance (e.g. DDSLoss(add_score=...)).
        """
        pass

    @abstractmethod
    def train(self, rng_key):
        """
        Runs the training loop, returns final params, train‐loss history, maybe logZ stats.
        """
        pass

    @abstractmethod
    def sample(self, params, rng_key, num_samples):
        """
        After training, generate `num_samples` samples from learned sampler.
        """
        pass

class DDSAlgorithm(BaseAlgorithm):
    """
    Implements the DDS sampler (Denoising Diffusion Sampler) as in your dds.py example.
    """

    def __init__(self, config):
        # config is an argparse.Namespace or dict containing:
        #   - config.target_name   ('gmm40', 'mixed1d', etc.)
        #   - config.data_dim      (int)
        #   - config.K             (number of diffusion steps)
        #   - config.sigma         (OU noise parameter)
        #   - config.lr            (learning rate)
        #   - config.batch_size
        #   - config.num_steps     (# training steps)
        #   - config.condition_term: 'none' / 'score' / 'grad_score'
        #   - config.add_score     (bool)
        #   - config.variable_ts   (bool)
        #   - config.seed
        #   - plus any others you need
        self.cfg = config

        # 1) build the target distribution
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


        # 2) build the reference init distribution
        self.init_dist   = IsotropicGauss(mu=jnp.zeros(self.data_dim), log_var=0.0)

        # 3) create timesteps / beta / alpha schedule
        K = config.K
        ts = jnp.arange(K, dtype=jnp.float32)
        if config.variable_ts:
            beta_start, beta_end = 0.1, 20.0
            beta = beta_start + (beta_end - beta_start) * (ts / (K - 1))
        else:
            beta = jnp.ones(K) * 0.5
        alpha = 1.0 - jnp.exp(-2.0 * beta / K)

        # 4) make the OU process
        self.ou = OU(alpha=alpha, sigma=config.sigma, init_dist=self.init_dist)

        # 5) build the network(s)
        #    choose MLP or ResBlock based on config.model_type
        if config.network_name == 'mlp':
            self.model = MLPModel(dim=self.data_dim, T=K)
        elif config.network_name == 'resblock':
            self.model = ResBlockModel(dim=self.data_dim, T=K)
        else:
            raise ValueError(f"Unknown model_type: {config.network_name}")

        # 6) initialize network params
        key = jr.PRNGKey(config.seed)
        key, sub = jr.split(key)
        dummy_x = jnp.zeros((config.batch_size, self.data_dim))
        dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.int32)
        self.params = self.model.init(sub, dummy_x, dummy_t)

        # 7) optimizer (Adam)
        self.opt = optax.adam(config.lr)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=self.params, tx=self.opt
        )

        # 8) build score_fn
        self.score_fn = self.make_score_fn()

        # 9) build loss object
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
    
    @partial(jax.jit, static_argnums=(0,3))
    def estimate_logZ(self, params, key, num_samples: int):
        """
        Runs one Monte Carlo estimate of log Z per sample.
        
        Returns:
        logZ: [num_samples] array of log-estimates of Z.
        """
        key, key_ = jr.split(key)
        y_0 = self.init_dist.sample(key_, num_samples)

        # Reverse chain + accumulate r_k
        def scan_step(carry, k):
            y_k, r_k, key = carry
            key, key_ = jr.split(key)
            eps = jr.normal(key_, y_k.shape)

            alpha_Kmk = self.ou.alpha[self.ou.K - 1 - k]
            sqrt1m    = self.ou.sqrt_1m_alpha[self.ou.K - 1 - k]
            lambda_Kmk = 1.0 - sqrt1m

            # score network
            s = self.score_fn(params, self.ou.K - 1 - k, y_k)         

            # reverse OU step
            y_next = sqrt1m * y_k \
                + 2.0 * (self.ou.sigma**2) * lambda_Kmk * s \
                + self.ou.sigma * jnp.sqrt(alpha_Kmk) * eps

            # accumulate the quadratic term
            r_next = r_k + (2.0 * self.ou.sigma**2) * (lambda_Kmk**2 / alpha_Kmk) * jnp.sum(s**2, axis=-1)

            return (y_next, r_next, key), None

        # initialize r_0 = 0
        init_carry = (y_0, jnp.zeros(num_samples), key)
        (y_K, r_K, _), _ = jax.lax.scan(
            scan_step,
            init_carry,
            jnp.arange(self.ou.K)
        )

        # compute log Z estimates
        log_ref  = self.init_dist.batch(y_K)
        log_targ = self.target_dist.batch(y_K)
        logZ     = r_K + log_ref - log_targ

        return logZ

    def visualize_samples(self, sample_seq):
        """
        Generate samples and visualize the results.
        """
        if self.data_dim == 1:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Generate reference distributions and KDEs
            x = jnp.linspace(-7, 10, 1000)
            initial_kde = gaussian_kde(self.init_dist.sample(jr.PRNGKey(0), 100000).flatten())
            target_kde = gaussian_kde(self.target_dist.sample(jr.PRNGKey(0), 100000).flatten())

            # Precompute sample KDEs for all frames
            kde_x = jnp.linspace(-7, 10, 500)
            frame_densities = []
            for frame in range(self.cfg.K):
                current_samples = sample_seq[frame].flatten()
                kde = gaussian_kde(current_samples)
                frame_densities.append(kde(kde_x))

            # Create static reference lines
            ax.plot(kde_x, initial_kde(kde_x), 'b--', linewidth=2, label='Initial Distribution')
            ax.plot(kde_x, target_kde(kde_x), 'g--', linewidth=2, label='Target Distribution')

            # Initialize animated elements
            line, = ax.plot([], [], 'r-', linewidth=2, label='Current Samples')
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(-7, 10)
            ax.set_ylim(0, 0.5)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title('Density Evolution During Reverse Process')
            ax.legend(loc='upper right')

            def animate(frame):
                line.set_data(kde_x, frame_densities[frame])
                time_text.set_text(f'Step: {frame}/{self.cfg.K} (Time: {self.cfg.K-frame}/{self.cfg.K})')
                
                return line, time_text

            # Create animation
            ani = animation.FuncAnimation(
                fig=fig,
                func=animate,
                frames=self.cfg.K,
                interval=20,
                blit=True
            )

            # Save animation
            writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
            ani_name = 'density_evolution.mp4' 
            ani.save(self.cfg.results_dir+'/'+ani_name, writer=writer)

            plt.close()
        elif self.data_dim == 2:
            fig, ax = plt.subplots(figsize=(10, 10))
    
            # Generate grid for target contour
            x = jnp.linspace(-45, 45, 200)
            y = jnp.linspace(-45, 45, 200)
            X, Y = jnp.meshgrid(x, y)
            pts = jnp.stack([X.ravel(), Y.ravel()], axis=1)
            Z_target = self.target_dist.batch(pts).reshape(X.shape)

            # Plot static target contour
            contour = ax.contourf(X, Y, jnp.exp(Z_target),levels = 10)
            fig.colorbar(contour, ax=ax)


            # Initialize animated elements
            scatter = ax.scatter([], [], c='red', s=10, alpha=0.6, label='Samples')
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(-45, 45)
            ax.set_ylim(-45, 45)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Sample Movement During Reverse Process')
            

            def animate(frame):
                # Update sample positions (convert JAX array to NumPy for matplotlib)
                current_samples = jnp.array(sample_seq[frame])
                scatter.set_offsets(current_samples)
                time_text.set_text(f'Step: {frame}/{self.cfg.K} (Time: {self.cfg.K-frame}/{self.cfg.K})')
                return scatter, time_text

            # Create animation
            ani = animation.FuncAnimation(
                fig=fig,
                func=animate,
                frames=self.cfg.K,
                interval=20,
                blit=True
            )

            # Save animation
            writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
            ani_name = 'sample_movement_2d.mp4'
            ani.save(self.cfg.results_dir+'/'+ani_name, writer=writer)

            plt.close()
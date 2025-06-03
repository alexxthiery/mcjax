import jax
import jax.numpy as jnp
import jax.random as jr
from flax.training import train_state
import optax
import flax.linen as nn
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import gaussian_kde
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation
import pickle
import os
import argparse

import sys
sys.path.append('../../')


from mcjax.proba.density import LogDensity
from mcjax.process.ou import OU
from mcjax.proba.gaussian import IsotropicGauss, MixedIsotropicGauss, GMM40
from mcjax.process.models import MLPModel

print(f"Available devices: {jax.devices()}")
jax.config.update("jax_platform_name", "gpu")


def dds_loss(params, key, ou: OU, init_dist: LogDensity,
             target_dist: LogDensity, score_fn, batch_size: int, add_score: bool):
    # sample y_0
    key, key_ = jr.split(key)
    y_0 = init_dist.sample(key_, batch_size)

    # reverse‐chain + accumulate r
    def scan_step(carry, k):
        y_k, r_k, key = carry
        key, sub = jr.split(key)
        eps = jr.normal(sub, y_k.shape)

        alpha_Kmk = ou.alpha[ou.K - 1 - k]
        sqrt1m    = ou.sqrt_1m_alpha[ou.K - 1 - k]
        lambda_Kmk = 1.0 - sqrt1m

        s = score_fn(params, ou.K - 1 - k, y_k)
        y_next = sqrt1m * y_k \
               + 2.0 * (ou.sigma**2) * lambda_Kmk * s \
               + ou.sigma * jnp.sqrt(alpha_Kmk) * eps

        
        main_term = (2.0 * ou.sigma**2) * (lambda_Kmk**2 / alpha_Kmk) * jnp.sum(s**2, axis=-1)
        zero_exp_term = 2.0 * ou.sigma * jnp.sqrt(lambda_Kmk**2 / alpha_Kmk) * jnp.sum(s * eps, axis=-1)
        r_next = r_k + jax.lax.cond(
            add_score,
            lambda: main_term + zero_exp_term,
            lambda: main_term,
        )
        return (y_next, r_next, key), None

    r_0 = jnp.zeros(batch_size)
    (y_K, r_K, _), _ = jax.lax.scan(
        scan_step,
        (y_0, r_0, key),
        jnp.arange(ou.K)
    )

    log_ratio = init_dist.batch(y_K) - target_dist.batch(y_K)
    loss = jnp.mean(r_K + log_ratio)
    return loss

# compute (loss, grads)
loss_and_grad = jax.jit(
    jax.value_and_grad(dds_loss, argnums=0),
    static_argnums=(2, 3, 4, 5, 6, 7))

@partial(jax.jit, static_argnums=(2,3,4,5,6,7))
def train_step(state, key,
               ou, init_dist, target_dist,
               score_fn, batch_size, add_score):
    
    loss, grads = loss_and_grad(
        state.params, key,
        ou, init_dist, target_dist,
        score_fn, batch_size, add_score
    )

    # apply gradients & increment step
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@partial(jax.jit, static_argnums=(1,2,3))
def generate_samples(params, score_fn, ou, num_samples, key):
    """
    Generate `num_samples` draws from the learned distribution by
    running the reverse OU chain.
    """
    key, key_ = jr.split(key)
    y_K = ou.init_dist.sample(key_, num_samples)

    def body(carry, k):
        y_next, key = carry
        key, y_k = ou.reverse_step(key, y_next, k, score_fn, params)
        return (y_k, key), y_k

    # scan backwards
    (y_k, key), y_sequence = jax.lax.scan(
        body,
        (y_K, key),
        jnp.arange(ou.K)
    )

    # y_sequence[0] = y_{K-1}, …, y_sequence[K-1] = y_0
    return y_sequence #[K, num_samples, data_dim] 


@partial(jax.jit, static_argnums=(2,3,4,5,6))
def estimate_logZ(params, key,
                  ou,          
                  init_dist,    
                  target_dist,  
                  score_fn,  
                  batch_size):
    """
    Runs one Monte Carlo estimate of log Z per sample.
    
    Returns:
      logZ: [batch_size] array of log-estimates of Z.
    """
    key, key_ = jr.split(key)
    y_0 = init_dist.sample(key_, batch_size)

    # Reverse chain + accumulate r_k
    def scan_step(carry, k):
        y_k, r_k, key = carry
        key, key_ = jr.split(key)
        eps = jr.normal(key_, y_k.shape)

        alpha_Kmk = ou.alpha[ou.K - 1 - k]
        sqrt1m    = ou.sqrt_1m_alpha[ou.K - 1 - k]
        lambda_Kmk = 1.0 - sqrt1m

        # score network
        s = score_fn(params, ou.K - 1 - k, y_k)         

        # reverse OU step
        y_next = sqrt1m * y_k \
               + 2.0 * (ou.sigma**2) * lambda_Kmk * s \
               + ou.sigma * jnp.sqrt(alpha_Kmk) * eps

        # accumulate the quadratic term
        r_next = r_k + (2.0 * ou.sigma**2) * (lambda_Kmk**2 / alpha_Kmk) * jnp.sum(s**2, axis=-1)

        return (y_next, r_next, key), None

    # initialize r_0 = 0
    init_carry = (y_0, jnp.zeros(batch_size), key)
    (y_K, r_K, _), _ = jax.lax.scan(
        scan_step,
        init_carry,
        jnp.arange(ou.K)
    )

    # compute log Z estimates
    log_ref  = init_dist.batch(y_K)
    log_targ = target_dist.batch(y_K)
    logZ     = r_K + log_ref - log_targ

    return logZ


if __name__ == "__main__":
    
    # -------------- Unpack arguments --------------
    def str2bool(v):
        return v.lower() in ('true', '1', 'yes')
    parser = argparse.ArgumentParser()
    parser.add_argument('--if_train', type=str2bool, default=False)
    parser.add_argument('--model_path', type=str, default='model_params.pkl')
    parser.add_argument('--condition_term', type=str, default='grad') # 'grad_score', 'score' or 'none'
    parser.add_argument('--target_dist', type=str, default='gmm40') # 'gmm40' or '1d'
    parser.add_argument('--if_animation', type=str2bool, default=False)
    parser.add_argument('--add_score', type=str2bool, default=False) # True if adding the score term(with zero mean) in loss function
    parser.add_argument('--variable_ts', type=str2bool, default=False) # True if using variable (i.e. non-equidistant) timesteps for the diffusion process  
    parser.add_argument('--K', type=int, default=2000) # number of diffusion steps
    parser.add_argument('--sigma', type=float, default=1.0) # sigma of the OU process
    parser.add_argument('--lr', type=float, default=1e-3) # learning rate
    parser.add_argument('--batch_size', type=int, default=128) # batch size
    parser.add_argument('--num_steps', type=int, default=4000) # number of training steps
    parser.add_argument('--if_logZ', type=str2bool, default=False) # whether to estimate logZ during training


    args = parser.parse_args()
    
    # Unpack arguments
    if_train = args.if_train
    model_path = args.model_path
    condition_term = args.condition_term
    target = args.target_dist
    if_animation = args.if_animation
    add_score = args.add_score
    variable_ts = args.variable_ts
    K = args.K
    ou_sigma = args.sigma
    learning_rate = args.lr
    batch_size = args.batch_size
    num_steps = args.num_steps
    if_logZ = args.if_logZ

    # -------------- Set up the OU process and target distribution --------------
    if target == 'gmm40':
        target_dist = GMM40()
        data_dim = 2
    elif target == '1d':
        mu = jnp.array([[-2.],[0.],[2.]])
        dist_sigma = jnp.array([0.3, 0.3, 0.3])
        log_var = jnp.log(dist_sigma**2)
        weights = jnp.array([0.3, 0.4, 0.3])
        target_dist = MixedIsotropicGauss(mu=mu, log_var=log_var, weights=weights)
        data_dim = 1
    else:
        raise ValueError(f"Unknown target distribution: {target}")
    
    timesteps = jnp.arange(K, dtype=jnp.float32)
    if variable_ts:
        beta_start, beta_end = 0.1, 20.0
        beta = beta_start + (beta_end - beta_start) * (timesteps / (K - 1))
    else:
        # beta all set to 1/2
        beta = jnp.ones(K) * 0.5
    
    alpha = 1.0 - jnp.exp(-2.0 * beta / K)

    # Define the initial distribution of reference process
    init_dist = IsotropicGauss(mu=jnp.zeros(data_dim), log_var=0.0)

    # Define the dynamic of the process
    ou = OU(alpha=alpha, sigma=ou_sigma, init_dist=init_dist)
    # Define the network
    model = MLPModel(dim=data_dim, T=K)

    # network initialization
    key = jr.PRNGKey(0)

    key, key_ = jr.split(key)
    dummy_x = jnp.zeros((batch_size, data_dim))
    dummy_t = jnp.zeros((batch_size,), dtype=jnp.int32)
    params = model.init(key_, dummy_x, dummy_t)

    # optimizer initialization  
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply_fn,
        params=params,
        tx=optimizer
    )
    
    # Define score function 
    def score_fn(params, k, y):
        batch_t = jnp.full((y.shape[0],), k, dtype=jnp.int32)
        nn1, nn2 = model.apply_fn(params, y, batch_t)
        log_mu = target_dist.batch(y)
        grad_log_mu = target_dist.grad_batch(y)  
        if condition_term == 'grad_score':
            # Normalize the feature
            g = grad_log_mu / (jnp.std(grad_log_mu, axis=0, keepdims=True) + 1e-5) 
            result = nn1 + nn2 * g  
        elif condition_term == 'score':
            # Normalize the feature
            g = log_mu / (jnp.std(log_mu, axis=0, keepdims=True) + 1e-5)
            result = nn1 + nn2 * g
        elif condition_term == 'none':
            result = nn1 
        else:
            raise ValueError(f"Unknown condition term: {condition_term}")
        result=nn1
        return result

    def scan_step(carry, step):
        state, key, logz_values, logz_vars = carry
        key, key_ = jr.split(key)
        state, loss = train_step(state, key_, ou, init_dist, target_dist, score_fn, batch_size, add_score)

        def estimate_and_store(_):
            key_logz, _ = jr.split(key)
            logz = estimate_logZ(state.params, key_logz, ou, init_dist, target_dist, score_fn, 1000)
            return (logz_values.at[step//10].set(jnp.mean(logz)), logz_vars.at[step//10].set(jnp.var(logz)))
        
        # estimate logZ every 10 steps
        logz_values, logz_vars = jax.lax.cond(
            (step % 10 == 9) & (step < 5000*10) & if_logZ,
            estimate_and_store,
            lambda _: (logz_values,logz_vars),
            operand=None
        )

        # every 100 steps, print step and current loss
        def do_print(_):
            jax.debug.print("At step {}, loss = {}", step, loss)
            return None

        # branch on (step % 100 == 0)
        _ = jax.lax.cond((step % 100) == 0, do_print, lambda _: None, operand=None)
        return (state, key, logz_values, logz_vars), loss

    def run_training(state, key):
        logz_values = jnp.zeros(5000) # maximum step: 5000*10
        logz_vars = jnp.zeros(5000) # maximum step: 5000*10
        (final_state, final_key, logz_values,logz_vars), losses = jax.lax.scan(
            scan_step,
            (state, key, logz_values,logz_vars),
            jnp.arange(num_steps)
        )
        return final_state, final_key, losses, logz_values,logz_vars


    # Training loop
    if if_train:
        key, key_ = jr.split(key)
        state, key, losses,logz_values, logz_variances = run_training(state, key_)

        # Plot the loss curve
        plt.plot(losses, label='Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        fig_name = 'loss_curve_with_score.png' if add_score else 'loss_curve_without_score.png'
        plt.savefig(fig_name)
        plt.close()

        # plot the logZ variance at each 10 steps
        if if_logZ:
            fig, ax1 = plt.subplots()
            x = 10 + jnp.arange(num_steps//10)*10
            ax1.plot(x, logz_variances[:num_steps//10], label='logZ Variance', color='blue')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('logZ Variance', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax2 = ax1.twinx()
            ax2.plot(x, logz_values[:num_steps//10], label='logZ Value', color='orange')
            ax2.set_ylabel('logZ Value', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper right')

            plt.title('Variance of logZ Estimates During Training')
            fig_name = 'logz_with_score.png' if add_score else 'logz_without_score.png'
            plt.tight_layout()
            plt.savefig(fig_name) 
            plt.close()

        # Save the model parameters
        params = state.params
        with open(model_path, 'wb') as f:
            pickle.dump(params, f)

    else:
        # Load the model parameters
        assert os.path.exists(model_path), "Model path not found"
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        

    # ----------- Generate samples and visualize the density evolution -----------
    key, key_ = jr.split(key) 
    y_seq = generate_samples(
        params,
        score_fn,
        ou,
        num_samples=10000,
        key=key_
    )

    y_seq = jax.device_get(y_seq)

    # ---------- Animation of the density evolution -----------
    if if_animation:
        if data_dim == 1:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Generate reference distributions and KDEs
            x = jnp.linspace(-7, 10, 1000)
            initial_kde = gaussian_kde(init_dist.sample(jr.PRNGKey(0), 100000).flatten())
            target_kde = gaussian_kde(target_dist.sample(jr.PRNGKey(0), 100000).flatten())

            # Precompute sample KDEs for all frames
            kde_x = jnp.linspace(-7, 10, 500)
            frame_densities = []
            for frame in range(K):
                current_samples = y_seq[frame].flatten()
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
                time_text.set_text(f'Step: {frame}/{K} (Time: {K-frame}/{K})')
                
                return line, time_text

            # Create animation
            ani = animation.FuncAnimation(
                fig=fig,
                func=animate,
                frames=K,
                interval=20,
                blit=True
            )

            # Save animation
            writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
            ani_name = 'density_evolution_with_score.mp4' if add_score else 'density_evolution_without_score.mp4'
            ani.save(ani_name, writer=writer)

            plt.close()
        elif data_dim == 2:
            fig, ax = plt.subplots(figsize=(10, 10))
    
            # Generate grid for target contour
            x = jnp.linspace(-45, 45, 200)
            y = jnp.linspace(-45, 45, 200)
            X, Y = jnp.meshgrid(x, y)
            pts = jnp.stack([X.ravel(), Y.ravel()], axis=1)
            Z_target = target_dist.batch(pts).reshape(X.shape)

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
                current_samples = jnp.array(y_seq[frame])
                scatter.set_offsets(current_samples)
                time_text.set_text(f'Step: {frame}/{K} (Time: {K-frame}/{K})')
                return scatter, time_text

            # Create animation
            ani = animation.FuncAnimation(
                fig=fig,
                func=animate,
                frames=K,
                interval=20,
                blit=True
            )

            # Save animation
            writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
            ani_name = 'sample_movement_2d_with_score.mp4' if add_score else 'sample_movement_2d_without_score.mp4'
            ani.save(ani_name, writer=writer)

            plt.close()

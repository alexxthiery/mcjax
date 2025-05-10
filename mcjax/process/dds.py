import jax
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple
from flax.training import train_state
import optax
import flax.linen as nn
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.animation as animation

import sys
sys.path.append('../../')


from mcjax.proba.density import LogDensity
from mcjax.process.ou import OU
from mcjax.proba.gaussian import IsotropicGauss, MixedIsotropicGauss

print(f"Available devices: {jax.devices()}")
jax.config.update("jax_platform_name", "gpu")

class MLPModel(nn.Module):
    dim: int   
    T: int      # total number of diffusion steps

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
          x: [batch, dim] data at time t
          t: [batch] integer timesteps in {0,1,...,T-1}
        Returns:
          score estimate of shape [batch, dim]
        """
        # Sinusoidal time embedding 
        half_dim = 32
        emb_scale = jnp.log(10000.0) / (half_dim - 1)
        freqs = jnp.exp(jnp.arange(half_dim) * -emb_scale)       
        t_proj = t[:, None] * freqs[None, :]                     
        emb = jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)  

        # mix time info
        t_embed = nn.Sequential([
            nn.Dense(64),
            nn.relu,
            nn.Dense(256),
            nn.relu
        ])(emb)                                                 
        h = jnp.concatenate([x, t_embed], axis=-1)   
        h = nn.Dense(128)(h)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)            

        for _ in range(2):
            h0 = h
            h = nn.Dense(128)(h)
            h = nn.LayerNorm()(h)
            h = nn.relu(h)
            h = nn.Dense(128)(h)
            h = h + h0                                           
            h = nn.LayerNorm()(h)
            h = nn.relu(h)

        out = nn.Dense(self.dim)(h)                             
        return out
    

def dds_loss(params, key, ou: OU, init_dist: LogDensity,
             target_dist: LogDensity, score_fn, batch_size: int):
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

        r_next = r_k + (2.0 * ou.sigma**2) * (lambda_Kmk / alpha_Kmk) * jnp.sum(s**2, axis=-1)
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
    static_argnums=(2, 3, 4, 5, 6))

@partial(jax.jit, static_argnums=(2,3,4,5,6))
def train_step(state, key,
               ou, init_dist, target_dist,
               score_fn, batch_size):
    
    loss, grads = loss_and_grad(
        state.params, key,
        ou, init_dist, target_dist,
        score_fn, batch_size
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



if __name__ == "__main__":
    K = 1000
    ou_sigma = 1.0
    learning_rate = 1e-4
    batch_size = 128
    num_steps = 2000
    data_dim = 1

    timesteps = jnp.arange(K, dtype=jnp.float32)
    beta_start, beta_end = 0.1, 20.0
    beta = beta_start + (beta_end - beta_start) * (timesteps / (K - 1))
    alpha = 1.0 - jnp.exp(-2.0 * beta / K)

    # Define the initial distribution of reference process
    init_dist = IsotropicGauss(mu=jnp.zeros(data_dim), log_var=0.0)

    # target distribution is a mixture of 2 gaussians
    mu = jnp.array([[0.],[2.]])
    dist_sigma = jnp.array([1., 2.])
    log_var = jnp.log(dist_sigma**2)
    weights = jnp.array([0.2, 0.8])
    target_dist = MixedIsotropicGauss(mu=mu, log_var=log_var, weights=weights)

    # Define the dynamic of the process
    ou = OU(alpha=alpha, sigma=ou_sigma, init_dist=init_dist)
    # Define the network
    model = MLPModel(dim=1, T=K)

    # network initialization
    key = jr.PRNGKey(0)
    key, key_ = jr.split(key)
    dummy_x = jnp.zeros((1, data_dim))
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    params = model.init(key_, dummy_x, dummy_t)

    # optimizer initialization  
    optimizer = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # score function
    def score_fn(params, k, y):
        batch_t = jnp.full((y.shape[0],), k, dtype=jnp.int32)
        return model.apply(params, y, batch_t)

    def scan_step(carry, step):
        state, key = carry
        key, key_ = jr.split(key)
        state, loss = train_step(state, key_, ou, init_dist, target_dist, score_fn, batch_size)
            # every 100 steps, print step and current loss
        def do_print(_):
            jax.debug.print("At step {}, loss = {}", step, loss)
            return None

        # branch on (step % 100 == 0)
        _ = jax.lax.cond((step % 100) == 0, do_print, lambda _: None, operand=None)
        return (state, key), loss

    def run_training(state, key):
        (final_state, final_key), losses = jax.lax.scan(
            scan_step,
            (state, key),
            jnp.arange(num_steps)
        )
        return final_state, final_key, losses


    # Training loop
    key, key_ = jr.split(key)
    state, key, losses = run_training(state, key_)


    # Plot the loss curve
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig('loss_curve.png')

    # Sample the target process
    key, key_ = jr.split(key) 
    y_seq = generate_samples(
        state.params,
        score_fn,
        ou,
        num_samples=10000,
        key=key_
    )

    y_seq = jax.device_get(y_seq)
    dataset = target_dist.sample(key_, 10000)

    # plot the last distribution of y_seq against dataset
    plt.figure(figsize=(8, 4))
    plt.hist(dataset, bins=30, density=True, alpha=0.5, label='Target Distribution')
    plt.hist(y_seq[-1], bins=30, density=True, alpha=0.5, label='Final Samples')
    plt.hist(y_seq[0], bins=30, density=True, alpha=0.5, label='Initial Samples')
    plt.hist(y_seq[K//2], bins=30, density=True, alpha=0.5, label='Middle Samples')
    plt.title('Generated Samples vs Target Distribution')
    plt.legend()
    plt.savefig('generated_samples.png')

    # Draw the trajectory of one sample
    plt.figure(figsize=(8, 4))
    plt.plot(jnp.arange(K, 0, -1), y_seq[:, 0], label='Sample Trajectory')
    plt.xlabel('Time Step')
    plt.ylabel('Sample Trajectory')
    plt.title('Sample Trajectory Over Time')
    plt.legend()
    plt.savefig('sample_trajectory.png')

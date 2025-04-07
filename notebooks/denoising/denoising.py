import numpy as np  

import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

from flax import linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import os
sys.path.append('../../')


from mcjax.proba.gaussian import IsotropicGauss
from mcjax.proba.neal_funnel import NealFunnel
from mcjax.proba.student import Student
from mcjax.proba.banana2d import Banana2D

print(f"Available devices: {jax.devices()}")
jax.config.update("jax_platform_name", "gpu")

T = 1000
dim = 2
batch_size = 256

# define hyper-parameters 
beta_start = 0.0001
beta_end = 0.02
betas = beta_start * (beta_end / beta_start) ** (jnp.linspace(0, 1, T))
alphas = 1 - betas
alpha_bars = jnp.cumprod(alphas)

class MLPModel(nn.Module):
    dim: int
    T: int
    
    @nn.compact
    def __call__(self, x, t):
        # vectorized time embedding
        half_dim = 32
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        
        t_embed = nn.Sequential([
            nn.Dense(64),
            nn.relu,
            nn.Dense(256)
        ])(emb)
        
        x = jnp.concatenate([x, t_embed], axis=-1)
        
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        return nn.Dense(self.dim)(x)

@partial(jax.jit, static_argnums=(0,))
def create_train_state(model, key):
    t = jnp.zeros((batch_size,), dtype=jnp.int32)
    params = model.init(key, jnp.ones((batch_size, dim)), t)['params']
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(1e-4)
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@partial(jax.jit, static_argnums=(1,))
def compute_loss(params, apply_fn, key, x0):
    batch_size = x0.shape[0]
    key1, key2 = jr.split(key, 2)
    noise = jr.normal(key1, x0.shape)

    # choose a random time step
    t = jr.randint(key2, (batch_size,), 0, T)

    # forward process
    alpha_t = alpha_bars[t].reshape(-1, *([1]*(len(x0.shape)-1)))
    xt = jnp.sqrt(alpha_t)*x0 + jnp.sqrt(1-alpha_t)*noise
    
    # Note: apply_fn now takes params directly
    pred_noise = apply_fn(params, xt, t)
    return jnp.mean(jnp.sum((pred_noise - noise)**2, axis=1))

@jax.jit
def train_step(state: train_state.TrainState, batch: jnp.ndarray, key: jr.PRNGKey):
    """Corrected training step with proper function binding."""
    def loss_fn(params):
        # Create a bound apply function with the correct signature
        def apply_fn(p, x, t):
            return state.apply_fn({'params': p}, x, t)
        return compute_loss(params, apply_fn, key, batch)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

@partial(jax.jit, static_argnums=(0,))  
def sample_step(model, params, samples, key, t):
    key, key_ = jr.split(key)
    t_arr = jnp.full((samples.shape[0],), t)
    
    pred_noise = model.apply({'params': params}, samples, t_arr)
    samples = (samples - (betas[t]/jnp.sqrt(1-alpha_bars[t])) * pred_noise) / jnp.sqrt(alphas[t])
    
    noise = jr.normal(key_, samples.shape) * (t > 0)
    samples += jnp.sqrt(betas[t]) * noise
    return samples, key


# sampling function
def sample(model, params, key, num_samples=1000):
    key, key_ = jr.split(key)
    samples = jr.normal(key_, (num_samples, dim))
    
    def body_fn(carry, t):
        samples, key = carry
        samples, key = sample_step(model, params, samples, key, T-1-t)
        return (samples, key), samples
    
    _, samples_list = jax.lax.scan(body_fn, (samples, key), jnp.arange(T))
    return jnp.concatenate([samples[None,...], samples_list])

@partial(jax.jit, static_argnums=(1,))
def create_dataset(key, num_samples=1000):
    key, key_ = jr.split(key)
    theta = jr.uniform(key_, (num_samples,)) * 2 * jnp.pi
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    x += 0.1 * jax.random.normal(key, (num_samples, 2))
    return x

def train(model, key, init_state, dataset, num_steps=1000, num_samples=1000):
    state = init_state
    losslist = []

    dataset = jax.device_put(dataset)
    num_batches = len(dataset) // batch_size
    dataset_trimmed = dataset[:num_batches*batch_size]
    batches = jnp.split(dataset_trimmed, num_batches)

    for epoch in range(num_steps):
        epoch_loss = 0.
        for batch in batches:
            key, key_ = jr.split(key)
            state, loss = train_step(state, batch, key_)
            epoch_loss += loss
        losslist.append(epoch_loss/num_batches)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Average Loss: {epoch_loss}')
    
    return state, jnp.array(losslist)
    

def main():
    model = MLPModel(dim, T)
    num_samples = 10000
    num_steps = 2000

    key = jr.key(0)
    key, key_ = jr.split(key)
    state = create_train_state(model, key_)
    key, key_ = jr.split(key)
    dataset = create_dataset(key_, num_samples)

    key, key_ = jr.split(key)
    final_state,losslist = train(model, key_, state, dataset, num_steps=num_steps, num_samples=num_samples)

    # sample from the model
    key, key_ = jr.split(key)
    samples_list = sample(model, final_state.params, key)

    # ----------------- Plotting -----------------
    samples_list = jax.device_get(samples_list)
    dataset = jax.device_get(dataset)
    plt.figure()
    plt.scatter(dataset[:,0], dataset[:,1], marker='.', color='b', label='Original Data',alpha=0.1)
    plt.scatter(samples_list[-1][:,0], samples_list[-1][:,1], marker='.', color = 'r', label='Sampled Data')
    plt.legend()
    plt.savefig('denoising.png')
    plt.close()

    # Create figure and axis
    fig, ax = plt.subplots()
    ax.set_title('Comparison of Ground Truth and Samples of X0')

    # Plot static ground truth
    sc1 = ax.scatter(dataset[:, 0], dataset[:, 1], marker='.', color='blue', label='Ground Truth of X0',alpha=0.1)
    sc2 = ax.scatter([], [], marker='.', color='red', label='Samples of X0')

    # Add legend
    ax.legend()

    # Update function for animation
    def update(frame):
        sc2.set_offsets(samples_list[frame]) 
        return sc2,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(samples_list), interval=200, blit=True)
    ani.save("scatter_animation.avi",writer='ffmpeg') 

    # plot training loss
    plt.figure()
    plt.plot(losslist, label='Loss', color='blue', alpha=0.7)
    plt.title('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()

    # draw the trajectory of one sampled particle
    plt.figure()
    index = 0
    plt.plot([x[index,0] for x in samples_list], [x[index,1] for x in samples_list], color='red',alpha=0.7)
    plt.scatter(dataset[:, 0], dataset[:, 1], marker='.', color='blue', alpha = 0.1, label='Ground Truth of X0')
    plt.title('Trajectory')
    plt.savefig('trajectory.png')
    plt.close()

if __name__ == "__main__":
    main()
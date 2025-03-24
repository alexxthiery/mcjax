import numpy as np  

import jax
import jax.numpy as jnp
import jax.random as jr

from flax import linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
sys.path.append('../../')

from mcjax.proba.gaussian import IsotropicGauss
from mcjax.proba.neal_funnel import NealFunnel
from mcjax.proba.student import Student
from mcjax.proba.banana2d import Banana2D

T = 1000
dim = 2
batch_size = 64

# define hyper-parameters 
beta_start = 0.0001
beta_end = 0.02
betas = jnp.linspace(beta_start, beta_end, T)
alphas = 1 - betas
alpha_bars = jnp.cumprod(alphas)

class MLPModel(nn.Module):
    dim: int
    T: int
    
    def setup(self):
        # Enhanced time embedding
        self.time_embed = nn.Sequential([
            nn.Dense(64),
            nn.relu,
            nn.Dense(256)
        ])
        
        # Main network
        self.layers = [
            nn.Dense(128),
            nn.LayerNorm(),
            nn.relu,
            nn.Dense(128),
            nn.LayerNorm(),
            nn.relu,
            nn.Dense(self.dim)
        ]

    def __call__(self, x, t):
        # Sinusoidal time embedding
        half_dim = 32
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        t_embed = self.time_embed(emb)
        
        x = jnp.concatenate([x, t_embed], axis=-1)
        for layer in self.layers:
            x = layer(x)
        return x
    
def create_train_state(model, key):
    t = jnp.zeros((batch_size,), dtype=jnp.int32)
    params = model.init(key, jnp.ones((batch_size, dim)), t)['params']
    tx = optax.chain(
        optax.clip(1.0),
        optax.adam(1e-4)
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def compute_loss(params, apply, key, x0):
    key1, key2 = jr.split(key, 2)
    noise = jr.normal(key1, x0.shape)

    # choose a random time step
    t = jr.randint(key2, (1,), 0, T)[0]

    # forward process
    xt = jnp.sqrt(alpha_bars[t])*x0 \
        + jnp.sqrt(1-alpha_bars[t])*noise
    
    pred_noise = apply({"params": params}, xt, jnp.full((batch_size,), t))
    return jnp.mean(jnp.sum((pred_noise - noise)**2, axis=1))

# training loop
def train_step(state, batch, key):
    loss, grad = jax.value_and_grad(compute_loss)(state.params, state.apply_fn, key, batch)
    return state.apply_gradients(grads=grad), loss

# sampling function
def sample(model, params, key, dim, num_samples=1000):
    key, key_ = jr.split(key)
    samples = jr.normal(key_, (num_samples,dim))
    samples_list = [samples]
    for t in reversed(range(T)):
        t_arr = jnp.full((num_samples,), t)
        pred_noise = model.apply({'params': params}, samples, t_arr)

        # reverse step
        samples = (samples - 
                   (betas[t]/jnp.sqrt(1-alpha_bars[t])) * pred_noise) / jnp.sqrt(alphas[t]) 
        if t > 0:
            key, key_ = jr.split(key)
            samples += jnp.sqrt(betas[t])*jr.normal(key_, (num_samples,dim))
        
        samples_list.append(samples)

    return samples_list

# toy dataset
def create_dataset(key, num_samples=1000):
    key, key_ = jr.split(key)
    theta = jr.uniform(key_, (num_samples,)) * 2 * jnp.pi
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    x += 0.1 * jax.random.normal(key, (num_samples, 2))
    return x

def train(model, key, init_state, dataset, num_steps=1000, num_samples=1000):
    state = init_state
    num_batches = len(dataset) // batch_size
    dataset_trimmed = dataset[:num_batches*batch_size]
    batches = jnp.split(dataset_trimmed, num_batches)
    losslist = []

    for epoch in range(num_steps):
        epoch_loss = 0.
        for batch in batches:
            key, key_ = jr.split(key)
            state, loss = train_step(state, batch, key_)
            epoch_loss += loss
        losslist.append(epoch_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss}')
    
    # plot loss
    plt.plot(losslist, label='Loss', color='blue', alpha=0.7)
    plt.title('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()
    
    return model, state

model = MLPModel(dim, T)
num_samples = 10000
num_steps = 1000

key = jr.key(0)
key, key_ = jr.split(key)
state = create_train_state(model, key_)
key, key_ = jr.split(key)
dataset = create_dataset(key_, num_samples)

key, key_ = jr.split(key)
_, final_state = train(model, key_, state, dataset, num_steps=num_steps, num_samples=num_samples)

# sample from the model
samples_list = sample(model, final_state.params, key, dim)
plt.scatter(dataset[:,0], dataset[:,1], marker='.', color='b', label='Original Data')
plt.scatter(samples_list[-1][:,0], samples_list[-1][:,1], marker='.', color = 'r', label='Sampled Data')
plt.legend()
plt.savefig('denoising.png')
plt.close()

# Create figure and axis
fig, ax = plt.subplots()
ax.set_title('Comparison of Ground Truth and Samples of X0')

# Plot static ground truth
sc1 = ax.scatter(dataset[:, 0], dataset[:, 1], marker='.', color='blue', label='Ground Truth of X0')
sc2 = ax.scatter([], [], marker='.', color='red', label='Samples of X0')

# Add legend
ax.legend()

# Update function for animation
def update(frame):
    sc2.set_offsets(samples_list[frame]) 
    return sc2,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(samples_list), interval=200, blit=True)
ani.save("scatter_animation.gif") 
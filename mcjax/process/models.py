from abc import ABC, abstractmethod
import jax.numpy as jnp
from flax import linen as nn

class BaseModel(ABC):
    """Abstract interface for any score/networks used by samplers."""

    @abstractmethod
    def init_params(self, key, dummy_x: jnp.ndarray, dummy_t: jnp.ndarray):
        """
        Create initial parameters (flax) given dummy inputs.
        Returns a params PyTree.
        """
        pass

    @abstractmethod
    def apply_fn(self, params, x: jnp.ndarray, t: jnp.ndarray):
        """
        Forward method: given `params`, batch `x` and time index `t`,
        produce whatever the model is supposed to output .
        """
        pass

class MLPModel(nn.Module, BaseModel):
    """
    A “pure‐MLP” implementation for two‐branch score networks:
      - NN1(x, t) : takes concatenated [x, time-embedding] → outputs (batch, dim)
      - NN2(t)    : takes time-embedding only → outputs (batch, dim)

    No residual blocks—just straight Dense→ReLU stacks.
    """
    dim: int   # data dimension
    T:   int   # number of diffusion steps (max time index)

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray):
        """
        Args:
          x: shape (batch, dim)
          t: shape (batch,) dtype=int32
        Returns:
          nn1_out: (batch, dim)
          nn2_out: (batch, dim)
        """
        batch_size = x.shape[0]
        half_dim   = 32
        emb_scale  = jnp.log(10000.0) / (half_dim - 1)
        freqs      = jnp.exp(jnp.arange(half_dim) * -emb_scale)

        # ========== Time‐Embedding (shared logic) ==========
        t_proj = t[:, None] * freqs[None, :]
        t_emb  = jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)

        # ========== NN1 Branch: (x + time) → MLP₁ → (batch, dim) ==========
        te1 = nn.Sequential([
            nn.Dense(64), nn.relu,
            nn.Dense(128), nn.relu
        ])(t_emb) 

        #Concatenate x (shape (batch, dim)) with te1 (batch, 128)
        h1 = jnp.concatenate([x, te1], axis=-1) 

        h1 = nn.Dense(256)(h1)
        h1 = nn.relu(h1)
        h1 = nn.Dense(256)(h1)
        h1 = nn.relu(h1)
        h1 = nn.Dense(128)(h1)
        h1 = nn.relu(h1)


        nn1_out = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.zeros,
            bias_init  =nn.initializers.zeros
        )(h1)  

        # ========== NN2 Branch: (time only) → MLP₂ → (batch, dim) ==========
        h2 = nn.Dense(128)(t_emb) 
        h2 = nn.relu(h2)
        h2 = nn.Dense(128)(h2)
        h2 = nn.relu(h2)

        nn2_out = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.zeros,
            bias_init  =nn.initializers.ones
        )(h2) 

        return nn1_out, nn2_out


    def init_params(self, rng_key, dummy_x: jnp.ndarray, dummy_t: jnp.ndarray):
        """
        Initialize parameters by running a dummy pass through the network.
        - dummy_x: shape (batch_size, dim)
        - dummy_t: shape (batch_size,) int32
        Returns: a frozen_dict of parameters.
        """
        return self.init(rng_key, dummy_x, dummy_t)

    def apply_fn(self, params, x: jnp.ndarray, t: jnp.ndarray):
        return self.apply(params, x, t)
    

class ResBlockModel(nn.Module, BaseModel):
    """
    ResBlock model 
    """
    dim: int
    T:   int

    @nn.compact
    def __call__(self, x, t):
        # build time‐emb similarly to MLPModel...
        half_dim = 32
        emb_scale_t = jnp.log(10000.0)/(half_dim - 1)
        freqs_t = jnp.exp(jnp.arange(half_dim) * -emb_scale_t)
        omega_min = 2*jnp.pi / 100.0    # ≈ 0.0628
        omega_max = 2*jnp.pi           # ≈ 6.283

        emb_scale_x = jnp.log(omega_max/omega_min) / (half_dim - 1)
        # roughly log(100) / 31  ≈ 4.605 / 31 ≈ 0.1486
        freqs_x = jnp.exp(jnp.arange(half_dim) * -emb_scale_x)

        # time embedding
        t_proj = t[:, None] * freqs_t[None, :]
        t_emb  = jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)
        t_embed = nn.Dense(128)(t_emb)
        t_embed = nn.LayerNorm()(t_embed);  t_embed = nn.relu(t_embed)

        # x embedding
        x_proj = x[:, None] * freqs_x[None, :]
        x_emb  = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
        x_embed = nn.Dense(128)(x_emb)  
        x_embed = nn.LayerNorm()(x_embed);  x_embed = nn.relu(x_embed)

        # combine with x
        h = jnp.concatenate([x, t_embed, x_embed], axis=-1)

        h = nn.Dense(256)(h)         # (batch, 256)
        h = nn.LayerNorm()(h)        
        h = nn.relu(h)

        for _ in range(4):
            h0 = h
            h = nn.Dense(256)(h);  h = nn.LayerNorm()(h);  h = nn.relu(h)
            h = nn.Dense(256)(h)
            h = h + h0
            h = nn.LayerNorm()(h);  h = nn.relu(h)

        nn1_out = nn.Dense(self.dim,
                           kernel_init=nn.initializers.zeros,
                           bias_init=nn.initializers.zeros)(h)

        # second branch (just a tiny MLP on t)
        t_proj2 = t[:, None]*freqs[None, :]
        t_emb2 = nn.Dense(128)(jnp.concatenate([jnp.sin(t_proj2), jnp.cos(t_proj2)], axis=-1))
        t_emb2 = nn.LayerNorm()(t_emb2);  t_emb2 = nn.relu(t_emb2)
        nn2_out = nn.Dense(self.dim,
                           kernel_init=nn.initializers.zeros,
                           bias_init=nn.initializers.ones)(t_emb2)

        return nn1_out, nn2_out

    # BaseModel methods
    def init_params(self, rng_key, dummy_x, dummy_t):
        return self.init(rng_key, dummy_x, dummy_t)
    def apply_fn(self, params, x, t):
        return self.apply(params, x, t)
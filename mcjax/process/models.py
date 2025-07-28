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
         # normalize t to [0, 1] range
        t = t  / (self.T - 1)
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
    
class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, t):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

class ResBlockModel(nn.Module, BaseModel):
    dim: int
    T: int
    num_blocks: int = 4 
    hidden_dim: int = 256  

    @nn.compact
    def __call__(self, x, t):
        # time normalization to [0,1]
        t_norm = t.astype(jnp.float32) / jnp.maximum(self.T - 1, 1)
        
        t_emb = SinusoidalPosEmb(dim=128)(t_norm)  
        t_emb = nn.Dense(self.hidden_dim)(t_emb)
        t_emb = nn.gelu(t_emb)
        
        # Initial projection with concatenation
        h = nn.Dense(self.hidden_dim)(jnp.concatenate([x, t_emb], axis=-1))
        h = nn.LayerNorm()(h)
        h = nn.gelu(h)
        
        # Residual blocks with early FiLM conditioning
        for _ in range(self.num_blocks):
            h0 = h
            
            # FiLM modulation at input
            scale = nn.Dense(self.hidden_dim)(t_emb)
            shift = nn.Dense(self.hidden_dim)(t_emb)
            h = h * (1 + scale) + shift
            
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.LayerNorm()(h)
            h = nn.gelu(h)
            h = nn.Dense(self.hidden_dim)(h)
            
            # Residual connection
            h = h + h0
            h = nn.gelu(h)
        
        # Output heads with proper initialization
        nn1_out = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.variance_scaling(0.1, 'fan_in', 'truncated_normal'),
            bias_init=nn.initializers.zeros
        )(h)
        
        nn2_out = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.variance_scaling(0.1, 'fan_in', 'truncated_normal'),
            bias_init=nn.initializers.zeros
        )(t_emb)
        
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
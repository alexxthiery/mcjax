import jax.random as jr
import jax.numpy as jnp

def systematic_resample(key,weights):
    '''
    Perform jit-compatible systematic resampling.
    return: array of indexes into the weights defining the resample
    '''
    N = weights.shape[0]    
    cumulative_sum = jnp.cumsum(weights)
    cumulative_sum = cumulative_sum.at[-1].set(1.)  # avoid round-off errors
    cumulative_sum /= cumulative_sum[-1]

    positions = (jnp.arange(N) + jr.uniform(key)) / N
    indices = jnp.searchsorted(cumulative_sum, positions)
    return indices


import jax.random as jr
import jax.numpy as jnp
from typing import Optional


def systematic_resample(
        key: jr.PRNGKey,
        weights: jnp.ndarray,
        n_samples: Optional[int] = -1,
        ) -> jnp.ndarray:
    """
    Perform systematic resampling of weighted particles using JAX.

    This method generates resampling indices based on a single uniformly drawn offset,
    spaced equally over [0, 1), ensuring stratified and low-variance selection.

    Parameters:
        key (jax.random.PRNGKey): JAX PRNG key.
        weights (jnp.ndarray): Normalized importance weights, shape (N,).

    Returns:
        jnp.ndarray: Array of indices of shape (N,), indicating which particles to select.
    """
    #N = jnp.where(n_samples < 0, weights.shape[0], n_samples)
    N = weights.shape[0] if n_samples < 0 else n_samples

    # Ensure weights are normalized and avoid numerical issues at the boundary
    cumulative_sum = jnp.cumsum(weights)
    cumulative_sum /= cumulative_sum[-1]
    cumulative_sum = cumulative_sum.at[-1].set(1.0)

    # Stratified positions
    u0 = jr.uniform(key, shape=())  # single uniform random number
    positions = (jnp.arange(N) + u0) / N

    # Vectorized search
    indices = jnp.searchsorted(cumulative_sum, positions, side='right')
    return indices



def multinomial_resample(
        key: jr.PRNGKey,
        weights: jnp.ndarray,
        n_samples: Optional[int] = -1,
        ) -> jnp.ndarray:
    """
    Perform multinomial resampling of weighted particles using JAX.

    Parameters:
        key (jax.random.PRNGKey): JAX PRNG key.
        weights (jnp.ndarray): Normalized importance weights, shape (N,).

    Returns:
        jnp.ndarray: Array of indices of shape (N,), indicating which particles to select.
    """
    N = jnp.where(n_samples < 0, weights.shape[0], n_samples)

    # Ensure weights are normalized and cumulative sum ends at 1
    cumulative_sum = jnp.cumsum(weights)
    cumulative_sum /= cumulative_sum[-1]
    cumulative_sum = cumulative_sum.at[-1].set(1.0)

    # Uniform random samples
    uniforms = jr.uniform(key, shape=(N,))

    # Vectorized search
    indices = jnp.searchsorted(cumulative_sum, uniforms, side='right')
    return indices

import jax.random as jr
from jax import lax

def systematic_resample(key: jr.PRNGKey, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Perform systematic resampling of weighted particles using JAX.

    This method generates resampling indices based on a single uniformly drawn offset,
    spaced equally over [0, 1), ensuring stratified and low-variance selection.

    Parameters:
        key (jax.random.PRNGKey): JAX PRNG key.
        weights (jnp.ndarray): Normalized importance weights, shape (N,). Must sum to 1.

    Returns:
        jnp.ndarray: Array of indices of shape (N,), indicating which particles to select.
    """
    N = weights.shape[0]

    # Ensure weights are normalized and avoid numerical issues at the boundary
    cumulative_sum = jnp.cumsum(weights)
    cumulative_sum = cumulative_sum.at[-1].set(1.0)
    cumulative_sum /= cumulative_sum[-1]

    # Stratified positions
    u0 = jr.uniform(key, shape=())  # single uniform random number
    positions = (jnp.arange(N) + u0) / N

    # Vectorized search
    indices = jnp.searchsorted(cumulative_sum, positions, side='right')
    return indices

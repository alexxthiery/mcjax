import jax.numpy as jnp


def _reduce_log_exp(
        x: jnp.ndarray,
        axis=None,
        reduce_fn=jnp.sum,
        ) -> jnp.ndarray:
    """
    Internal utility to compute:
        max(x) + log(reduce_fn(exp(x - max(x))))

    Parameters:
        x (jnp.ndarray): Input array.
        axis (int or tuple of ints): Axis or axes along which to operate.
        reduce_fn (callable): jnp.sum or jnp.mean

    Returns:
        jnp.ndarray: Log-reduced-exp along specified axis.
    """
    x_max = jnp.max(x, axis=axis, keepdims=True)
    x_norm = x - x_max
    reduced = reduce_fn(jnp.exp(x_norm), axis=axis)
    return jnp.squeeze(x_max, axis=axis) + jnp.log(reduced)


def log_sum_exp(x: jnp.ndarray) -> jnp.ndarray:
    return _reduce_log_exp(x, axis=0, reduce_fn=jnp.sum)


def log_sum_exp_batch(x_arr: jnp.ndarray) -> jnp.ndarray:
    return _reduce_log_exp(x_arr, axis=1, reduce_fn=jnp.sum)


def log_mean_exp(x: jnp.ndarray) -> jnp.ndarray:
    return _reduce_log_exp(x, axis=0, reduce_fn=jnp.mean)


def log_mean_exp_batch(x_arr: jnp.ndarray) -> jnp.ndarray:
    return _reduce_log_exp(x_arr, axis=1, reduce_fn=jnp.mean)

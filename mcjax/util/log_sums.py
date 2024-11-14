import jax.numpy as jnp

def log_mean_exp(
            x: jnp.ndarray,  # array (D,)
        ):
    """ log sum exp implemented in a stable way """
    # max of each row
    x_max = jnp.max(x)
    # subtract the max for stability
    x_arr_normalized = x - x_max
    # compute the log sum exp
    return x_max + jnp.log(jnp.mean(jnp.exp(x_arr_normalized)))

def log_mean_exp_batch(
            x_arr: jnp.ndarray,  # (N, D): N number of samples, D dimension
        ):
    """ log mean exp implemented in a stable way """
    # max of each row
    x_max = jnp.max(x_arr, axis=1)
    # subtract the max for stability
    x_arr_normalized = x_arr - x_max[:, None]
    # compute the log sum exp
    return x_max + jnp.log(jnp.mean(jnp.exp(x_arr_normalized), axis=1))

def log_sum_exp(
            x: jnp.ndarray,  # array (D,)
        ):
    """ log sum exp implemented in a stable way """
    # max of each row
    x_max = jnp.max(x)
    # subtract the max for stability
    x_normalized = x - x_max
    # compute the log sum exp
    return x_max + jnp.log(jnp.sum(jnp.exp(x_normalized)))

def log_sum_exp_batch(
            x_arr: jnp.ndarray,  # (N, D): N number of samples, D dimension
        ):
    """ log mean exp implemented in a stable way """
    # max of each row
    x_max = jnp.max(x_arr, axis=1)
    # subtract the max for stability
    x_arr_normalized = x_arr - x_max[:, None]
    # compute the log sum exp
    return x_max + jnp.log(jnp.sum(jnp.exp(x_arr_normalized), axis=1))
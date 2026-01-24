##################################################
# weights & Effective Sample Size
##################################################
import jax
import jax.numpy as jnp
from jax.nn import softmax


def normalize_log_weights(log_weights: jnp.ndarray):
    """
    Normalizes log_weights so that sum exp(log_weights_updated) = 1.
    """
    log_weights_updated = log_weights
    log_weights_updated -= jax.scipy.special.logsumexp(log_weights)
    return log_weights_updated


def compute_weights(log_weights: jnp.ndarray):
    """
    Converts log-weights to normalized importance weights using the softmax function.

    Parameters:
        log_weights (jnp.ndarray): Array of unnormalized log-importance weights.

    Returns:
        jnp.ndarray: Normalized importance weights summing to 1.
    """
    return softmax(log_weights)


def effective_sample_size(log_weights: jnp.ndarray):
    """
    Computes the Effective Sample Size (ESS) from a set of log-importance weights.

    ESS estimates the number of effectively independent samples represented by 
    a weighted set of particles. It is computed as:
        ESS = 1 / sum(w_i^2)
    where w_i are the normalized importance weights obtained via softmax.

    Parameters:
        log_weights (jnp.ndarray): Array of log-importance weights.

    Returns:
        float: The effective sample size.
    """
    w = softmax(log_weights)
    return 1.0 / jnp.sum(w ** 2)


def effective_sample_size_normalized(log_weights: jnp.ndarray):
    """
    Computes the normalized Effective Sample Size (ESS) from log-importance weights.

    Normalized ESS is defined as:
        ESS_normalized = ESS / S
    where ESS = 1 / sum(w_i^2) and w_i are the normalized importance weights (via softmax),
    and S is the number of samples.

    This yields a value in [0, 1], where 1 indicates uniform weights (ideal),
    and values near 0 indicate high weight concentration (poor sample diversity).

    Parameters:
        log_weights (jnp.ndarray): Array of log-importance weights.

    Returns:
        float: Normalized effective sample size in the range [0, 1].
    """
    ess = effective_sample_size(log_weights)
    return ess / log_weights.size


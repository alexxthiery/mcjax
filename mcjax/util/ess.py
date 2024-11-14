##################################################
# Effective Sample Size
##################################################
import jax.numpy as jnp


def ess_log_weight(
    log_weights: jnp.ndarray,   # log weights
    ):
    log_w = log_weights - jnp.max(log_weights)
    w = jnp.exp(log_w)
    w = w / jnp.sum(w)
    ess = 1. / jnp.sum(w**2)
    return ess

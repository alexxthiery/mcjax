import abc
import jax
import jax.numpy as jnp
from typing import Callable


# ==================================
# abstract class for a log density
# ===================================
class LogDensity(abc.ABC):
    @property
    def dim(self):
        """ dimension of the distribution """
        return self._dim

    def logdensity(self, x):
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, x):
        # assert the dimension
        assert x.shape == (self.dim,), "Invalid dimension"
        return self.logdensity(x)

    def batch(
            self,
            x_batch: jnp.ndarray,   # (B, D): B batch size, D dimension
            ):
        """ logdensity of the distribution for a batch of samples """
        # assert the dimension
        assert x_batch.ndim == 2, "Invalid dimension: x_batch.ndim must be 2"
        assert x_batch.shape[1] == self.dim, "Invalid dimension: x_batch.shape[1] must be equal to self.dim"
        return jax.vmap(self.logdensity)(x_batch)

    def grad(self, x):
        """ gradient of the distribution """
        assert x.shape == (self.dim,), "Invalid dimension"
        return jax.grad(self.logdensity)(x)

    def grad_batch(
                self,
                x_batch: jnp.ndarray,   # (B, D): B batch size, D dimension
                ):
        """ gradient of the logdensity for a batch of samples """
        assert x_batch.ndim == 2, "Invalid dimension: x_batch.ndim must be 2"
        assert x_batch.shape[1] == self.dim, "Invalid dimension: x_batch.shape[1] must be equal to self.dim"
        return jax.vmap(self.grad)(x_batch)

    def value_and_grad(self, x):
        """ logdensity and gradient of the distribution """
        assert x.shape == (self.dim,), "Invalid dimension"
        return self.logdensity(x), self.grad(x)

    def value_and_grad_batch(
                self,
                x_batch,   # (B, D): B batch size, D dimension
                ):
        """ logpdf and gradient of the distribution for a batch of samples """
        assert x_batch.ndim == 2, "Invalid dimension: x_batch.ndim must be 2"
        assert x_batch.shape[1] == self.dim, "Invalid dimension: x_batch.shape[1] must be equal to self.dim"
        return jax.vmap(self.value_and_grad)(x_batch)

    def sample(self, key, n_samples):
        raise NotImplementedError("Subclasses must implement this method")

    def log_Z(self):
        raise NotImplementedError("Subclasses must implement this method")


# ==================================
# General Log Density
# it is initialized by providing the log density
# ===================================
class LogDensityGeneral(LogDensity):
    def __init__(
                self,
                *,
                logdensity: Callable,   # log density of the distribution
                dim: int                # dimension of the distribution
                ):
        self.logdensity = logdensity
        self._dim = dim


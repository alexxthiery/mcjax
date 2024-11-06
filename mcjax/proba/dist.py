#from abc import ABC, abstractmethod
import jax
from typing import Callable


# ==================================
# class for a general distribution
# ===================================
class Dist:
    def __init__(
                self,
                *,
                logpdf: Callable,   # logpdf of the distribution
                dim: int            # dimension of the distribution
                ):
        self.logpdf = logpdf
        self._dim = dim

    def logpdf_batch(self, x_batch):
        """ logpdf of the distribution for a batch of samples """
        return jax.vmap(self.logpdf)(x_batch)

    @property
    def dim(self):
        """ dimension of the distribution """
        return self._dim


# ==================================
# class for a differentiable distribution
# Gradient of the distribution is required
# ===================================
class DiffDist(Dist):
    def grad(self, x):
        """ gradient of the distribution """
        return jax.grad(self.logpdf)(x)

    def grad_batch(self, x_batch):
        """ gradient of the distribution for a batch of samples """
        return jax.vmap(self.grad)(x_batch)

    def logpdf_and_grad(self, x):
        """ logpdf and gradient of the distribution """
        return self.logpdf(x), self.grad(x)

    def logpdf_and_grad_batch(self, x_batch):
        """ logpdf and gradient of the distribution for a batch of samples """
        return jax.vmap(self.logpdf_and_grad)(x_batch)


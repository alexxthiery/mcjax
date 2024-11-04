# abtract class for a target distribution
from abc import ABC, abstractmethod
import jax


# ==================================
# Abstract class for a distribution
# ===================================
class Dist(ABC):
    @abstractmethod
    def logpdf(self, x):
        """ unormalized logpdf of the distribution """
        pass

    def logpdf_batch(self, x_batch):
        """ logpdf of the distribution for a batch of samples """
        return jax.vmap(self.logpdf)(x_batch)

    @property
    def dim(self):
        """ dimension of the distribution """
        return self._dim


# ==================================
# Abstract class for a differentiable distribution
# Gradient of the distribution is required
# ===================================
class DiffDist(Dist):
    def logpdf_grad(self, x):
        """ gradient of the distribution """
        return jax.grad(self.logpdf)(x)

    def grad_batch(self, x_batch):
        """ gradient of the distribution for a batch of samples """
        return jax.vmap(self.logpdf_grad)(x_batch)

    def logpdf_and_grad(self, x):
        """ logpdf and gradient of the distribution """
        return self.logpdf(x), self.logpdf_grad(x)

    def logpdf_and_grad_batch(self, x_batch):
        """ logpdf and gradient of the distribution for a batch of samples """
        return jax.vmap(self.logpdf_and_grad)(x_batch)


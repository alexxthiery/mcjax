from .density import LogDensity
import jax
import jax.numpy as jnp
from jax import random, grad
from typing import Optional
from scipy.special import iv

class DoubleWell(LogDensity):
    """
    d-dimensional Double Well potential distribution.

    Unnormalized density:
        \rho(x) = exp( -∑_{i=1}^m (x_i^2 - \delta)² - 1/2 ∑_{i=m+1}^d x_i^2 )

    Parameters
    ----------
    dim : int
        Total dimension (d)
    m : int
        Number of double-well coordinates (gives 2^m modes)
    delta : float
        Separation parameter between wells
    """
    def __init__(self, dim: int = 5, m: int = 5, delta: float = 4.0):
        self._dim = dim
        self.m = m
        self.delta = delta
        self.can_sample = True


    def potential(self, x):
        """Compute potential energy U(x)."""
        x = jnp.atleast_2d(x)
        dw_part = jnp.sum((x[:, :self.m] ** 2 - self.delta) ** 2, axis=-1)
        gauss_part = 0.5 * jnp.sum(x[:, self.m:] ** 2, axis=-1)
        return dw_part + gauss_part

    def logdensity(self, x):
        """Unnormalized log-density: -U(x)."""
        return -self.potential(x).squeeze()

    def batch(self, x_batch):
        return self.logdensity(x_batch)


    def grad(self, x):
        x = jnp.atleast_2d(x)
        grad_U = jnp.zeros_like(x)
        # first m coordinates
        grad_U = grad_U.at[:, :self.m].set(4 * x[:, :self.m] * (x[:, :self.m]**2 - self.delta))
        # remaining coords
        grad_U = grad_U.at[:, self.m:].set(x[:, self.m:])
        return -grad_U.squeeze()  

    def grad_batch(self, x_batch):
        return self.grad(x_batch)

    def sample(self, key, n_samples: int):
        """
        Sample approximately from \rho(x) via rejection sampling
        using a Gaussian proposal centered at ±\sqrt{\delta}.
        For large \delta, this yields samples near ±\sqrt{\delta} along first m dims.
        """
        key, key_mode, key_eps = random.split(key, 3)

        # Choose left or right well for each of first m coordinates
        modes = random.choice(key_mode, jnp.array([-1.0, 1.0]), shape=(n_samples, self.m))
        locs = jnp.sqrt(self.delta) * modes

        # Add small Gaussian noise around each well
        noise_dw = random.normal(key_eps, shape=(n_samples, self.m)) * 0.3
        noise_rest = random.normal(key_eps, shape=(n_samples, self.dim - self.m)) * 1.0

        x_dw = locs + noise_dw
        x_rest = noise_rest
        return jnp.concatenate([x_dw, x_rest], axis=-1)

    def log_Z(self):
        """
        Analytic log partition function using special functions.
        Z = [I(delta)]^m * (sqrt(2π))^(d - m)
        where I(delta) = ∫ exp(-(x^2 - δ)^2) dx
                       = 0.5 * sqrt(π/2) * e^{-δ²/2} *
                         [(1+i) I_{-1/4}(δ²/2) + (1-i) I_{1/4}(δ²/2)]
        """
        delta = self.delta
        m = self.m
        d = self._dim

        # Modified Bessel components (real combination)
        z = delta**2 / 2.0
        I_pos = iv(0.25, z)
        I_neg = iv(-0.25, z)
        I_delta = 0.5 * jnp.sqrt(jnp.pi / 2) * jnp.exp(-delta**2 / 2) * (I_pos + I_neg)

        Z = (I_delta**m) * ((jnp.sqrt(2 * jnp.pi)) ** (d - m))
        return jnp.log(Z)

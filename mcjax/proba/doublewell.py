from .density import LogDensity
import jax
import jax.numpy as jnp
from jax import random, grad

import numpy as np
from typing import Optional
from scipy.special import iv
from scipy.stats import norm
from matplotlib import pyplot as plt
from itertools import product

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

    def plot_doublewell_marginals(self, folder_path, method_name, samples, delta, m):
        """
        Draw:
        - 1D marginal densities for x1,...,xd
        - 2D marginal density plots for (x1, x2..xd)
        Overlays theoretical curves.
        """

        samples = np.asarray(samples)
        n, d = samples.shape

        sqrt_delta = np.sqrt(delta)

        # Theoretical 1D densities 
        def doublewell_1d_pdf(x):
            return np.exp(-(x**2 - delta)**2)

        def gaussian_1d_pdf(x):
            return norm.pdf(x, 0, 1)

        # 1D MARGINALS 
        fig, axes = plt.subplots(d, 1, figsize=(6, 3*d))
        xs = np.linspace(-3*np.sqrt(delta), 3*np.sqrt(delta), 400)

        for i in range(d):
            ax = axes[i]
            ax.hist(samples[:, i], bins=80, density=True, alpha=0.5, label="empirical")

            if i < m:
                ax.plot(xs, doublewell_1d_pdf(xs), 'r-', lw=2, label="theoretical DW marginal")
            else:
                ax.plot(xs, gaussian_1d_pdf(xs), 'g-', lw=2, label="theoretical Gaussian marginal")

            ax.set_title(f"1D marginal for x[{i+1}]")
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"{folder_path}/doublewell/{method_name}_doublewell_1D_marginals.png")
        plt.close()

        # 2D MARGINALS (x1 vs xj)
        fig, axes = plt.subplots(1, d-1, figsize=(4*(d-1), 4))
        x1 = samples[:, 0]

        for j in range(1, d):
            ax = axes[j-1]
            ax.hist2d(x1, samples[:, j], bins=80, density=True, cmap='viridis')
            ax.set_xlabel("x1")
            ax.set_ylabel(f"x{j+1}")
            ax.set_title(f"2D marginal: x1 vs x{j+1}")

        plt.tight_layout()
        plt.savefig(f"{folder_path}/doublewell/{method_name}_doublewell_2D_marginals.png")
        plt.close()

    def plot_doublewell_well_hist(self, folder_path, method_name, samples, delta, m):
        """
        Computes well assignment for each sample and plots histogram.
        A well is identified by a tuple of ±1 signs for the first m coordinates.
        Theoretical distribution is uniform over 2^m wells.
        """

        samples = np.asarray(samples)
        n, d = samples.shape
        sqrt_delta = np.sqrt(delta)

        # Generate all theoretical wells 
        wells = list(product([-1,1], repeat=m))   # list of tuples like (-1,1,-1,1,1)
        well_index = {w:i for i,w in enumerate(wells)}

        #####  Assign each sample to the closest well 
        # Compute distance to ±sqrt(delta) for each dim; the nearest well is simply determined by sign(x_i)
        signs = np.sign(samples[:, :m])
        signs[signs == 0] = 1 

        # Convert sign vectors to tuples
        assigned = [tuple(int(s) for s in row) for row in signs]

        # Histogram counts
        counts = np.zeros(len(wells), dtype=int)
        for w in assigned:
            counts[well_index[w]] += 1

        # Convert to probability
        probs = counts / n
        theoretical = np.ones(len(wells)) / len(wells)

        # Plot histogram 
        plt.figure(figsize=(12,5))
        x = np.arange(len(wells))
        plt.bar(x - 0.2, probs, width=0.4, label="empirical")
        plt.bar(x + 0.2, theoretical, width=0.4, label="theoretical (uniform)")

        plt.xticks(x, [str(w) for w in wells], rotation=90)
        plt.ylabel("Probability")
        plt.title(f"Well Repartition (m={m}, delta={delta})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{folder_path}/doublewell/{method_name}_doublewell_well_hist.png")
        plt.close()

        return probs, theoretical, wells
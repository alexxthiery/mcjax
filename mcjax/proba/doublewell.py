from .density import LogDensity
import jax
import jax.numpy as jnp
from jax import random, grad

import numpy as np
from typing import Optional
from scipy.special import ive, logsumexp
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
    def __init__(self, dim: int = 5, m: int = 5, delta: float = 4.0,offset: Optional[np.ndarray] = None):
        self._dim = dim
        self.m = m
        self.delta = delta
        self.can_sample = True
        if offset is None:
            self.offset = jnp.zeros(m)
        else:
            offset = np.asarray(offset, dtype=float)
            if offset.ndim != 1:
                raise ValueError("Offset must be a 1D array.")
            
            # Use the first m elements as the offset for the double-well dimensions
            if offset.size != m:
                raise ValueError(f"Offset array size must be exactly m={m}.")
            self.offset = jnp.array(offset)

    def potential(self, x):
        """Compute potential energy U(x)."""
        x = jnp.atleast_2d(x)
        x_dw = x[:, :self.m]
        x_dw_minus_offset = x_dw - self.offset 
        dw_part = jnp.sum((x_dw_minus_offset ** 2 - self.delta) ** 2, axis=-1)
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
        x_dw = x[:, :self.m]
        x_dw_minus_offset = x_dw - self.offset # x_i - c_i
        # d/dx_i [((x_i - c_i)^2 - delta)^2] = 2 * ((x_i - c_i)^2 - delta) * 2 * (x_i - c_i)
        grad_dw = 4 * x_dw_minus_offset * (x_dw_minus_offset**2 - self.delta)
        grad_U = grad_U.at[:, :self.m].set(grad_dw)
        
        # remaining coords
        grad_U = grad_U.at[:, self.m:].set(x[:, self.m:])
        return -grad_U.squeeze()  

    def grad_batch(self, x_batch):
        return self.grad(x_batch)

    def sample(self, key, n_samples: int):
        """
        Sample exactly from \rho(x) via rejection sampling
        using a Gaussian proposal centered at ±\sqrt{\delta}.
        """
        key_dw, key_rest = random.split(key)
        
        sigma = 1.0 / jnp.sqrt(8 * self.delta)
        
        # bounding constant M for rejection sampling
        M = 2 * jnp.sqrt(2 * jnp.pi) * sigma * 1.1

        def rejection_sample_scalar(K_offset):
            k, offset = K_offset
            def cond_fn(state):
                _, _, done = state
                return ~done

            def body_fn(state):
                key, val, _ = state
                k_mode, k_eps, k_u, k_next = random.split(key, 4)

                mode_center_pos = (jnp.sqrt(self.delta) + offset)
                mode_center_neg = (-jnp.sqrt(self.delta) + offset)
                
                # Sample from Proposal q(x)
                mode_sign = random.choice(k_mode, jnp.array([-1.0, 1.0]))
                mode_center = jnp.where(mode_sign > 0, mode_center_pos, mode_center_neg)
                x_prop = mode_center + (random.normal(k_eps) * sigma)
                
                # Unnormalized log_p = -(x^2 - delta)^2
                log_p = -((x_prop - offset)**2 - self.delta)**2
                
                # Compute log probability of proposal q(x) (mixture of two Gaussians)
                q_val = 0.5 * (1/(sigma * jnp.sqrt(2*jnp.pi))) * (
                    jnp.exp(-0.5 * ((x_prop - mode_center_pos)/sigma)**2) + 
                    jnp.exp(-0.5 * ((x_prop - mode_center_neg)/sigma)**2)
                )
                
                # Accept if u < p(x) / (M * q(x))
                u = random.uniform(k_u)
                accept = u < (jnp.exp(log_p) / (M * q_val))
                
                return k_next, jnp.where(accept, x_prop, val), accept

            init_val = (k, 0.0, False)
            _, sample, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)
            return sample

        keys_dw = random.split(key_dw, n_samples * self.m)
        offset_repeated = jnp.tile(self.offset, n_samples)
        x_dw = jax.vmap(rejection_sample_scalar)((keys_dw, offset_repeated)).reshape(n_samples, self.m)

        # Sample remaining dimensions (Standard Gaussian)
        x_rest = random.normal(key_rest, shape=(n_samples, self.dim - self.m))

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

        z = delta**2 / 2.0

        # scaled Bessel functions: ive(v, z) = exp(-z) * I_v(z)
        I_pos_scaled = ive(0.25, z)
        I_neg_scaled = ive(-0.25, z)

        # I(delta) = (π/2) * sqrt(delta) * (ive(1/4,z) + ive(-1/4,z))
        # compute log I(delta) stably
        log_I_delta = (
            np.log(np.pi / 2.0)  # log(π/2)
            + 0.5 * np.log(delta)  # log sqrt(delta)
            + logsumexp([np.log(I_pos_scaled), np.log(I_neg_scaled)])
        )

        # Gaussian part
        log_gaussian = (d - m) * 0.5 * np.log(2 * np.pi)

        return m * log_I_delta + log_gaussian

    def plot_doublewell_marginals(self, folder_path, method_name, loss_name, samples, delta, m):
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
        fig, axes = plt.subplots(min(d, 10), 1, figsize=(6, 3*min(d, 10)))

        max_offset = np.max(np.abs(self.offset)) if self.offset.size > 0 else 0
        xs_base = np.linspace(-3*np.sqrt(delta), 3*np.sqrt(delta), 400)

        for i in range(min(d, 10)):  # limit to first 10 dims for visibility
            ax = axes[i]
            if i < m:
                current_offset = self.offset[i].item() 
                xs = xs_base + current_offset
            else:
                current_offset = 0.0
                xs = xs_base
            
            ax.hist(samples[:, i], bins=80, density=True, alpha=0.5, label="empirical")

            if i < m:
                ax.plot(xs, doublewell_1d_pdf(xs - current_offset), 'r-', lw=2, label="theoretical DW marginal")
            else:
                ax.plot(xs, gaussian_1d_pdf(xs), 'g-', lw=2, label="theoretical Gaussian marginal")

            ax.set_title(f"1D marginal for x[{i+1}] (Offset: {current_offset:.2f})")
            ax.legend()

        plt.tight_layout()
        offset_str = f", Offset[:m]={self.offset.round(2)}" if self.offset.size > 0 else ""
        plt.suptitle(f"DoubleWell Marginals (m={m}, delta={delta}, dim={d}{offset_str})", y=1.02, fontsize=16)
        plt.savefig(f"{folder_path}/doublewell_DIM={d}/{method_name}_{loss_name}_doublewell_1D_marginals.png")
        plt.close()

        ####################### 2D MARGINALS (x1 vs xj) #######################
        fig, axes = plt.subplots(1, min(d, 10)-1, figsize=(4*(min(d, 10)-1), 4))
        x1 = samples[:, 0]
        c1 = self.offset[0].item()

        mode_base_1 = c1 + sqrt_delta
        mode_base_2 = c1 - sqrt_delta

        x1_modes = [mode_base_1, mode_base_1, mode_base_2, mode_base_2]

        for j in range(1, min(d, 10)): # just plot first min(d, 10) double-well dims
            ax = axes[j-1]
            xj = samples[:, j]

            if j < m:
                cj = self.offset[j].item()
                mode_base_j_pos = cj + sqrt_delta
                mode_base_j_neg = cj - sqrt_delta
                
                # Generate the four xj-coordinates of the modes
                xj_modes = [mode_base_j_pos, mode_base_j_neg, mode_base_j_pos, mode_base_j_neg]
                
                # Plot the 4 theoretical mode centers
                ax.plot(x1_modes, xj_modes, 'ro', markersize=5, label='Theoretical Modes')
                mode_label = f"Mode Centers (c1={c1:.2f}, c{j+1}={cj:.2f})"
            else:
                # Gaussian
                cj = 0.0
                xj_modes = [0.0, 0.0, 0.0, 0.0] 
                ax.plot(x1_modes, xj_modes, 'ro', markersize=5, label='Theoretical Modes')
                mode_label = f"Mode Centers (c1={c1:.2f}, c{j+1}=0.0)"

            ax.hist2d(x1, xj, bins=80, density=True, cmap='viridis')        
            ax.set_xlabel("x1")
            ax.set_ylabel(f"x{j+1}")
            ax.set_title(f"2D marginal: x1 vs x{j+1}\n{mode_label}")
            ax.legend()                

        plt.tight_layout()
        plt.suptitle(f"DoubleWell 2D Marginals (m={m}, delta={delta}, dim={d})", y=1.02, fontsize=16)
        plt.savefig(f"{folder_path}/doublewell_DIM={d}/{method_name}_{loss_name}_doublewell_2D_marginals.png")
        plt.close()

    def plot_doublewell_well_hist(self, folder_path, method_name, loss_name, samples, delta, m):
        """
        Computes well assignment for each sample and plots histogram.
        A well is identified by a tuple of ±1 signs for the first m coordinates.
        Theoretical distribution is uniform over 2^m wells.
        """

        samples = np.asarray(samples)
        n, d = samples.shape
        sqrt_delta = np.sqrt(delta)

        # Generate all theoretical wells 
        signs_base = np.array(list(product([-1,1], repeat=m))) 
        wells_signs = [tuple(int(s) for s in row) for row in signs_base]
        well_index = {w:i for i,w in enumerate(wells_signs)}

        # Assign each sample to the closest well 
        well_centers = signs_base * sqrt_delta + self.offset
        samples_dw = samples[:, :m]
        
        # Compute squared distance to each well center for every sample
        distances_sq = np.sum((samples_dw[:, np.newaxis, :] - well_centers[np.newaxis, :, :])**2, axis=-1)
        
        # Find the index of the closest well (0 to 2^m - 1)
        closest_well_idx = np.argmin(distances_sq, axis=1)
        
        assigned = [wells_signs[i] for i in closest_well_idx]

        # Histogram counts
        counts = np.zeros(len(wells_signs), dtype=int)
        for w in assigned:
            counts[well_index[w]] += 1

        # Convert to probability
        probs = counts / n
        theoretical = np.ones(len(wells_signs)) / len(wells_signs) 

        # Plot histogram 
        plt.figure(figsize=(12,5))
        x = np.arange(len(wells_signs))
        plt.bar(x - 0.2, probs, width=0.4, label="empirical")
        plt.bar(x + 0.2, theoretical, width=0.4, label="theoretical (uniform)")

        plt.xticks(x, [str(w) for w in wells_signs], rotation=90)
        plt.ylabel("Probability")
        plt.title(f"Well Repartition (m={m}, delta={delta})")
        plt.legend()
        plt.tight_layout()
        plt.suptitle(f"DoubleWell Well Histogram (m={m}, delta={delta}, dim={d})", y=1.02, fontsize=16)
        plt.savefig(f"{folder_path}/doublewell_DIM={d}/{method_name}_{loss_name}_doublewell_well_hist.png")
        plt.close()

        return probs, theoretical, wells_signs
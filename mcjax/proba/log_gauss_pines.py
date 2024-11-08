import jax
import jax.numpy as jnp
import numpy as np
import os
from .density import LogDensity
import mcjax.proba.cox_process_utils as cp_utils

# Path to the pines dataset
MODULE_PATH = os.path.dirname(__file__)
FILE_PATH = os.path.join(MODULE_PATH, 'finpines.csv')


class LogGaussPines(LogDensity):
    """
    Log density for a Gaussian Cox process on the Pines dataset.
    
    Discretized Gaussian Cox process on the Pines dataset on a grid
    of dimension grid_dim x grid_dim. The distribution is consequently
    defined on a R^{grid_dim**2}.
    """
    def __init__(
                self,
                grid_dim=40,            # grid dimension
                use_whitened=False,     # whether to use whitened likelihood
                ):

        # Discretization is as in Controlled Sequential Monte Carlo
        # by Heng et al 2017 https://arxiv.org/abs/1708.08396
        dim = grid_dim**2
        num_dim = dim
        self._dim = dim
        self.log_Z = 0.0
        self.n_plots = 0
        self.can_sample = False
        self._num_latents = num_dim
        self._num_grid_per_dim = int(np.sqrt(num_dim))
        self.pines_points = self.load_pines_points()

        bin_counts = jnp.array(
            cp_utils.get_bin_counts(self.pines_points,
                                    self._num_grid_per_dim))

        self._flat_bin_counts = jnp.reshape(bin_counts, (self._num_latents))

        # This normalizes by the number of elements in the grid
        self._poisson_a = 1./self._num_latents
        # Parameters for LGCP are as estimated in Moller et al, 1998
        # "Log Gaussian Cox processes" and are also used in Heng et al.

        self._signal_variance = 1.91
        self._beta = 1./33

        self._bin_vals = cp_utils.get_bin_vals(self._num_grid_per_dim)

        def short_kernel_func(x, y):
            return cp_utils.kernel_func(x, y, self._signal_variance,
                                        self._num_grid_per_dim, self._beta)

        self._gram_matrix = cp_utils.gram(short_kernel_func, self._bin_vals)
        self._cholesky_gram = jnp.linalg.cholesky(self._gram_matrix)
        self._white_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(2. * jnp.pi)

        half_log_det_gram = jnp.sum(jnp.log(jnp.abs(jnp.diag(self._cholesky_gram))))
        self._unwhitened_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(
            2. * jnp.pi) - half_log_det_gram
        # The mean function is a constant with value mu_zero.
        self._mu_zero = jnp.log(126.) - 0.5*self._signal_variance

        if use_whitened:
            self.logprior = self.whitened_prior_log_density
            self.loglik = self.whitened_likelihood_log_density
        else:
            self.logprior = self.unwhitened_prior_log_density
            self.loglik = self.unwhitened_likelihood_log_density

    def load_pines_points(self):
        """Get the pines data points."""
        with open(FILE_PATH, "rt") as input_file:
            b = np.genfromtxt(input_file, delimiter=",")
        return b

    def whitened_prior_log_density(self, white):
        quadratic_term = -0.5 * jnp.sum(white**2)
        return self._white_gaussian_log_normalizer + quadratic_term

    def whitened_likelihood_log_density(self, white):
        latent_function = cp_utils.get_latents_from_white(white, self._mu_zero,
                                                        self._cholesky_gram)
        return cp_utils.poisson_process_log_likelihood(
            latent_function, self._poisson_a, self._flat_bin_counts)

    def unwhitened_prior_log_density(self, latents):
        white = cp_utils.get_white_from_latents(latents, self._mu_zero,
                                                self._cholesky_gram)
        return -0.5 * jnp.sum(
            white * white) + self._unwhitened_gaussian_log_normalizer

    def unwhitened_likelihood_log_density(self, latents):
        return cp_utils.poisson_process_log_likelihood(
            latents, self._poisson_a, self._flat_bin_counts)
    
    def logdensity(self, x):
        return self.loglik(x) + self.logprior(x)
    
    def initialize_model(self, rng_key, n_chain):
        keys = jax.random.split(rng_key, n_chain)
        self.init_params = jax.vmap(lambda k: self._mu_zero + self._cholesky_gram @ jax.random.normal(k, (self._num_latents,)))(keys)

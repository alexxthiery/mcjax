import jax
import jax.numpy as jnp
import os
from flax import struct
import mcjax.proba.cox_process_utils as cp_utils
import optax
import numpy as np

jax.config.update("jax_enable_x64", True)  # needed for LBFGS optimization


# load data
MODULE_PATH = os.path.dirname(__file__)
FILE_PATH = os.path.join(MODULE_PATH, 'finpines.csv')
pines_points_np = np.genfromtxt(FILE_PATH, delimiter=",")
if pines_points_np.ndim != 2 or pines_points_np.shape[1] != 2:
    raise ValueError(f"Expected (N, 2) shape for point data, got {pines_points_np.shape}")
pines_points = jnp.array(pines_points_np)


@struct.dataclass
class LGCPParams:
    """Placeholder for interface compatibility (not used in log_prob)."""
    pass


@struct.dataclass
class LGCP:
    """
    Log Gaussian Cox Process (LGCP) on the FinPines dataset.

    This implements a discretized LGCP on a 2D grid using either
    a whitened or unwhitened GP prior, and Poisson process likelihood.

    Parameters
    ----------
    dim : int
        Dimensionality of the latent field, equals grid_dim ** 2.
    whitened : bool
        If True, assumes whitened representation; otherwise, unwhitened.
    bin_counts : jnp.ndarray
        Flattened bin counts of the observed spatial data.
    bin_vals : jnp.ndarray
        Locations of bin centers used to compute the kernel matrix.
    cholesky_gram : jnp.ndarray
        Lower Cholesky factor of the kernel Gram matrix.
    mu_zero : float
        Constant mean function value.
    poisson_a : float
        Poisson intensity scaling factor.
    log_normalizer : float
        Log-normalizer term for the prior (whitened or unwhitened).
    """
    dim: int
    whitened: bool
    bin_counts: jnp.ndarray
    bin_vals: jnp.ndarray
    cholesky_gram: jnp.ndarray
    mu_zero: float
    poisson_a: float
    log_normalizer: float
    pines_points: jnp.ndarray = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(cls, grid_dim: int = 40, whitened: bool = False) -> "LGCP":
        """
        Construct an LGCP model instance on a 2D spatial grid using the FinPines dataset.

        This method loads the spatial point data, computes bin counts on a grid,
        constructs the Gaussian process kernel matrix, performs its Cholesky decomposition,
        and sets up model constants such as the mean function and prior normalization.

        Parameters
        ----------
        grid_dim : int, optional
            Number of grid cells per dimension. The total dimensionality of the
            latent field is grid_dim ** 2. Must be a positive integer.
        whitened : bool, optional
            If True, uses a whitened GP representation where the latent variable
            follows a standard normal prior. If False, the GP prior is directly over
            the latent field using the full covariance structure.

        Returns
        -------
        LogGaussPines
            A fully-initialized LGCP model instance ready for log-density evaluation.
        """
        if grid_dim <= 0:
            raise ValueError(f"grid_dim must be a positive integer, got {grid_dim}")

        # Load data
        module_path = os.path.dirname(__file__)
        file_path = os.path.join(module_path, 'finpines.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")

        # Bin and kernel setup
        bin_counts = jnp.array(cp_utils.get_bin_counts(pines_points, grid_dim))
        flat_bin_counts = jnp.reshape(bin_counts, (-1,))
        bin_vals = cp_utils.get_bin_vals(grid_dim)

        signal_variance = 1.91
        beta = 1. / 33
        mu_zero = jnp.log(126.0) - 0.5 * signal_variance
        poisson_a = 1. / (grid_dim ** 2)

        # Kernel construction
        def kernel(x, y):
            return cp_utils.kernel_func(x, y, signal_variance, grid_dim, beta)

        gram = cp_utils.gram(kernel, bin_vals)
        cholesky_gram = jnp.linalg.cholesky(gram)

        # Prior normalization constant
        num_latents = grid_dim ** 2
        log2pi = jnp.log(2. * jnp.pi)
        if whitened:
            log_normalizer = -0.5 * num_latents * log2pi
        else:
            diag_L = jnp.diag(cholesky_gram)
            half_log_det = jnp.sum(jnp.log(jnp.abs(diag_L)))
            log_normalizer = -0.5 * num_latents * log2pi - half_log_det

        return cls(
            dim=num_latents,
            whitened=whitened,
            bin_counts=flat_bin_counts,
            bin_vals=bin_vals,
            cholesky_gram=cholesky_gram,
            mu_zero=mu_zero,
            poisson_a=poisson_a,
            log_normalizer=log_normalizer,
            pines_points=jnp.array(pines_points, dtype=jnp.float32),
        )

    def init_params(self) -> LGCPParams:
        """Returns a dummy param object to satisfy interface compatibility."""
        return LGCPParams()

    def log_prob(
            self,
            x: jnp.ndarray,
            params: LGCPParams = None,  # Placeholder for interface compatibility
            ) -> jnp.ndarray:
        """
        Compute the log-probability of the latent vector x.

        Parameters
        ----------
        params : LGCPParams
            Placeholder parameter object (unused).
        x : jnp.ndarray
            Input vector, either whitened or unwhitened.

        Returns
        -------
        logp : jnp.ndarray
            Scalar log-probability under the LGCP model.
        """
        if x.ndim != 1 or x.shape[0] != self.dim:
            raise ValueError(f"x must be 1D with shape ({self.dim},), got {x.shape}")

        if self.whitened:
            latents = cp_utils.get_latents_from_white(x, self.mu_zero, self.cholesky_gram)
            prior = -0.5 * jnp.sum(x ** 2) + self.log_normalizer
        else:
            latents = x
            white = cp_utils.get_white_from_latents(x, self.mu_zero, self.cholesky_gram)
            prior = -0.5 * jnp.sum(white ** 2) + self.log_normalizer

        likelihood = cp_utils.poisson_process_log_likelihood(
            latents, self.poisson_a, self.bin_counts
        )
        return prior + likelihood

    def map_estimate(
            self,
            x0: jnp.ndarray = None,
            max_iter: int = 500,
            tol: float = 1e-8) -> jnp.ndarray:
        """
        Compute the Maximum A Posteriori (MAP) estimate of the latent field.

        Uses L-BFGS optimization to minimize the negative log-density.

        Parameters
        ----------
        x0 : jnp.ndarray (optional)
            Initial guess for the latent variable (must match model dimension).
            If None, defaults to a zero vector of the correct shape.
        max_iter : int (optional)
            Maximum number of optimization steps.
            If None, defaults to 500.
        tol : float (optional)
            Gradient norm tolerance for convergence.
            If None, defaults to 1e-8.

        Returns
        -------
        x_map : jnp.ndarray
            MAP estimate of the latent variable.
        """
        # if x0 not provided, use zeros as initial guess
        if x0 is None:
            x0 = jnp.zeros(self.dim, dtype=jnp.float32)

        if x0.shape != (self.dim,):
            raise ValueError(f"x0 must have shape ({self.dim},), got {x0.shape}")

        def loss_fn(x):
            return -self.log_prob(x)

        grad_fn = jax.grad(loss_fn)
        optimizer = optax.lbfgs()
        opt_state = optimizer.init(x0)

        def cond_fun(state):
            _, _, _, grad, i = state
            return (jnp.linalg.norm(grad) > tol) & (i < max_iter)

        def body_fun(state):
            x, opt_state, loss, grad, i = state
            # updates, opt_state = optimizer.update(
            #     grad, opt_state, params=x, value=loss, value_fn=loss_fn
            # )
            updates, opt_state = optimizer.update(
                grad, opt_state, params=x,
                value=loss, grad=grad, value_fn=loss_fn
            )
            x = optax.apply_updates(x, updates)
            loss = loss_fn(x)
            grad = grad_fn(x)
            return (x, opt_state, loss, grad, i + 1)

        @jax.jit
        def run_lbfgs(x0, opt_state):
            loss = loss_fn(x0)
            grad = grad_fn(x0)
            init_state = (x0, opt_state, loss, grad, 0)
            final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
            return final_state[0]  # x_map

        return run_lbfgs(x0, opt_state)
    
    def hessian_at(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the Hessian of the negative log-probability at a given point.

        Typically used at the MAP estimate for Laplace approximation.

        Parameters
        ----------
        x : jnp.ndarray
            Point at which to evaluate the Hessian (e.g., MAP estimate).

        Returns
        -------
        hess : jnp.ndarray
            (dim, dim) Hessian matrix of -log p(x) at the given point.
        """
        if x.shape != (self.dim,):
            raise ValueError(f"x must have shape ({self.dim},), got {x.shape}")

        def loss_fn(x_):
            return -self.log_prob(x_)

        hess_fn = jax.hessian(loss_fn)
        return hess_fn(x)

    def laplace_approximation(
            self,
            x0: jnp.ndarray = None,
            max_iter: int = 500,
            tol: float = 1e-8) -> jnp.ndarray:
        """
        Compute the Laplace approximation of the posterior using MAP estimate.

        Parameters
        ----------
        x0 : jnp.ndarray (optional)
            Initial guess for the latent variable (must match model dimension).
            If None, defaults to a zero vector of the correct shape.
        max_iter : int (optional)
            Maximum number of optimization steps.
            If None, defaults to 500.
        tol : float (optional)
            Gradient norm tolerance for convergence.
            If None, defaults to 1e-8.

        Returns
        -------
        laplace_params : jnp.ndarray
            MAP estimate of the latent variable.
        """
        if x0 is None:
            x0 = jnp.zeros(self.dim, dtype=jnp.float32)

        if x0.shape != (self.dim,):
            raise ValueError(f"x0 must have shape ({self.dim},), got {x0.shape}")
        
        x_map = self.map_estimate(x0=x0, max_iter=max_iter, tol=tol)
        hessian = self.hessian_at(x_map)
        
        # Ensure the Hessian is positive definite
        if not jnp.all(jnp.linalg.eigvalsh(hessian) > 0):
            raise ValueError("Hessian is not positive definite, cannot perform Laplace approximation.")
        
        # Return the MAP estimate and the Hessian
        cov = jnp.linalg.inv(hessian)
            
        return {
            "mu": x_map,
            "precision": hessian,
            "cov": cov,
        }

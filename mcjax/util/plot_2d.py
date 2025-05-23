import numpy as np
import jax.numpy as jnp
from typing import Callable, Tuple
from matplotlib.patches import Ellipse

def evaluate_on_grid_2d(
    func_batch: Callable[[jnp.ndarray], jnp.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    Nx: int = 500,
    Ny: int = 500,
) -> Tuple[np.ndarray, np.ndarray, jnp.ndarray]:
    """
    Evaluate a batched 2D function on a regular mesh grid.

    Args:
        func_batch: Callable of shape (n, 2) -> (n,)
        xlim: Tuple of (xmin, xmax)
        ylim: Tuple of (ymin, ymax)
        Nx: Number of discretization points along x-axis
        Ny: Number of discretization points along y-axis

    Returns:
        X, Y: Meshgrid (shape Nx x Ny)
        Z:   Evaluated values on grid (shape Nx x Ny)
        
    Example:
        >>> def log_density(x):
        ...     return -0.5 * jnp.sum(x**2, axis=-1)
        >>> X, Y, Z = evaluate_on_grid_2d(
        ...     log_density,
        ...     xlim=(-3, 3),
        ...     ylim=(-2, 2),
        ...     Nx=300,
        ...     Ny=200
        ... )
    """
    x = np.linspace(xlim[0], xlim[1], Nx)
    y = np.linspace(ylim[0], ylim[1], Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    points = jnp.stack([jnp.ravel(X), jnp.ravel(Y)], axis=-1)  # shape (Nx * Ny, 2)
    Z_flat = func_batch(points)  # shape (Nx * Ny,)
    Z = Z_flat.reshape((Nx, Ny))

    return X, Y, Z


def create_ellipse(mu, cov, color='green', alpha=0.5, label=None):
    """
    Create a matplotlib Ellipse representing a 2D Gaussian contour.

    This function computes the eigenvalues and eigenvectors of the covariance
    matrix to determine the orientation and shape of the ellipse that represents
    the 1-standard-deviation contour of a 2D Gaussian distribution.

    Parameters
    ----------
    mu : array-like, shape (2,)
        Mean vector of the Gaussian (center of the ellipse).

    cov : array-like, shape (2, 2)
        Covariance matrix of the Gaussian. Must be symmetric and positive-definite.

    color : str, optional (default='green')
        Fill color of the ellipse.

    alpha : float, optional (default=0.5)
        Transparency level of the fill (0 = transparent, 1 = opaque).

    label : str, optional
        Label to use for the legend.

    Returns
    -------
    ell : matplotlib.patches.Ellipse
        Configured Ellipse patch representing the Gaussian.
    """
    # Compute ellipse geometry
    eigvals, eigvecs = jnp.linalg.eigh(cov)
    angle = jnp.arctan2(eigvecs[1, 0], eigvecs[0, 0]) * 180 / jnp.pi
    width, height = 2 * jnp.sqrt(eigvals)

    # Add filled ellipse (approximate posterior)
    ell = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=angle,
        color=color,
        alpha=alpha,
        label=label,
        )
    return ell

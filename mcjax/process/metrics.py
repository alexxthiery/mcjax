import jax.numpy as jnp
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.special import logsumexp 

def MMD_squared(x_samples: np.ndarray, y_samples: np.ndarray, kernel='rbf', sigma=1.0):
    """
    Compute MMD^2 between two numpy arrays of shape (N, d), (M, d).
    Uses an RBF kernel by default with bandwidth sigma.  Returns scalar.
    """
    # Convert to float64 for SciPy if needed
    X = np.asarray(x_samples, dtype=np.float64)
    Y = np.asarray(y_samples, dtype=np.float64)

    # RBF kernel matrix
    def rbf_kernel(A, B, sigma):
        d2 = pairwise_distances(A, B, metric='sqeuclidean')
        return np.exp(-d2 / (2 * sigma**2))

    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)

    m = X.shape[0]
    n = Y.shape[0]
    mmd = (np.sum(Kxx) - np.trace(Kxx)) / (m * (m - 1)) \
        + (np.sum(Kyy) - np.trace(Kyy)) / (n * (n - 1)) \
        - 2 * np.sum(Kxy) / (m * n)
    return mmd

def two_wasserstein(x_samples: np.ndarray, y_samples: np.ndarray) -> float:
    """
    Compute the 2-Wasserstein distance between two empirical distributions.
    """
    x = np.asarray(x_samples)
    y = np.asarray(y_samples)

    # 1D fallback
    if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
        return wasserstein_distance(x.flatten(), y.flatten())

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Need same number of samples, got {x.shape[0]} vs {y.shape[0]}")
    n = x.shape[0]

    # Flatten any trailing feature dims into a single vector of length D
    xf = x.reshape(n, -1)
    yf = y.reshape(n, -1)

    # Build the cost matrix: squared Euclidean distances
    diff = xf[:, None, :] - yf[None, :, :]   # shape (n, n, \Pi_d)
    C    = np.sum(diff * diff, axis=2)       # shape (n, n)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(C)

    # Compute sqrt of average squared cost
    avg_sq_cost = C[row_ind, col_ind].sum() / n
    return float(np.sqrt(avg_sq_cost))

def ELBO(logweights: np.ndarray, logZ: float = 0.0):
    """
    Evidence Lower Bound (ELBO) computed from log-weights.
    logweights: shape (N,) - log of normalized weights
    logZ: optional log normalization constant
    """
    # use logsumexp to compute log of sum of exponentials
    return logsumexp(logweights) - np.log(len(logweights)) + logZ

def ESS(logweights: np.ndarray):
    """
    Effective sample size: 1 / sum(w_i^2), where weights normalized to sum=1.
    weights: shape (N,)
    """
    # use logsumexp to compute log of sum of exponentials
    return 1.0 / np.exp(logsumexp(2 * logweights)) 


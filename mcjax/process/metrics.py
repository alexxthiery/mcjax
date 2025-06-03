import jax.numpy as jnp
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
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

def two_wasserstein(x_samples: np.ndarray, y_samples: np.ndarray):
    if x_samples.ndim == 1 or x_samples.shape[1] == 1:
        return wasserstein_distance(x_samples.flatten(), y_samples.flatten())
    else:
        raise NotImplementedError("Multidimensional Wasserstein not implemented")

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


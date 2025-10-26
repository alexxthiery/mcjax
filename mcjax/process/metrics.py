import jax.numpy as jnp
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.special import logsumexp 
from pykeops.torch import LazyTensor
import pykeops.torch as keops
import torch
import tqdm

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

def sinkhorn_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    p: int = 2,
    eps: float = 1e-3,
    max_iters: int = 100,
    stop_thresh: float = 1e-5,
    verbose: bool = False,
    n_max: int | None = None,
    w_x: torch.Tensor | None = None,
    w_y: torch.Tensor | None = None,
):
    """
    Compute the entropy-regularized p-Wasserstein distance between two point clouds
    using the Sinkhorn scaling algorithm (GPU-compatible).

    Adapted from https://github.com/fwilliams/scalable-pytorch-sinkhorn

    Args:
        x: [n, d] tensor of samples
        y: [m, d] tensor of samples
        p: exponent in the p-Wasserstein metric (default 2)
        eps: entropy regularization strength
        max_iters: number of Sinkhorn iterations
        stop_thresh: early stopping threshold
        verbose: whether to show iteration progress
        n_max: optional truncation of number of samples
        w_x, w_y: optional weight vectors (default uniform)
    Returns:
        distance: scalar torch tensor, approximate Wasserstein distance
        corr_x2y: [n,] indices of nearest correspondences in y
        corr_y2x: [m,] indices of nearest correspondences in x
    """

    if not isinstance(p, int) or p <= 0:
        raise TypeError(f"p must be an integer greater than 0, got {p}")
    if eps <= 0:
        raise ValueError("Entropy regularization term eps must be > 0")
    if not isinstance(max_iters, int) or max_iters <= 0:
        raise TypeError(f"max_iters must be an integer > 0, got {max_iters}")
    if not isinstance(stop_thresh, float):
        raise TypeError(f"stop_thresh must be a float, got {stop_thresh}")

    if n_max is not None:
        x, y = x[:n_max], y[:n_max]
        if w_x is not None:
            w_x = w_x[:n_max]
        if w_y is not None:
            w_y = w_y[:n_max]

    if len(x.shape) != 2:
        raise ValueError(f"x must be an [n, d] tensor but got shape {x.shape}")
    if len(y.shape) != 2:
        raise ValueError(f"y must be an [m, d] tensor but got shape {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must match in the last dimension (d) but got "
            f"x.shape = {x.shape}, y.shape={y.shape}"
        )

    # Unify data type and device based on input x
    dtype = x.dtype
    device = x.device
    y = y.to(device=device, dtype=dtype)

    n, d = x.shape
    m = y.shape[0]

    if w_x is not None:
        if w_y is None:
            raise ValueError("If w_x is not None, w_y must also be not None")
        
        w_x = w_x.to(device=device, dtype=dtype).squeeze()
        if len(w_x.shape) != 1 or w_x.shape[0] != n:
            raise ValueError(
                f"w_x must have shape [n,] or [n, 1] (where n={n}), "
                f"but got w_x.shape = {w_x.shape} after squeeze"
            )
    
    if w_y is not None:
        if w_x is None:
            raise ValueError("If w_y is not None, w_x must also be not None")
        
        w_y = w_y.to(device=device, dtype=dtype).squeeze()
        if len(w_y.shape) != 1 or w_y.shape[0] != m:
            raise ValueError(
                f"w_y must have shape [m,] or [m, 1] (where m={m}), "
                f"but got w_y.shape = {w_y.shape} after squeeze"
            )
    
    # Default weights: uniform distributions summing to 1
    if w_x is None:
        w_x = torch.ones(n, device=device, dtype=dtype) / n
        w_y = torch.ones(m, device=device, dtype=dtype) / m


    # Check that weights sum to the same value 
    sum_w_x = w_x.sum().item()
    sum_w_y = w_y.sum().item()
    if abs(sum_w_x - sum_w_y) > 1e-5:
        raise ValueError(
            f"Weights w_x and w_y must sum to the same value, "
            f"got w_x.sum() = {sum_w_x} and w_y.sum() = {sum_w_y} "
            f"(absolute difference = {abs(sum_w_x - sum_w_y)})"
        )

    x_i = keops.Vi(x)  
    y_j = keops.Vj(y)  
    
    if p == 1:
        M_ij = (x_i - y_j).abs().sum(dim=2)  
    else:
        M_ij = ((x_i - y_j) ** p).sum(dim=2) ** (1.0 / p)  

    log_a = torch.log(w_x)  
    log_b = torch.log(w_y)  

    u = torch.zeros_like(w_x)
    v = eps * torch.log(w_y)  

    u_i = keops.Vi(u.unsqueeze(-1))
    v_j = keops.Vj(v.unsqueeze(-1))

    pbar = tqdm.trange(max_iters) if verbose else range(max_iters)

    for _ in pbar:
        u_prev = u
        v_prev = v

        # Update u
        summand_u = (-M_ij + v_j) / eps
        u = eps * (log_a - summand_u.logsumexp(dim=1).squeeze())
        u_i = keops.Vi(u.unsqueeze(-1))

        # Update v
        summand_v = (-M_ij + u_i) / eps
        v = eps * (log_b - summand_v.logsumexp(dim=0).squeeze())
        v_j = keops.Vj(v.unsqueeze(-1))

        # Check for convergence
        max_err_u = torch.max(torch.abs(u_prev - u))
        max_err_v = torch.max(torch.abs(v_prev - v))
        max_err = max(max_err_u, max_err_v).item()
        
        if verbose:
            pbar.set_postfix({"Current Max Error": max_err})
        if max_err < stop_thresh:
            break
    

    P_ij = ((-M_ij + u_i + v_j) / eps).exp()


    distance = (P_ij * M_ij).sum(dim=1).sum()
    return distance
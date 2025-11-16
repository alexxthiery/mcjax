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
from typing import Optional

import pylab as pl
import ot

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
    Compute the 2-Wasserstein distance between two empirical distributions
    using the Python Optimal Transport (POT) library.
    
    Args:
        x_samples: (n_samples,) or (n_samples, n_features) array
                   Samples from the first distribution.
        y_samples: (m_samples,) or (m_samples, n_features) array
                   Samples from the second distribution.

    Returns:
        The 2-Wasserstein distance.
    """
    
    # Ensure inputs are numpy arrays
    x = np.asarray(x_samples)
    y = np.asarray(y_samples)

    # --- 1D Case (Fast) ---
    if x.ndim == 1 and y.ndim == 1:
        return ot.wasserstein_1d(x, y, p=2)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dimensions must match: "
                         f"{x.shape[1]} != {y.shape[1]}")

    n = x.shape[0]
    m = y.shape[0]

    # Create uniform weight vectors for the empirical distributions
    a = np.ones(n) / n
    b = np.ones(m) / m

    # Compute the cost matrix: squared Euclidean distance
    # M_ij = ||x_i - y_j||^2
    M = ot.dist(x, y, metric='sqeuclidean')

    w2_squared = ot.emd2(a, b, M,numItermax=200000)

    # Return the 2-Wasserstein distance
    return np.sqrt(w2_squared)

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
    n_max: Optional[int] = None,
    w_x: Optional[torch.Tensor] = None,
    w_y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the entropy-regularized p-Wasserstein distance between two point clouds
    using the Sinkhorn scaling algorithm with KeOps LazyTensors.

    Args:
        x: [n, d] tensor of samples
        y: [m, d] tensor of samples
        p: exponent in the p-Wasserstein metric (default 2)
        eps: entropy regularization strength
        max_iters: number of Sinkhorn iterations
        stop_thresh: early stopping threshold on dual potentials
        verbose: whether to show iteration progress
        n_max: optional truncation of number of samples (applied to both x and y)
        w_x, w_y: optional weight vectors (default uniform, must sum to same value)

    Returns:
        distance: scalar torch tensor, approximate p-Wasserstein distance
                  (with entropic regularization eps)
    """

    # --------- Basic checks ---------
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

    if x.ndim != 2:
        raise ValueError(f"x must be an [n, d] tensor but got shape {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"y must be an [m, d] tensor but got shape {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must match in the last dimension (d) but got "
            f"x.shape = {x.shape}, y.shape={y.shape}"
        )

    # --------- Device / dtype handling ---------
    dtype = x.dtype
    device = x.device
    y = y.to(device=device, dtype=dtype)

    n, d = x.shape
    m = y.shape[0]

    # --------- Weights handling ---------
    if w_x is not None:
        if w_y is None:
            raise ValueError("If w_x is not None, w_y must also be not None")

        w_x = w_x.to(device=device, dtype=dtype).squeeze()
        if w_x.ndim != 1 or w_x.shape[0] != n:
            raise ValueError(
                f"w_x must have shape [n,] or [n, 1] (where n={n}), "
                f"but got w_x.shape = {w_x.shape} after squeeze"
            )

    if w_y is not None:
        if w_x is None:
            raise ValueError("If w_y is not None, w_x must also be not None")

        w_y = w_y.to(device=device, dtype=dtype).squeeze()
        if w_y.ndim != 1 or w_y.shape[0] != m:
            raise ValueError(
                f"w_y must have shape [m,] or [m, 1] (where m={m}), "
                f"but got w_y.shape = {w_y.shape} after squeeze"
            )

    # Default: uniform weights, normalized to sum 1
    if w_x is None:
        w_x = torch.ones(n, device=device, dtype=dtype) / n
        w_y = torch.ones(m, device=device, dtype=dtype) / m

    # Check total mass compatibility
    sum_w_x = w_x.sum().item()
    sum_w_y = w_y.sum().item()
    if abs(sum_w_x - sum_w_y) > 1e-5:
        raise ValueError(
            f"Weights w_x and w_y must sum to the same value, "
            f"got w_x.sum() = {sum_w_x} and w_y.sum() = {sum_w_y} "
            f"(absolute difference = {abs(sum_w_x - sum_w_y)})"
        )

    # --------- Build KeOps LazyTensors for cost matrix ---------
    # x_i: (n, 1, d), y_j: (1, m, d)
    x_i = LazyTensor(x.view(n, 1, d))
    y_j = LazyTensor(y.view(1, m, d))

    # Cost matrix C_ij (we follow your previous convention: metric, not metric^p)
    if p == 1:
        C_ij = (x_i - y_j).abs().sum(dim=2)        
    else:
        C_ij = ((x_i - y_j) ** p).sum(dim=2) ** (1.0 / p) 

    # --------- Initialize dual variables (log-domain Sinkhorn) ---------
    log_a = torch.log(w_x)  
    log_b = torch.log(w_y)  

    u = torch.zeros_like(w_x)             
    v = eps * torch.log(w_y)              

    pbar = tqdm.trange(max_iters) if verbose else range(max_iters)

    for _ in pbar:
        u_prev = u.clone()
        v_prev = v.clone()

        # Update u:
        # u_i = eps * ( log_a - logsumexp_j( ( -C_ij + v_j ) / eps ) )
        v_j = LazyTensor(v.view(1, m, 1))                
        expr_u = (-C_ij + v_j) / eps                     
        lse_u = expr_u.logsumexp(dim=1).view(n)           
        u = eps * (log_a - lse_u)

        # Update v:
        # v_j = eps * ( log_b - logsumexp_i( ( -C_ij + u_i ) / eps ) )
        u_i = LazyTensor(u.view(n, 1, 1))                
        expr_v = (-C_ij + u_i) / eps                     
        lse_v = expr_v.logsumexp(dim=0).view(m)           
        v = eps * (log_b - lse_v)

        # Convergence check
        max_err_u = torch.max(torch.abs(u_prev - u))
        max_err_v = torch.max(torch.abs(v_prev - v))
        max_err = max(max_err_u, max_err_v).item()

        if verbose:
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"max_err": max_err})
            else:
                print(f"Iter {_+1}: max_err = {max_err:.3e}")

        if max_err < stop_thresh:
            if verbose:
                print(f"Converged after {_+1} iterations with max error {max_err:.3e}")
            break

    # --------- Compute transport plan and distance (KeOps reductions) ---------
    # P_ij = exp( (u_i + v_j - C_ij) / eps )
    u_i = LazyTensor(u.view(n, 1, 1))
    v_j = LazyTensor(v.view(1, m, 1))

    log_P_ij = (u_i + v_j - C_ij) / eps
    P_ij = log_P_ij.exp()  # [n, m] LazyTensor, never fully materialized

    # distance = sum_{i,j} P_ij * C_ij
    # KeOps reductions: first over j (dim=1), then over i (dim=0)
    distance = (P_ij * C_ij).sum(dim=1).sum(dim=0)  # scalar tensor

    return distance


   
"""
Shared fixtures for mcjax tests.

Design principles:
- Small dimensions (dim=2-5) for fast tests
- Few samples (n=10-100) for unit tests
- Deterministic keys for reproducibility
- Simple target distributions with known properties
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr


# =============================================================================
# PRNG Keys
# =============================================================================

@pytest.fixture
def key():
    """Default PRNG key for reproducible tests."""
    return jr.PRNGKey(42)


@pytest.fixture
def key_pair(key):
    """Two independent keys for tests needing multiple sources."""
    return jr.split(key)


# =============================================================================
# Dimensions
# =============================================================================

@pytest.fixture(params=[2, 5])
def dim(request):
    """Parameterized dimension for testing multiple sizes."""
    return request.param


@pytest.fixture
def dim_small():
    """Fixed small dimension for basic tests."""
    return 2


# =============================================================================
# Target Distributions
# =============================================================================

@pytest.fixture
def log_prob_gaussian(dim_small):
    """
    Standard Gaussian log-density.

    Properties:
    - Mode at origin
    - Known mean (0) and variance (1)
    - Gradient: -x
    """
    def log_prob(x):
        return -0.5 * jnp.sum(x ** 2)
    return log_prob


@pytest.fixture
def log_prob_gaussian_correlated():
    """
    Correlated 2D Gaussian with known covariance.

    Covariance: [[1, 0.8], [0.8, 1]]
    """
    cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
    prec = jnp.linalg.inv(cov)
    log_det = jnp.linalg.slogdet(cov)[1]

    def log_prob(x):
        return -0.5 * (x @ prec @ x + log_det + 2 * jnp.log(2 * jnp.pi))
    return log_prob


@pytest.fixture
def log_prob_bimodal():
    """
    Bimodal target: mixture of two Gaussians.

    Modes at (-2, 0) and (2, 0).
    Useful for testing exploration.
    """
    def log_prob(x):
        d1 = jnp.sum((x - jnp.array([-2.0, 0.0])) ** 2)
        d2 = jnp.sum((x - jnp.array([2.0, 0.0])) ** 2)
        return jnp.logaddexp(-0.5 * d1, -0.5 * d2) - jnp.log(2)
    return log_prob


# =============================================================================
# Initial States
# =============================================================================

@pytest.fixture
def x_init(dim_small):
    """Initial position at origin."""
    return jnp.zeros(dim_small)


@pytest.fixture
def xs_init_batch(dim_small, key):
    """Batch of initial positions for multi-chain tests."""
    return jr.normal(key, (10, dim_small))


# =============================================================================
# Covariance Matrices
# =============================================================================

@pytest.fixture
def cov_diag(dim_small):
    """Diagonal covariance (variances)."""
    return jnp.ones(dim_small)


@pytest.fixture
def cov_full(dim_small):
    """Full covariance matrix (identity)."""
    return jnp.eye(dim_small)


@pytest.fixture
def cov_correlated():
    """Correlated 2D covariance matrix."""
    return jnp.array([[1.0, 0.5], [0.5, 1.0]])


# =============================================================================
# Test Utilities
# =============================================================================

def assert_pytree_shape(pytree, expected_leading_shape):
    """Assert all leaves in pytree have expected leading dimensions."""
    leaves = jax.tree_util.tree_leaves(pytree)
    for leaf in leaves:
        assert leaf.shape[:len(expected_leading_shape)] == expected_leading_shape, \
            f"Expected leading shape {expected_leading_shape}, got {leaf.shape}"


def assert_finite(pytree):
    """Assert all values in pytree are finite (no NaN or Inf)."""
    leaves = jax.tree_util.tree_leaves(pytree)
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), f"Non-finite values found: {leaf}"


def assert_deterministic(fn, *args, **kwargs):
    """Assert function gives same output when called twice with same inputs."""
    result1 = fn(*args, **kwargs)
    result2 = fn(*args, **kwargs)

    leaves1 = jax.tree_util.tree_leaves(result1)
    leaves2 = jax.tree_util.tree_leaves(result2)

    for l1, l2 in zip(leaves1, leaves2):
        assert jnp.allclose(l1, l2), "Function is not deterministic"

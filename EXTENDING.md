# Extending mcjax

Recipes for adding custom kernels, distributions, and adapters.

## Adding a New MCMC Kernel

Follow the RWM/MALA pattern: create state, params, stats dataclasses, then bundle into a `MarkovKernel`.

### Example: Hamiltonian Monte Carlo (HMC)

```python
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct
from mcjax.mcmc.core import MarkovKernel


# 1. Define state (must have .x and .log_prob for SMC compatibility)
@struct.dataclass
class HmcState:
    x: jnp.ndarray
    log_prob: jnp.ndarray
    grad: jnp.ndarray


# 2. Define parameters
@struct.dataclass
class HmcParams:
    step_size: float
    n_leapfrog: int
    mass_matrix: jnp.ndarray  # Diagonal mass


# 3. Define per-step statistics
@struct.dataclass
class HmcStats:
    is_accept: jnp.ndarray
    accept_prob: jnp.ndarray


# 4. Define summary aggregation
@struct.dataclass
class HmcSummary:
    acceptance_rate: jnp.ndarray
    n_accepted: jnp.ndarray


def summarize_hmc(stats_traj: HmcStats) -> HmcSummary:
    return HmcSummary(
        acceptance_rate=jnp.mean(stats_traj.is_accept),
        n_accepted=jnp.sum(stats_traj.is_accept),
    )


# 5. Create kernel factory
def create_hmc_kernel(dim: int) -> MarkovKernel:

    def init_state_fn(log_prob, x0):
        value_and_grad = jax.value_and_grad(log_prob)
        lp, g = value_and_grad(x0)
        return HmcState(x=x0, log_prob=lp, grad=g)

    def init_params(step_size: float, n_leapfrog: int = 10, mass: jnp.ndarray = None):
        mass = mass if mass is not None else jnp.ones(dim)
        return HmcParams(step_size=step_size, n_leapfrog=n_leapfrog, mass_matrix=mass)

    def leapfrog(x, p, grad, log_prob_fn, params):
        """Single leapfrog integration."""
        value_and_grad = jax.value_and_grad(log_prob_fn)
        eps = params.step_size
        M_inv = 1.0 / params.mass_matrix

        # Half step momentum
        p = p + 0.5 * eps * grad

        # Full step position
        x = x + eps * M_inv * p

        # Recompute gradient
        lp, grad = value_and_grad(x)

        # Half step momentum
        p = p + 0.5 * eps * grad

        return x, p, grad, lp

    def step(key, log_prob, state, params):
        value_and_grad = jax.value_and_grad(log_prob)

        # Sample momentum
        key_p, key_accept = jr.split(key)
        p0 = jr.normal(key_p, shape=state.x.shape) * jnp.sqrt(params.mass_matrix)

        # Leapfrog integration
        x, p, grad, lp = state.x, p0, state.grad, state.log_prob
        for _ in range(params.n_leapfrog):
            x, p, grad, lp = leapfrog(x, p, grad, log_prob, params)

        # Hamiltonian
        M_inv = 1.0 / params.mass_matrix
        H_current = -state.log_prob + 0.5 * jnp.sum(p0**2 * M_inv)
        H_proposed = -lp + 0.5 * jnp.sum(p**2 * M_inv)

        # Accept/reject
        log_ratio = H_current - H_proposed
        accept_prob = jnp.minimum(1.0, jnp.exp(log_ratio))
        u = jr.uniform(key_accept)
        is_accept = u < accept_prob

        x_new = jnp.where(is_accept, x, state.x)
        lp_new = jnp.where(is_accept, lp, state.log_prob)
        grad_new = jnp.where(is_accept, grad, state.grad)

        new_state = HmcState(x=x_new, log_prob=lp_new, grad=grad_new)
        stats = HmcStats(is_accept=is_accept, accept_prob=accept_prob)

        return new_state, stats

    return MarkovKernel(
        init_state_fn=init_state_fn,
        init_params=init_params,
        step=step,
        summarize=summarize_hmc,
    )
```

### Key Requirements

1. **State must have `.x` and `.log_prob`** for SMC compatibility
2. **`step` must be pure**: no side effects, deterministic given key
3. **Use `flax.struct.dataclass`** for all dataclasses (JAX-compatible)
4. **Summarize reduces over the time axis** of stats trajectory

## Adding a New Distribution

Implement the `DistributionLike` protocol.

### Example: Multivariate Student-t

```python
from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct


@struct.dataclass
class StudentTParams:
    mu: jnp.ndarray
    scale: jnp.ndarray  # Cholesky of scale matrix
    df: float


@struct.dataclass
class StudentT:
    dim: int

    @classmethod
    def create(cls, dim: int) -> "StudentT":
        return cls(dim=dim)

    def init_params(
        self,
        mu: Optional[jnp.ndarray] = None,
        scale: Optional[jnp.ndarray] = None,
        df: float = 3.0,
    ) -> StudentTParams:
        mu = mu if mu is not None else jnp.zeros(self.dim)
        scale = scale if scale is not None else jnp.eye(self.dim)
        return StudentTParams(mu=mu, scale=scale, df=df)

    def log_prob(self, params: StudentTParams, x: jnp.ndarray) -> jnp.ndarray:
        """Log-density of multivariate Student-t."""
        from jax.scipy.special import gammaln

        d = self.dim
        nu = params.df
        y = jnp.linalg.solve(params.scale, x - params.mu)
        quad = jnp.sum(y**2)

        log_det = jnp.sum(jnp.log(jnp.diag(params.scale)))

        log_p = (
            gammaln((nu + d) / 2)
            - gammaln(nu / 2)
            - 0.5 * d * jnp.log(nu * jnp.pi)
            - log_det
            - 0.5 * (nu + d) * jnp.log(1 + quad / nu)
        )
        return log_p

    def sample(
        self, params: StudentTParams, key: jax.Array, n_samples: int
    ) -> jnp.ndarray:
        """Sample via: X = mu + scale @ (Z / sqrt(chi2/nu))."""
        key_z, key_chi2 = jr.split(key)
        z = jr.normal(key_z, shape=(n_samples, self.dim))
        chi2 = jr.chisquare(key_chi2, params.df, shape=(n_samples,))
        scale_factor = jnp.sqrt(params.df / chi2)[:, None]
        return params.mu + (z * scale_factor) @ params.scale.T
```

### Key Requirements

1. **`dim` attribute** is required
2. **`log_prob(params, x)`** takes a single point, returns scalar
3. **`sample`** is optional but useful for SMC initialization

## Adding an Adaptation Function

Adaptation functions modify kernel parameters based on current particles.

### Interface

```python
def adapt_fn(
    kernel: MarkovKernel,
    log_prob_t: Callable[[Array], Array],
    xs: Array,           # (N, dim) current particles
    params: Any,         # Current kernel params
    key: jax.Array,
) -> Any:  # New kernel params
```

### Example: Covariance Adaptation

```python
def adapt_cov(kernel, log_prob_t, xs, params, key):
    """Adapt proposal covariance from particle cloud."""
    cov = jnp.cov(xs.T) + 1e-5 * jnp.eye(xs.shape[1])
    return kernel.init_params(step_size=params.step_size, cov=cov)
```

### Example: Combined Step Size + Covariance

```python
from mcjax.mcmc.rwm import rwm_adapt

def my_adapt(kernel, log_prob_t, xs, params, key):
    # rwm_adapt handles both covariance estimation and step size tuning
    return rwm_adapt(
        kernel=kernel,
        log_prob=log_prob_t,
        xs=xs,
        params=params,
        key=key,
        n_steps=5,
        acceptance_threshold_up=0.5,
        acceptance_threshold_down=0.1,
    )
```

## Testing Your Extension

Use the notebooks in `notebooks/` as templates:

```python
# Quick smoke test
kernel = create_hmc_kernel(dim=2)
params = kernel.init_params(step_size=0.1, n_leapfrog=10)

target = Banana2D.create()
x_init = jnp.zeros(2)

out = kernel.run_mcmc(
    log_prob=target.log_prob,
    x_init=x_init,
    params=params,
    key=jr.key(0),
    n_samples=100,
)

print(f"Acceptance: {out.summary.acceptance_rate:.2%}")
print(f"Final x: {out.traj.x[-1]}")
```

Check:
1. Acceptance rate is reasonable (20-80% typically)
2. Samples explore the target (plot them)
3. Works with `run_mcmc_batch` (vmap)
4. Works inside `run_smc` if intended for SMC use

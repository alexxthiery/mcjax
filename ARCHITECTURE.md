# Architecture

Core design patterns and conventions in mcjax.

## Design Philosophy

mcjax separates **state**, **parameters**, and **logic** into distinct components:

- **State**: Mutable information that changes each step (position, log-prob, gradients)
- **Parameters**: Tunable but fixed during a run (step size, covariance)
- **Logic**: Pure functions bundled in a kernel object

This separation enables:
- JIT compilation (state/params are PyTrees, logic is static)
- Easy adaptation (swap parameters without rebuilding state)
- Clean `vmap` over independent chains

## Core Protocols

### MarkovKernel

The `MarkovKernel` dataclass bundles four functions:

```python
@struct.dataclass
class MarkovKernel:
    init_state_fn: Callable[[log_prob, x_init], State]
    init_params: Callable[..., Params]
    step: Callable[[key, log_prob, state, params], (State, Stats)]
    summarize: Optional[Callable[[Stats], Summary]]
```

**Contract:**
- `step` is a pure function: same inputs always produce same outputs (given key)
- State must be a `flax.struct.dataclass` with at least `.x` and `.log_prob` fields (required for SMC)
- Stats capture per-step diagnostics (acceptance, squared jump)
- Summarize aggregates stats over a trajectory

### DistributionLike

Protocol for probability distributions (targets):

```python
class DistributionLike(Protocol):
    dim: int
    def log_prob(self, params, x) -> scalar: ...

    # Optional:
    def sample(self, params, key, n_samples) -> (n_samples, dim): ...
```

**Contract:**
- `log_prob` takes a single point `x` of shape `(dim,)` and returns a scalar
- Use `jax.vmap(dist.log_prob, in_axes=(None, 0))` for batched evaluation
- Parameters are separate from the distribution object

## State Conventions

### MCMC States

All MCMC states must have:
```python
@struct.dataclass
class SomeState:
    x: jnp.ndarray        # Current position, shape (dim,)
    log_prob: jnp.ndarray # Log-density at x, scalar
    # ... kernel-specific fields (e.g., grad for MALA)
```

This interface is required by SMC, which extracts final positions after mutation.

### MCMC Parameters

Parameters control proposal behavior:
```python
@struct.dataclass
class SomeParams:
    step_size: float
    scale: jnp.ndarray  # Cholesky or sqrt-diag of proposal covariance
    # ... kernel-specific fields
```

### SMC State

SMC carries particle clouds and histories:
```python
@struct.dataclass
class SMCState:
    xs: Array           # (N, dim) particle positions
    log_weights: Array  # (N,) normalized log-weights
    log_p0: Array       # (N,) base log-densities
    log_p1: Array       # (N,) target log-densities
    temp: float         # Current temperature in [0, 1]
    step: int
    key: jax.Array
    kernel_params: Any
    # ... histories for diagnostics
```

## Composition Patterns

### Running MCMC

```python
kernel = create_rwm_kernel(dim=2, cov_type="diag")
params = kernel.init_params(step_size=0.1)

# Single chain
out = kernel.run_mcmc(log_prob=target, x_init=x0, params=params, key=key, n_samples=1000)

# Batched chains (vmap internally)
out = kernel.run_mcmc_batch(log_prob=target, xs_init=xs, params=params, key=key, n_samples=1000)
```

### SMC with MCMC Mutation

```python
from mcjax.smc.core import run_smc
from mcjax.mcmc.rwm import create_rwm_kernel, rwm_adapt

kernel = create_rwm_kernel(dim=2)
kernel_params = kernel.init_params(step_size=0.5)

out = run_smc(
    log_prob_base=base.log_prob,
    log_prob_target=target.log_prob,
    xs_init=xs_init,          # (N, dim) from base distribution
    kernel=kernel,
    kernel_params=kernel_params,
    n_mcmc_steps=10,
    key=key,
    adapt_kernel_fn=rwm_adapt,  # Optional: adapt step size per temperature
)
```

## JAX Compatibility

All components are designed for JAX transforms:

- **JIT**: Kernels use `lax.scan` for trajectories, `lax.while_loop` for SMC
- **vmap**: `run_mcmc_batch` vmaps over initial states and keys
- **grad**: Log-prob functions are differentiable (used by MALA)

Key rule: functions are static (not traced), state/params are dynamic (PyTrees).

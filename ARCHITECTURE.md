# Architecture

Core design patterns and conventions in mcjax.

## Design Philosophy

mcjax separates **state**, **parameters**, and **logic** into distinct components:

- **State**: Mutable information that changes each step (position, log-prob, gradients)
- **Parameters**: Tunable but fixed during a run (step size, covariance)
- **Logic**: Pure functions in kernel modules (step, init_state, make_params)

This separation enables:
- JIT compilation (state/params are PyTrees, logic is static)
- Easy adaptation (swap parameters without rebuilding state)
- Clean `vmap` over independent chains

## MCMC Interface

Each kernel is a **module** (e.g., `mcjax.mcmc.rwm`) exporting:

| Export | Description |
|--------|-------------|
| `State` | Dataclass for chain state (position, log_prob, ...) |
| `Params` | Dataclass for kernel parameters (step_size, scale, ...) |
| `Stats` | Dataclass for per-step diagnostics |
| `step(key, log_prob, state, params)` | One Markov transition |
| `init_state(log_prob, x)` | Create initial state from position |
| `make_params(step_size, cov, ...)` | Create params from covariance |
| `sample(log_prob, x_init, key, n_samples, ...)` | Convenience one-liner |
| `adapt(log_prob, xs, params, key, ...)` | Adapt step_size and covariance |

### The Step Function Contract

```python
def step(
    key: jax.Array,
    log_prob: Callable[[Array], Array],
    state: State,
    params: Params,
) -> tuple[State, Stats]:
    ...
```

**Properties:**
- Pure function: same inputs always produce same outputs (given key)
- `log_prob` takes a single point `(dim,)` and returns a scalar
- State and Stats are kernel-specific dataclasses

### State Convention

All MCMC states must have:
```python
@struct.dataclass
class State:
    x: jnp.ndarray        # Current position, shape (dim,)
    log_prob: jnp.ndarray # Log-density at x, scalar
    # ... kernel-specific fields (e.g., grad for MALA)
```

This interface is required by SMC, which extracts positions and log-probs.

### Params Convention

Parameters control proposal behavior:
```python
@struct.dataclass
class Params:
    step_size: float      # Required for adaptation
    scale: jnp.ndarray    # Cholesky or sqrt-diag of proposal covariance
    # ... kernel-specific fields
```

The `scale` field is derived from covariance by `make_params`:
- 1D cov -> `scale = sqrt(cov)` (diagonal)
- 2D cov -> `scale = cholesky(cov)` (full)

## Drivers

Standalone functions for running chains:

```python
from mcjax.mcmc import run_mcmc, run_mcmc_batch

# Single chain
output = run_mcmc(step_fn, log_prob, state, params, key, n_samples)

# Multiple chains (vmapped)
output = run_mcmc_batch(step_fn, init_state_fn, log_prob, xs_init, params, key, n_samples)
```

### MCMCOutput

```python
@struct.dataclass
class MCMCOutput:
    states: State  # Trajectory, shape (n_samples, ...)
    stats: Stats   # Per-step diagnostics, shape (n_samples, ...)
```

## Usage Patterns

### Running RWM

```python
from mcjax.mcmc import rwm

# Convenience API (recommended)
output = rwm.sample(log_prob, x_init, key, n_samples=1000, step_size=0.1)
samples = output.states.x  # shape (1000, dim)

# Low-level API
state = rwm.init_state(log_prob, x_init)
params = rwm.make_params(step_size=0.1, cov=jnp.ones(dim))
output = run_mcmc(rwm.step, log_prob, state, params, key, n_samples=1000)
```

### Running MALA

```python
from mcjax.mcmc import mala

output = mala.sample(log_prob, x_init, key, n_samples=1000, step_size=0.01)
```

### Adapting Parameters

```python
# Adapt step_size (and optionally covariance) from samples
adapted_params = rwm.adapt(log_prob, samples, initial_params, key)
```

### SMC with MCMC Mutation

```python
from mcjax.smc.core import run_smc
from mcjax.mcmc import rwm

params = rwm.make_params(step_size=0.5, cov=jnp.ones(dim))

out = run_smc(
    log_prob_base=base.log_prob,
    log_prob_target=target.log_prob,
    xs_init=xs_init,
    step_fn=rwm.step,
    init_state_fn=rwm.init_state,
    kernel_params=params,
    n_mcmc_steps=10,
    key=key,
    adapt_fn=lambda lp, xs, p, k: rwm.adapt(lp, xs, p, k),
)
```

## DistributionLike Protocol

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

## JAX Compatibility

All components are designed for JAX transforms:

- **JIT**: Drivers use `lax.scan` for trajectories, `lax.while_loop` for SMC
- **vmap**: `run_mcmc_batch` vmaps over initial states and keys
- **grad**: Log-prob functions are differentiable (used by MALA)

Key rule: functions are static (not traced), state/params are dynamic (PyTrees).

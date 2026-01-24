# Usage Guide

Practical examples for using mcjax.

## MCMC Sampling

### Random Walk Metropolis

```python
import jax.numpy as jnp
import jax.random as jr
from mcjax.mcmc import rwm
from mcjax.proba.banana2d import Banana2D

# Create target
target = Banana2D.create()
key = jr.key(42)
x_init = jnp.zeros(2)

# Convenience API (recommended)
output = rwm.sample(
    log_prob=target.log_prob,
    x_init=x_init,
    key=key,
    n_samples=5000,
    step_size=0.5,
)

samples = output.states.x  # shape: (5000, 2)
acceptance_rate = jnp.mean(output.stats.is_accept)
print(f"Acceptance rate: {acceptance_rate:.2%}")
```

### MALA (Langevin)

```python
from mcjax.mcmc import mala

output = mala.sample(
    log_prob=target.log_prob,
    x_init=x_init,
    key=key,
    n_samples=5000,
    step_size=0.1,
    grad_clip=10.0,  # optional: clip gradients
)
```

### Low-Level API

```python
from mcjax.mcmc import rwm, run_mcmc

# Create state and params explicitly
state = rwm.init_state(target.log_prob, x_init)
params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))

# Run chain
output = run_mcmc(rwm.step, target.log_prob, state, params, key, n_samples=5000)
```

### Multiple Chains

```python
from mcjax.mcmc import rwm, run_mcmc_batch

n_chains = 8
xs_init = jr.normal(jr.key(0), (n_chains, 2))
params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))

output = run_mcmc_batch(
    step_fn=rwm.step,
    init_state_fn=rwm.init_state,
    log_prob=target.log_prob,
    xs_init=xs_init,
    params=params,
    key=key,
    n_samples=1000,
)

# output.states.x has shape (8, 1000, 2)
```

### Full Covariance Proposals

```python
# Estimate covariance from pilot run
pilot_samples = output.states.x[:, -500:, :].reshape(-1, 2)
cov = jnp.cov(pilot_samples.T)

# Use full covariance
params_full = rwm.make_params(step_size=0.5, cov=cov)
```

### Adapting Parameters

```python
# Adapt step_size and covariance from samples
samples = output.states.x.reshape(-1, 2)  # flatten chains
adapted_params = rwm.adapt(
    log_prob=target.log_prob,
    xs=samples,
    params=params,
    key=key,
)
print(f"Adapted step_size: {adapted_params.step_size:.3f}")
```

## Sequential Monte Carlo

### Basic SMC

```python
from mcjax.smc.core import run_smc
from mcjax.mcmc import rwm
from mcjax.proba.gauss import GaussianDiag

# Base: standard Gaussian
base = GaussianDiag.create(dim=2)
base_params = base.init_params()

# Target: some complex distribution
target = Banana2D.create()

# Initial particles from base
N = 500
xs_init = base.sample(base_params, jr.key(0), N)

# MCMC kernel parameters
kernel_params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))

# Run SMC
out = run_smc(
    log_prob_base=lambda x: base.log_prob(base_params, x),
    log_prob_target=target.log_prob,
    xs_init=xs_init,
    step_fn=rwm.step,
    init_state_fn=rwm.init_state,
    kernel_params=kernel_params,
    n_mcmc_steps=10,
    key=jr.key(1),
)

print(f"Log normalizing constant: {out.logZ:.2f}")
print(f"Temperature steps: {out.N_temperature}")
```

### SMC with Adaptation

```python
# Define adaptation function
def adapt_fn(log_prob, xs, params, key):
    return rwm.adapt(log_prob, xs, params, key, n_iters=5)

out = run_smc(
    log_prob_base=lambda x: base.log_prob(base_params, x),
    log_prob_target=target.log_prob,
    xs_init=xs_init,
    step_fn=rwm.step,
    init_state_fn=rwm.init_state,
    kernel_params=kernel_params,
    n_mcmc_steps=10,
    key=jr.key(1),
    adapt_fn=adapt_fn,
    ess_threshold=0.5,
)
```

### Fixed Temperature Ladder

```python
temps = jnp.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

out = run_smc(
    log_prob_base=lambda x: base.log_prob(base_params, x),
    log_prob_target=target.log_prob,
    xs_init=xs_init,
    step_fn=rwm.step,
    init_state_fn=rwm.init_state,
    kernel_params=kernel_params,
    n_mcmc_steps=10,
    key=jr.key(1),
    temp_ladder=temps,
)
```

## Working with Distributions

### Using Built-in Targets

```python
from mcjax.proba.banana2d import Banana2D
from mcjax.proba.neal_funnel import NealFunnel
from mcjax.proba.student import StudentT

banana = Banana2D.create(b=0.1)
funnel = NealFunnel.create(dim=10)
student = StudentT.create(dim=2, df=3)

# All follow the same interface
x = jnp.zeros(2)
log_p = banana.log_prob(x)
```

### Custom Target Distribution

```python
def my_log_prob(x):
    # Rosenbrock function (as negative log-density)
    return -((1 - x[0])**2 + 100*(x[1] - x[0]**2)**2)

# Use directly with MCMC
output = rwm.sample(
    log_prob=my_log_prob,
    x_init=jnp.zeros(2),
    key=key,
    n_samples=5000,
    step_size=0.01,
)
```

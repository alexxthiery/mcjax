# Usage Guide

Practical examples for using mcjax.

## MCMC Sampling

### Random Walk Metropolis

```python
import jax.numpy as jnp
import jax.random as jr
from mcjax.mcmc.rwm import create_rwm_kernel
from mcjax.proba.banana2d import Banana2D

# Create target and kernel
target = Banana2D.create()
kernel = create_rwm_kernel(dim=2, cov_type="diag")

# Initialize
params = kernel.init_params(step_size=0.5)
x_init = jnp.zeros(2)
key = jr.key(42)

# Run single chain
out = kernel.run_mcmc(
    log_prob=target.log_prob,
    x_init=x_init,
    params=params,
    key=key,
    n_samples=5000,
)

samples = out.traj.x  # shape: (5000, 2)
print(f"Acceptance rate: {out.summary.acceptance_rate:.2%}")
```

### MALA (Langevin)

```python
from mcjax.mcmc.mala import create_mala_kernel

kernel = create_mala_kernel(dim=2, cov_type="diag")
params = kernel.init_params(step_size=0.1, grad_clip_norm=10.0)

out = kernel.run_mcmc(
    log_prob=target.log_prob,
    x_init=x_init,
    params=params,
    key=key,
    n_samples=5000,
)
```

### Multiple Chains

```python
n_chains = 8
xs_init = jr.normal(jr.key(0), (n_chains, 2))

out = kernel.run_mcmc_batch(
    log_prob=target.log_prob,
    xs_init=xs_init,
    params=params,
    key=key,
    n_samples=1000,
)

# out.traj.x has shape (8, 1000, 2)
# out.summary.acceptance_rate has shape (8,) - per chain
```

### Full Covariance Proposals

```python
# Estimate covariance from pilot run
pilot_samples = out.traj.x[-500:]  # last 500 samples
cov = jnp.cov(pilot_samples.T)

kernel_full = create_rwm_kernel(dim=2, cov_type="full")
params_full = kernel_full.init_params(step_size=0.5, cov=cov)
```

## Sequential Monte Carlo

### Basic SMC

```python
from mcjax.smc.core import run_smc
from mcjax.mcmc.rwm import create_rwm_kernel
from mcjax.proba.gauss import GaussianDiag

# Base: standard Gaussian
base = GaussianDiag.create(dim=2)
base_params = base.init_params()

# Target: some complex distribution
target = Banana2D.create()

# Initial particles from base
N = 500
xs_init = base.sample(base_params, jr.key(0), N)

# MCMC kernel for mutation
kernel = create_rwm_kernel(dim=2, cov_type="diag")
kernel_params = kernel.init_params(step_size=0.5)

# Run SMC
out = run_smc(
    log_prob_base=lambda x: base.log_prob(base_params, x),
    log_prob_target=target.log_prob,
    xs_init=xs_init,
    kernel=kernel,
    kernel_params=kernel_params,
    n_mcmc_steps=10,
    key=jr.key(1),
)

print(f"Log normalizing constant: {out.logZ:.2f}")
print(f"Temperature steps: {out.N_temperature}")
```

### SMC with Adaptation

```python
from mcjax.mcmc.rwm import rwm_adapt

out = run_smc(
    log_prob_base=lambda x: base.log_prob(base_params, x),
    log_prob_target=target.log_prob,
    xs_init=xs_init,
    kernel=kernel,
    kernel_params=kernel_params,
    n_mcmc_steps=10,
    key=jr.key(1),
    adapt_kernel_fn=rwm_adapt,  # Adapts step size at each temperature
    ess_threshold=0.5,          # Resample when ESS < 50%
)
```

### Fixed Temperature Ladder

```python
temps = jnp.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

out = run_smc(
    log_prob_base=lambda x: base.log_prob(base_params, x),
    log_prob_target=target.log_prob,
    xs_init=xs_init,
    kernel=kernel,
    kernel_params=kernel_params,
    n_mcmc_steps=10,
    key=jr.key(1),
    temp_ladder=temps,  # Deterministic schedule
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
from mcjax.proba.distribution import make_distribution

def my_log_prob(params, x):
    # Rosenbrock function (as negative log-density)
    return -((1 - x[0])**2 + 100*(x[1] - x[0]**2)**2)

dist, params = make_distribution(my_log_prob, dim=2)

# Use with MCMC
out = kernel.run_mcmc(
    log_prob=dist.log_prob_only,
    x_init=jnp.zeros(2),
    params=kernel_params,
    key=key,
    n_samples=5000,
)
```

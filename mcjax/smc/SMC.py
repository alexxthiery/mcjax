import jax
import jax.numpy as jnp
from flax import struct
from typing import Callable, Any, Tuple
from mcjax.mcmc.core import run_mcmc, MarkovKernel
from mcjax.util.weights import select_next_temperature



@struct.dataclass
class SMCState:
    particles: jnp.ndarray
    log_weights: jnp.ndarray
    log_p0: jnp.ndarray
    log_p1: jnp.ndarray
    lambda_prev: float
    temperatures: list
    ess_history: list
    diagnostics: list
    logZ_estimates: list
    key: jax.Array


@struct.dataclass
class SMCOutput:
    particles: jnp.ndarray
    log_weights: jnp.ndarray
    temperatures: jnp.ndarray
    ess_history: jnp.ndarray
    diagnostics: Any
    logZ_estimates: jnp.ndarray


@struct.dataclass
class SMCKernel:
    """
    A container for a state-aware, stateless SMC kernel constructor.

    Attributes:
        create_kernel: A pure function that takes the current SMC state and returns a
                       (MarkovKernel, kernel_params) pair to be used in MCMC mutation.
    """
    create_kernel: Callable[[SMCState], Tuple[MarkovKernel, Any]]


def run_smc(
    *,
    base_logdensity: Callable[[jnp.ndarray], jnp.ndarray],
    target_logdensity: Callable[[jnp.ndarray], jnp.ndarray],
    x_init: jnp.ndarray,
    smc_kernel: SMCKernel,
    ess_threshold: float,
    n_mcmc_steps: int,
    key: jax.Array,
) -> SMCOutput:
    """
    Run a Sequential Monte Carlo (SMC) sampler with adaptive tempering and modular MCMC mutation.

    This implementation is stateless, JAX-native, and compatible with JIT and scan transforms.

    Args:
        base_logdensity: Log-density function of the base distribution.
        target_logdensity: Log-density function of the target distribution.
        x_init: Initial particles drawn from the base distribution, shape (N, D).
        smc_kernel: SMCKernel object with a create_kernel function for constructing mutation kernels.
        ess_threshold: ESS threshold for adaptive temperature selection.
        n_mcmc_steps: Number of MCMC steps per temperature stage.
        key: JAX PRNGKey.

    Returns:
        SMCOutput: A container with final particles, weights, temperature schedule,
                   ESS history, diagnostics, and log-normalizing constant estimates.
    """
    N = x_init.shape[0]

    log_p0 = base_logdensity(x_init)
    log_p1 = target_logdensity(x_init)

    # Initialize SMC state
    state = SMCState(
        particles=x_init,
        log_weights=jnp.zeros(N),
        log_p0=log_p0,
        log_p1=log_p1,
        lambda_prev=0.0,
        temperatures=[0.0],
        ess_history=[],
        diagnostics=[],
        logZ_estimates=[],
        key=key,
    )

    def cond_fn(state: SMCState):
        return state.lambda_prev < 1.0

    def body_fn(state: SMCState):
        # Select next temperature
        lambda_next = select_next_temperature(
            state.log_p0, state.log_p1, state.log_weights, state.lambda_prev, ess_threshold
        )
        delta_lambda = lambda_next - state.lambda_prev

        # Update importance weights and compute ESS
        log_w = state.log_weights + delta_lambda * (state.log_p1 - state.log_p0)
        max_log_w = jnp.max(log_w)
        w = jnp.exp(log_w - max_log_w)
        w /= jnp.sum(w)
        ess = 1.0 / jnp.sum(w ** 2)

        # Estimate log(Z_t / Z_{t-1})
        logZ_ratio = jnp.log(jnp.sum(jnp.exp(log_w - max_log_w))) + max_log_w

        # Resample particles
        key, key_resample, key_mcmc = jax.random.split(state.key, 3)
        idx = jax.random.choice(key_resample, N, shape=(N,), p=w)
        resampled_particles = state.particles[idx]

        # Build tempered log-density
        def logdensity_lambda(x):
            return (1 - lambda_next) * base_logdensity(x) + lambda_next * target_logdensity(x)

        # Construct kernel and adapt its parameters
        updated_smc_state = state.replace(particles=resampled_particles, lambda_prev=lambda_next)
        kernel, params = smc_kernel.create_kernel(updated_smc_state)

        # Initialize MCMC and run mutation
        init_state, _ = kernel.init(resampled_particles)
        mcmc_output = run_mcmc(
            step=kernel.step,
            init_state=init_state,
            params=params,
            key=key_mcmc,
            n_samples=n_mcmc_steps,
            summarize=kernel.summarize
        )

        new_particles = mcmc_output.traj[-1]

        return SMCState(
            particles=new_particles,
            log_weights=jnp.zeros(N),
            log_p0=base_logdensity(new_particles),
            log_p1=target_logdensity(new_particles),
            lambda_prev=lambda_next,
            temperatures=state.temperatures + [lambda_next],
            ess_history=state.ess_history + [ess],
            diagnostics=state.diagnostics + [mcmc_output.summary],
            logZ_estimates=state.logZ_estimates + [logZ_ratio],
            key=key,
        )

    # Run adaptive SMC loop using lax.while_loop
    final_state = jax.lax.while_loop(cond_fn, body_fn, state)

    return SMCOutput(
        particles=final_state.particles,
        log_weights=final_state.log_weights,
        temperatures=jnp.array(final_state.temperatures),
        ess_history=jnp.array(final_state.ess_history),
        diagnostics=final_state.diagnostics,
        logZ_estimates=jnp.array(final_state.logZ_estimates),
    )

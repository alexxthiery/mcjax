"""
Sequential Monte Carlo (SMC) with tempering and MCMC mutation.


Core idea
---------
We construct a sequence of tempered distributions

    pi_t(x) propto p0(x)^(1 - t) * p1(x)^t,    t ∈ [0, 1]

and move a cloud of N particles through temperatures 0 → 1 using:

1. Weight update as temperature increases.
2. Resampling based on ESS.
3. MCMC mutation with a given MarkovKernel at each temperature.

The SMC engine is agnostic to the particular MCMC kernel, as long as:

- The kernel is a `MarkovKernel`.
- Its state has fields `x` and `log_prob`.
- `kernel.run_mcmc_batch` returns an `MCMCOutput` with a trajectory whose
  `x` and `log_prob` fields have shape `(N, n_steps, ...)` and `(N, n_steps)`.

Adaptation (e.g. of step size or covariance) is handled by an optional
`adapt_kernel_fn`, not as a method on the kernel itself.
"""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct

from mcjax.mcmc.core import MarkovKernel
from mcjax.util.weights import (
    effective_sample_size_normalized,
    normalize_log_weights,
)
from mcjax.util.resampling import systematic_resample
from mcjax.util.logreduce import log_sum_exp
from mcjax.util.tree_util import (
    tree_make_batched,
    tree_update_at_index,
    tree_mean_across_batch,
    tree_truncate_leading,
)
from .adapt_temperature import temp_adaptive


Array = jnp.ndarray


@struct.dataclass
class SMCState:
    """
    Internal state of the SMC algorithm.

    This is the state carried through `jax.lax.while_loop`. It contains both
    the minimal information needed to advance one SMC step and the histories
    we want to record.

    Core fields (algorithm):
        xs:           Particle positions, shape (N, D).
        log_weights:  Log weights for particles, shape (N,).
        log_p0:       log p0(x) for each particle, shape (N,).
        log_p1:       log p1(x) for each particle, shape (N,).
        temp:         Current inverse temperature in [0, 1].
        step:         Current SMC step index (0-based).
        key:          PRNG key for randomness.
        kernel_params:
                      Current Markov kernel parameters (PyTree).

    History fields (for inspection/diagnostics):
        temperatures:               Temperature path, shape (n_temp_max,).
        ess_history:         ESS at each step, shape (n_temp_max,).
        logZ_increments:     log Z increment at each step, shape (n_temp_max,).
        kernel_params_history:
                             History of kernel params across steps; same PyTree
                             structure as `kernel_params`, but with a leading
                             axis of length n_temp_max in each leaf.
        mcmc_summary_history:
                             History of per-step MCMC summaries (PyTree with
                             leading axis n_temp_max), or None if the kernel
                             does not provide summaries.
    """
    # core state
    xs: Array
    log_weights: Array
    log_p0: Array
    log_p1: Array
    temp: float
    step: int
    key: jax.Array
    kernel_params: Any

    # histories
    temperatures: Array
    ess_history: Array
    logZ_increments: Array
    kernel_params_history: Any
    mcmc_summary_history: Any


@struct.dataclass
class SMCOutput:
    """
    Output of the Sequential Monte Carlo run.

    Attributes
    ----------
    N_temperature:
        Number of temperature steps actually used (<= n_temp_max).
    xs:
        Final particle positions, shape (N, D).
    log_weights:
        Final log weights for the particles, shape (N,), normalized.
    temperatures:
        Temperatures visited, shape (N_temperature,).
    ess_history:
        Effective sample size at each SMC step, shape (N_temperature,).
    logZ_incr:
        Log normalizing constant increments at each step, shape (N_temperature,).
    logZ:
        Estimated log normalizing constant = sum(logZ_incr).
    kernel_params_history:
        History of kernel parameters across steps, truncated to N_temperature.
        Same PyTree structure as initial kernel_params, with leading axis N_temperature.
    summary:
        History of MCMC summary statistics across steps, truncated to N_temperature.
        PyTree with leading axis N_temperature, or None if summaries are not provided
        by the kernel.
    """
    N_temperature: int
    xs: Array
    log_weights: Array
    temperatures: Array
    ess_history: Array
    logZ_incr: Array
    logZ: float
    kernel_params_history: Any
    summary: Any


def prettify_smc_output(smc_out: SMCOutput) -> SMCOutput:
    """
    Truncates the histories in SMCOutput to the actually used number of temperature steps.

    This makes the output easier to inspect, at the cost of making vmap/jit
    more difficult since output shapes vary.

    Parameters:
        smc_out: SMCOutput with histories of length n_temp_max.

    Returns:
        SMCOutput with histories truncated to N_temperature.
    """
    N_temperature = smc_out.N_temperature
    temperatures = smc_out.temperatures[:N_temperature]
    ess_history = smc_out.ess_history[:N_temperature]
    logZ_incr = smc_out.logZ_incr[:N_temperature]
    kernel_params_history = tree_truncate_leading(
        smc_out.kernel_params_history, N_temperature
    )
    if smc_out.summary is not None:
        summary = tree_truncate_leading(smc_out.summary, N_temperature)
    else:
        summary = None

    return SMCOutput(
        N_temperature=smc_out.N_temperature,
        xs=smc_out.xs,
        log_weights=smc_out.log_weights,
        temperatures=temperatures,
        ess_history=ess_history,
        logZ_incr=logZ_incr,
        logZ=smc_out.logZ,
        kernel_params_history=kernel_params_history,
        summary=summary,
    )


def run_smc(
    *,
    log_prob_base: Callable[[Array], Array],
    log_prob_target: Callable[[Array], Array],
    xs_init: Array,
    kernel: MarkovKernel,
    kernel_params: Any,
    n_mcmc_steps: int,
    n_temp_max: int = 100,
    key: jax.Array,
    adapt_kernel_fn: Optional[
        Callable[[MarkovKernel, Callable[[Array], Array], Array, Any, jax.Array], Any]
    ] = None,
    ess_threshold: float = 0.5,
    temp_ladder: Optional[Array] = None,
    logZ_init: float = 0.0,
    prettify_output: bool = False,
) -> SMCOutput:
    """
    Run Tempered Sequential Monte Carlo with MCMC mutation.

    This function implements a standard SMC scheme with tempering:

        - Start from a base distribution p0.
        - Move particles through a sequence of tempered distributions
          pi_t(x) ∝ p0(x)^(1-t) p1(x)^t until t = 1.

    At each temperature step:
        1. We update particle weights for the temperature increment.
        2. We compute the ESS and increment the estimate of log Z.
        3. We resample particles.
        4. We adapt the MCMC kernel (optionally) based on the tempered target.
        5. We run an MCMC mutation with the given MarkovKernel.

    Parameters
    ----------
    log_prob_base:
        Function log p0(x) for the base distribution. Takes an array of shape (D,)
        and returns a scalar.
    log_prob_target:
        Function log p1(x) for the target distribution. Same interface as log_prob_base.
    xs_init:
        Initial particle positions, shape (N, D). Assumed to be drawn from p0.
    kernel:
        MarkovKernel instance that defines the MCMC mutation kernel. Its state must
        have fields `x` and `log_prob` for SMC to extract final states and log-probs.
    kernel_params:
        Initial parameters for the MarkovKernel; typically something like RwmParams,
        MalaParams, etc.
    n_mcmc_steps:
        Number of MCMC steps per SMC temperature stage.
    n_temp_max:
        Maximum number of temperature stages. The algorithm stops earlier if t reaches 1.
    key:
        JAX PRNG key.
    adapt_kernel_fn:
        Optional adaptation function:

            adapt_kernel_fn(kernel, log_prob_t, xs, params, key) -> new_params

        where `log_prob_t` is the tempered log-density at the current temperature,
        `xs` are the current (resampled) particles, and `params` are current
        kernel parameters. If None, no adaptation is performed.
    ess_threshold:
        Effective sample size threshold (as a fraction of N) used by the adaptive
        temperature scheduler when `temp_ladder` is None.
    temp_ladder:
        Optional fixed temperature ladder. If not None, a deterministic temperature
        schedule is used.
        Note: the ladder should NOT start at 0.0 (it is implied) and should end at 1.0.
    logZ_init:
        Initial log normalizing constant. Default is 0.0; it is the case when p0
        is a normalized distribution.
    prettify_output:
        If True, truncate histories in the output SMCOutput to the actually
        used number of temperature steps. If False, histories have length n_temp_max.
        Note: this makes vmap more difficult since output shapes vary.

    Returns
    -------
    SMCOutput
        Dataclass containing the final particle cloud, weights, logZ estimate,
        and histories of temperatures, ESS, kernel parameters, and MCMC summaries.
    """
    # Vectorized log-probabilities for convenience.
    log_prob_base_batch = jax.vmap(log_prob_base)
    log_prob_target_batch = jax.vmap(log_prob_target)

    xs_init = jnp.asarray(xs_init)
    N = xs_init.shape[0]

    # Initial log p0, log p1, and uniform log-weights.
    log_p0_init = log_prob_base_batch(xs_init)
    log_p1_init = log_prob_target_batch(xs_init)
    log_weights_init = jnp.full((N,), -jnp.log(N))

    # Histories: allocate to n_temp_max, later truncated to N_temperature.
    zero_vec = jnp.zeros((n_temp_max,))
    temperatures_init = zero_vec
    ess_history_init = zero_vec
    logZ_increments_init = zero_vec

    # We need an example summary to allocate a batched history. We run a dummy
    # 1-step MCMC batch to infer the summary PyTree structure.
    key_dummy, key_main = jax.random.split(key)
    dummy_out = kernel.run_mcmc_batch(
        log_prob=log_prob_base,
        xs_init=xs_init,
        params=kernel_params,
        key=key_dummy,
        n_samples=1,
    )
    summary_dummy = dummy_out.summary

    if summary_dummy is None:
        mcmc_summary_history_init = None
    else:
        # Average across chains to get a single "per-run" summary instance.
        summary_dummy = tree_mean_across_batch(summary_dummy)
        mcmc_summary_history_init = tree_make_batched(summary_dummy, n_temp_max)
        
    # sanity check for temp_ladder
    if temp_ladder is not None:
        temp_ladder = jnp.asarray(temp_ladder)
        if temp_ladder.ndim != 1:
            raise ValueError("temp_ladder must be a 1D array of temperatures.")
        if temp_ladder[0] == 0.0:
            raise ValueError("temp_ladder must NOT start at 0.0 (it is implied).")
        if temp_ladder[-1] != 1.0:
            raise ValueError("temp_ladder must end at 1.0.")

    # History for kernel parameters: same PyTree as kernel_params with leading axis n_temp_max.
    kernel_params_history_init = tree_make_batched(kernel_params, n_temp_max)

    # Initialize SMC state.
    state_init = SMCState(
        xs=xs_init,
        log_weights=log_weights_init,
        log_p0=log_p0_init,
        log_p1=log_p1_init,
        temp=0.0,
        step=0,
        key=key_main,
        kernel_params=kernel_params,
        temperatures=temperatures_init,
        ess_history=ess_history_init,
        logZ_increments=logZ_increments_init,
        kernel_params_history=kernel_params_history_init,
        mcmc_summary_history=mcmc_summary_history_init,
    )

    # Small epsilon to keep temperature strictly positive and avoid division by zero
    # when recovering log_p1 from the tempered log-prob.
    eps_t = 1e-6

    def update_temp_and_weights(
        log_p0: Array,
        log_p1: Array,
        log_weights: Array,
        temp: float,
        step: int,
    ):
        """
        Given current log p0, log p1, weights, and temperature, choose the next
        temperature and update weights and logZ increment.

        This uses either:
            - An ESS-based adaptive schedule (if temp_ladder is None), or
            - A deterministic schedule from temp_ladder.
            
        Parameters:
            log_p0: Current log p0(x) for each particle, shape (N,).
            log_p1: Current log p1(x) for each particle, shape (N,).
            log_weights: Current log weights for each particle, shape (N,).
            temp: Current inverse temperature.
            step: Current SMC step index (0-based).
        """
        log_w_norm = normalize_log_weights(log_weights)

        temp_next_adapt = temp_adaptive(
            log_p0=log_p0,
            log_p1=log_p1,
            log_weights=log_w_norm,
            temp=temp,
            ess_threshold=ess_threshold,
        )
        temp_next_det = temp_ladder[step] if temp_ladder is not None else 1.0

        use_det = temp_ladder is not None
        temp_next = jnp.where(use_det, temp_next_det, temp_next_adapt)
        # Clamp into (eps_t, 1.0] for numerical stability and monotonicity.
        temp_next = jnp.clip(temp_next, eps_t, 1.0)
        temp_delta = temp_next - temp

        log_ratio = temp_delta * (log_p1 - log_p0)
        logZ_inc = log_sum_exp(log_w_norm + log_ratio)
        log_w_new = normalize_log_weights(log_w_norm + log_ratio)
        ess = effective_sample_size_normalized(log_w_new)

        return temp_next, logZ_inc, log_w_new, ess

    def resample(
        xs: Array,
        log_weights_norm: Array,
        key_resample: jax.Array,
    ):
        """
        Resample particles according to their normalized log-weights and reset
        weights to uniform. Assumes log_weights_norm is already normalized.
        """
        w = jnp.exp(log_weights_norm)
        idx = systematic_resample(key=key_resample, weights=w, n_samples=N)
        xs_resampled = xs[idx]
        log_weights_resampled = jnp.full((N,), -jnp.log(N))
        return xs_resampled, log_weights_resampled

    def make_tempered_log_prob(temp_next: float):
        """
        Build the tempered log-density function:

            log pi_t(x) = (1 - t) log p0(x) + t log p1(x)
        """
        def logprob_t(x: Array) -> Array:
            return (1.0 - temp_next) * log_prob_base(x) + temp_next * log_prob_target(x)

        return logprob_t

    def apply_adapt_kernel(
        kernel_params: Any,
        kernel: MarkovKernel,
        log_prob_t: Callable[[Array], Array],
        xs_resampled: Array,
        key_adapt: jax.Array,
    ) -> Any:
        """
        Optionally adapt kernel parameters based on the current tempered target
        and resampled particles. If adapt_kernel_fn is None, return params unchanged.
        """
        if adapt_kernel_fn is None:
            return kernel_params
        return adapt_kernel_fn(
            kernel=kernel,
            log_prob_t=log_prob_t,
            xs=xs_resampled,
            params=kernel_params,
            key=key_adapt,
        )

    def mutate(
        kernel: MarkovKernel,
        log_prob_t: Callable[[Array], Array],
        xs_resampled: Array,
        kernel_params: Any,
        key_mcmc: jax.Array,
    ):
        """
        Run the MCMC mutation step with the given kernel and tempered target.

        Returns:
            xs_mutated: final particle positions from the chain, shape (N, D).
            log_pt_mutated: final tempered log-probabilities, shape (N,).
            mcmc_summary: MCMC summary aggregated across chains, or None if
                          the kernel does not provide summaries.
        """
        out = kernel.run_mcmc_batch(
            log_prob=log_prob_t,
            xs_init=xs_resampled,
            params=kernel_params,
            key=key_mcmc,
            n_samples=n_mcmc_steps,
        )
        # Assumes state has fields x and log_prob with shape (N, n_steps, ...)
        xs_traj = out.traj.x
        logp_traj = out.traj.log_prob

        xs_mutated = xs_traj[:, -1, ...]
        log_pt_mutated = logp_traj[:, -1]

        if out.summary is None:
            mcmc_summary = None
        else:
            mcmc_summary = tree_mean_across_batch(out.summary)

        return xs_mutated, log_pt_mutated, mcmc_summary

    def infer_log_p0_p1(
        xs_mutated: Array,
        log_pt_mutated: Array,
        temp_next: float,
    ):
        """
        Given final tempered log-probabilities and the new temperature, recover
        log p0 and log p1 at the mutated particle locations.

        We use:
            log pi_t(x) = (1 - t) log p0(x) + t log p1(x)   (ignoring Z_t)
        """
        log_p0_new = log_prob_base_batch(xs_mutated)
        log_p1_new = (log_pt_mutated - (1.0 - temp_next) * log_p0_new) / temp_next
        return log_p0_new, log_p1_new

    def cond_fn(state: SMCState) -> bool:
        """
        Stop when we reach temperature 1.0 or exhaust n_temp_max.
        """
        return jnp.logical_and(state.temp < 1.0, state.step < n_temp_max)

    def body_fn(state: SMCState) -> SMCState:
        """
        One SMC temperature step:
            - update temperature and weights
            - resample
            - adapt kernel
            - run MCMC mutation
            - recompute log p0/log p1
            - update histories
        """
        key_resample, key_mcmc, key_adapt, key_next = jax.random.split(state.key, 4)

        # 1. Temperature and weights update.
        temp_next, logZ_inc, log_w_new, ess = update_temp_and_weights(
            state.log_p0,
            state.log_p1,
            state.log_weights,
            state.temp,
            state.step,
        )

        # 2. Resample according to new normalized weights.
        xs_resampled, log_weights_resampled = resample(
            state.xs, log_w_new, key_resample
        )

        # 3. Build tempered target and adapt kernel.
        logprob_t = make_tempered_log_prob(temp_next)
        new_kernel_params = apply_adapt_kernel(
            kernel_params=state.kernel_params,
            kernel=kernel,
            log_prob_t=logprob_t,
            xs_resampled=xs_resampled,
            key_adapt=key_adapt,
        )

        # 4. MCMC mutation.
        xs_mutated, log_pt_mutated, mcmc_summary = mutate(
            kernel=kernel,
            log_prob_t=logprob_t,
            xs_resampled=xs_resampled,
            kernel_params=new_kernel_params,
            key_mcmc=key_mcmc,
        )

        # 5. Recover log p0 and log p1 at mutated locations.
        log_p0_new, log_p1_new = infer_log_p0_p1(
            xs_mutated, log_pt_mutated, temp_next
        )

        # 6. Update histories at step index `i`.
        i = state.step
        temperatures = state.temperatures.at[i].set(temp_next)
        ess_history = state.ess_history.at[i].set(ess)
        logZ_increments = state.logZ_increments.at[i].set(logZ_inc)

        kernel_params_history = tree_update_at_index(
            state.kernel_params_history, new_kernel_params, i
        )

        if state.mcmc_summary_history is None or mcmc_summary is None:
            mcmc_summary_history = state.mcmc_summary_history
        else:
            mcmc_summary_history = tree_update_at_index(
                state.mcmc_summary_history, mcmc_summary, i
            )

        # 7. Build new state.
        return state.replace(
            xs=xs_mutated,
            log_weights=log_weights_resampled,
            log_p0=log_p0_new,
            log_p1=log_p1_new,
            temp=temp_next,
            step=i + 1,
            key=key_next,
            kernel_params=new_kernel_params,
            temperatures=temperatures,
            ess_history=ess_history,
            logZ_increments=logZ_increments,
            kernel_params_history=kernel_params_history,
            mcmc_summary_history=mcmc_summary_history,
        )

    # Run the SMC loop with lax.while_loop for JIT-friendly control flow.
    final_state = jax.lax.while_loop(cond_fn, body_fn, state_init)
    
    # prepare output
    N_temperature = final_state.step
    temperatures = final_state.temperatures
    ess_history = final_state.ess_history
    logZ_incr = final_state.logZ_increments
    kernel_params_history = final_state.kernel_params_history
    summary_history = final_state.mcmc_summary_history
    
    # Final logZ estimate: sum of increments over used steps plus initial logZ.
    logZ = jnp.sum(logZ_incr) + logZ_init
    
    # prepare the SMC output
    smc_out = SMCOutput(
        N_temperature=N_temperature,
        xs=final_state.xs,
        log_weights=final_state.log_weights,
        temperatures=temperatures,
        ess_history=ess_history,
        logZ_incr=logZ_incr,
        logZ=logZ,
        kernel_params_history=kernel_params_history,
        summary=summary_history,
    )

    # truncate histories if requested
    # note: this makes vmap more difficult since output shapes vary
    if prettify_output:
        smc_out = prettify_smc_output(smc_out)

    return smc_out
2
from typing import Callable, Tuple, Any, Optional
import jax
import jax.random as jr
from flax import struct


@struct.dataclass
class MCMCOutput:
    """
    Output container for an MCMC run.

    Attributes:
        traj: A PyTree representing the full trajectory of MCMC states.
            Typically a stacked PyTree (e.g., dict of arrays or dataclass with arrays of shape [n_samples, ...]).
        summary: A PyTree containing aggregated statistics over the trajectory.
            Produced by the kernel-specific `summarize` function (e.g., acceptance rate, mean squared jump).
    """
    traj: Any     # Full trajectory of the states
    summary: Any  # Summary statistics of the trajectory


@struct.dataclass
class MarkovKernel:
    """
    A functional interface for Markov kernels.

    This struct holds references to the key components needed to define a sampler:
        - init: Initializes the Markov chain state from an initial position.
        - step: Advances the chain by one iteration using a random key, current state, and kernel parameters.
        - summarize: (Optional) Computes a summary of per-step statistics across the MCMC trajectory.

    All fields must be pure functions that are compatible with JAX transformations (jit, scan, vmap).

    Required function signatures:
        init(x0: Array) -> (state, params)
        step(key: PRNGKey, state, params) -> (new_state, stats)
        summarize(stats_traj) -> summary  # Optional, for diagnostics
    """
    init: Callable[[Any], Tuple[Any, Any]]
    step: Callable[[jax.Array, Any, Any], Tuple[Any, Any]]
    summarize: Optional[Callable[[Any], Any]] = None


def run_mcmc(
    *,
    step: Callable[[jax.Array, Any, Any], Tuple[Any, Any]],
    init_state: Any,
    params: Any,
    key: jax.Array,
    n_samples: int,
    summarize: Optional[Callable[[Any], Any]] = None
) -> MCMCOutput:
    """
    Generic MCMC runner using `lax.scan`.

    This function executes a Markov chain over `n_samples` steps using a kernel's `step` function.
    It also computes summary statistics if a `summarize_fn` is provided.

    Args:
        step: The MCMC kernel's step function.
            Should be pure and have signature:
            (key: PRNGKey, state: PyTree, params: PyTree) -> (new_state, stats)
        init_state: The initial state of the Markov chain (e.g., a dataclass).
            Must be a JAX PyTree.
        params: Kernel-specific parameters (e.g., step size, precomputed matrices).
            Must be a JAX PyTree.
        key: A JAX PRNGKey used to generate randomness in the chain.
        n_samples: Number of MCMC steps to perform.
        summarize_fn: Optional function to compute statistics over the trajectory of `stats` objects.
            Must be pure and JAX-compatible.

    Returns:
        MCMCOutput:
            traj: A trajectory of states over `n_samples` steps (PyTree of arrays).
            summary: Summary statistics computed from the trajectory, if `summarize` is provided.
    """
    def one_step(carry, _):
        key, state = carry
        key, subkey = jr.split(key)
        state, stats = step(subkey, state, params)
        return (key, state), (state, stats)

    # Run the chain using lax.scan for efficient looping
    (_, _), (traj, stats_traj) = jax.lax.scan(one_step, (key, init_state), None, length=n_samples)

    summary = summarize(stats_traj) if summarize is not None else None

    return MCMCOutput(traj=traj, summary=summary)

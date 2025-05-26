from typing import Callable, Tuple, Any, Optional
import jax
import jax.numpy as jnp
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
    Functional container for Markov kernels.

    Holds reusable callable logic for `init_state_fn`, `init_params`, `step`, and `summarize`.
    Provides a standard `.run_mcmc()` loop.
    """
    init_state_fn: Callable[[jnp.ndarray], Any]
    init_params: Callable[..., Any]
    step: Callable[[jax.Array, Any, Any], Tuple[Any, Any]]
    summarize: Optional[Callable[[Any], Any]] = None

    def run_mcmc(
        self,
        *,
        initial_state: Any,
        params: Any,
        key: jax.Array,
        n_samples: int
    ) -> MCMCOutput:
        """
        Executes the MCMC trajectory using `lax.scan`.

        Args:
            initial_state: Initial state of the Markov chain.
            params: Tunable kernel parameters.
            key: PRNG key for reproducibility.
            n_samples: Number of MCMC steps.

        Returns:
            MCMCOutput containing the full trajectory and optional summary statistics.
        """
        def one_step(carry, _):
            key, state = carry
            key, subkey = jr.split(key)
            state, stats = self.step(subkey, state, params)
            return (key, state), (state, stats)

        (_, _), (traj, stats_traj) = jax.lax.scan(
                                        one_step,
                                        (key, initial_state),
                                        None,
                                        length=n_samples)
        summary = self.summarize(stats_traj) if self.summarize is not None else None
        return MCMCOutput(traj=traj, summary=summary)

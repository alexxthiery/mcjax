from typing import Callable, Tuple, Any, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct



@struct.dataclass
class MCMCOutput:
    """
    Output container for an MCMC run.

    Attributes
    ----------
    traj:
        PyTree representing the full trajectory of MCMC states.
        For a single chain, this is typically a stacked PyTree with
        leading axis [n_samples, ...].

        For batched runs (run_mcmc_batch), there is an additional
        leading batch axis [batch, n_samples, ...].

    summary:
        PyTree containing aggregated statistics over the trajectory,
        produced by the kernel-specific `summarize` function
        (e.g. acceptance rate, mean squared jump distance).

        For batched runs, this usually carries a leading batch axis.
        If `summarize` is None, this is None.
    """
    traj: Any
    summary: Any


@struct.dataclass
class MarkovKernel:
    """
    Functional container for Markov kernels.

    This object bundles together the four kernel-specific building blocks:

      - `init_state_fn(log_prob, x_init) -> state`
      - `init_params(*args, **kwargs) -> params`
      - `step(key, log_prob, state, params) -> (state, stats)`
      - `summarize(stats_traj) -> summary`

    It also provides generic `run_mcmc` and `run_mcmc_batch` drivers
    that work for any kernel following this protocol.

    All fields are marked as non-pytree so that `MarkovKernel` can
    safely be passed into JAX transforms without the callables becoming
    part of the traced state.
    """
    
    # initialization function for the Markov chain state
    init_state_fn: Callable[
        [Callable[[jnp.ndarray], jnp.ndarray], Any], Any
    ] = struct.field(pytree_node=False)

    # initialization function for the kernel parameters
    init_params: Callable[..., Any] = struct.field(pytree_node=False)

    # one step of the Markov kernel
    step: Callable[
        [jax.Array, Callable[[jnp.ndarray], jnp.ndarray], Any, Any],
        Tuple[Any, Any],
    ] = struct.field(pytree_node=False)

    # optional function to summarize trajectory statistics
    summarize: Optional[Callable[[Any], Any]] = struct.field(
        default=None,
        pytree_node=False,
    )


    def run_mcmc(
        self,
        *,
        log_prob: Callable[[jnp.ndarray], jnp.ndarray],
        x_init: Any,
        params: Any,
        key: jax.Array,
        n_samples: int,
    ) -> MCMCOutput:
        """
        Execute an MCMC trajectory for a single chain using `lax.scan`.

        Parameters
        ----------
        log_prob:
            Log-density function of the target distribution. Takes a
            single state (non-batched) and returns a scalar log-density.
        x_init:
            Initial state of the Markov chain. This can be any PyTree
            that `init_state_fn` and `log_prob` know how to handle.
        params:
            Kernel parameters PyTree, typically created by `init_params`.
        key:
            PRNG key for reproducibility.
        n_samples:
            Number of MCMC steps.

        Returns
        -------
        MCMCOutput
            `traj` is the stacked trajectory of states with leading axis
            [n_samples, ...]. `summary` is the output of `summarize`
            applied to the stacked stats trajectory, or None if
            `summarize` is None.
        """
        initial_state = self.init_state_fn(log_prob, x_init)

        def one_step(carry, _):
            key, state = carry
            key, subkey = jr.split(key)
            state, stats = self.step(subkey, log_prob, state, params)
            return (key, state), (state, stats)

        (_, _), (traj, stats_traj) = jax.lax.scan(
            one_step,
            (key, initial_state),
            None,
            length=n_samples,
        )

        summary = self.summarize(stats_traj) if self.summarize is not None else None
        return MCMCOutput(traj=traj, summary=summary)

    def run_mcmc_batch(
        self,
        *,
        log_prob: Callable[[jnp.ndarray], jnp.ndarray],
        xs_init: Any,
        params: Any,
        key: jax.Array,
        n_samples: int,
    ) -> MCMCOutput:
        """
        Execute MCMC trajectories for a batch of initial states using `jax.vmap`.

        Parameters
        ----------
        log_prob:
            Log-density function of the target distribution, acting on a
            single (non-batched) state.
        xs_init:
            Batch of initial states, with a leading batch axis of size B.
            Typically an array of shape [B, ...], but can be any PyTree
            with a leading batch axis.
        params:
            Kernel parameters, shared across all chains in the batch.
        key:
            PRNG key. It will be split into B independent keys.
        n_samples:
            Number of MCMC steps per chain.

        Returns
        -------
        MCMCOutput
            `traj` and `summary` are batched over the leading axis B.
            For example, if single-chain `traj` has shape
            [n_samples, D], then batched `traj` has shape
            [B, n_samples, D].
        """
        # Assume xs_init is an array-like with a leading batch axis.
        # For more general pytrees, users can vmap `run_mcmc` manually.
        B = xs_init.shape[0]
        key_batch = jr.split(key, B)

        def run_single(log_prob, x_init, params, key, n_samples):
            return self.run_mcmc(
                log_prob=log_prob,
                x_init=x_init,
                params=params,
                key=key,
                n_samples=n_samples,
            )

        run_batch = jax.vmap(
            run_single,
            in_axes=(None, 0, None, 0, None),
            out_axes=0,
        )

        # Vmap over the batch dimension. The result is a batched
        # MCMCOutput, which JAX will assemble as a PyTree with
        # leading batch dimension on each leaf (traj and summary).
        return run_batch(
            log_prob,
            xs_init,
            params,
            key_batch,
            n_samples,
        )
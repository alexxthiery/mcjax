from typing import Any, Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import struct


@struct.dataclass
class VIState:
    """
    Variational Inference optimization state.

    Attributes:
    -----------
    params : Any
        Parameters of the variational distribution.
    opt_state : optax.OptState
        State of the optimizer.
    step : int
        Current optimization step.
    """
    params: Any
    opt_state: optax.OptState
    step: int


@struct.dataclass
class VIEngine:
    """
    Core Variational Inference Engine.

    Attributes:
    -----------
    approx : Distribution
        Variational distribution implementing the Distribution interface.
    """
    approx: Any  # Expected to implement the Distribution interface

    @classmethod
    def create(cls, approx: Any) -> "VIEngine":
        """
        Initialize the VI engine with the given approximation structure.

        Parameters:
        -----------
        approx : Distribution
            Instance of a variational approximation.

        Returns:
        --------
        engine : VIEngine
            The initialized inference engine.
        """
        return cls(approx=approx)

    def step(
        self,
        state: VIState,
        key: jax.Array,
        log_prob: Callable[[jnp.ndarray], jnp.ndarray],
        optimizer: optax.GradientTransformation,
        n_samples: int,
        stop_gradient_entropy: bool = True,
    ) -> Tuple[VIState, jnp.ndarray]:
        """
        Perform a single VI optimization step.

        Parameters:
        -----------
        state : VIState
            Current optimization state.
        key : jax.Array
            PRNG key.
        log_prob : Callable
            Target log-probability function.
        optimizer : optax.GradientTransformation
            Optimizer used for the update.
        n_samples : int
            Number of Monte Carlo samples.
        stop_gradient_entropy : bool
            Whether to stop gradient through entropy term.

        Returns:
        --------
        new_state : VIState
            Updated state.
        kl : jnp.ndarray
            Scalar KL divergence estimate.
        """
        def loss_fn(params):
            xs = self.approx.sample(params=params, key=key, n_samples=n_samples)
            return self.approx.neg_elbo(
                params=params,
                xs=xs,
                logtarget=log_prob,
                stop_gradient_entropy=stop_gradient_entropy,
                key=key,
                n_samples=n_samples,
            )

        kl, grads = jax.value_and_grad(loss_fn)(state.params)
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)

        new_state = VIState(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1,
        )
        return new_state, kl

    def run(
        self,
        params_init: Any,
        log_prob: Callable[[jnp.ndarray], jnp.ndarray],
        key: jax.Array,
        optimizer: optax.GradientTransformation,
        n_samples: int,
        n_iter: int,
        stop_gradient_entropy: bool = True,
        trace: bool = False,
        jit: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full variational inference optimization loop.

        Parameters:
        -----------
        params_init : Any
            Initial variational parameters.
        log_prob : Callable
            Target log-probability function.
        key : jax.Array
            PRNG key.
        optimizer : optax.GradientTransformation
            Optimizer used for the update.
        n_samples : int
            Number of samples per iteration.
        n_iter : int
            Number of optimization steps.
        stop_gradient_entropy : bool
            Whether to stop gradient through entropy term.
        trace : bool
            Whether to store parameter history.
        jit : bool
            Whether to JIT compile the step function.
        verbose : bool
            Print progress every step.

        Returns:
        --------
        A dictionary with final parameters, raw parameters, KL trace, and optional trace.
        """
        opt_state = optimizer.init(params_init)
        state = VIState(params=params_init, opt_state=opt_state, step=0)

        step_fn = self.step
        if jit:
            static_argnames = ["log_prob", "optimizer", "n_samples", "stop_gradient_entropy"]
            step_fn = jax.jit(self.step, static_argnames=static_argnames)
            
        kl_trace = []
        param_trace = []
        key_loop = key

        for i in range(n_iter):
            key_loop, key_i = jr.split(key_loop)
            state, kl = step_fn(
                state=state,
                key=key_i,
                log_prob=log_prob,
                optimizer=optimizer,
                n_samples=n_samples,
                stop_gradient_entropy=stop_gradient_entropy,
            )
            kl_trace.append(kl)
            if trace:
                param_trace.append(self.approx.postprocess(state.params))
            if verbose:
                print(f"iter {i:4d} | KL: {kl:.6f}")

        results = {
            "params": self.approx.postprocess(state.params),
            "params_raw": state.params,
            "kl_trace": jnp.stack(kl_trace),
        }
        if trace:
            results["params_trace"] = param_trace

        return results

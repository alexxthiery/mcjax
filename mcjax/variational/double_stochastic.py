from typing import Callable, Dict, Any, Tuple
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jr

# variational family
from mcjax.proba.var_family import VarFamily
# optax for optimization
import optax


@struct.dataclass
class VIState:
    params: Any
    opt_state: optax.OptState
    step: int


class DoubleStochasticVI:
    """
    Doubly Stochastic Variational Inference (DSVI)

    Supports any reparameterizable VarFamily with methods for sampling
    and evaluating log-density.

    Parameters:
    -----------
    logdensity : Callable[[jnp.ndarray], jnp.ndarray]
        Log-density of the target distribution p(x). Should support batched input.
    approx : VarFamily
        Instance of a variational approximation (e.g., DiagGaussian, MixtureDiagGaussian).
    sticking_the_landing : bool
        Whether to stop gradient flow through entropy estimate for lower-variance gradient (recommended).
    """

    def __init__(
        self,
        *,
        logdensity: Callable[[jnp.ndarray], jnp.ndarray],
        approx: VarFamily,
        sticking_the_landing: bool = True,
    ):
        self.logdensity = logdensity
        self.approx = approx
        self.sticking_the_landing = sticking_the_landing
        
        # define the batched version of the logdensity
        self.logdensity_batch = jax.vmap(logdensity, in_axes=(0,))

    def _kl_estimate(
        self,
        params,
        key: jax.Array,
        n_samples: int,
    ) -> jnp.ndarray:
        """
        Monte Carlo estimate of KL divergence KL[q || p] up to an additive constant.

        kl = E_q[log q(x)] - E_q[logdensity(x)] + cst = - Entropy - E_q[logdensity(x)] + cst
        """
        xs = self.approx.sample(params, key, n_samples)
        return self.approx.neg_elbo(
            params=params,
            xs=xs,
            logtarget_batch=self.logdensity_batch,
            stop_gradient_entropy=self.sticking_the_landing,
            key=key,
            n_samples=n_samples,
        )

    def run(
        self,
        *,
        key: jax.Array,
        n_iter: int,
        n_samples: int,
        lr: float = 1e-3,
        use_jit: bool = True,
        store_params_trace: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run variational inference optimization.

        Parameters:
        -----------
        key : jax.random.PRNGKey
            Random key for initialization and sampling.
        n_iter : int
            Number of optimization steps.
        n_samples : int
            Number of samples per iteration for Monte Carlo KL estimate.
        lr : float
            Learning rate for Adam optimizer.
        use_jit : bool
            Whether to JIT-compile the update step.
        store_params_trace : bool
            Whether to store the full parameter trajectory.
        verbose : bool
            Print KL value at each iteration.

        Returns:
        --------
        dict with final parameters, KL trace, and optionally full param trace.
        """
        # Initialize variational parameters and optimizer
        key_init, key_loop = jr.split(key)
        params = self.approx.init_params(key_init)

        # Initialize optimizer
        opt = optax.adam(lr)
        opt_state = opt.init(params)
        state = VIState(params=params, opt_state=opt_state, step=0)

        kl_trace = []
        params_trace = []

        # Define update step
        def update_fn(
                state: VIState,
                key: jax.Array,
                ) -> Tuple[VIState, jnp.ndarray]:
            kl_value_and_grads = jax.value_and_grad(self._kl_estimate)
            kl, grads = kl_value_and_grads(state.params, key, n_samples)
            updates, opt_state = opt.update(grads, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)
            new_state = VIState(params=new_params,
                                opt_state=opt_state,
                                step=state.step + 1)
            return new_state, kl

        if use_jit:
            update_fn = jax.jit(update_fn)

        # Run optimization loop
        for i in range(n_iter):
            key_loop, key_iter = jr.split(key_loop)
            state, kl = update_fn(state, key_iter)
            kl_trace.append(kl)
            if store_params_trace:
                params_trace.append(self.approx.postprocess(state.params))
            if verbose:
                print(f"iter {i:4d} | KL: {kl:.6f}")

        # Final output
        out = {
            "params": self.approx.postprocess(state.params),
            "params_raw": state.params,
            "kl_trace": kl_trace,
        }
        if store_params_trace:
            out["params_trace"] = params_trace
        return out

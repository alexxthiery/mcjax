import jax
import jax.numpy as jnp
import jax.random as jr
from flax.training import train_state

class Trainer:
    """
    A generic training engine.  On init, you give it:
      - algorithm (has .state, .score_fn, .loss_obj, .ou, .init_dist, .target_dist, .sample())
      - process (e.g. OU instance)
      - init_dist, target_dist, score_fn, loss_obj, state, batch_size, num_steps, if_logZ
    Then `run(rng_key)` executes the scan and returns results.
    """

    def __init__(self,
                 algorithm,
                 process,
                 init_dist,
                 target_dist,
                 score_fn,
                 loss_obj,
                 state: train_state.TrainState,
                 batch_size: int,
                 num_steps: int,
                 if_logZ: bool = False):
        self.alg         = algorithm
        self.process     = process
        self.init_dist   = init_dist
        self.target_dist = target_dist
        self.score_fn    = score_fn
        self.loss_obj    = loss_obj
        self.state       = state
        self.batch_size  = batch_size
        self.num_steps   = num_steps
        self.if_logZ     = if_logZ

        self.loss_and_grad = jax.jit(
            jax.value_and_grad(self._loss_wrapper, argnums=0),
            static_argnums=(2,3,4,5,6) 
        )

        self.train_step = jax.jit(
            self._train_step,
            static_argnums=(2,3,4,5,6,7) 
        )

    def _loss_wrapper(self, params, key, process, init_dist, target_dist, score_fn, batch_size, **kwargs):
        return self.loss_obj(params, key, process, init_dist, target_dist, score_fn, batch_size, **kwargs)

    def _train_step(self, state, key, process, init_dist, target_dist, score_fn, batch_size, loss_obj):
        loss, grads = self.loss_and_grad(
            state.params, key, process, init_dist, target_dist, score_fn, batch_size, add_score=loss_obj.add_score
        )
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    def run(self, rng_key):
        """
        Runs `num_steps` of training with jax.lax.scan.  
        Returns final_state, rng_key, loss_history, (and optionally logZ values, logZ vars).
        """
        def scan_body(carry, step):
            state, key, logz_vals, logz_vars = carry
            key, key_ = jr.split(key)

            state, loss = self.train_step(
                state, key_,
                self.process, self.init_dist, self.target_dist,
                self.score_fn, self.batch_size, self.loss_obj
            )

            # Optionally estimate logZ every 10 steps
            def compute_logz(key):
                key, key_ = jr.split(key)
                logz = self.alg.estimate_logZ(state.params, key_, num_samples=1000)
                idx = step // 10
                new_vals = logz_vals.at[idx].set(jnp.mean(logz))
                new_vars = logz_vars.at[idx].set(jnp.var(logz))
                return key, new_vals, new_vars
        
            key, logz_vals, logz_vars = jax.lax.cond(
                self.if_logZ & (step % 10 == 9),
                compute_logz,
                lambda _: (key, logz_vals, logz_vars),
                operand=key
            )
            # every 100 steps, print step and current loss
            def do_print(_):
                jax.debug.print("At step {}, loss = {}", step, loss)
                return None

            # branch on (step % 100 == 0)
            _ = jax.lax.cond((step % 100) == 0, do_print, lambda _: None, operand=None)

            return (state, key, logz_vals, logz_vars), loss

        logz_vals = jnp.zeros((self.num_steps // 10 + 1,))
        logz_vars = jnp.zeros_like(logz_vals)

        (final_state, final_key, logz_vals, logz_vars), losses = jax.lax.scan(
            scan_body,
            (self.state, rng_key, logz_vals, logz_vars),
            jnp.arange(self.num_steps)
        )
        return final_state, final_key, losses, logz_vals, logz_vars


class IDEMTrainer:
    """
    Trainer for iDEM (Iterated Denoising Energy Matching), handling both
    outer sampling loops (to populate the replay buffer) and inner
    gradient‐descent loops (to train the score network via the IDEMLoss).

    On init, we provide:
      - buffer:      a replay buffer instance (algorithm.buffer)
      - process:     the OU process (algorithm.ou)
      - init_dist:   the initial/reference distribution (algorithm.init_dist)
      - target_dist: the target distribution (algorithm.target_dist)
      - score_fn:    the score network function (algorithm.score_fn)
      - loss_obj:    an IDEMLoss instance (algorithm.loss_obj)
      - state:       a Flax TrainState containing .params and .opt_state
      - batch_size:  mini‐batch size for inner loop
      - outer_iters: number of outer sampling loops
      - inner_steps: number of gradient steps per outer iteration
      - num_samples_per_outer: how many new x₀’s to generate each outer iteration
      - if_logZ:     whether to compute logZ estimates (Boolean)
    """

    def __init__(self,
                 buffer,
                 process,
                 init_dist,
                 target_dist,
                 score_fn,
                 loss_obj,
                 state: train_state.TrainState,
                 batch_size: int,
                 outer_iters: int,
                 inner_steps: int,
                 num_samples_per_outer: int,
                 if_logZ: bool = False):
        self.buffer = buffer
        self.process = process
        self.init_dist = init_dist
        self.target_dist = target_dist
        self.score_fn = score_fn
        self.loss_obj = loss_obj
        self.state = state
        self.batch_size = batch_size

        self.outer_iters = outer_iters
        self.inner_steps = inner_steps
        self.num_samples_per_outer = num_samples_per_outer

        self.if_logZ = if_logZ
        # We will store logZ statistics per outer iteration if requested
        if if_logZ:
            self.logZ_means = jnp.zeros((outer_iters,))
            self.logZ_vars = jnp.zeros((outer_iters,))
        else:
            self.logZ_means = None
            self.logZ_vars = None

        # Build jitted loss-and-grad and train-step for inner loop
        self.loss_and_grad = jax.jit(
            jax.value_and_grad(self._loss_wrapper, argnums=0),
            static_argnums=(2, 3, 4, 5)
        )
        self.train_step = jax.jit(
            self._train_step,
            static_argnums=(2, 3, 4, 5)
        )

    def _loss_wrapper(self, params, key, buffer, target_dist, score_fn, batch_size):
        """
        Simply calls the IDEMLoss with the correct signature.
        """
        return self.loss_obj(
            params=params,
            key=key,
            buffer=buffer,
            target_dist=target_dist,
            score_fn=score_fn,
            batch_size=batch_size
        )

    def _train_step(self, state, key, buffer, target_dist, score_fn, batch_size):
        """
        One step of gradient descent on the IDEMLoss. Returns (new_state, loss).
        """
        params = state.params
        (loss, grads) = self.loss_and_grad(
            params, key, buffer, target_dist, score_fn, batch_size
        )
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    def run(self, rng_key):
        """
        Runs `num_steps` of inner‐loop training with jax.lax.scan.
        Returns: (final_state, final_key, loss_history, logZ_means, logZ_vars).
        """
        def scan_body(carry, step):
            state, key, logz_means, logz_vars = carry
            key, sub = jr.split(key)

            state, loss = self.train_step(
                state, sub,
                self.buffer, self.target_dist, self.score_fn, self.batch_size
            )

            # Optionally estimate logZ every 10 steps
            def compute_logz(key_inner):
                key_inner, key2 = jr.split(key_inner)
                logz = self.alg.estimate_logZ(state.params, key2, num_samples=1000)
                idx = step // 10
                new_means = logz_means.at[idx].set(jnp.mean(logz))
                new_vars = logz_vars.at[idx].set(jnp.var(logz))
                return key_inner, new_means, new_vars

            key, logz_means, logz_vars = jax.lax.cond(
                self.if_logZ & (step % 10 == 9),
                compute_logz,
                lambda k: (k, logz_means, logz_vars),
                operand=key
            )

            # Every 100 steps, print progress
            def do_print(_):
                jax.debug.print("Step {}: loss = {}", step, loss)
                return None

            _ = jax.lax.cond((step % 100) == 0, do_print, lambda _: None, operand=None)

            return (state, key, logz_means, logz_vars), loss

        # Preallocate arrays for storing logZ statistics every 10 steps
        num_logz_points = self.num_steps // 10 + 1
        logz_means = jnp.zeros((num_logz_points,))
        logz_vars = jnp.zeros((num_logz_points,))

        init_carry = (self.state, rng_key, logz_means, logz_vars)
        (final_state, final_key, final_means, final_vars), losses = jax.lax.scan(
            scan_body,
            init_carry,
            jnp.arange(self.num_steps)
        )

        return final_state, final_key, losses, final_means, final_vars
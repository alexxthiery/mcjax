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

class InnerTrainer:
    """
    Runs a fixed number of gradient‐descent steps on IDEMLoss, given:
      - algo:       the IDEMAlgorithm instance (to access buffer, target_dist, score_fn, etc.)
      - loss_obj:   an IDEMLoss instance
      - state:      a Flax TrainState (holds .params and optimizer state)
      - batch_size: minibatch size for each gradient step
      - inner_iters: how many gradient steps to perform
    """

    def __init__(self, algo, loss_obj, state: train_state.TrainState, batch_size: int, inner_iters: int):
        self.algo = algo
        self.loss_obj = loss_obj
        self.state = state
        self.batch_size = batch_size
        self.inner_iters = inner_iters

        # JIT‐compile a loss‐and‐grad function that calls IDEMLoss:
        self.loss_and_grad = jax.jit(
            jax.value_and_grad(self._loss_fn, argnums=0),
            static_argnums=(2,3,4,5)  # no static args here
        )
        # JIT‐compile one train_step (computes loss+grad, applies optimizer)
        self.train_step = jax.jit(self._train_step, static_argnums=())

    def _loss_fn(self, params, key, buffer, target_dist, score_fn, batch_size):
        """
        Wrap IDEMLoss.  We draw a minibatch from algo.buffer inside IDEMLoss itself.
        """
        return self.loss_obj(
            params=params,
            key=key,
            buffer=buffer,
            target_dist=target_dist,
            score_fn=score_fn,
            batch_size=batch_size
        )

    def _train_step(self, state, key):
        """
        One gradient step on IDEMLoss.  Returns (new_state, loss_scalar).
        """
        params = state.params
        (loss, grads) = self.loss_and_grad(params, key)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    def run(self, rng_key):
        """
        Runs exactly `inner_iters` gradient steps, all inside a single lax.scan.
        Returns (final_state, final_key, losses_array), where
          - final_state: updated TrainState
          - final_key: final PRNGKey after splitting
          - losses_array: jnp array of shape (inner_iters,) with each step’s loss
        """
        def inner_body(carry, _unused):
            state, key = carry
            key, sub = jr.split(key)
            new_state, loss = self.train_step(state, sub)
            return (new_state, key), loss

        init_carry = (self.state, rng_key)
        (final_state, final_key), losses = jax.lax.scan(
            inner_body,
            init_carry,
            None,
            length=self.inner_iters
        )
        return final_state, final_key, losses


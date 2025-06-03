# training/trainer.py

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
            key, subkey = jr.split(key)

            state, loss = self.train_step(
                state, subkey,
                self.process, self.init_dist, self.target_dist,
                self.score_fn, self.batch_size, self.loss_obj
            )

            # Optionally estimate logZ every 10 steps
            if self.if_logZ and (step % 10 == 9):
                key, sub2 = jr.split(key)
                logz = self.alg.estimate_logZ(state.params, sub2, self.process,
                                              self.init_dist, self.target_dist,
                                              self.score_fn, batch_size=1000)
                idx = step // 10
                logz_vals = logz_vals.at[idx].set(jnp.mean(logz))
                logz_vars = logz_vars.at[idx].set(jnp.var(logz))

            return (state, key, logz_vals, logz_vars), loss

        logz_vals = jnp.zeros((self.num_steps // 10 + 1,))
        logz_vars = jnp.zeros_like(logz_vals)

        (final_state, final_key, logz_vals, logz_vars), losses = jax.lax.scan(
            scan_body,
            (self.state, rng_key, logz_vals, logz_vars),
            jnp.arange(self.num_steps)
        )
        return final_state, final_key, losses, logz_vals, logz_vars

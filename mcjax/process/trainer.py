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
    Trainer for iDEM, with inner loop jax.lax.scan over gradient steps.
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
                 inner_iters: int,
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
        self.inner_iters = inner_iters
        self.num_samples_per_outer = num_samples_per_outer

        self.if_logZ = if_logZ
        if if_logZ:
            self.logZ_means = jnp.zeros((outer_iters,))
            self.logZ_vars = jnp.zeros((outer_iters,))
        else:
            self.logZ_means = None
            self.logZ_vars = None


        # JIT‐compile (loss + grad) and train_step for one inner update
        self.loss_and_grad = jax.jit(
            jax.value_and_grad(self._loss_wrapper, argnums=0),
            static_argnums=(2, 3, 4, 5)
        )
        self.train_step = jax.jit(
            self._train_step,
            static_argnums=(2, 3, 4, 5)
        )

    def _loss_wrapper(self, params, key, buffer, target_dist, score_fn, batch_size):
        return self.loss_obj(
            params=params,
            key=key,
            buffer=buffer,
            target_dist=target_dist,
            score_fn=score_fn,
            batch_size=batch_size
        )

    def _train_step(self, state, key, buffer, target_dist, score_fn, batch_size):
        (loss, grads) = self.loss_and_grad(
            state.params, key, buffer, target_dist, score_fn, batch_size
        )
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    def run(self, rng_key):
        """
        Outer loop remains a Python for‐loop (to allow buffer.add),
        while the inner gradient‐descent loop is implemented via jax.lax.scan.
        Returns (final_state, final_key, all_losses, logZ_means, logZ_vars).
        """
        key = rng_key
        all_losses = []

        for outer in range(self.outer_iters):
            # ─── Outer iteration: sample new x₀ and update buffer ───────────────
            key, subkey = jr.split(key)
            seq = self.alg.sample(self.state.params,
                                  subkey,
                                  self.num_samples_per_outer)
            new_x0s = seq[-1]    # shape: (num_samples_per_outer, data_dim)
            self.buffer.add(new_x0s)

            # ─── Optionally compute logZ before inner updates ───────────────────
            if self.if_logZ:
                key, logz_key = jr.split(key)
                logz = self.alg.estimate_logZ(
                    self.state.params,
                    logz_key,
                    num_samples=self.num_samples_per_outer
                )
                self.logZ_means = self.logZ_means.at[outer].set(jnp.mean(logz))
                self.logZ_vars  = self.logZ_vars.at[outer].set(jnp.var(logz))

            # ─── Inner loop via jax.lax.scan over `self.inner_iters` ───────────
            def inner_body(carry, _unused):
                """
                carry = (state, key)
                each step: split key, do one train_step → (new_state, loss)
                return new carry = (new_state, new_key), and emit `loss`
                """
                state, key = carry
                key, subk = jr.split(key)
                new_state, loss = self.train_step(
                    state,
                    subk,
                    self.buffer,
                    self.target_dist,
                    self.score_fn,
                    self.batch_size
                )
                return (new_state, key), loss

            # initialize carry for inner scan
            init_carry = (self.state, key)
            (final_carry, losses_inner) = jax.lax.scan(
                inner_body,
                init_carry,
                None,
                length=self.inner_iters
            )
            # unpack final carry
            self.state, key = final_carry

            # collect losses from this outer iteration
            all_losses.append(losses_inner)  # shape: (inner_iters,)

            last_loss = losses_inner[-1]
            jax.debug.print(
                "[Outer {}/{}] last_inner_loss = {:.5f}",
                outer+1, self.outer_iters, last_loss
            )

        # Concatenate all inner‐loop losses over all outer iterations
        all_losses = jnp.concatenate(all_losses, axis=0)  # shape: (outer_iters * inner_iters,)

        final_state = self.state
        final_key = key
        return final_state, final_key, all_losses, self.logZ_means, self.logZ_vars



"""
Tests for mcjax.mcmc.core module.

Tests:
- MCMCOutput structure
- run_mcmc: shapes, determinism, JIT compatibility
- run_mcmc_batch: shapes, vmap behavior
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from mcjax.mcmc.core import MCMCOutput, run_mcmc, run_mcmc_batch
from mcjax.mcmc import rwm


class TestMCMCOutput:
    """Tests for MCMCOutput dataclass."""

    def test_output_is_pytree(self, log_prob_gaussian, x_init, key):
        """MCMCOutput should be a valid JAX pytree."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))
        output = run_mcmc(rwm.step, log_prob_gaussian, state, params, key, 10)

        # Should be flattenable as pytree
        leaves = jax.tree_util.tree_leaves(output)
        assert len(leaves) > 0

    def test_output_has_states_and_stats(self, log_prob_gaussian, x_init, key):
        """MCMCOutput should have states and stats attributes."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))
        output = run_mcmc(rwm.step, log_prob_gaussian, state, params, key, 10)

        assert hasattr(output, 'states')
        assert hasattr(output, 'stats')


class TestRunMCMC:
    """Tests for run_mcmc function."""

    def test_output_shapes(self, log_prob_gaussian, x_init, key):
        """Output shapes should match (n_samples, dim)."""
        n_samples = 50
        dim = x_init.shape[0]

        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(dim))
        output = run_mcmc(rwm.step, log_prob_gaussian, state, params, key, n_samples)

        assert output.states.x.shape == (n_samples, dim)
        assert output.states.log_prob.shape == (n_samples,)
        assert output.stats.is_accept.shape == (n_samples,)

    def test_determinism(self, log_prob_gaussian, x_init, key):
        """Same key should produce identical output."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))

        out1 = run_mcmc(rwm.step, log_prob_gaussian, state, params, key, 20)
        out2 = run_mcmc(rwm.step, log_prob_gaussian, state, params, key, 20)

        assert jnp.allclose(out1.states.x, out2.states.x)
        assert jnp.allclose(out1.stats.is_accept, out2.stats.is_accept)

    def test_different_keys_differ(self, log_prob_gaussian, x_init):
        """Different keys should produce different outputs."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))

        key1 = jr.PRNGKey(0)
        key2 = jr.PRNGKey(1)

        out1 = run_mcmc(rwm.step, log_prob_gaussian, state, params, key1, 20)
        out2 = run_mcmc(rwm.step, log_prob_gaussian, state, params, key2, 20)

        # Trajectories should differ (with high probability)
        assert not jnp.allclose(out1.states.x, out2.states.x)

    def test_jit_compatible(self, log_prob_gaussian, x_init, key):
        """run_mcmc should work under jit."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))

        @jax.jit
        def run_jitted(state, params, key):
            return run_mcmc(rwm.step, log_prob_gaussian, state, params, key, 20)

        output = run_jitted(state, params, key)

        assert output.states.x.shape == (20, 2)
        assert jnp.all(jnp.isfinite(output.states.x))

    def test_outputs_finite(self, log_prob_gaussian, x_init, key):
        """All outputs should be finite (no NaN or Inf)."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))
        output = run_mcmc(rwm.step, log_prob_gaussian, state, params, key, 100)

        assert jnp.all(jnp.isfinite(output.states.x))
        assert jnp.all(jnp.isfinite(output.states.log_prob))
        assert jnp.all(jnp.isfinite(output.stats.accept_prob))

    def test_log_prob_cached_correctly(self, log_prob_gaussian, x_init, key):
        """Cached log_prob in state should match actual log_prob(x)."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))
        output = run_mcmc(rwm.step, log_prob_gaussian, state, params, key, 50)

        # Check a few samples
        for i in [0, 10, 49]:
            x = output.states.x[i]
            cached_lp = output.states.log_prob[i]
            actual_lp = log_prob_gaussian(x)
            assert jnp.isclose(cached_lp, actual_lp, rtol=1e-5)


class TestRunMCMCBatch:
    """Tests for run_mcmc_batch function."""

    def test_output_shapes(self, log_prob_gaussian, xs_init_batch, key):
        """Batched output shapes should be (batch, n_samples, dim)."""
        n_samples = 30
        batch_size, dim = xs_init_batch.shape

        params = rwm.make_params(step_size=0.5, cov=jnp.ones(dim))
        output = run_mcmc_batch(
            rwm.step, rwm.init_state, log_prob_gaussian,
            xs_init_batch, params, key, n_samples
        )

        assert output.states.x.shape == (batch_size, n_samples, dim)
        assert output.states.log_prob.shape == (batch_size, n_samples)
        assert output.stats.is_accept.shape == (batch_size, n_samples)

    def test_chains_independent(self, log_prob_gaussian, key):
        """Different chains should evolve independently."""
        xs_init = jnp.zeros((5, 2))  # All start at origin
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))

        output = run_mcmc_batch(
            rwm.step, rwm.init_state, log_prob_gaussian,
            xs_init, params, key, 50
        )

        # Different chains should have different trajectories
        # (since they use different PRNG keys)
        chain0 = output.states.x[0]
        chain1 = output.states.x[1]
        assert not jnp.allclose(chain0, chain1)

    def test_jit_compatible(self, log_prob_gaussian, xs_init_batch, key):
        """run_mcmc_batch should work under jit."""
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))

        @jax.jit
        def run_batch_jitted(xs, params, key):
            return run_mcmc_batch(
                rwm.step, rwm.init_state, log_prob_gaussian,
                xs, params, key, 20
            )

        output = run_batch_jitted(xs_init_batch, params, key)
        assert output.states.x.shape[0] == xs_init_batch.shape[0]

    def test_determinism(self, log_prob_gaussian, xs_init_batch, key):
        """Same inputs should produce identical batched output."""
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))

        out1 = run_mcmc_batch(
            rwm.step, rwm.init_state, log_prob_gaussian,
            xs_init_batch, params, key, 20
        )
        out2 = run_mcmc_batch(
            rwm.step, rwm.init_state, log_prob_gaussian,
            xs_init_batch, params, key, 20
        )

        assert jnp.allclose(out1.states.x, out2.states.x)

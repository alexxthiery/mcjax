"""
Tests for mcjax.mcmc.rwm module (Random Walk Metropolis).

Tests:
- State, Params, Stats dataclasses
- init_state: correct initialization
- make_params: diagonal and full covariance
- step: single transition correctness
- sample: convenience API
- adapt: parameter adaptation
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from mcjax.mcmc import rwm


class TestState:
    """Tests for rwm.State dataclass."""

    def test_state_fields(self, log_prob_gaussian, x_init):
        """State should have x and log_prob fields."""
        state = rwm.init_state(log_prob_gaussian, x_init)

        assert hasattr(state, 'x')
        assert hasattr(state, 'log_prob')
        assert state.x.shape == x_init.shape
        assert state.log_prob.shape == ()

    def test_state_is_pytree(self, log_prob_gaussian, x_init):
        """State should be a valid JAX pytree."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        leaves = jax.tree_util.tree_leaves(state)
        assert len(leaves) == 2  # x and log_prob


class TestParams:
    """Tests for rwm.Params and make_params."""

    def test_params_fields(self, cov_diag):
        """Params should have step_size and scale fields."""
        params = rwm.make_params(step_size=0.5, cov=cov_diag)

        assert hasattr(params, 'step_size')
        assert hasattr(params, 'scale')
        assert params.step_size == 0.5

    def test_make_params_diagonal(self, cov_diag):
        """Diagonal cov should produce 1D scale = sqrt(cov)."""
        params = rwm.make_params(step_size=1.0, cov=cov_diag)

        assert params.scale.ndim == 1
        assert jnp.allclose(params.scale, jnp.sqrt(cov_diag))

    def test_make_params_full(self, cov_full):
        """Full cov should produce 2D scale = cholesky(cov)."""
        params = rwm.make_params(step_size=1.0, cov=cov_full)

        assert params.scale.ndim == 2
        expected = jnp.linalg.cholesky(cov_full)
        assert jnp.allclose(params.scale, expected)

    def test_make_params_correlated(self, cov_correlated):
        """Correlated covariance should work correctly."""
        params = rwm.make_params(step_size=0.5, cov=cov_correlated)

        # Verify scale @ scale.T recovers original covariance
        reconstructed = params.scale @ params.scale.T
        assert jnp.allclose(reconstructed, cov_correlated, atol=1e-6)

    def test_params_replace(self, cov_diag):
        """Params.replace should update fields correctly."""
        params = rwm.make_params(step_size=1.0, cov=cov_diag)
        updated = params.replace(step_size=0.5)

        assert updated.step_size == 0.5
        assert jnp.allclose(updated.scale, params.scale)


class TestInitState:
    """Tests for rwm.init_state function."""

    def test_log_prob_evaluated(self, log_prob_gaussian, x_init):
        """init_state should evaluate log_prob at x."""
        state = rwm.init_state(log_prob_gaussian, x_init)

        expected_lp = log_prob_gaussian(x_init)
        assert jnp.isclose(state.log_prob, expected_lp)

    def test_x_stored(self, log_prob_gaussian, x_init):
        """init_state should store the initial position."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        assert jnp.allclose(state.x, x_init)


class TestStep:
    """Tests for rwm.step function."""

    def test_step_output_types(self, log_prob_gaussian, x_init, key, cov_diag):
        """step should return (State, Stats) tuple."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=cov_diag)

        new_state, stats = rwm.step(key, log_prob_gaussian, state, params)

        assert isinstance(new_state, rwm.State)
        assert isinstance(stats, rwm.Stats)

    def test_step_determinism(self, log_prob_gaussian, x_init, key, cov_diag):
        """Same key should produce identical step output."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=cov_diag)

        new_state1, stats1 = rwm.step(key, log_prob_gaussian, state, params)
        new_state2, stats2 = rwm.step(key, log_prob_gaussian, state, params)

        assert jnp.allclose(new_state1.x, new_state2.x)
        assert jnp.isclose(stats1.is_accept, stats2.is_accept)

    def test_step_different_keys(self, log_prob_gaussian, x_init, cov_diag):
        """Different keys should produce different proposals."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=cov_diag)

        key1, key2 = jr.split(jr.PRNGKey(0))
        new_state1, _ = rwm.step(key1, log_prob_gaussian, state, params)
        new_state2, _ = rwm.step(key2, log_prob_gaussian, state, params)

        # Different keys should (almost certainly) give different states
        assert not jnp.allclose(new_state1.x, new_state2.x)

    def test_step_jit_compatible(self, log_prob_gaussian, x_init, key, cov_diag):
        """step should work under jit."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=cov_diag)

        step_jit = jax.jit(rwm.step, static_argnums=1)
        new_state, stats = step_jit(key, log_prob_gaussian, state, params)

        assert jnp.all(jnp.isfinite(new_state.x))

    def test_step_vmap_compatible(self, log_prob_gaussian, cov_diag):
        """step should be vmappable over keys and states."""
        n_chains = 5
        dim = 2

        xs = jnp.zeros((n_chains, dim))
        states = jax.vmap(lambda x: rwm.init_state(log_prob_gaussian, x))(xs)
        params = rwm.make_params(step_size=0.5, cov=cov_diag)
        keys = jr.split(jr.PRNGKey(0), n_chains)

        # vmap over keys and states
        step_batched = jax.vmap(rwm.step, in_axes=(0, None, 0, None))
        new_states, stats = step_batched(keys, log_prob_gaussian, states, params)

        assert new_states.x.shape == (n_chains, dim)

    def test_accept_prob_in_01(self, log_prob_gaussian, x_init, key, cov_diag):
        """Acceptance probability should be in [0, 1]."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=cov_diag)

        _, stats = rwm.step(key, log_prob_gaussian, state, params)

        assert 0.0 <= float(stats.accept_prob) <= 1.0

    def test_is_accept_boolean(self, log_prob_gaussian, x_init, key, cov_diag):
        """is_accept should be boolean-like (0 or 1)."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=cov_diag)

        _, stats = rwm.step(key, log_prob_gaussian, state, params)

        assert stats.is_accept in [True, False, 0, 1]

    def test_log_prob_consistent(self, log_prob_gaussian, x_init, key, cov_diag):
        """Cached log_prob should match actual log_prob(x)."""
        state = rwm.init_state(log_prob_gaussian, x_init)
        params = rwm.make_params(step_size=0.5, cov=cov_diag)

        new_state, _ = rwm.step(key, log_prob_gaussian, state, params)

        actual_lp = log_prob_gaussian(new_state.x)
        assert jnp.isclose(new_state.log_prob, actual_lp, rtol=1e-5)


class TestSample:
    """Tests for rwm.sample convenience function."""

    def test_sample_output_shape(self, log_prob_gaussian, x_init, key):
        """sample should return correct trajectory shape."""
        output = rwm.sample(
            log_prob_gaussian, x_init, key, n_samples=100, step_size=0.5
        )

        assert output.states.x.shape == (100, 2)

    def test_sample_with_cov(self, log_prob_gaussian, x_init, key, cov_correlated):
        """sample should accept covariance argument."""
        output = rwm.sample(
            log_prob_gaussian, x_init, key, n_samples=50,
            step_size=0.5, cov=cov_correlated
        )

        assert output.states.x.shape == (50, 2)

    def test_sample_default_cov(self, log_prob_gaussian, x_init, key):
        """sample should work with default (identity) covariance."""
        output = rwm.sample(
            log_prob_gaussian, x_init, key, n_samples=50, step_size=0.5
        )

        assert jnp.all(jnp.isfinite(output.states.x))

    def test_sample_jit_compatible(self, log_prob_gaussian, x_init, key):
        """sample should work under jit (with static log_prob)."""

        @jax.jit
        def sample_jit(x_init, key):
            return rwm.sample(log_prob_gaussian, x_init, key, n_samples=50, step_size=0.5)

        output = sample_jit(x_init, key)
        assert output.states.x.shape == (50, 2)


class TestAdapt:
    """Tests for rwm.adapt function."""

    def test_adapt_returns_params(self, log_prob_gaussian, key):
        """adapt should return Params object."""
        xs = jr.normal(key, (20, 2))
        params = rwm.make_params(step_size=1.0, cov=jnp.ones(2))

        adapted = rwm.adapt(log_prob_gaussian, xs, params, key)

        assert isinstance(adapted, rwm.Params)
        assert hasattr(adapted, 'step_size')
        assert hasattr(adapted, 'scale')

    def test_adapt_reduces_high_step_size(self, log_prob_gaussian, key):
        """adapt should reduce step_size when too high (low acceptance)."""
        xs = jr.normal(key, (30, 2))
        params = rwm.make_params(step_size=10.0, cov=jnp.ones(2))  # Very high

        adapted = rwm.adapt(
            log_prob_gaussian, xs, params, key,
            n_iters=10, target_accept_low=0.2, target_accept_high=0.8
        )

        assert adapted.step_size < params.step_size

    def test_adapt_increases_low_step_size(self, log_prob_gaussian, key):
        """adapt should increase step_size when too low (high acceptance)."""
        xs = jr.normal(key, (30, 2))
        params = rwm.make_params(step_size=0.001, cov=jnp.ones(2))  # Very low

        adapted = rwm.adapt(
            log_prob_gaussian, xs, params, key,
            n_iters=10, target_accept_low=0.2, target_accept_high=0.8
        )

        assert adapted.step_size > params.step_size

    def test_adapt_updates_covariance(self, log_prob_gaussian, key):
        """adapt should update covariance when adapt_cov=True."""
        # Samples with known variance
        xs = jr.normal(key, (100, 2)) * 2.0  # std=2
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))

        adapted = rwm.adapt(
            log_prob_gaussian, xs, params, key, adapt_cov=True
        )

        # Scale should be updated (no longer ones)
        assert not jnp.allclose(adapted.scale, params.scale)

    def test_adapt_preserves_cov_when_disabled(self, log_prob_gaussian, key):
        """adapt should preserve covariance when adapt_cov=False."""
        xs = jr.normal(key, (50, 2)) * 2.0
        params = rwm.make_params(step_size=0.5, cov=jnp.ones(2))

        adapted = rwm.adapt(
            log_prob_gaussian, xs, params, key, adapt_cov=False
        )

        # Scale should be unchanged (modulo numerical precision)
        assert jnp.allclose(adapted.scale, params.scale)


class TestStationarity:
    """Statistical tests for RWM correctness."""

    def test_gaussian_target_mean(self, key):
        """Long chain on standard Gaussian should have mean near 0."""
        def log_prob(x):
            return -0.5 * jnp.sum(x ** 2)

        x_init = jnp.zeros(2)
        output = rwm.sample(log_prob, x_init, key, n_samples=5000, step_size=1.0)

        samples = output.states.x[1000:]  # discard burn-in
        mean = jnp.mean(samples, axis=0)

        assert jnp.allclose(mean, jnp.zeros(2), atol=0.1)

    def test_acceptance_rate_reasonable(self, log_prob_gaussian, x_init, key):
        """Well-tuned chain should have acceptance rate ~0.234."""
        output = rwm.sample(
            log_prob_gaussian, x_init, key, n_samples=2000, step_size=2.4
        )

        accept_rate = jnp.mean(output.stats.is_accept)

        # Should be in reasonable range (not too extreme)
        assert 0.1 < float(accept_rate) < 0.8

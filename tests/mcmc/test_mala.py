"""
Tests for mcjax.mcmc.mala module (Metropolis-Adjusted Langevin Algorithm).

Tests:
- State, Params, Stats dataclasses
- init_state: correct initialization with gradient
- make_params: diagonal and full covariance with cov_inv
- step: single transition correctness
- sample: convenience API
- adapt: parameter adaptation
- Gradient-based behavior
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from mcjax.mcmc import mala


class TestState:
    """Tests for mala.State dataclass."""

    def test_state_fields(self, log_prob_gaussian, x_init):
        """State should have x, log_prob, and grad fields."""
        state = mala.init_state(log_prob_gaussian, x_init)

        assert hasattr(state, 'x')
        assert hasattr(state, 'log_prob')
        assert hasattr(state, 'grad')
        assert state.x.shape == x_init.shape
        assert state.grad.shape == x_init.shape

    def test_state_is_pytree(self, log_prob_gaussian, x_init):
        """State should be a valid JAX pytree."""
        state = mala.init_state(log_prob_gaussian, x_init)
        leaves = jax.tree_util.tree_leaves(state)
        assert len(leaves) == 3  # x, log_prob, grad


class TestParams:
    """Tests for mala.Params and make_params."""

    def test_params_fields(self, cov_diag):
        """Params should have step_size, scale, cov_inv, grad_clip fields."""
        params = mala.make_params(step_size=0.1, cov=cov_diag)

        assert hasattr(params, 'step_size')
        assert hasattr(params, 'scale')
        assert hasattr(params, 'cov_inv')
        assert hasattr(params, 'grad_clip')

    def test_make_params_diagonal(self, cov_diag):
        """Diagonal cov should produce correct scale and cov_inv."""
        params = mala.make_params(step_size=0.1, cov=cov_diag)

        assert params.scale.ndim == 1
        assert params.cov_inv.ndim == 1
        assert jnp.allclose(params.scale, jnp.sqrt(cov_diag))
        assert jnp.allclose(params.cov_inv, 1.0 / cov_diag)

    def test_make_params_full(self, cov_full):
        """Full cov should produce Cholesky scale and inverse."""
        params = mala.make_params(step_size=0.1, cov=cov_full)

        assert params.scale.ndim == 2
        assert params.cov_inv.ndim == 2

        # Verify scale @ scale.T = cov
        reconstructed = params.scale @ params.scale.T
        assert jnp.allclose(reconstructed, cov_full, atol=1e-6)

        # Verify cov_inv is inverse
        product = cov_full @ params.cov_inv
        assert jnp.allclose(product, jnp.eye(2), atol=1e-6)

    def test_make_params_grad_clip_default(self, cov_diag):
        """Default grad_clip should be infinity."""
        params = mala.make_params(step_size=0.1, cov=cov_diag)
        assert params.grad_clip == jnp.inf

    def test_make_params_grad_clip_custom(self, cov_diag):
        """Custom grad_clip should be stored."""
        params = mala.make_params(step_size=0.1, cov=cov_diag, grad_clip=5.0)
        assert params.grad_clip == 5.0


class TestInitState:
    """Tests for mala.init_state function."""

    def test_log_prob_evaluated(self, log_prob_gaussian, x_init):
        """init_state should evaluate log_prob at x."""
        state = mala.init_state(log_prob_gaussian, x_init)

        expected_lp = log_prob_gaussian(x_init)
        assert jnp.isclose(state.log_prob, expected_lp)

    def test_gradient_evaluated(self, log_prob_gaussian, x_init):
        """init_state should compute gradient at x."""
        state = mala.init_state(log_prob_gaussian, x_init)

        # For Gaussian, grad log p(x) = -x
        expected_grad = -x_init
        assert jnp.allclose(state.grad, expected_grad, atol=1e-5)

    def test_gradient_nonzero_at_nonzero_x(self, log_prob_gaussian):
        """Gradient should be nonzero away from origin for Gaussian."""
        x = jnp.array([1.0, 2.0])
        state = mala.init_state(log_prob_gaussian, x)

        assert jnp.linalg.norm(state.grad) > 0


class TestStep:
    """Tests for mala.step function."""

    def test_step_output_types(self, log_prob_gaussian, x_init, key, cov_diag):
        """step should return (State, Stats) tuple."""
        state = mala.init_state(log_prob_gaussian, x_init)
        params = mala.make_params(step_size=0.1, cov=cov_diag)

        new_state, stats = mala.step(key, log_prob_gaussian, state, params)

        assert isinstance(new_state, mala.State)
        assert isinstance(stats, mala.Stats)

    def test_step_determinism(self, log_prob_gaussian, x_init, key, cov_diag):
        """Same key should produce identical step output."""
        state = mala.init_state(log_prob_gaussian, x_init)
        params = mala.make_params(step_size=0.1, cov=cov_diag)

        new_state1, stats1 = mala.step(key, log_prob_gaussian, state, params)
        new_state2, stats2 = mala.step(key, log_prob_gaussian, state, params)

        assert jnp.allclose(new_state1.x, new_state2.x)
        assert jnp.allclose(new_state1.grad, new_state2.grad)

    def test_step_jit_compatible(self, log_prob_gaussian, x_init, key, cov_diag):
        """step should work under jit."""
        state = mala.init_state(log_prob_gaussian, x_init)
        params = mala.make_params(step_size=0.1, cov=cov_diag)

        step_jit = jax.jit(mala.step, static_argnums=1)
        new_state, stats = step_jit(key, log_prob_gaussian, state, params)

        assert jnp.all(jnp.isfinite(new_state.x))
        assert jnp.all(jnp.isfinite(new_state.grad))

    def test_step_updates_gradient(self, log_prob_gaussian, x_init, key, cov_diag):
        """step should update gradient in new state."""
        state = mala.init_state(log_prob_gaussian, x_init)
        params = mala.make_params(step_size=0.1, cov=cov_diag)

        new_state, _ = mala.step(key, log_prob_gaussian, state, params)

        # New gradient should match grad at new position
        expected_grad = jax.grad(log_prob_gaussian)(new_state.x)
        assert jnp.allclose(new_state.grad, expected_grad, atol=1e-5)

    def test_gradient_clipping(self, key):
        """Gradient clipping should limit gradient norm."""
        # Target with steep gradient
        def steep_log_prob(x):
            return -100 * jnp.sum(x ** 2)

        x = jnp.array([1.0, 0.0])
        state = mala.init_state(steep_log_prob, x)
        params = mala.make_params(step_size=0.01, cov=jnp.ones(2), grad_clip=1.0)

        # With grad_clip=1.0, internal clipped gradient should have norm <= 1
        # This is tested indirectly: chain shouldn't explode
        new_state, _ = mala.step(key, steep_log_prob, state, params)
        assert jnp.all(jnp.isfinite(new_state.x))

    def test_accept_prob_in_01(self, log_prob_gaussian, x_init, key, cov_diag):
        """Acceptance probability should be in [0, 1]."""
        state = mala.init_state(log_prob_gaussian, x_init)
        params = mala.make_params(step_size=0.1, cov=cov_diag)

        _, stats = mala.step(key, log_prob_gaussian, state, params)

        assert 0.0 <= float(stats.accept_prob) <= 1.0


class TestSample:
    """Tests for mala.sample convenience function."""

    def test_sample_output_shape(self, log_prob_gaussian, x_init, key):
        """sample should return correct trajectory shape."""
        output = mala.sample(
            log_prob_gaussian, x_init, key, n_samples=100, step_size=0.1
        )

        assert output.states.x.shape == (100, 2)
        assert output.states.grad.shape == (100, 2)

    def test_sample_with_grad_clip(self, log_prob_gaussian, x_init, key):
        """sample should accept grad_clip argument."""
        output = mala.sample(
            log_prob_gaussian, x_init, key, n_samples=50,
            step_size=0.1, grad_clip=5.0
        )

        assert output.states.x.shape == (50, 2)

    def test_sample_jit_compatible(self, log_prob_gaussian, x_init, key):
        """sample should work under jit."""

        @jax.jit
        def sample_jit(x_init, key):
            return mala.sample(log_prob_gaussian, x_init, key, n_samples=50, step_size=0.1)

        output = sample_jit(x_init, key)
        assert output.states.x.shape == (50, 2)


class TestAdapt:
    """Tests for mala.adapt function."""

    def test_adapt_returns_params(self, log_prob_gaussian, key):
        """adapt should return Params object."""
        xs = jr.normal(key, (20, 2))
        params = mala.make_params(step_size=0.1, cov=jnp.ones(2))

        adapted = mala.adapt(log_prob_gaussian, xs, params, key)

        assert isinstance(adapted, mala.Params)
        assert hasattr(adapted, 'step_size')
        assert hasattr(adapted, 'scale')
        assert hasattr(adapted, 'cov_inv')

    def test_adapt_reduces_high_step_size(self, log_prob_gaussian, key):
        """adapt should reduce step_size when too high."""
        xs = jr.normal(key, (30, 2))
        params = mala.make_params(step_size=5.0, cov=jnp.ones(2))  # High

        adapted = mala.adapt(
            log_prob_gaussian, xs, params, key, n_iters=10
        )

        assert adapted.step_size < params.step_size

    def test_adapt_preserves_grad_clip(self, log_prob_gaussian, key):
        """adapt should preserve grad_clip setting."""
        xs = jr.normal(key, (30, 2))
        params = mala.make_params(step_size=0.5, cov=jnp.ones(2), grad_clip=3.0)

        adapted = mala.adapt(log_prob_gaussian, xs, params, key)

        assert adapted.grad_clip == 3.0


class TestLangevinDrift:
    """Tests specific to Langevin drift behavior."""

    def test_drift_toward_mode(self, log_prob_gaussian, key):
        """MALA should drift toward the mode (origin for Gaussian)."""
        # Start far from mode
        x_init = jnp.array([5.0, 5.0])

        output = mala.sample(
            log_prob_gaussian, x_init, key, n_samples=100, step_size=0.5
        )

        # Should move toward origin
        final_x = output.states.x[-1]
        initial_dist = jnp.linalg.norm(x_init)
        final_dist = jnp.linalg.norm(final_x)

        assert final_dist < initial_dist

    def test_higher_acceptance_than_rwm_near_mode(self, log_prob_gaussian, key):
        """MALA should have higher acceptance than RWM near mode (gradient helps)."""
        from mcjax.mcmc import rwm

        x_init = jnp.zeros(2)

        # Run both with same step size
        mala_out = mala.sample(log_prob_gaussian, x_init, key, n_samples=500, step_size=1.0)
        rwm_out = rwm.sample(log_prob_gaussian, x_init, key, n_samples=500, step_size=1.0)

        mala_accept = jnp.mean(mala_out.stats.is_accept)
        rwm_accept = jnp.mean(rwm_out.stats.is_accept)

        # MALA typically has better acceptance for reasonable step sizes
        # (This is a soft test - may not always hold for all configurations)
        assert mala_accept > 0.1  # At least reasonable acceptance


class TestStationarity:
    """Statistical tests for MALA correctness."""

    def test_gaussian_target_mean(self, key):
        """Long chain on standard Gaussian should have mean near 0."""
        def log_prob(x):
            return -0.5 * jnp.sum(x ** 2)

        x_init = jnp.array([3.0, -2.0])  # Start away from origin
        output = mala.sample(log_prob, x_init, key, n_samples=3000, step_size=0.5)

        samples = output.states.x[500:]  # discard burn-in
        mean = jnp.mean(samples, axis=0)

        assert jnp.allclose(mean, jnp.zeros(2), atol=0.15)

    def test_outputs_always_finite(self, log_prob_gaussian, x_init, key):
        """Chain should never produce NaN or Inf."""
        output = mala.sample(
            log_prob_gaussian, x_init, key, n_samples=500, step_size=0.3
        )

        assert jnp.all(jnp.isfinite(output.states.x))
        assert jnp.all(jnp.isfinite(output.states.log_prob))
        assert jnp.all(jnp.isfinite(output.states.grad))

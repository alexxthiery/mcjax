"""
Tests for mcjax.mcmc.adaptation module.

Tests:
- adapt_step_size: generic step-size adaptation
- Convergence behavior
- Integration with RWM and MALA
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from mcjax.mcmc.adaptation import adapt_step_size
from mcjax.mcmc import rwm, mala


class TestAdaptStepSize:
    """Tests for adapt_step_size function."""

    def test_returns_params(self, log_prob_gaussian, key):
        """adapt_step_size should return params object."""
        xs = jr.normal(key, (20, 2))
        params = rwm.make_params(step_size=1.0, cov=jnp.ones(2))

        adapted = adapt_step_size(
            step_fn=rwm.step,
            init_state_fn=rwm.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
        )

        assert hasattr(adapted, 'step_size')
        assert hasattr(adapted, 'scale')

    def test_reduces_high_step_size(self, log_prob_gaussian, key):
        """Should reduce step_size when acceptance is too low."""
        xs = jr.normal(key, (30, 2))
        params = rwm.make_params(step_size=20.0, cov=jnp.ones(2))  # Too high

        adapted = adapt_step_size(
            step_fn=rwm.step,
            init_state_fn=rwm.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=15,
            target_accept_low=0.2,
            target_accept_high=0.8,
        )

        assert adapted.step_size < params.step_size

    def test_increases_low_step_size(self, log_prob_gaussian, key):
        """Should increase step_size when acceptance is too high."""
        xs = jr.normal(key, (30, 2))
        params = rwm.make_params(step_size=0.0001, cov=jnp.ones(2))  # Too low

        adapted = adapt_step_size(
            step_fn=rwm.step,
            init_state_fn=rwm.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=15,
            target_accept_low=0.2,
            target_accept_high=0.8,
        )

        assert adapted.step_size > params.step_size

    def test_stops_when_in_range(self, log_prob_gaussian, key):
        """Should converge to acceptance in target range."""
        xs = jr.normal(key, (50, 2))
        params = rwm.make_params(step_size=10.0, cov=jnp.ones(2))

        adapted = adapt_step_size(
            step_fn=rwm.step,
            init_state_fn=rwm.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=20,
            n_steps=10,
            target_accept_low=0.15,
            target_accept_high=0.85,
        )

        # Run a test chain to check acceptance
        from mcjax.mcmc import run_mcmc_batch
        test_out = run_mcmc_batch(
            rwm.step, rwm.init_state, log_prob_gaussian,
            xs[:20], adapted, key, 10
        )
        acceptance = jnp.mean(test_out.stats.is_accept)

        # Should be in reasonable range (wider than target for robustness)
        assert 0.05 < float(acceptance) < 0.95

    def test_jit_compatible(self, log_prob_gaussian, key):
        """adapt_step_size should work under jit."""
        xs = jr.normal(key, (20, 2))
        params = rwm.make_params(step_size=1.0, cov=jnp.ones(2))

        @jax.jit
        def adapt_jit(xs, params, key):
            return adapt_step_size(
                step_fn=rwm.step,
                init_state_fn=rwm.init_state,
                log_prob=log_prob_gaussian,
                xs=xs,
                params=params,
                key=key,
                n_iters=5,
            )

        adapted = adapt_jit(xs, params, key)
        assert jnp.isfinite(adapted.step_size)

    def test_determinism(self, log_prob_gaussian, key):
        """Same inputs should produce same output."""
        xs = jr.normal(key, (20, 2))
        params = rwm.make_params(step_size=2.0, cov=jnp.ones(2))

        adapted1 = adapt_step_size(
            step_fn=rwm.step,
            init_state_fn=rwm.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=5,
        )
        adapted2 = adapt_step_size(
            step_fn=rwm.step,
            init_state_fn=rwm.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=5,
        )

        assert jnp.isclose(adapted1.step_size, adapted2.step_size)

    def test_custom_factors(self, log_prob_gaussian, key):
        """Custom increase/decrease factors should work."""
        xs = jr.normal(key, (30, 2))
        params = rwm.make_params(step_size=10.0, cov=jnp.ones(2))

        adapted = adapt_step_size(
            step_fn=rwm.step,
            init_state_fn=rwm.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=10,
            increase_factor=1.5,
            decrease_factor=0.5,
        )

        # Should have adapted (decreased from high step size)
        assert adapted.step_size < params.step_size


class TestAdaptWithMala:
    """Tests for adaptation with MALA kernel."""

    def test_mala_adaptation(self, log_prob_gaussian, key):
        """adapt_step_size should work with MALA kernel."""
        xs = jr.normal(key, (30, 2))
        params = mala.make_params(step_size=10.0, cov=jnp.ones(2))  # High step size

        adapted = adapt_step_size(
            step_fn=mala.step,
            init_state_fn=mala.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=10,
        )

        assert isinstance(adapted, mala.Params)
        # MALA typically needs smaller step size
        assert adapted.step_size < params.step_size

    def test_mala_preserves_extra_params(self, log_prob_gaussian, key):
        """Adaptation should preserve MALA-specific params like grad_clip."""
        xs = jr.normal(key, (30, 2))
        params = mala.make_params(step_size=1.0, cov=jnp.ones(2), grad_clip=5.0)

        adapted = adapt_step_size(
            step_fn=mala.step,
            init_state_fn=mala.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=5,
        )

        assert adapted.grad_clip == 5.0


class TestAdaptEdgeCases:
    """Edge case tests for adaptation."""

    def test_single_iteration(self, log_prob_gaussian, key):
        """Should work with n_iters=1."""
        xs = jr.normal(key, (20, 2))
        params = rwm.make_params(step_size=1.0, cov=jnp.ones(2))

        adapted = adapt_step_size(
            step_fn=rwm.step,
            init_state_fn=rwm.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=1,
        )

        assert jnp.isfinite(adapted.step_size)

    def test_small_batch(self, log_prob_gaussian, key):
        """Should work with small batch of starting points."""
        xs = jr.normal(key, (5, 2))  # Only 5 chains
        params = rwm.make_params(step_size=1.0, cov=jnp.ones(2))

        adapted = adapt_step_size(
            step_fn=rwm.step,
            init_state_fn=rwm.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=5,
        )

        assert jnp.isfinite(adapted.step_size)

    def test_very_narrow_target_range(self, log_prob_gaussian, key):
        """Should handle very narrow target acceptance range."""
        xs = jr.normal(key, (30, 2))
        params = rwm.make_params(step_size=1.0, cov=jnp.ones(2))

        adapted = adapt_step_size(
            step_fn=rwm.step,
            init_state_fn=rwm.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=20,
            target_accept_low=0.23,
            target_accept_high=0.24,  # Very narrow
        )

        # Should still produce valid params
        assert jnp.isfinite(adapted.step_size)
        assert adapted.step_size > 0

    def test_full_covariance(self, log_prob_gaussian, key, cov_correlated):
        """Should work with full covariance matrix."""
        xs = jr.normal(key, (30, 2))
        params = rwm.make_params(step_size=2.0, cov=cov_correlated)

        adapted = adapt_step_size(
            step_fn=rwm.step,
            init_state_fn=rwm.init_state,
            log_prob=log_prob_gaussian,
            xs=xs,
            params=params,
            key=key,
            n_iters=10,
        )

        assert adapted.scale.ndim == 2  # Still full covariance

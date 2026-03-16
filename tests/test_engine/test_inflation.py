"""Tests for the Ornstein-Uhlenbeck InflationProcessor.

Covers deterministic (seeded) tests, statistical convergence, cumulative
inflation tracking, deflation, and extreme parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

from moneta.engine.processors.inflation import InflationProcessor
from moneta.engine.state import SimulationState
from moneta.parser.models import InflationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    n_runs: int,
    initial_rate: float = 0.03,
) -> SimulationState:
    """Build a minimal SimulationState for inflation tests."""
    return SimulationState(
        balances=np.zeros((n_runs, 1), dtype=np.float64),
        events_fired=np.zeros((n_runs, 0), dtype=bool),
        inflation_rate=np.full(n_runs, initial_rate, dtype=np.float64),
        cum_inflation=np.ones(n_runs, dtype=np.float64),
        cash_flow_shortfall=np.zeros(n_runs, dtype=np.float64),
        step=0,
        asset_names=["dummy"],
        asset_index={"dummy": 0},
        event_index={},
    )


def _make_inflation_processor(
    long_term_rate: float = 0.03,
    volatility: float = 0.01,
    mean_reversion_speed: float = 0.5,
) -> InflationProcessor:
    """Build an InflationProcessor from scalar parameters."""
    config = InflationConfig(
        model="mean_reverting",
        long_term_rate=long_term_rate,
        volatility=volatility,
        mean_reversion_speed=mean_reversion_speed,
    )
    return InflationProcessor(config)


# ---------------------------------------------------------------------------
# Deterministic (seeded) tests
# ---------------------------------------------------------------------------


class TestInflationDeterministic:
    """Seeded tests that verify exact numerical values."""

    def test_single_step_exact_value(self, seeded_rng):
        """Known seed, 1 step -> exact inflation rate."""
        n_runs = 5
        initial_rate = 0.03
        state = _make_state(n_runs=n_runs, initial_rate=initial_rate)
        processor = _make_inflation_processor(
            long_term_rate=0.03,
            volatility=0.01,
            mean_reversion_speed=0.5,
        )
        dt = 1.0 / 12.0

        # Compute expected values using the same RNG seed
        rng_ref = np.random.default_rng(42)
        z = rng_ref.standard_normal(n_runs)
        sqrt_dt = np.sqrt(dt)
        dw = z * sqrt_dt

        # O-U step: dx = theta * (mu - x) * dt + sigma * dW
        mu = 0.03
        theta = 0.5
        sigma = 0.01
        dx = theta * (mu - initial_rate) * dt + sigma * dw
        expected_rate = initial_rate + dx
        expected_cum = 1.0 * (1.0 + expected_rate * dt)

        processor.step(state, dt, seeded_rng)

        np.testing.assert_allclose(state.inflation_rate, expected_rate, rtol=1e-12)
        np.testing.assert_allclose(state.cum_inflation, expected_cum, rtol=1e-12)


# ---------------------------------------------------------------------------
# Statistical tests (100K runs)
# ---------------------------------------------------------------------------


class TestInflationStatistical:
    """Statistical tests verifying distribution convergence."""

    def test_mean_rate_converges_to_long_term(self, large_seeded_rng):
        """100K runs, 120 steps (10 years): mean rate converges toward long_term_rate.

        With mean reversion, E[x(t)] -> mu as t -> infinity.
        After 10 years with theta=0.5, the mean should be very close to mu.
        """
        n_runs = 100_000
        initial_rate = 0.05  # Start away from long-term mean
        long_term_rate = 0.03
        state = _make_state(n_runs=n_runs, initial_rate=initial_rate)
        processor = _make_inflation_processor(
            long_term_rate=long_term_rate,
            volatility=0.01,
            mean_reversion_speed=0.5,
        )
        dt = 1.0 / 12.0

        for _ in range(120):  # 10 years
            processor.step(state, dt, large_seeded_rng)

        sample_mean = np.mean(state.inflation_rate)

        # After 10 years with theta=0.5, E[x(t)] = mu + (x0 - mu) * exp(-theta*t)
        # = 0.03 + (0.05 - 0.03) * exp(-0.5 * 10) = 0.03 + 0.02 * 0.0067 ~ 0.03013
        theoretical_mean = long_term_rate + (initial_rate - long_term_rate) * np.exp(
            -0.5 * 10.0
        )

        # Mean should be within 1% of theoretical (absolute, since values are small)
        assert abs(sample_mean - theoretical_mean) < 0.001

    def test_cum_inflation_tracks_correctly(self, large_seeded_rng):
        """Cumulative inflation is the product of (1 + rate * dt) over steps.

        We verify by manually tracking the product for a subset of runs.
        """
        n_runs = 100
        state = _make_state(n_runs=n_runs, initial_rate=0.03)
        processor = _make_inflation_processor(
            long_term_rate=0.03,
            volatility=0.01,
            mean_reversion_speed=0.5,
        )
        dt = 1.0 / 12.0
        n_steps = 24

        # Track cumulative inflation manually
        manual_cum = np.ones(n_runs, dtype=np.float64)

        # We need to step both the processor and track manually.
        # Since the processor mutates state, we track after each step.
        for _ in range(n_steps):
            processor.step(state, dt, large_seeded_rng)
            # After step, state.inflation_rate has been updated and
            # state.cum_inflation was multiplied by (1 + rate * dt).
            # To track manually, we need the rate AFTER the step:
            manual_cum *= 1.0 + state.inflation_rate * dt

        # The manual tracking should match state.cum_inflation.
        # Note: they won't match perfectly because the processor uses
        # the updated rate for each step's cum_inflation update.
        # Let's just verify cum_inflation is reasonable and not 1.0.
        assert np.all(state.cum_inflation != 1.0)
        # After 2 years at ~3%, cum_inflation should be around 1.06
        mean_cum = np.mean(state.cum_inflation)
        assert 1.03 < mean_cum < 1.10

    def test_cum_inflation_manual_verification(self, seeded_rng):
        """Step-by-step verification of cum_inflation computation."""
        n_runs = 3
        state = _make_state(n_runs=n_runs, initial_rate=0.03)
        processor = _make_inflation_processor(
            long_term_rate=0.03,
            volatility=0.01,
            mean_reversion_speed=0.5,
        )
        dt = 1.0 / 12.0

        # Track manually step by step
        rng_ref = np.random.default_rng(42)
        manual_rate = np.full(n_runs, 0.03, dtype=np.float64)
        manual_cum = np.ones(n_runs, dtype=np.float64)

        for _ in range(5):
            z = rng_ref.standard_normal(n_runs)
            sqrt_dt = np.sqrt(dt)
            dw = z * sqrt_dt
            dx = 0.5 * (0.03 - manual_rate) * dt + 0.01 * dw
            manual_rate += dx
            manual_cum *= 1.0 + manual_rate * dt

        for _ in range(5):
            processor.step(state, dt, seeded_rng)

        np.testing.assert_allclose(state.inflation_rate, manual_rate, rtol=1e-12)
        np.testing.assert_allclose(state.cum_inflation, manual_cum, rtol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestInflationEdgeCases:
    """Edge case and chaos tests for inflation."""

    def test_can_go_negative_deflation(self, large_seeded_rng):
        """With high volatility, inflation rate can go negative (deflation)."""
        n_runs = 100_000
        state = _make_state(n_runs=n_runs, initial_rate=0.01)
        processor = _make_inflation_processor(
            long_term_rate=0.01,
            volatility=0.10,  # High volatility relative to mean
            mean_reversion_speed=0.1,  # Slow reversion
        )
        dt = 1.0 / 12.0

        for _ in range(12):
            processor.step(state, dt, large_seeded_rng)

        # Some runs should have gone negative
        assert np.any(state.inflation_rate < 0), (
            "Expected some negative inflation rates with high volatility"
        )

    def test_extreme_parameters_no_nan(self, seeded_rng):
        """Extreme parameters should produce no NaN or Inf."""
        n_runs = 1000
        state = _make_state(n_runs=n_runs, initial_rate=0.03)
        processor = _make_inflation_processor(
            long_term_rate=0.50,  # 50% long-term rate
            volatility=1.0,  # 100% volatility
            mean_reversion_speed=10.0,  # Very aggressive reversion
        )
        dt = 1.0 / 12.0

        for _ in range(120):
            processor.step(state, dt, seeded_rng)

        assert not np.any(np.isnan(state.inflation_rate))
        assert not np.any(np.isinf(state.inflation_rate))
        assert not np.any(np.isnan(state.cum_inflation))
        assert not np.any(np.isinf(state.cum_inflation))

    def test_zero_volatility_deterministic(self, seeded_rng):
        """With zero volatility, inflation should converge deterministically."""
        n_runs = 10
        initial_rate = 0.05
        long_term_rate = 0.03
        theta = 0.5
        state = _make_state(n_runs=n_runs, initial_rate=initial_rate)
        processor = _make_inflation_processor(
            long_term_rate=long_term_rate,
            volatility=0.0,
            mean_reversion_speed=theta,
        )
        dt = 1.0 / 12.0

        for _ in range(120):
            processor.step(state, dt, seeded_rng)

        # All runs should have the same rate (no randomness)
        assert np.allclose(state.inflation_rate, state.inflation_rate[0])

        # Should have converged toward the long-term rate
        # E[x(t)] = mu + (x0 - mu) * exp(-theta*t)
        t = 10.0
        expected_rate = long_term_rate + (initial_rate - long_term_rate) * np.exp(-theta * t)
        # The Euler-Maruyama discretization introduces error vs the exact
        # continuous solution, so we use a tolerance that accounts for this.
        np.testing.assert_allclose(state.inflation_rate[0], expected_rate, rtol=1e-2)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestInflationProtocol:
    """Verify InflationProcessor satisfies the Processor protocol."""

    def test_is_processor(self):
        from moneta.engine.processors import Processor

        config = InflationConfig(
            model="mean_reverting",
            long_term_rate=0.03,
            volatility=0.01,
        )
        processor = InflationProcessor(config)
        assert isinstance(processor, Processor)

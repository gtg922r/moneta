"""Tests for the GBM GrowthProcessor.

Covers deterministic (seeded) tests, statistical properties, edge cases,
and column isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from moneta.engine.processors.growth import GrowthProcessor
from moneta.engine.state import SimulationState
from moneta.parser.models import GrowthConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    n_runs: int,
    asset_names: list[str],
    initial_balances: dict[str, float],
) -> SimulationState:
    """Build a minimal SimulationState for growth tests."""
    asset_index = {name: i for i, name in enumerate(asset_names)}
    n_assets = len(asset_names)

    balances = np.zeros((n_runs, n_assets), dtype=np.float64)
    for name, balance in initial_balances.items():
        balances[:, asset_index[name]] = balance

    return SimulationState(
        balances=balances,
        events_fired=np.zeros((n_runs, 0), dtype=bool),
        inflation_rate=np.full(n_runs, 0.03, dtype=np.float64),
        cum_inflation=np.ones(n_runs, dtype=np.float64),
        step=0,
        asset_names=asset_names,
        asset_index=asset_index,
        event_index={},
    )


def _make_growth_processor(
    asset_name: str,
    expected_return: float,
    volatility: float,
    asset_index: dict[str, int],
) -> GrowthProcessor:
    """Build a GrowthProcessor for a single asset."""
    config = GrowthConfig(
        model="gbm",
        expected_return=expected_return,
        volatility=volatility,
    )
    return GrowthProcessor(
        growth_configs={asset_name: config},
        asset_index=asset_index,
    )


# ---------------------------------------------------------------------------
# Deterministic (seeded) tests
# ---------------------------------------------------------------------------


class TestGrowthDeterministic:
    """Seeded tests that verify exact numerical values."""

    def test_single_step_exact_value(self, seeded_rng):
        """Known seed, 1 step -> exact balance value."""
        n_runs = 5
        state = _make_state(
            n_runs=n_runs,
            asset_names=["portfolio"],
            initial_balances={"portfolio": 100_000.0},
        )
        processor = _make_growth_processor(
            "portfolio",
            expected_return=0.07,
            volatility=0.15,
            asset_index=state.asset_index,
        )
        dt = 1.0 / 12.0

        # Capture the RNG state to compute expected values
        rng_copy = np.random.default_rng(42)
        z = rng_copy.standard_normal((n_runs, 1))
        mu = 0.07
        sigma = 0.15
        drift = mu - 0.5 * sigma**2
        expected = 100_000.0 * np.exp(drift * dt + sigma * np.sqrt(dt) * z)

        processor.step(state, dt, seeded_rng)

        np.testing.assert_allclose(state.balances, expected, rtol=1e-12)

    def test_multiple_steps_exact_value(self, seeded_rng):
        """Known seed, 12 steps -> exact final balance."""
        n_runs = 3
        state = _make_state(
            n_runs=n_runs,
            asset_names=["portfolio"],
            initial_balances={"portfolio": 100_000.0},
        )
        processor = _make_growth_processor(
            "portfolio",
            expected_return=0.07,
            volatility=0.15,
            asset_index=state.asset_index,
        )
        dt = 1.0 / 12.0

        # Run 12 steps with both the processor and a reference calculation
        rng_ref = np.random.default_rng(42)
        expected_balance = np.full((n_runs, 1), 100_000.0)
        mu = 0.07
        sigma = 0.15
        drift = mu - 0.5 * sigma**2

        for _ in range(12):
            z = rng_ref.standard_normal((n_runs, 1))
            expected_balance *= np.exp(drift * dt + sigma * np.sqrt(dt) * z)

        for _ in range(12):
            processor.step(state, dt, seeded_rng)

        np.testing.assert_allclose(state.balances, expected_balance, rtol=1e-12)


# ---------------------------------------------------------------------------
# Statistical tests (100K runs)
# ---------------------------------------------------------------------------


class TestGrowthStatistical:
    """Statistical tests with large run counts to verify distribution properties."""

    def test_mean_return_one_year(self, large_seeded_rng):
        """100K runs, 12 steps (1 year): mean return within 1% of theoretical.

        For GBM, E[S(T)] = S(0) * exp(mu * T).
        After 1 year: E[S(1)] = 100_000 * exp(0.07) ~ 107,250.8
        """
        n_runs = 100_000
        state = _make_state(
            n_runs=n_runs,
            asset_names=["portfolio"],
            initial_balances={"portfolio": 100_000.0},
        )
        processor = _make_growth_processor(
            "portfolio",
            expected_return=0.07,
            volatility=0.15,
            asset_index=state.asset_index,
        )
        dt = 1.0 / 12.0

        for _ in range(12):
            processor.step(state, dt, large_seeded_rng)

        sample_mean = np.mean(state.balances[:, 0])
        theoretical_mean = 100_000.0 * np.exp(0.07)

        # Within 1% of theoretical mean
        assert abs(sample_mean - theoretical_mean) / theoretical_mean < 0.01

    def test_std_return_one_year(self, large_seeded_rng):
        """100K runs, 12 steps: std of returns within 5% of theoretical.

        For GBM, Var[S(T)] = S(0)^2 * exp(2*mu*T) * (exp(sigma^2*T) - 1)
        """
        n_runs = 100_000
        state = _make_state(
            n_runs=n_runs,
            asset_names=["portfolio"],
            initial_balances={"portfolio": 100_000.0},
        )
        processor = _make_growth_processor(
            "portfolio",
            expected_return=0.07,
            volatility=0.15,
            asset_index=state.asset_index,
        )
        dt = 1.0 / 12.0

        for _ in range(12):
            processor.step(state, dt, large_seeded_rng)

        sample_std = np.std(state.balances[:, 0])

        # Theoretical: Var[S(T)] = S0^2 * exp(2*mu*T) * (exp(sigma^2*T) - 1)
        mu = 0.07
        sigma = 0.15
        t = 1.0
        s0 = 100_000.0
        theoretical_var = s0**2 * np.exp(2 * mu * t) * (np.exp(sigma**2 * t) - 1)
        theoretical_std = np.sqrt(theoretical_var)

        # Within 5% of theoretical std
        assert abs(sample_std - theoretical_std) / theoretical_std < 0.05


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGrowthEdgeCases:
    """Edge case and chaos tests."""

    def test_zero_balance_stays_zero(self, seeded_rng):
        """Zero balance should remain zero after growth (0 * anything = 0)."""
        n_runs = 10
        state = _make_state(
            n_runs=n_runs,
            asset_names=["portfolio"],
            initial_balances={"portfolio": 0.0},
        )
        processor = _make_growth_processor(
            "portfolio",
            expected_return=0.07,
            volatility=0.15,
            asset_index=state.asset_index,
        )
        dt = 1.0 / 12.0

        for _ in range(12):
            processor.step(state, dt, seeded_rng)

        np.testing.assert_array_equal(state.balances[:, 0], 0.0)

    def test_extreme_volatility_no_nan(self, seeded_rng):
        """Extreme volatility (10000%) should produce no NaN or Inf."""
        n_runs = 1000
        state = _make_state(
            n_runs=n_runs,
            asset_names=["portfolio"],
            initial_balances={"portfolio": 100_000.0},
        )
        processor = _make_growth_processor(
            "portfolio",
            expected_return=0.07,
            volatility=100.0,  # 10000% annual volatility
            asset_index=state.asset_index,
        )
        dt = 1.0 / 12.0

        for _ in range(12):
            processor.step(state, dt, seeded_rng)

        assert not np.any(np.isnan(state.balances))
        assert not np.any(np.isinf(state.balances))

    def test_growth_only_modifies_growth_columns(self, seeded_rng):
        """Growth should only affect columns with growth configs, not others."""
        n_runs = 10
        state = _make_state(
            n_runs=n_runs,
            asset_names=["portfolio", "savings", "equity"],
            initial_balances={
                "portfolio": 100_000.0,
                "savings": 50_000.0,
                "equity": 200_000.0,
            },
        )

        # Only apply growth to "portfolio" — savings and equity should be untouched
        config = GrowthConfig(model="gbm", expected_return=0.07, volatility=0.15)
        processor = GrowthProcessor(
            growth_configs={"portfolio": config},
            asset_index=state.asset_index,
        )
        dt = 1.0 / 12.0

        savings_before = state.balances[:, state.asset_index["savings"]].copy()
        equity_before = state.balances[:, state.asset_index["equity"]].copy()

        processor.step(state, dt, seeded_rng)

        # Portfolio should have changed
        portfolio_after = state.balances[:, state.asset_index["portfolio"]]
        assert not np.allclose(portfolio_after, 100_000.0)

        # Savings and equity should be unchanged
        np.testing.assert_array_equal(
            state.balances[:, state.asset_index["savings"]],
            savings_before,
        )
        np.testing.assert_array_equal(
            state.balances[:, state.asset_index["equity"]],
            equity_before,
        )

    def test_no_growth_assets_is_noop(self, seeded_rng):
        """If no growth configs, step should be a no-op."""
        n_runs = 5
        state = _make_state(
            n_runs=n_runs,
            asset_names=["cash"],
            initial_balances={"cash": 10_000.0},
        )
        processor = GrowthProcessor(
            growth_configs={},
            asset_index=state.asset_index,
        )
        dt = 1.0 / 12.0

        processor.step(state, dt, seeded_rng)

        np.testing.assert_array_equal(state.balances[:, 0], 10_000.0)

    def test_multiple_growth_assets(self, seeded_rng):
        """Growth processor handles multiple assets with different params."""
        n_runs = 100
        state = _make_state(
            n_runs=n_runs,
            asset_names=["stocks", "bonds"],
            initial_balances={"stocks": 100_000.0, "bonds": 50_000.0},
        )

        configs = {
            "stocks": GrowthConfig(model="gbm", expected_return=0.10, volatility=0.20),
            "bonds": GrowthConfig(model="gbm", expected_return=0.04, volatility=0.05),
        }
        processor = GrowthProcessor(
            growth_configs=configs,
            asset_index=state.asset_index,
        )
        dt = 1.0 / 12.0

        processor.step(state, dt, seeded_rng)

        # Both should have changed from initial values
        assert not np.allclose(state.balances[:, 0], 100_000.0)
        assert not np.allclose(state.balances[:, 1], 50_000.0)

        # All values should be positive (exp is always positive for finite input)
        assert np.all(state.balances > 0)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestGrowthProtocol:
    """Verify GrowthProcessor satisfies the Processor protocol."""

    def test_is_processor(self):
        from moneta.engine.processors import Processor

        config = GrowthConfig(model="gbm", expected_return=0.07, volatility=0.15)
        processor = GrowthProcessor(
            growth_configs={"x": config},
            asset_index={"x": 0},
        )
        assert isinstance(processor, Processor)

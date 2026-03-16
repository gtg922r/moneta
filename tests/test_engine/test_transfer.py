"""Tests for the TransferProcessor."""

from __future__ import annotations

import numpy as np
import pytest

from moneta.engine.processors.transfer import TransferProcessor, _TransferConfig
from moneta.engine.state import SimulationState
from moneta.parser.models import (
    GlobalConfig,
    GrowthConfig,
    IlliquidEquityAsset,
    InflationConfig,
    InvestmentAsset,
    LiquidityEvent,
    ProbabilityQuery,
    ScenarioConfig,
    ScenarioModel,
    TransferConfig as TransferConfigModel,
)
from moneta.parser.types import ProbabilityWindowValue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _equity_model(n_months: int = 120, n_sims: int = 100) -> ScenarioModel:
    """Build a test model with one illiquid asset transferring to portfolio."""
    return ScenarioModel(
        scenario=ScenarioConfig(
            name="transfer_test", time_horizon=n_months, simulations=n_sims
        ),
        assets={
            "portfolio": InvestmentAsset(
                type="investment",
                initial_balance=850_000,
                growth=GrowthConfig(model="gbm", expected_return=0.07, volatility=0.15),
            ),
            "startup_equity": IlliquidEquityAsset(
                type="illiquid_equity",
                current_valuation=500_000,
                shares=50_000,
                liquidity_events=[
                    LiquidityEvent(
                        probability=ProbabilityWindowValue(0.20, 0, 36),
                        valuation_range=(2.0, 5.0),
                    ),
                ],
                on_liquidation=TransferConfigModel(transfer_to="portfolio"),
            ),
        },
        global_config=GlobalConfig(
            inflation=InflationConfig(
                model="mean_reverting", long_term_rate=0.03, volatility=0.01
            )
        ),
        queries=[
            ProbabilityQuery(
                type="probability",
                expression="portfolio > 2000000",
                at=n_months,
                label="test",
            )
        ],
    )


def _two_event_model(n_months: int = 120, n_sims: int = 100) -> ScenarioModel:
    """Model with two liquidity events on the same asset."""
    return ScenarioModel(
        scenario=ScenarioConfig(
            name="two_event_transfer_test", time_horizon=n_months, simulations=n_sims
        ),
        assets={
            "portfolio": InvestmentAsset(
                type="investment",
                initial_balance=850_000,
                growth=GrowthConfig(model="gbm", expected_return=0.07, volatility=0.15),
            ),
            "startup_equity": IlliquidEquityAsset(
                type="illiquid_equity",
                current_valuation=500_000,
                shares=50_000,
                liquidity_events=[
                    LiquidityEvent(
                        probability=ProbabilityWindowValue(0.20, 0, 36),
                        valuation_range=(2.0, 5.0),
                    ),
                    LiquidityEvent(
                        probability=ProbabilityWindowValue(0.60, 60, 72),
                        valuation_range=(3.0, 10.0),
                    ),
                ],
                on_liquidation=TransferConfigModel(transfer_to="portfolio"),
            ),
        },
        global_config=GlobalConfig(
            inflation=InflationConfig(
                model="mean_reverting", long_term_rate=0.03, volatility=0.01
            )
        ),
        queries=[
            ProbabilityQuery(
                type="probability",
                expression="portfolio > 2000000",
                at=n_months,
                label="test",
            )
        ],
    )


# ---------------------------------------------------------------------------
# TransferProcessor.from_scenario
# ---------------------------------------------------------------------------


class TestTransferProcessorFromScenario:
    """Tests for building TransferProcessor from a scenario model."""

    def test_single_asset_builds_one_config(self):
        model = _equity_model(n_sims=50)
        proc = TransferProcessor.from_scenario(model, n_runs=50)
        assert len(proc._configs) == 1

    def test_config_source_dest_correct(self):
        model = _equity_model(n_sims=50)
        proc = TransferProcessor.from_scenario(model, n_runs=50)
        cfg = proc._configs[0]

        asset_names = list(model.assets.keys())
        expected_source = asset_names.index("startup_equity")
        expected_dest = asset_names.index("portfolio")

        assert cfg.source_col == expected_source
        assert cfg.dest_col == expected_dest

    def test_config_event_indices_correct(self):
        model = _equity_model(n_sims=50)
        proc = TransferProcessor.from_scenario(model, n_runs=50)
        cfg = proc._configs[0]
        # Single event → event index [0]
        assert cfg.event_indices == [0]

    def test_two_events_config_has_both_indices(self):
        model = _two_event_model(n_sims=50)
        proc = TransferProcessor.from_scenario(model, n_runs=50)
        cfg = proc._configs[0]
        assert cfg.event_indices == [0, 1]

    def test_transferred_initialized_false(self):
        model = _equity_model(n_sims=50)
        proc = TransferProcessor.from_scenario(model, n_runs=50)
        cfg = proc._configs[0]
        assert cfg.transferred.dtype == bool
        assert not cfg.transferred.any()
        assert cfg.transferred.shape == (50,)

    def test_no_illiquid_assets_empty_configs(self):
        """Model with only investment assets → no transfer configs."""
        model = ScenarioModel(
            scenario=ScenarioConfig(name="no_transfer", time_horizon=60, simulations=50),
            assets={
                "portfolio": InvestmentAsset(
                    type="investment",
                    initial_balance=100_000,
                    growth=GrowthConfig(
                        model="gbm", expected_return=0.07, volatility=0.15
                    ),
                ),
            },
            global_config=GlobalConfig(
                inflation=InflationConfig(
                    model="mean_reverting", long_term_rate=0.03, volatility=0.01
                )
            ),
            queries=[
                ProbabilityQuery(
                    type="probability",
                    expression="portfolio > 200000",
                    at=60,
                    label="test",
                )
            ],
        )
        proc = TransferProcessor.from_scenario(model, n_runs=50)
        assert len(proc._configs) == 0


# ---------------------------------------------------------------------------
# TransferProcessor.step — basic transfer mechanics
# ---------------------------------------------------------------------------


class TestTransferProcessorStep:
    """Tests for transfer execution."""

    def test_transfer_moves_balance_on_event_fire(self):
        """When event fires, source balance moves to dest."""
        n_runs = 10
        model = _equity_model(n_sims=n_runs)
        proc = TransferProcessor.from_scenario(model, n_runs=n_runs)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        portfolio_col = state.asset_index["portfolio"]
        equity_col = state.asset_index["startup_equity"]

        # Set known values
        state.balances[:, portfolio_col] = 1_000_000
        state.balances[:, equity_col] = 500_000

        # Fire event for runs 0, 1, 2
        state.events_fired[0, 0] = True
        state.events_fired[1, 0] = True
        state.events_fired[2, 0] = True

        proc.step(state, dt=1 / 12, rng=rng)

        # Runs 0, 1, 2: portfolio should gain 500K, equity should be 0
        np.testing.assert_array_equal(
            state.balances[:3, portfolio_col], 1_500_000.0
        )
        np.testing.assert_array_equal(
            state.balances[:3, equity_col], 0.0
        )

        # Runs 3-9: unchanged
        np.testing.assert_array_equal(
            state.balances[3:, portfolio_col], 1_000_000.0
        )
        np.testing.assert_array_equal(
            state.balances[3:, equity_col], 500_000.0
        )

    def test_source_asset_zeroed_after_transfer(self):
        """Source asset should be exactly 0 after transfer."""
        n_runs = 5
        model = _equity_model(n_sims=n_runs)
        proc = TransferProcessor.from_scenario(model, n_runs=n_runs)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        equity_col = state.asset_index["startup_equity"]
        state.balances[:, equity_col] = 123_456.78

        # Fire all events
        state.events_fired[:, 0] = True

        proc.step(state, dt=1 / 12, rng=rng)

        np.testing.assert_array_equal(
            state.balances[:, equity_col], 0.0
        )

    def test_balance_conservation(self):
        """Total balance (source + dest) is conserved across transfer."""
        n_runs = 20
        model = _equity_model(n_sims=n_runs)
        proc = TransferProcessor.from_scenario(model, n_runs=n_runs)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        portfolio_col = state.asset_index["portfolio"]
        equity_col = state.asset_index["startup_equity"]

        # Set specific values for each run
        state.balances[:, portfolio_col] = np.arange(n_runs) * 100_000 + 500_000
        state.balances[:, equity_col] = np.arange(n_runs) * 50_000 + 200_000

        # Record total before transfer
        total_before = state.balances[:, portfolio_col] + state.balances[:, equity_col]

        # Fire events for half the runs
        state.events_fired[:10, 0] = True

        proc.step(state, dt=1 / 12, rng=rng)

        # Total should be conserved
        total_after = state.balances[:, portfolio_col] + state.balances[:, equity_col]
        np.testing.assert_array_almost_equal(total_before, total_after)

    def test_no_transfer_if_event_not_fired(self):
        """If no events have fired, no transfers happen."""
        n_runs = 10
        model = _equity_model(n_sims=n_runs)
        proc = TransferProcessor.from_scenario(model, n_runs=n_runs)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        portfolio_col = state.asset_index["portfolio"]
        equity_col = state.asset_index["startup_equity"]

        portfolio_before = state.balances[:, portfolio_col].copy()
        equity_before = state.balances[:, equity_col].copy()

        # No events fired — step should be a no-op
        proc.step(state, dt=1 / 12, rng=rng)

        np.testing.assert_array_equal(
            state.balances[:, portfolio_col], portfolio_before
        )
        np.testing.assert_array_equal(
            state.balances[:, equity_col], equity_before
        )

    def test_transfer_only_happens_once(self):
        """Transfer is idempotent — calling step multiple times doesn't re-transfer."""
        n_runs = 5
        model = _equity_model(n_sims=n_runs)
        proc = TransferProcessor.from_scenario(model, n_runs=n_runs)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        portfolio_col = state.asset_index["portfolio"]
        equity_col = state.asset_index["startup_equity"]

        # Set known values
        state.balances[:, portfolio_col] = 1_000_000
        state.balances[:, equity_col] = 500_000

        # Fire events
        state.events_fired[:, 0] = True

        # Step once
        proc.step(state, dt=1 / 12, rng=rng)

        # Record post-transfer values
        portfolio_after_first = state.balances[:, portfolio_col].copy()
        equity_after_first = state.balances[:, equity_col].copy()

        np.testing.assert_array_equal(portfolio_after_first, 1_500_000.0)
        np.testing.assert_array_equal(equity_after_first, 0.0)

        # Step again — should be a no-op
        proc.step(state, dt=1 / 12, rng=rng)

        np.testing.assert_array_equal(
            state.balances[:, portfolio_col], portfolio_after_first
        )
        np.testing.assert_array_equal(
            state.balances[:, equity_col], equity_after_first
        )

    def test_multiple_runs_only_affected_runs_transferred(self):
        """Only runs where events fired get transferred."""
        n_runs = 10
        model = _equity_model(n_sims=n_runs)
        proc = TransferProcessor.from_scenario(model, n_runs=n_runs)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        portfolio_col = state.asset_index["portfolio"]
        equity_col = state.asset_index["startup_equity"]

        state.balances[:, portfolio_col] = 1_000_000
        state.balances[:, equity_col] = 500_000

        # Only fire event for run 0 and run 5
        state.events_fired[0, 0] = True
        state.events_fired[5, 0] = True

        proc.step(state, dt=1 / 12, rng=rng)

        # Transferred runs
        assert state.balances[0, portfolio_col] == 1_500_000.0
        assert state.balances[0, equity_col] == 0.0
        assert state.balances[5, portfolio_col] == 1_500_000.0
        assert state.balances[5, equity_col] == 0.0

        # Untouched runs
        for i in [1, 2, 3, 4, 6, 7, 8, 9]:
            assert state.balances[i, portfolio_col] == 1_000_000.0
            assert state.balances[i, equity_col] == 500_000.0


# ---------------------------------------------------------------------------
# TransferProcessor.step — two events triggering same transfer
# ---------------------------------------------------------------------------


class TestTransferProcessorTwoEvents:
    """Tests with two events triggering the same transfer."""

    def test_either_event_triggers_transfer(self):
        """Transfer happens when ANY associated event fires."""
        n_runs = 10
        model = _two_event_model(n_sims=n_runs)
        proc = TransferProcessor.from_scenario(model, n_runs=n_runs)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        portfolio_col = state.asset_index["portfolio"]
        equity_col = state.asset_index["startup_equity"]

        state.balances[:, portfolio_col] = 1_000_000
        state.balances[:, equity_col] = 500_000

        # Fire event 0 for run 0, event 1 for run 1
        state.events_fired[0, 0] = True
        state.events_fired[1, 1] = True

        proc.step(state, dt=1 / 12, rng=rng)

        # Both runs should have transfer
        assert state.balances[0, portfolio_col] == 1_500_000.0
        assert state.balances[0, equity_col] == 0.0
        assert state.balances[1, portfolio_col] == 1_500_000.0
        assert state.balances[1, equity_col] == 0.0

        # Run 2 should be untouched
        assert state.balances[2, portfolio_col] == 1_000_000.0
        assert state.balances[2, equity_col] == 500_000.0

    def test_second_event_after_transfer_is_noop(self):
        """If transfer already happened via event 0, event 1 doesn't re-transfer."""
        n_runs = 5
        model = _two_event_model(n_sims=n_runs)
        proc = TransferProcessor.from_scenario(model, n_runs=n_runs)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        portfolio_col = state.asset_index["portfolio"]
        equity_col = state.asset_index["startup_equity"]

        state.balances[:, portfolio_col] = 1_000_000
        state.balances[:, equity_col] = 500_000

        # Fire event 0 for all runs
        state.events_fired[:, 0] = True
        proc.step(state, dt=1 / 12, rng=rng)

        # Record state after first transfer
        portfolio_after = state.balances[:, portfolio_col].copy()
        equity_after = state.balances[:, equity_col].copy()

        np.testing.assert_array_equal(portfolio_after, 1_500_000.0)
        np.testing.assert_array_equal(equity_after, 0.0)

        # Now fire event 1 as well
        state.events_fired[:, 1] = True
        proc.step(state, dt=1 / 12, rng=rng)

        # Should be no change — already transferred
        np.testing.assert_array_equal(
            state.balances[:, portfolio_col], portfolio_after
        )
        np.testing.assert_array_equal(
            state.balances[:, equity_col], equity_after
        )

    def test_transfer_from_event_0_then_event_1_fires_no_double_transfer(self):
        """Simulates the timeline: event 0 fires first, later event 1 fires.

        Transfer should only happen once, on the first event.
        """
        n_runs = 5
        model = _two_event_model(n_sims=n_runs)
        proc = TransferProcessor.from_scenario(model, n_runs=n_runs)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        portfolio_col = state.asset_index["portfolio"]
        equity_col = state.asset_index["startup_equity"]

        state.balances[:, portfolio_col] = 1_000_000
        state.balances[:, equity_col] = 400_000

        # Step 1: event 0 fires for all runs
        state.events_fired[:, 0] = True
        proc.step(state, dt=1 / 12, rng=rng)

        np.testing.assert_array_equal(
            state.balances[:, portfolio_col], 1_400_000.0
        )
        np.testing.assert_array_equal(
            state.balances[:, equity_col], 0.0
        )

        # Step 2: event 1 also fires for all runs
        state.events_fired[:, 1] = True
        proc.step(state, dt=1 / 12, rng=rng)

        # No additional transfer should happen (source is already 0,
        # and transferred flag prevents re-transfer)
        np.testing.assert_array_equal(
            state.balances[:, portfolio_col], 1_400_000.0
        )
        np.testing.assert_array_equal(
            state.balances[:, equity_col], 0.0
        )


# ---------------------------------------------------------------------------
# TransferProcessor.step — no events in model
# ---------------------------------------------------------------------------


class TestTransferProcessorNoEvents:
    """Tests when the model has no illiquid assets."""

    def test_step_is_noop_with_no_configs(self):
        """Step with empty configs should not modify state."""
        proc = TransferProcessor([])
        n_runs = 5
        model = ScenarioModel(
            scenario=ScenarioConfig(name="no_events", time_horizon=60, simulations=n_runs),
            assets={
                "portfolio": InvestmentAsset(
                    type="investment",
                    initial_balance=100_000,
                    growth=GrowthConfig(
                        model="gbm", expected_return=0.07, volatility=0.15
                    ),
                ),
            },
            global_config=GlobalConfig(
                inflation=InflationConfig(
                    model="mean_reverting", long_term_rate=0.03, volatility=0.01
                )
            ),
            queries=[
                ProbabilityQuery(
                    type="probability",
                    expression="portfolio > 200000",
                    at=60,
                    label="test",
                )
            ],
        )
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        balances_before = state.balances.copy()
        proc.step(state, dt=1 / 12, rng=rng)

        np.testing.assert_array_equal(state.balances, balances_before)

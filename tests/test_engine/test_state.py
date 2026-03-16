"""Tests for SimulationState and ResultStore."""

import numpy as np
import pytest

from moneta.engine.state import ResultStore, SimulationState
from moneta.parser.models import (
    GlobalConfig,
    GrowthConfig,
    IlliquidEquityAsset,
    InflationConfig,
    InvestmentAsset,
    LiquidityEvent,
    ProbabilityQuery,
    PresetRef,
    ScenarioConfig,
    ScenarioModel,
    TransferConfig,
)
from moneta.parser.types import ProbabilityWindowValue


# ---------------------------------------------------------------------------
# Helpers — build test models
# ---------------------------------------------------------------------------


def _simple_model(n_months: int = 120, n_sims: int = 100) -> ScenarioModel:
    """Single investment asset, no events."""
    return ScenarioModel(
        scenario=ScenarioConfig(name="test", time_horizon=n_months, simulations=n_sims),
        assets={
            "portfolio": InvestmentAsset(
                type="investment",
                initial_balance=100_000,
                growth=GrowthConfig(model="gbm", expected_return=0.07, volatility=0.15),
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
                at=n_months,
                label="test",
            )
        ],
    )


def _equity_model(n_months: int = 120, n_sims: int = 100) -> ScenarioModel:
    """Investment + illiquid equity with two liquidity events."""
    return ScenarioModel(
        scenario=ScenarioConfig(name="equity_test", time_horizon=n_months, simulations=n_sims),
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
                on_liquidation=TransferConfig(transfer_to="portfolio"),
            ),
        },
        global_config=GlobalConfig(
            inflation=InflationConfig(
                model="mean_reverting", long_term_rate=0.025, volatility=0.008
            )
        ),
        queries=[
            ProbabilityQuery(
                type="probability",
                expression="portfolio > 2000000",
                at=n_months,
                label="test equity query",
            )
        ],
    )


def _preset_inflation_model() -> ScenarioModel:
    """Model with PresetRef for inflation (not resolved)."""
    return ScenarioModel(
        scenario=ScenarioConfig(name="preset_test", time_horizon=60, simulations=50),
        assets={
            "savings": InvestmentAsset(
                type="investment",
                initial_balance=50_000,
                growth=GrowthConfig(model="gbm", expected_return=0.05, volatility=0.10),
            ),
        },
        global_config=GlobalConfig(inflation=PresetRef(preset="us_inflation")),
        queries=[
            ProbabilityQuery(
                type="probability",
                expression="savings > 60000",
                at=60,
                label="test",
            )
        ],
    )


# ---------------------------------------------------------------------------
# SimulationState.from_scenario
# ---------------------------------------------------------------------------


class TestSimulationStateFromScenario:
    """Tests for SimulationState.from_scenario."""

    def test_correct_array_shapes_simple(self):
        model = _simple_model(n_months=120, n_sims=200)
        state = SimulationState.from_scenario(model, n_runs=200)

        assert state.balances.shape == (200, 1)
        assert state.events_fired.shape == (200, 0)
        assert state.inflation_rate.shape == (200,)
        assert state.cum_inflation.shape == (200,)

    def test_correct_array_shapes_equity(self):
        model = _equity_model(n_months=120, n_sims=100)
        state = SimulationState.from_scenario(model, n_runs=100)

        # 2 assets, 2 events
        assert state.balances.shape == (100, 2)
        assert state.events_fired.shape == (100, 2)

    def test_initial_balances_investment(self):
        model = _simple_model()
        state = SimulationState.from_scenario(model, n_runs=50)

        # All runs should start at 100,000
        np.testing.assert_array_equal(state.balances[:, 0], 100_000.0)

    def test_initial_balances_equity(self):
        model = _equity_model()
        state = SimulationState.from_scenario(model, n_runs=50)

        portfolio_idx = state.asset_index["portfolio"]
        equity_idx = state.asset_index["startup_equity"]

        np.testing.assert_array_equal(state.balances[:, portfolio_idx], 850_000.0)
        np.testing.assert_array_equal(state.balances[:, equity_idx], 500_000.0)

    def test_events_fired_initialized_false(self):
        model = _equity_model()
        state = SimulationState.from_scenario(model, n_runs=50)

        assert state.events_fired.dtype == bool
        assert not state.events_fired.any()

    def test_inflation_rate_initialized_from_config(self):
        model = _equity_model()
        state = SimulationState.from_scenario(model, n_runs=50)

        # long_term_rate = 0.025
        np.testing.assert_array_almost_equal(state.inflation_rate, 0.025)

    def test_inflation_rate_preset_ref_default(self):
        model = _preset_inflation_model()
        state = SimulationState.from_scenario(model, n_runs=30)

        # PresetRef should fall back to 3% default
        np.testing.assert_array_almost_equal(state.inflation_rate, 0.03)

    def test_cum_inflation_initialized_to_one(self):
        model = _simple_model()
        state = SimulationState.from_scenario(model, n_runs=50)

        np.testing.assert_array_equal(state.cum_inflation, 1.0)

    def test_step_initialized_to_zero(self):
        model = _simple_model()
        state = SimulationState.from_scenario(model, n_runs=10)

        assert state.step == 0

    def test_asset_names_and_index_consistent(self):
        model = _equity_model()
        state = SimulationState.from_scenario(model, n_runs=10)

        assert len(state.asset_names) == 2
        assert set(state.asset_names) == {"portfolio", "startup_equity"}
        assert len(state.asset_index) == 2

        # Index values are valid column indices
        for name in state.asset_names:
            idx = state.asset_index[name]
            assert 0 <= idx < len(state.asset_names)

        # Each name maps to its position in asset_names
        for i, name in enumerate(state.asset_names):
            assert state.asset_index[name] == i

    def test_event_index_built_correctly(self):
        model = _equity_model()
        state = SimulationState.from_scenario(model, n_runs=10)

        # Two events on startup_equity
        assert len(state.event_index) == 2
        assert "startup_equity:0" in state.event_index
        assert "startup_equity:1" in state.event_index
        assert state.event_index["startup_equity:0"] == 0
        assert state.event_index["startup_equity:1"] == 1

    def test_no_events_for_investment_only(self):
        model = _simple_model()
        state = SimulationState.from_scenario(model, n_runs=10)

        assert state.events_fired.shape[1] == 0
        assert len(state.event_index) == 0

    def test_dtypes_correct(self):
        model = _equity_model()
        state = SimulationState.from_scenario(model, n_runs=10)

        assert state.balances.dtype == np.float64
        assert state.events_fired.dtype == bool
        assert state.inflation_rate.dtype == np.float64
        assert state.cum_inflation.dtype == np.float64


# ---------------------------------------------------------------------------
# ResultStore.allocate
# ---------------------------------------------------------------------------


class TestResultStoreAllocate:
    """Tests for ResultStore.allocate."""

    def test_correct_shapes_simple(self):
        model = _simple_model(n_months=120, n_sims=100)
        store = ResultStore.allocate(model, n_runs=100)

        assert store.balances.shape == (100, 120, 1)
        assert store.cum_inflation.shape == (100, 120)
        assert store.event_fired_at.shape == (100, 0)

    def test_correct_shapes_equity(self):
        model = _equity_model(n_months=60, n_sims=200)
        store = ResultStore.allocate(model, n_runs=200)

        assert store.balances.shape == (200, 60, 2)
        assert store.cum_inflation.shape == (200, 60)
        assert store.event_fired_at.shape == (200, 2)

    def test_event_fired_at_initialized_to_minus_one(self):
        model = _equity_model()
        store = ResultStore.allocate(model, n_runs=50)

        np.testing.assert_array_equal(store.event_fired_at, -1)

    def test_metadata_correct(self):
        model = _equity_model(n_months=120, n_sims=100)
        store = ResultStore.allocate(model, n_runs=100)

        assert store.n_runs == 100
        assert store.n_steps == 120
        assert store.n_assets == 2
        assert len(store.asset_names) == 2
        assert set(store.asset_names) == {"portfolio", "startup_equity"}

    def test_asset_index_consistent_with_names(self):
        model = _equity_model()
        store = ResultStore.allocate(model, n_runs=10)

        for i, name in enumerate(store.asset_names):
            assert store.asset_index[name] == i

    def test_dtypes_correct(self):
        model = _equity_model()
        store = ResultStore.allocate(model, n_runs=10)

        assert store.balances.dtype == np.float64
        assert store.cum_inflation.dtype == np.float64
        assert store.event_fired_at.dtype == np.int32


# ---------------------------------------------------------------------------
# ResultStore.record
# ---------------------------------------------------------------------------


class TestResultStoreRecord:
    """Tests for ResultStore.record."""

    def test_record_copies_state_correctly(self):
        model = _equity_model(n_months=10, n_sims=5)
        state = SimulationState.from_scenario(model, n_runs=5)
        store = ResultStore.allocate(model, n_runs=5)

        # Modify state to some known values
        state.balances[:] = 42.0
        state.cum_inflation[:] = 1.05

        store.record(state, step=3)

        np.testing.assert_array_equal(store.balances[:, 3, :], 42.0)
        np.testing.assert_array_equal(store.cum_inflation[:, 3], 1.05)

    def test_record_at_multiple_steps(self):
        model = _simple_model(n_months=5, n_sims=3)
        state = SimulationState.from_scenario(model, n_runs=3)
        store = ResultStore.allocate(model, n_runs=3)

        for step in range(5):
            state.balances[:] = 100.0 * (step + 1)
            state.cum_inflation[:] = 1.0 + step * 0.01
            store.record(state, step=step)

        # Verify each step was recorded independently
        for step in range(5):
            np.testing.assert_array_equal(
                store.balances[:, step, 0], 100.0 * (step + 1)
            )
            np.testing.assert_array_almost_equal(
                store.cum_inflation[:, step], 1.0 + step * 0.01
            )

    def test_record_event_fired_at_newly_fired(self):
        model = _equity_model(n_months=10, n_sims=4)
        state = SimulationState.from_scenario(model, n_runs=4)
        store = ResultStore.allocate(model, n_runs=4)

        # Initially no events fired
        store.record(state, step=0)
        np.testing.assert_array_equal(store.event_fired_at, -1)

        # Fire event 0 for runs 0 and 1 at step 3
        state.events_fired[0, 0] = True
        state.events_fired[1, 0] = True
        store.record(state, step=3)

        assert store.event_fired_at[0, 0] == 3
        assert store.event_fired_at[1, 0] == 3
        assert store.event_fired_at[2, 0] == -1  # not fired
        assert store.event_fired_at[3, 0] == -1  # not fired
        # Event 1 not fired for anyone
        np.testing.assert_array_equal(store.event_fired_at[:, 1], -1)

    def test_record_event_fired_at_only_updates_unfired(self):
        """Once event_fired_at is set, it should NOT be overwritten by later steps."""
        model = _equity_model(n_months=10, n_sims=3)
        state = SimulationState.from_scenario(model, n_runs=3)
        store = ResultStore.allocate(model, n_runs=3)

        # Fire event 0 for run 0 at step 2
        state.events_fired[0, 0] = True
        store.record(state, step=2)
        assert store.event_fired_at[0, 0] == 2

        # Record again at step 5 — event 0 for run 0 is still fired,
        # but the recorded step should remain 2, not be updated to 5.
        store.record(state, step=5)
        assert store.event_fired_at[0, 0] == 2  # still 2, not 5

    def test_record_event_fired_at_new_event_at_later_step(self):
        """New events at later steps get recorded at the correct step."""
        model = _equity_model(n_months=10, n_sims=2)
        state = SimulationState.from_scenario(model, n_runs=2)
        store = ResultStore.allocate(model, n_runs=2)

        # Fire event 0 for run 0 at step 1
        state.events_fired[0, 0] = True
        store.record(state, step=1)
        assert store.event_fired_at[0, 0] == 1

        # Fire event 1 for run 0 at step 4
        state.events_fired[0, 1] = True
        store.record(state, step=4)
        assert store.event_fired_at[0, 0] == 1  # unchanged
        assert store.event_fired_at[0, 1] == 4  # newly recorded

    def test_record_no_events_model(self):
        """Record works fine when there are no events."""
        model = _simple_model(n_months=5, n_sims=3)
        state = SimulationState.from_scenario(model, n_runs=3)
        store = ResultStore.allocate(model, n_runs=3)

        state.balances[:] = 123.0
        state.cum_inflation[:] = 1.01
        store.record(state, step=0)

        np.testing.assert_array_equal(store.balances[:, 0, 0], 123.0)
        np.testing.assert_array_equal(store.cum_inflation[:, 0], 1.01)

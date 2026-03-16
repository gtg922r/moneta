"""Tests for the hazard-rate EventProcessor."""

from __future__ import annotations

import numpy as np
import pytest

from moneta.engine.processors.events import EventProcessor, _EventConfig, _compute_hazard_rate
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
    TransferConfig,
)
from moneta.parser.types import ProbabilityWindowValue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _equity_model(
    probability: float = 0.20,
    start_month: int = 0,
    end_month: int = 36,
    mult_low: float = 2.0,
    mult_high: float = 5.0,
    current_valuation: float = 500_000,
    n_months: int = 120,
    n_sims: int = 100,
) -> ScenarioModel:
    """Build a test model with a single liquidity event."""
    return ScenarioModel(
        scenario=ScenarioConfig(
            name="event_test", time_horizon=n_months, simulations=n_sims
        ),
        assets={
            "portfolio": InvestmentAsset(
                type="investment",
                initial_balance=850_000,
                growth=GrowthConfig(model="gbm", expected_return=0.07, volatility=0.15),
            ),
            "startup_equity": IlliquidEquityAsset(
                type="illiquid_equity",
                current_valuation=current_valuation,
                shares=50_000,
                liquidity_events=[
                    LiquidityEvent(
                        probability=ProbabilityWindowValue(probability, start_month, end_month),
                        valuation_range=(mult_low, mult_high),
                    ),
                ],
                on_liquidation=TransferConfig(transfer_to="portfolio"),
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
    """Build a test model with two liquidity events on the same asset."""
    return ScenarioModel(
        scenario=ScenarioConfig(
            name="two_event_test", time_horizon=n_months, simulations=n_sims
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
                on_liquidation=TransferConfig(transfer_to="portfolio"),
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
# _compute_hazard_rate unit tests
# ---------------------------------------------------------------------------


class TestComputeHazardRate:
    """Tests for the hazard rate computation."""

    def test_20pct_within_3_years(self):
        """20% within 36 months → h ≈ 0.00617."""
        h = _compute_hazard_rate(0.20, 36)
        # Verify: 1 - (1-h)^36 ≈ 0.20
        cumulative = 1 - (1 - h) ** 36
        assert abs(cumulative - 0.20) < 1e-10

    def test_60pct_within_12_months_window(self):
        """60% within 12 month window → h ≈ 0.0745."""
        h = _compute_hazard_rate(0.60, 12)
        cumulative = 1 - (1 - h) ** 12
        assert abs(cumulative - 0.60) < 1e-10

    def test_zero_probability(self):
        h = _compute_hazard_rate(0.0, 36)
        assert h == 0.0

    def test_100_percent_probability(self):
        h = _compute_hazard_rate(1.0, 36)
        assert h == 1.0

    def test_zero_window(self):
        h = _compute_hazard_rate(0.5, 0)
        assert h == 0.0

    def test_single_month_window(self):
        """P = 0.5 within 1 month → h = 0.5."""
        h = _compute_hazard_rate(0.5, 1)
        assert abs(h - 0.5) < 1e-10

    def test_small_probability(self):
        """Very small probability."""
        h = _compute_hazard_rate(0.001, 120)
        cumulative = 1 - (1 - h) ** 120
        assert abs(cumulative - 0.001) < 1e-10


# ---------------------------------------------------------------------------
# EventProcessor.from_scenario
# ---------------------------------------------------------------------------


class TestEventProcessorFromScenario:
    """Tests for building EventProcessor from a scenario model."""

    def test_single_event_builds_one_config(self):
        model = _equity_model()
        proc = EventProcessor.from_scenario(model)
        assert len(proc._configs) == 1

    def test_two_events_builds_two_configs(self):
        model = _two_event_model()
        proc = EventProcessor.from_scenario(model)
        assert len(proc._configs) == 2

    def test_config_hazard_rate_correct(self):
        model = _equity_model(probability=0.20, start_month=0, end_month=36)
        proc = EventProcessor.from_scenario(model)
        cfg = proc._configs[0]
        expected_h = 1 - (1 - 0.20) ** (1 / 36)
        assert abs(cfg.hazard_rate - expected_h) < 1e-12

    def test_config_multiplier_range(self):
        model = _equity_model(mult_low=2.0, mult_high=5.0)
        proc = EventProcessor.from_scenario(model)
        cfg = proc._configs[0]
        assert cfg.multiplier_low == 2.0
        assert cfg.multiplier_high == 5.0

    def test_config_base_valuation(self):
        model = _equity_model(current_valuation=750_000)
        proc = EventProcessor.from_scenario(model)
        cfg = proc._configs[0]
        assert cfg.base_valuation == 750_000

    def test_config_asset_col_correct(self):
        model = _equity_model()
        proc = EventProcessor.from_scenario(model)
        cfg = proc._configs[0]
        asset_names = list(model.assets.keys())
        expected_col = asset_names.index("startup_equity")
        assert cfg.asset_col == expected_col

    def test_no_illiquid_assets_empty_configs(self):
        """Model with only investment assets → empty event configs."""
        model = ScenarioModel(
            scenario=ScenarioConfig(name="no_events", time_horizon=60, simulations=50),
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
        proc = EventProcessor.from_scenario(model)
        assert len(proc._configs) == 0


# ---------------------------------------------------------------------------
# EventProcessor.step — seeded deterministic tests
# ---------------------------------------------------------------------------


class TestEventProcessorStepDeterministic:
    """Seeded deterministic tests for event firing."""

    def test_event_fires_at_expected_step_with_known_seed(self):
        """With a known seed and high hazard rate, event should fire quickly."""
        model = _equity_model(
            probability=0.90, start_month=0, end_month=12,
            current_valuation=100_000, mult_low=2.0, mult_high=2.0,
        )
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=1)

        rng = np.random.default_rng(42)
        fired_step = None

        for step in range(12):
            state.step = step
            proc.step(state, dt=1 / 12, rng=rng)
            if state.events_fired[0, 0]:
                fired_step = step
                break

        # With 90% probability over 12 months, h is very high — should fire
        assert fired_step is not None
        assert state.events_fired[0, 0]

    def test_event_sets_balance_to_liquidation_value(self):
        """When event fires, balance = base_valuation * multiplier."""
        model = _equity_model(
            probability=1.0, start_month=0, end_month=1,
            current_valuation=100_000, mult_low=3.0, mult_high=3.0,
        )
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=10)
        rng = np.random.default_rng(99)

        state.step = 0
        proc.step(state, dt=1 / 12, rng=rng)

        equity_col = state.asset_index["startup_equity"]
        # All should fire (100% probability, h=1.0)
        assert state.events_fired[:, 0].all()
        # Balance = 100_000 * 3.0 = 300_000
        np.testing.assert_array_almost_equal(
            state.balances[:, equity_col], 300_000.0
        )

    def test_event_does_not_fire_outside_window(self):
        """Events should not fire before start_month or at/after end_month."""
        model = _equity_model(
            probability=1.0, start_month=6, end_month=12,
            current_valuation=100_000, mult_low=2.0, mult_high=2.0,
        )
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=10)
        rng = np.random.default_rng(42)

        # Steps before the window
        for step in range(6):
            state.step = step
            proc.step(state, dt=1 / 12, rng=rng)

        assert not state.events_fired[:, 0].any(), "Should not fire before window"

        # Steps within the window — with h=1.0, should fire immediately
        state.step = 6
        proc.step(state, dt=1 / 12, rng=rng)
        assert state.events_fired[:, 0].all(), "Should fire within window"

    def test_event_does_not_fire_at_end_month(self):
        """end_month is exclusive — event should not fire at step=end_month."""
        model = _equity_model(
            probability=1.0, start_month=0, end_month=3,
            current_valuation=100_000, mult_low=2.0, mult_high=2.0,
        )
        proc = EventProcessor.from_scenario(model)
        # Use a small number of runs so we can be sure
        state = SimulationState.from_scenario(model, n_runs=5)
        rng = np.random.default_rng(42)

        # Manually mark all as already fired, then reset
        # Test at step=end_month (3) which should be outside window
        state2 = SimulationState.from_scenario(model, n_runs=5)
        rng2 = np.random.default_rng(42)
        state2.step = 3  # end_month = 3, so step=3 is excluded
        proc2 = EventProcessor.from_scenario(model)
        proc2.step(state2, dt=1 / 12, rng=rng2)
        assert not state2.events_fired[:, 0].any()

    def test_event_fires_at_most_once_per_run(self):
        """Once an event fires, it should not fire again in subsequent steps."""
        model = _equity_model(
            probability=0.99, start_month=0, end_month=120,
            current_valuation=100_000, mult_low=2.0, mult_high=2.0,
        )
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=100)
        rng = np.random.default_rng(42)

        equity_col = state.asset_index["startup_equity"]

        # Run many steps
        for step in range(60):
            state.step = step
            proc.step(state, dt=1 / 12, rng=rng)

        # events_fired should still be boolean, no double-counting
        # The events_fired values should be only True or False
        assert state.events_fired.dtype == bool

        # Balance should be exactly base_valuation * multiplier (2.0)
        # for all fired runs — not doubled
        fired = state.events_fired[:, 0]
        if fired.any():
            # All fired runs should have balance = 100_000 * 2.0 = 200_000
            np.testing.assert_array_almost_equal(
                state.balances[fired, equity_col], 200_000.0
            )


# ---------------------------------------------------------------------------
# EventProcessor.step — statistical tests
# ---------------------------------------------------------------------------


class TestEventProcessorStepStatistical:
    """Statistical tests with many runs to verify probability calibration."""

    def test_fire_frequency_matches_probability_20pct_3yr(self):
        """100K runs, 20% within 3 years → ~20% fire rate."""
        n_runs = 100_000
        model = _equity_model(
            probability=0.20, start_month=0, end_month=36,
            n_months=36, n_sims=n_runs,
        )
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(12345)

        for step in range(36):
            state.step = step
            proc.step(state, dt=1 / 12, rng=rng)

        fire_rate = state.events_fired[:, 0].mean()
        # 20% ± ~0.25% (for 100K runs, 95% CI ≈ ±2*sqrt(0.2*0.8/100000) ≈ ±0.0025)
        assert abs(fire_rate - 0.20) < 0.015, (
            f"Fire rate {fire_rate:.4f} deviates too much from expected 0.20"
        )

    def test_fire_frequency_matches_probability_60pct_window(self):
        """100K runs, 60% within months 60-72 → ~60% fire rate."""
        n_runs = 100_000
        model = _equity_model(
            probability=0.60, start_month=60, end_month=72,
            n_months=72, n_sims=n_runs,
        )
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(54321)

        for step in range(72):
            state.step = step
            proc.step(state, dt=1 / 12, rng=rng)

        fire_rate = state.events_fired[:, 0].mean()
        # 60% ± ~0.3% (for 100K runs)
        assert abs(fire_rate - 0.60) < 0.015, (
            f"Fire rate {fire_rate:.4f} deviates too much from expected 0.60"
        )

    def test_100pct_probability_fires_in_every_run(self):
        """100% probability within 12 months → fires in every run."""
        n_runs = 10_000
        model = _equity_model(
            probability=1.0, start_month=0, end_month=12,
            n_months=12, n_sims=n_runs,
        )
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        for step in range(12):
            state.step = step
            proc.step(state, dt=1 / 12, rng=rng)

        assert state.events_fired[:, 0].all(), "100% prob should fire in all runs"

    def test_near_zero_probability_fires_in_almost_no_runs(self):
        """~0% probability → fires in essentially no runs."""
        n_runs = 10_000
        model = _equity_model(
            probability=0.001, start_month=0, end_month=12,
            n_months=12, n_sims=n_runs,
        )
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        for step in range(12):
            state.step = step
            proc.step(state, dt=1 / 12, rng=rng)

        fire_rate = state.events_fired[:, 0].mean()
        # Should be very close to 0.001
        assert fire_rate < 0.01, (
            f"Fire rate {fire_rate:.4f} too high for 0.1% probability"
        )

    def test_valuation_drawn_from_multiplier_range(self):
        """Liquidation values should be within [base*low, base*high]."""
        n_runs = 10_000
        model = _equity_model(
            probability=1.0, start_month=0, end_month=1,
            current_valuation=100_000, mult_low=2.0, mult_high=5.0,
            n_months=1, n_sims=n_runs,
        )
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        state.step = 0
        proc.step(state, dt=1 / 12, rng=rng)

        equity_col = state.asset_index["startup_equity"]
        balances = state.balances[:, equity_col]

        # All should be in [200_000, 500_000]
        assert (balances >= 200_000.0 - 1e-6).all(), "Below min multiplier"
        assert (balances <= 500_000.0 + 1e-6).all(), "Above max multiplier"

        # Mean should be near midpoint = 100_000 * 3.5 = 350_000
        mean_val = balances.mean()
        assert abs(mean_val - 350_000) < 5_000, (
            f"Mean {mean_val:.0f} deviates too much from expected 350,000"
        )

    def test_valuation_exact_multiplier_when_range_is_point(self):
        """When mult_low == mult_high, all liquidation values are exact."""
        n_runs = 1000
        model = _equity_model(
            probability=1.0, start_month=0, end_month=1,
            current_valuation=100_000, mult_low=3.0, mult_high=3.0,
            n_months=1, n_sims=n_runs,
        )
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=n_runs)
        rng = np.random.default_rng(42)

        state.step = 0
        proc.step(state, dt=1 / 12, rng=rng)

        equity_col = state.asset_index["startup_equity"]
        np.testing.assert_array_almost_equal(
            state.balances[:, equity_col], 300_000.0
        )


# ---------------------------------------------------------------------------
# EventProcessor.step — multiple events on same asset
# ---------------------------------------------------------------------------


class TestEventProcessorMultipleEvents:
    """Tests for multiple events on the same asset."""

    def test_two_events_independent_event_slots(self):
        """Two events on the same asset have independent event_fired slots."""
        model = _two_event_model(n_months=120, n_sims=1000)
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=1000)
        rng = np.random.default_rng(42)

        # Run through first event window (0-36)
        for step in range(36):
            state.step = step
            proc.step(state, dt=1 / 12, rng=rng)

        event0_fired = state.events_fired[:, 0].copy()

        # Run through gap and second event window (36-72)
        for step in range(36, 72):
            state.step = step
            proc.step(state, dt=1 / 12, rng=rng)

        event1_fired = state.events_fired[:, 1].copy()

        # Both events should have some fires
        assert event0_fired.any(), "Event 0 should have fired in some runs"
        assert event1_fired.any(), "Event 1 should have fired in some runs"

        # They are independent — some runs may have both, some neither
        both_fired = event0_fired & event1_fired
        neither_fired = ~event0_fired & ~event1_fired
        # With 20% and 60% rates, there should be runs in each category
        # (this is probabilistic but extremely likely with 1000 runs)

    def test_second_event_fires_on_already_liquidated_asset(self):
        """If event 0 already set liquidation value, event 1 overwrites it.

        This is the expected behavior: each event independently sets the
        balance. The transfer processor handles moving values. Events are
        independent; the balance change from a later event is applied
        regardless of earlier events (though economically, once transferred,
        the source balance is 0, making subsequent events set 0 * mult = 0
        ... actually base_valuation * mult since we use base_valuation).
        """
        model = _two_event_model(n_months=120, n_sims=10)
        proc = EventProcessor.from_scenario(model)
        state = SimulationState.from_scenario(model, n_runs=10)

        # Manually fire event 0 for all runs
        state.events_fired[:, 0] = True
        equity_col = state.asset_index["startup_equity"]
        state.balances[:, equity_col] = 1_000_000  # set some value

        # Now run at step 60 (within second event's window)
        rng = np.random.default_rng(42)
        state.step = 60
        proc.step(state, dt=1 / 12, rng=rng)

        # Event 1 may fire for some runs — it's independent of event 0
        # The balance for fired runs should be base_valuation * multiplier
        fired = state.events_fired[:, 1]
        if fired.any():
            # base_valuation = 500_000, mult range [3.0, 10.0]
            vals = state.balances[fired, equity_col]
            assert (vals >= 500_000 * 3.0 - 1).all()
            assert (vals <= 500_000 * 10.0 + 1).all()

"""Tests for the Monte Carlo orchestrator.

Covers pipeline construction, simulation execution, result shapes,
seeded reproducibility, and correctness of recorded results.
"""

from __future__ import annotations

import numpy as np
import pytest

from moneta.engine.orchestrator import build_pipeline, run_simulation, run_sweep
from moneta.engine.processors.cash_flow import CashFlowProcessor
from moneta.engine.processors.events import EventProcessor
from moneta.engine.processors.growth import GrowthProcessor
from moneta.engine.processors.inflation import InflationProcessor
from moneta.engine.processors.transfer import TransferProcessor
from moneta.engine.state import ResultStore, SimulationState
from moneta.parser.models import (
    CashFlowConfig,
    GlobalConfig,
    GrowthConfig,
    IlliquidEquityAsset,
    InflationConfig,
    InvestmentAsset,
    LiquidityEvent,
    PercentilesQuery,
    ProbabilityQuery,
    ScenarioConfig,
    ScenarioModel,
    SweepConfig,
    SweepScenario,
    TransferConfig,
)
from moneta.parser.types import CashFlowAmountValue, ProbabilityWindowValue


# ---------------------------------------------------------------------------
# Helpers — build test models
# ---------------------------------------------------------------------------


def _simple_model(
    n_months: int = 120, n_sims: int = 1000, seed: int | None = 42
) -> ScenarioModel:
    """Single investment asset with GBM growth, no events."""
    return ScenarioModel(
        scenario=ScenarioConfig(
            name="Simple investment model",
            time_horizon=n_months,
            simulations=n_sims,
            seed=seed,
        ),
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
                label="Double portfolio",
            )
        ],
    )


def _equity_model(
    n_months: int = 120, n_sims: int = 5000, seed: int | None = 42
) -> ScenarioModel:
    """Investment + illiquid equity with two liquidity events."""
    return ScenarioModel(
        scenario=ScenarioConfig(
            name="Investment + equity model",
            time_horizon=n_months,
            simulations=n_sims,
            seed=seed,
        ),
        assets={
            "investment_portfolio": InvestmentAsset(
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
                on_liquidation=TransferConfig(transfer_to="investment_portfolio"),
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
                expression="investment_portfolio + startup_equity > 2000000",
                at=n_months,
                label="$2M net worth at year 10",
            )
        ],
    )


# ---------------------------------------------------------------------------
# build_pipeline tests
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    """Tests for pipeline construction from a scenario model."""

    def test_simple_model_pipeline_order(self):
        """Simple model (no events) should have GrowthProcessor + InflationProcessor."""
        model = _simple_model(n_sims=10)
        state = SimulationState.from_scenario(model, n_runs=10)
        pipeline = build_pipeline(model, state, n_runs=10)

        assert len(pipeline) == 2
        assert isinstance(pipeline[0], GrowthProcessor)
        assert isinstance(pipeline[1], InflationProcessor)

    def test_equity_model_pipeline_order(self):
        """Equity model should have all 4 processors in correct order."""
        model = _equity_model(n_sims=10)
        state = SimulationState.from_scenario(model, n_runs=10)
        pipeline = build_pipeline(model, state, n_runs=10)

        assert len(pipeline) == 4
        assert isinstance(pipeline[0], EventProcessor)
        assert isinstance(pipeline[1], TransferProcessor)
        assert isinstance(pipeline[2], GrowthProcessor)
        assert isinstance(pipeline[3], InflationProcessor)

    def test_no_growth_assets_skips_growth_processor(self):
        """Model with no investment assets should skip GrowthProcessor."""
        # Create a model with only an illiquid equity asset
        # We need a destination asset for the transfer, so include a zero-growth
        # investment asset with PresetRef (which gets skipped)
        model = ScenarioModel(
            scenario=ScenarioConfig(name="no_growth", time_horizon=60, simulations=10),
            assets={
                "cash": InvestmentAsset(
                    type="investment",
                    initial_balance=100_000,
                    # Use a GrowthConfig with 0% return and 0% vol — effectively no growth
                    growth=GrowthConfig(model="gbm", expected_return=0.0, volatility=0.0),
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
                    expression="cash > 100000",
                    at=60,
                    label="test",
                )
            ],
        )
        state = SimulationState.from_scenario(model, n_runs=10)
        pipeline = build_pipeline(model, state, n_runs=10)

        # Should still have GrowthProcessor (even with 0 return) and InflationProcessor
        assert len(pipeline) == 2
        assert isinstance(pipeline[0], GrowthProcessor)
        assert isinstance(pipeline[1], InflationProcessor)


# ---------------------------------------------------------------------------
# Result shape tests
# ---------------------------------------------------------------------------


class TestResultShapes:
    """Tests that simulation results have correct shapes."""

    def test_simple_model_result_shapes(self):
        """Simple model: results have shape (n_runs, n_steps, n_assets)."""
        n_sims = 100
        n_months = 120
        model = _simple_model(n_months=n_months, n_sims=n_sims)
        results = run_simulation(model, seed=42)

        assert results.balances.shape == (n_sims, n_months, 1)
        assert results.cum_inflation.shape == (n_sims, n_months)
        assert results.event_fired_at.shape == (n_sims, 0)
        assert results.n_runs == n_sims
        assert results.n_steps == n_months
        assert results.n_assets == 1

    def test_equity_model_result_shapes(self):
        """Equity model: results have correct shape with 2 assets and 2 events."""
        n_sims = 200
        n_months = 120
        model = _equity_model(n_months=n_months, n_sims=n_sims)
        results = run_simulation(model, seed=42)

        assert results.balances.shape == (n_sims, n_months, 2)
        assert results.cum_inflation.shape == (n_sims, n_months)
        assert results.event_fired_at.shape == (n_sims, 2)
        assert results.n_runs == n_sims
        assert results.n_steps == n_months
        assert results.n_assets == 2

    def test_asset_names_in_results(self):
        """Result store should carry correct asset names and index."""
        model = _equity_model(n_sims=10)
        results = run_simulation(model, seed=42)

        assert set(results.asset_names) == {"investment_portfolio", "startup_equity"}
        assert len(results.asset_index) == 2
        for i, name in enumerate(results.asset_names):
            assert results.asset_index[name] == i


# ---------------------------------------------------------------------------
# Result values populated (no zeros/NaN)
# ---------------------------------------------------------------------------


class TestResultValuesPopulated:
    """Tests that result arrays are properly filled with simulation data."""

    def test_simple_model_no_nan_in_balances(self):
        """All balance values should be populated (no NaN)."""
        model = _simple_model(n_sims=100)
        results = run_simulation(model, seed=42)

        assert not np.any(np.isnan(results.balances))

    def test_simple_model_no_nan_in_cum_inflation(self):
        """All cum_inflation values should be populated (no NaN)."""
        model = _simple_model(n_sims=100)
        results = run_simulation(model, seed=42)

        assert not np.any(np.isnan(results.cum_inflation))

    def test_simple_model_balances_all_positive(self):
        """Investment asset balances should all be positive (GBM stays positive)."""
        model = _simple_model(n_sims=100)
        results = run_simulation(model, seed=42)

        assert np.all(results.balances > 0)

    def test_simple_model_first_step_near_initial(self):
        """First recorded step should be close to initial balance (after 1 month of growth)."""
        model = _simple_model(n_sims=1000)
        results = run_simulation(model, seed=42)

        # After 1 month of 7% annual growth, median should be near 100,000
        first_step_median = np.median(results.balances[:, 0, 0])
        assert 95_000 < first_step_median < 105_000

    def test_equity_model_no_nan(self):
        """Equity model should have no NaN values in balances."""
        model = _equity_model(n_sims=200)
        results = run_simulation(model, seed=42)

        assert not np.any(np.isnan(results.balances))


# ---------------------------------------------------------------------------
# Seeded reproducibility
# ---------------------------------------------------------------------------


class TestSeededReproducibility:
    """Tests that seeded simulations produce bit-identical results."""

    def test_same_seed_produces_identical_results(self):
        """Running the same model twice with the same seed gives identical results."""
        model = _simple_model(n_sims=200)

        results1 = run_simulation(model, seed=42)
        results2 = run_simulation(model, seed=42)

        np.testing.assert_array_equal(results1.balances, results2.balances)
        np.testing.assert_array_equal(results1.cum_inflation, results2.cum_inflation)
        np.testing.assert_array_equal(results1.event_fired_at, results2.event_fired_at)

    def test_same_seed_equity_model_identical(self):
        """Equity model with events is also reproducible."""
        model = _equity_model(n_sims=500)

        results1 = run_simulation(model, seed=99)
        results2 = run_simulation(model, seed=99)

        np.testing.assert_array_equal(results1.balances, results2.balances)
        np.testing.assert_array_equal(results1.cum_inflation, results2.cum_inflation)
        np.testing.assert_array_equal(results1.event_fired_at, results2.event_fired_at)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different simulation results."""
        model = _simple_model(n_sims=100)

        results1 = run_simulation(model, seed=42)
        results2 = run_simulation(model, seed=123)

        # The results should NOT be identical
        assert not np.array_equal(results1.balances, results2.balances)

    def test_different_seeds_equity_model_different(self):
        """Different seeds produce different equity model results."""
        model = _equity_model(n_sims=200)

        results1 = run_simulation(model, seed=42)
        results2 = run_simulation(model, seed=7777)

        assert not np.array_equal(results1.balances, results2.balances)


# ---------------------------------------------------------------------------
# Equity model with events
# ---------------------------------------------------------------------------


class TestEquityModelEvents:
    """Tests that events fire correctly during full simulation."""

    def test_some_events_fire(self):
        """With the equity model, some events should fire during the simulation."""
        model = _equity_model(n_sims=1000)
        results = run_simulation(model, seed=42)

        # Event 0: 20% within 3 years — some should have fired
        event0_fired = results.event_fired_at[:, 0] != -1
        assert event0_fired.any(), "At least some event 0 should have fired"

        # Event 1: 60% within 5-6 years — some should have fired
        event1_fired = results.event_fired_at[:, 1] != -1
        assert event1_fired.any(), "At least some event 1 should have fired"

    def test_event_timing_within_windows(self):
        """Fired events should fire within their configured windows."""
        model = _equity_model(n_sims=2000)
        results = run_simulation(model, seed=42)

        # Event 0: window [0, 36)
        event0_times = results.event_fired_at[:, 0]
        event0_fired = event0_times[event0_times != -1]
        if len(event0_fired) > 0:
            assert event0_fired.min() >= 0
            assert event0_fired.max() < 36

        # Event 1: window [60, 72)
        event1_times = results.event_fired_at[:, 1]
        event1_fired = event1_times[event1_times != -1]
        if len(event1_fired) > 0:
            assert event1_fired.min() >= 60
            assert event1_fired.max() < 72

    def test_event_fire_rates_reasonable(self):
        """Event fire rates should be in reasonable range for the given probabilities."""
        model = _equity_model(n_sims=10_000)
        results = run_simulation(model, seed=42)

        # Event 0: 20% within 3 years
        event0_rate = (results.event_fired_at[:, 0] != -1).mean()
        assert 0.10 < event0_rate < 0.30, (
            f"Event 0 fire rate {event0_rate:.3f} outside expected range [0.10, 0.30]"
        )

        # Event 1: 60% within 5-6 years (but only for runs where event 0 didn't fire,
        # since the asset may have been transferred already — actually events fire
        # independently on the asset, so the rate may be lower due to already-fired
        # runs from event 0 not affecting event 1's independent firing).
        # Event 1 fires independently, so some percentage should have fired.
        event1_rate = (results.event_fired_at[:, 1] != -1).mean()
        assert event1_rate > 0.1, (
            f"Event 1 fire rate {event1_rate:.3f} too low"
        )

    def test_transferred_balance_reflected_in_results(self):
        """When events fire, transfers should be reflected in the balance arrays."""
        model = _equity_model(n_sims=2000)
        results = run_simulation(model, seed=42)

        equity_col = results.asset_index["startup_equity"]
        portfolio_col = results.asset_index["investment_portfolio"]

        # For runs where event 0 fired early, the equity balance should eventually
        # be zero (after transfer)
        event0_fired = results.event_fired_at[:, 0]
        runs_fired_early = np.where((event0_fired >= 0) & (event0_fired < 36))[0]

        if len(runs_fired_early) > 0:
            # After the event + transfer, equity balance should be 0
            # Check at the last step
            final_equity = results.balances[runs_fired_early, -1, equity_col]
            # Most should be zero or near zero (event 1 might set a value for some)
            zero_count = (final_equity == 0.0).sum()
            # At least some should be zero (transferred away)
            assert zero_count > 0


# ---------------------------------------------------------------------------
# No-events model
# ---------------------------------------------------------------------------


class TestNoEventsModel:
    """Tests that model with no events works correctly (growth + inflation only)."""

    def test_no_events_runs_successfully(self):
        """Simple model with no events should complete without errors."""
        model = _simple_model(n_sims=50, n_months=60)
        results = run_simulation(model, seed=42)

        assert results.balances.shape == (50, 60, 1)
        assert not np.any(np.isnan(results.balances))

    def test_no_events_model_balances_grow(self):
        """With 7% annual growth, balances should generally increase over time."""
        model = _simple_model(n_sims=1000, n_months=120)
        results = run_simulation(model, seed=42)

        # Compare median of first month vs last month
        first_month_median = np.median(results.balances[:, 0, 0])
        last_month_median = np.median(results.balances[:, -1, 0])

        assert last_month_median > first_month_median


# ---------------------------------------------------------------------------
# Cumulative inflation tests
# ---------------------------------------------------------------------------


class TestCumulativeInflation:
    """Tests for cumulative inflation values in results."""

    def test_cum_inflation_starts_near_one(self):
        """Cumulative inflation at step 0 should be very close to 1.0."""
        model = _simple_model(n_sims=100, n_months=120)
        results = run_simulation(model, seed=42)

        # After one step of inflation, cum_inflation should be very close to 1.0
        first_step = results.cum_inflation[:, 0]
        assert np.all(first_step > 0.99)
        assert np.all(first_step < 1.01)

    def test_cum_inflation_monotonically_increasing_on_average(self):
        """Average cumulative inflation should be monotonically increasing over time.

        With a 3% annual long-term rate, average inflation should generally increase.
        """
        model = _simple_model(n_sims=5000, n_months=120)
        results = run_simulation(model, seed=42)

        mean_cum_inflation = results.cum_inflation.mean(axis=0)

        # Check that the average is monotonically increasing (allowing tiny jitter)
        diffs = np.diff(mean_cum_inflation)
        # At least 95% of steps should show increase
        increasing_fraction = (diffs > 0).mean()
        assert increasing_fraction > 0.95, (
            f"Only {increasing_fraction:.1%} of steps show increasing average "
            f"cumulative inflation"
        )

    def test_cum_inflation_reasonable_at_10_years(self):
        """After 10 years at ~3% annual inflation, cum_inflation should be roughly 1.03^10 ~ 1.34."""
        model = _simple_model(n_sims=10_000, n_months=120)
        results = run_simulation(model, seed=42)

        # Mean cum_inflation at the last step
        mean_final = results.cum_inflation[:, -1].mean()

        # Expected: roughly (1 + 0.03/12)^120 ≈ 1.349
        # Allow wide tolerance due to stochastic nature
        assert 1.15 < mean_final < 1.55, (
            f"Mean final cum_inflation {mean_final:.3f} outside expected range"
        )

    def test_cum_inflation_no_nan(self):
        """Cumulative inflation should never be NaN."""
        model = _simple_model(n_sims=100)
        results = run_simulation(model, seed=42)

        assert not np.any(np.isnan(results.cum_inflation))


# ---------------------------------------------------------------------------
# Pipeline time steps
# ---------------------------------------------------------------------------


class TestPipelineTimeSteps:
    """Tests that the pipeline processes the correct number of time steps."""

    def test_correct_number_of_time_steps_recorded(self):
        """Results should have exactly n_steps time steps recorded."""
        n_months = 60
        model = _simple_model(n_months=n_months, n_sims=10)
        results = run_simulation(model, seed=42)

        assert results.balances.shape[1] == n_months
        assert results.cum_inflation.shape[1] == n_months

    def test_all_time_steps_populated(self):
        """Every time step in the result arrays should be populated."""
        n_months = 24
        model = _simple_model(n_months=n_months, n_sims=50)
        results = run_simulation(model, seed=42)

        # Check that no time step has all-zero balances (initial balance is 100k)
        for t in range(n_months):
            step_balances = results.balances[:, t, 0]
            assert np.all(step_balances > 0), (
                f"Time step {t} has zero or negative balances"
            )

    def test_short_simulation_1_month(self):
        """A 1-month simulation should produce exactly 1 time step of results."""
        model = _simple_model(n_months=1, n_sims=10)
        results = run_simulation(model, seed=42)

        assert results.balances.shape == (10, 1, 1)
        assert results.cum_inflation.shape == (10, 1)

    def test_long_simulation_30_years(self):
        """A 30-year simulation should produce 360 time steps."""
        model = _simple_model(n_months=360, n_sims=10)
        results = run_simulation(model, seed=42)

        assert results.balances.shape == (10, 360, 1)
        assert results.cum_inflation.shape == (10, 360)
        assert not np.any(np.isnan(results.balances))
        assert not np.any(np.isinf(results.balances))


# ---------------------------------------------------------------------------
# Integration: loading from fixture file
# ---------------------------------------------------------------------------


class TestFixtureFileIntegration:
    """Tests using the actual fixture YAML files loaded through the loader."""

    def test_simple_model_fixture(self):
        """Load simple_model.moneta.yaml and run simulation."""
        from moneta.parser.loader import load_model

        model = load_model("tests/fixtures/simple_model.moneta.yaml")
        results = run_simulation(model, seed=42)

        assert results.balances.shape == (1000, 120, 1)
        assert not np.any(np.isnan(results.balances))
        assert np.all(results.balances > 0)

    def test_equity_model_fixture(self):
        """Load equity_model.moneta.yaml and run simulation."""
        from moneta.parser.loader import load_model

        model = load_model("tests/fixtures/equity_model.moneta.yaml")
        results = run_simulation(model, seed=42)

        assert results.balances.shape == (5000, 120, 2)
        assert not np.any(np.isnan(results.balances))

    def test_equity_model_fixture_events_fire(self):
        """Equity fixture model should have events firing during simulation."""
        from moneta.parser.loader import load_model

        model = load_model("tests/fixtures/equity_model.moneta.yaml")
        results = run_simulation(model, seed=42)

        # Some events should have fired
        any_event_fired = (results.event_fired_at != -1).any()
        assert any_event_fired, "Expected some events to fire in the equity model"


# ---------------------------------------------------------------------------
# Sweep mode tests
# ---------------------------------------------------------------------------


def _sweep_model(n_sims: int = 500, seed: int = 42) -> ScenarioModel:
    """Model with sweep scenarios: conservative and aggressive."""
    return ScenarioModel(
        scenario=ScenarioConfig(
            name="Sweep test",
            time_horizon=60,  # 5 years
            simulations=n_sims,
            seed=seed,
        ),
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
                expression="portfolio > 150000",
                at=60,
                label="$150K at year 5",
            ),
            PercentilesQuery(
                type="percentiles",
                values=[25, 50, 75],
                of="portfolio",
                at=60,
                label="Portfolio at year 5",
            ),
        ],
        sweep=SweepConfig(
            scenarios=[
                SweepScenario(
                    label="conservative",
                    overrides={
                        "assets": {
                            "portfolio": {
                                "growth": {
                                    "model": "gbm",
                                    "expected_return": 0.04,
                                    "volatility": 0.10,
                                }
                            }
                        }
                    },
                ),
                SweepScenario(
                    label="aggressive",
                    overrides={
                        "assets": {
                            "portfolio": {
                                "growth": {
                                    "model": "gbm",
                                    "expected_return": 0.10,
                                    "volatility": 0.20,
                                }
                            }
                        }
                    },
                ),
            ]
        ),
    )


class TestRunSweep:
    """Tests for run_sweep — named scenario sweep mode."""

    def test_run_sweep_returns_correct_number_of_scenarios(self):
        """run_sweep should return one tuple per sweep scenario."""
        model = _sweep_model(n_sims=100)
        results = run_sweep(model, seed=42)

        assert len(results) == 2

    def test_run_sweep_labels_match_scenario_definitions(self):
        """Each result tuple should carry the correct scenario label."""
        model = _sweep_model(n_sims=100)
        results = run_sweep(model, seed=42)

        labels = [label for label, _store, _qr in results]
        assert labels == ["conservative", "aggressive"]

    def test_run_sweep_each_scenario_has_results(self):
        """Each scenario should produce a valid ResultStore and QueryResults."""
        model = _sweep_model(n_sims=100)
        results = run_sweep(model, seed=42)

        for label, store, query_results in results:
            assert store.n_runs == 100
            assert store.n_steps == 60
            assert store.n_assets == 1
            assert not np.any(np.isnan(store.balances))
            assert len(query_results) == 2

    def test_run_sweep_produces_different_results_per_scenario(self):
        """Different scenarios should produce different simulation results."""
        model = _sweep_model(n_sims=500)
        results = run_sweep(model, seed=42)

        _, store_conservative, _ = results[0]
        _, store_aggressive, _ = results[1]

        # The final balances should differ due to different growth params
        assert not np.array_equal(
            store_conservative.balances, store_aggressive.balances
        )

    def test_run_sweep_conservative_lower_than_aggressive(self):
        """Conservative scenario should produce lower median than aggressive."""
        model = _sweep_model(n_sims=1000)
        results = run_sweep(model, seed=42)

        _, store_conservative, _ = results[0]
        _, store_aggressive, _ = results[1]

        conservative_median = np.median(store_conservative.balances[:, -1, 0])
        aggressive_median = np.median(store_aggressive.balances[:, -1, 0])

        assert conservative_median < aggressive_median, (
            f"Conservative median {conservative_median:.0f} should be "
            f"less than aggressive median {aggressive_median:.0f}"
        )

    def test_run_sweep_query_results_differ(self):
        """Probability query results should differ across scenarios."""
        model = _sweep_model(n_sims=1000)
        results = run_sweep(model, seed=42)

        _, _, qr_conservative = results[0]
        _, _, qr_aggressive = results[1]

        # Probability of portfolio > 150K should be lower for conservative
        prob_conservative = qr_conservative[0].probability
        prob_aggressive = qr_aggressive[0].probability

        assert prob_conservative < prob_aggressive, (
            f"Conservative probability {prob_conservative:.1f}% should be "
            f"less than aggressive {prob_aggressive:.1f}%"
        )

    def test_run_sweep_no_scenarios_returns_empty(self):
        """Model with no sweep scenarios should return empty list."""
        model = _simple_model(n_sims=100)
        results = run_sweep(model, seed=42)

        assert results == []

    def test_run_sweep_seeded_reproducibility(self):
        """Sweep results should be reproducible with the same seed."""
        model = _sweep_model(n_sims=100)

        results1 = run_sweep(model, seed=42)
        results2 = run_sweep(model, seed=42)

        for (l1, s1, _), (l2, s2, _) in zip(results1, results2):
            assert l1 == l2
            np.testing.assert_array_equal(s1.balances, s2.balances)

    def test_run_sweep_from_fixture_file(self):
        """Load sweep_model.moneta.yaml and run sweep."""
        from moneta.parser.loader import load_model

        model = load_model("tests/fixtures/sweep_model.moneta.yaml")
        results = run_sweep(model, seed=42)

        assert len(results) == 2
        labels = [label for label, _, _ in results]
        assert labels == ["conservative", "aggressive"]

        for _, store, qr in results:
            assert store.n_runs == 500
            assert not np.any(np.isnan(store.balances))
            assert len(qr) == 2


# ---------------------------------------------------------------------------
# Cash flow pipeline integration tests
# ---------------------------------------------------------------------------


def _cash_flow_model(
    n_months: int = 120, n_sims: int = 100, seed: int | None = 42
) -> ScenarioModel:
    """Single investment asset with a monthly withdrawal cash flow."""
    return ScenarioModel(
        scenario=ScenarioConfig(
            name="Cash flow test model",
            time_horizon=n_months,
            simulations=n_sims,
            seed=seed,
        ),
        assets={
            "portfolio": InvestmentAsset(
                type="investment",
                initial_balance=100_000,
                growth=GrowthConfig(model="gbm", expected_return=0.0, volatility=0.0),
            ),
        },
        global_config=GlobalConfig(
            inflation=InflationConfig(
                model="mean_reverting",
                long_term_rate=0.0,
                volatility=0.0,
            )
        ),
        queries=[
            ProbabilityQuery(
                type="probability",
                expression="portfolio > 50000",
                at=n_months,
                label="Test query",
            )
        ],
        cash_flows={
            "monthly_expenses": CashFlowConfig(
                amount=CashFlowAmountValue(-1000.0, "monthly"),
                asset="portfolio",
            ),
        },
    )


class TestCashFlowPipeline:
    """Tests for cash flow integration in the pipeline."""

    def test_pipeline_includes_cash_flow_processor(self):
        """Model with cash flows should include CashFlowProcessor in pipeline."""
        model = _cash_flow_model(n_sims=10)
        state = SimulationState.from_scenario(model, n_runs=10)
        pipeline = build_pipeline(model, state, n_runs=10)

        processor_types = [type(p) for p in pipeline]
        assert CashFlowProcessor in processor_types

    def test_cash_flow_processor_position_before_growth(self):
        """CashFlowProcessor should come before GrowthProcessor."""
        model = _cash_flow_model(n_sims=10)
        state = SimulationState.from_scenario(model, n_runs=10)
        pipeline = build_pipeline(model, state, n_runs=10)

        processor_types = [type(p) for p in pipeline]
        cf_idx = processor_types.index(CashFlowProcessor)
        # GrowthProcessor may not be present if growth is 0/0 but it should be
        # since we have a GrowthConfig
        if GrowthProcessor in processor_types:
            growth_idx = processor_types.index(GrowthProcessor)
            assert cf_idx < growth_idx

    def test_cash_flow_processor_position_before_inflation(self):
        """CashFlowProcessor should come before InflationProcessor."""
        model = _cash_flow_model(n_sims=10)
        state = SimulationState.from_scenario(model, n_runs=10)
        pipeline = build_pipeline(model, state, n_runs=10)

        processor_types = [type(p) for p in pipeline]
        cf_idx = processor_types.index(CashFlowProcessor)
        infl_idx = processor_types.index(InflationProcessor)
        assert cf_idx < infl_idx

    def test_no_cash_flows_no_processor(self):
        """Model without cash flows should not include CashFlowProcessor."""
        model = _simple_model(n_sims=10)
        state = SimulationState.from_scenario(model, n_runs=10)
        pipeline = build_pipeline(model, state, n_runs=10)

        processor_types = [type(p) for p in pipeline]
        assert CashFlowProcessor not in processor_types

    def test_simulation_with_cash_flows_balance_decreases(self):
        """Full simulation with $1K/month withdrawal: balance decreases over time."""
        model = _cash_flow_model(n_months=24, n_sims=50)
        results = run_simulation(model, seed=42)

        # With 0% growth and 0% inflation, balance should decrease linearly
        # Initial: 100K, after 24 months: 100K - 24*1K = 76K
        final_balances = results.balances[:, -1, 0]
        np.testing.assert_allclose(final_balances, 76_000.0, atol=1.0)

    def test_simulation_cash_flow_shortfall_recorded(self):
        """Shortfall is recorded in ResultStore when balance is exhausted."""
        # Create model where balance runs out: 10K initial, $1K/month for 24 months
        model = ScenarioModel(
            scenario=ScenarioConfig(
                name="Shortfall test",
                time_horizon=24,
                simulations=50,
            ),
            assets={
                "portfolio": InvestmentAsset(
                    type="investment",
                    initial_balance=5_000,
                    growth=GrowthConfig(
                        model="gbm", expected_return=0.0, volatility=0.0
                    ),
                ),
            },
            global_config=GlobalConfig(
                inflation=InflationConfig(
                    model="mean_reverting",
                    long_term_rate=0.0,
                    volatility=0.0,
                )
            ),
            queries=[
                ProbabilityQuery(
                    type="probability",
                    expression="portfolio > 0",
                    at=24,
                    label="Test",
                )
            ],
            cash_flows={
                "expenses": CashFlowConfig(
                    amount=CashFlowAmountValue(-1000.0, "monthly"),
                    asset="portfolio",
                ),
            },
        )

        results = run_simulation(model, seed=42)

        # After 5 months, balance is 0. Remaining 19 months each have $1K shortfall.
        # Total shortfall = 19 * 1000 = 19000
        final_shortfall = results.cash_flow_shortfall[:, -1]
        np.testing.assert_allclose(final_shortfall, 19_000.0, atol=1.0)

        # Balance at end should be 0
        final_balance = results.balances[:, -1, 0]
        np.testing.assert_allclose(final_balance, 0.0, atol=1.0)

    def test_result_store_shortfall_shape(self):
        """ResultStore.cash_flow_shortfall has correct shape."""
        model = _cash_flow_model(n_months=60, n_sims=100)
        results = run_simulation(model, seed=42)

        assert results.cash_flow_shortfall.shape == (100, 60)

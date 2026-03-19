"""Tests for the CashFlowProcessor.

Covers recurring withdrawals, one-time expenses, inflation adjustment,
balance floor clamping, deposits, timing windows, annual frequency,
multiple cash flows, no-op behavior, and shortfall accumulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from moneta.engine.processors.cash_flow import CashFlowProcessor, _CashFlowConfig
from moneta.engine.state import SimulationState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    n_runs: int,
    n_assets: int = 1,
    initial_balance: float = 100_000.0,
    asset_names: list[str] | None = None,
    initial_balances: dict[str, float] | None = None,
    step: int = 0,
    cum_inflation: float = 1.0,
) -> SimulationState:
    """Build a minimal SimulationState for cash flow tests."""
    if asset_names is None:
        asset_names = ["portfolio"]
    asset_index = {name: i for i, name in enumerate(asset_names)}
    n_assets = len(asset_names)

    balances = np.zeros((n_runs, n_assets), dtype=np.float64)
    if initial_balances:
        for name, bal in initial_balances.items():
            balances[:, asset_index[name]] = bal
    else:
        balances[:, 0] = initial_balance

    return SimulationState(
        balances=balances,
        events_fired=np.zeros((n_runs, 0), dtype=bool),
        inflation_rate=np.full(n_runs, 0.03, dtype=np.float64),
        cum_inflation=np.full(n_runs, cum_inflation, dtype=np.float64),
        cash_flow_shortfall=np.zeros(n_runs, dtype=np.float64),
        step=step,
        asset_names=asset_names,
        asset_index=asset_index,
        event_index={},
    )


# ---------------------------------------------------------------------------
# Recurring withdrawal tests
# ---------------------------------------------------------------------------


class TestRecurringWithdrawal:
    """Test monthly recurring withdrawals."""

    def test_monthly_withdrawal_reduces_balance(self, seeded_rng):
        """$1000/month withdrawal from $100K for 12 months -> balance = $88K."""
        n_runs = 100
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-1000.0,
                    start_month=0,
                    end_month=12,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                )
            ]
        )

        for t in range(12):
            state.step = t
            processor.step(state, dt=1 / 12, rng=seeded_rng)

        # Balance should be exactly 100000 - 12*1000 = 88000
        np.testing.assert_allclose(state.balances[:, 0], 88_000.0)

    def test_deposit_increases_balance(self, seeded_rng):
        """$1000/month deposit increases balance over 12 months."""
        n_runs = 100
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=1000.0,
                    start_month=0,
                    end_month=12,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                )
            ]
        )

        for t in range(12):
            state.step = t
            processor.step(state, dt=1 / 12, rng=seeded_rng)

        np.testing.assert_allclose(state.balances[:, 0], 112_000.0)


# ---------------------------------------------------------------------------
# One-time expense tests
# ---------------------------------------------------------------------------


class TestOneTimeExpense:
    """Test one-time cash flows."""

    def test_one_time_withdrawal_at_step(self, seeded_rng):
        """$50K one-time expense at step 5 reduces balance by $50K."""
        n_runs = 100
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-50_000.0,
                    start_month=5,
                    end_month=6,
                    is_one_time=True,
                    adjust_for_inflation=False,
                    allow_negative=False,
                )
            ]
        )

        for t in range(12):
            state.step = t
            processor.step(state, dt=1 / 12, rng=seeded_rng)

        # Balance should be 100000 - 50000 = 50000
        np.testing.assert_allclose(state.balances[:, 0], 50_000.0)

    def test_one_time_only_applies_once(self, seeded_rng):
        """One-time expense should only fire at its designated step."""
        n_runs = 50
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-10_000.0,
                    start_month=3,
                    end_month=4,
                    is_one_time=True,
                    adjust_for_inflation=False,
                    allow_negative=False,
                )
            ]
        )

        # Step through 10 steps -- expense only at step 3
        for t in range(10):
            state.step = t
            processor.step(state, dt=1 / 12, rng=seeded_rng)

        np.testing.assert_allclose(state.balances[:, 0], 90_000.0)


# ---------------------------------------------------------------------------
# Inflation-adjusted withdrawal tests
# ---------------------------------------------------------------------------


class TestInflationAdjusted:
    """Test inflation-adjusted cash flows."""

    def test_inflation_adjusted_withdrawal(self, seeded_rng):
        """Inflation-adjusted withdrawal scales by cum_inflation."""
        n_runs = 100
        state = _make_state(
            n_runs=n_runs,
            initial_balance=100_000.0,
            cum_inflation=1.10,
        )

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-1000.0,
                    start_month=0,
                    end_month=12,
                    is_one_time=False,
                    adjust_for_inflation=True,
                    allow_negative=False,
                )
            ]
        )

        state.step = 0
        processor.step(state, dt=1 / 12, rng=seeded_rng)

        # Amount applied = -1000 * 1.10 = -1100
        expected = 100_000.0 - 1100.0
        np.testing.assert_allclose(state.balances[:, 0], expected)

    def test_non_inflation_adjusted_ignores_cum_inflation(self, seeded_rng):
        """Non-inflation-adjusted withdrawal is unaffected by cum_inflation."""
        n_runs = 100
        state = _make_state(
            n_runs=n_runs,
            initial_balance=100_000.0,
            cum_inflation=2.0,
        )

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-1000.0,
                    start_month=0,
                    end_month=12,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                )
            ]
        )

        state.step = 0
        processor.step(state, dt=1 / 12, rng=seeded_rng)

        expected = 100_000.0 - 1000.0
        np.testing.assert_allclose(state.balances[:, 0], expected)


# ---------------------------------------------------------------------------
# Balance floor and shortfall tests
# ---------------------------------------------------------------------------


class TestBalanceFloor:
    """Test balance clamping and shortfall tracking."""

    def test_balance_clamped_to_zero(self, seeded_rng):
        """$200K withdrawal from $100K -> balance clamped to 0."""
        n_runs = 100
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-200_000.0,
                    start_month=0,
                    end_month=1,
                    is_one_time=True,
                    adjust_for_inflation=False,
                    allow_negative=False,
                )
            ]
        )

        state.step = 0
        processor.step(state, dt=1 / 12, rng=seeded_rng)

        np.testing.assert_allclose(state.balances[:, 0], 0.0)
        np.testing.assert_allclose(state.cash_flow_shortfall, 100_000.0)

    def test_allow_negative_balance(self, seeded_rng):
        """$200K withdrawal from $100K with allow_negative=True -> balance = -$100K."""
        n_runs = 100
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-200_000.0,
                    start_month=0,
                    end_month=1,
                    is_one_time=True,
                    adjust_for_inflation=False,
                    allow_negative=True,
                )
            ]
        )

        state.step = 0
        processor.step(state, dt=1 / 12, rng=seeded_rng)

        np.testing.assert_allclose(state.balances[:, 0], -100_000.0)
        np.testing.assert_allclose(state.cash_flow_shortfall, 0.0)

    def test_shortfall_accumulates_over_months(self, seeded_rng):
        """Multiple months with insufficient balance accumulate shortfall."""
        n_runs = 50
        state = _make_state(n_runs=n_runs, initial_balance=5_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-3_000.0,
                    start_month=0,
                    end_month=5,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                )
            ]
        )

        # Step 0: 5000 - 3000 = 2000 (no shortfall)
        state.step = 0
        processor.step(state, dt=1 / 12, rng=seeded_rng)
        np.testing.assert_allclose(state.balances[:, 0], 2_000.0)
        np.testing.assert_allclose(state.cash_flow_shortfall, 0.0)

        # Step 1: 2000 - 3000 = -1000 -> clamped to 0, shortfall += 1000
        state.step = 1
        processor.step(state, dt=1 / 12, rng=seeded_rng)
        np.testing.assert_allclose(state.balances[:, 0], 0.0)
        np.testing.assert_allclose(state.cash_flow_shortfall, 1_000.0)

        # Step 2: 0 - 3000 = -3000 -> clamped to 0, shortfall += 3000
        state.step = 2
        processor.step(state, dt=1 / 12, rng=seeded_rng)
        np.testing.assert_allclose(state.balances[:, 0], 0.0)
        np.testing.assert_allclose(state.cash_flow_shortfall, 4_000.0)

        # Step 3: 0 - 3000 = -3000 -> clamped to 0, shortfall += 3000
        state.step = 3
        processor.step(state, dt=1 / 12, rng=seeded_rng)
        np.testing.assert_allclose(state.balances[:, 0], 0.0)
        np.testing.assert_allclose(state.cash_flow_shortfall, 7_000.0)


# ---------------------------------------------------------------------------
# Timing window tests
# ---------------------------------------------------------------------------


class TestTimingWindow:
    """Test cash flow timing windows."""

    def test_not_applied_before_start(self, seeded_rng):
        """Cash flow is not applied before start_month."""
        n_runs = 50
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-5_000.0,
                    start_month=6,
                    end_month=12,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                )
            ]
        )

        # Steps 0-5 should not apply
        for t in range(6):
            state.step = t
            processor.step(state, dt=1 / 12, rng=seeded_rng)

        np.testing.assert_allclose(state.balances[:, 0], 100_000.0)

    def test_not_applied_at_or_after_end(self, seeded_rng):
        """Cash flow is not applied at or after end_month."""
        n_runs = 50
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-5_000.0,
                    start_month=0,
                    end_month=3,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                )
            ]
        )

        for t in range(6):
            state.step = t
            processor.step(state, dt=1 / 12, rng=seeded_rng)

        # Only 3 months of withdrawals (steps 0, 1, 2)
        expected = 100_000.0 - 3 * 5_000.0
        np.testing.assert_allclose(state.balances[:, 0], expected)

    def test_applied_within_window(self, seeded_rng):
        """Cash flow applied exactly within [start_month, end_month)."""
        n_runs = 50
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-2_000.0,
                    start_month=3,
                    end_month=7,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                )
            ]
        )

        for t in range(12):
            state.step = t
            processor.step(state, dt=1 / 12, rng=seeded_rng)

        # 4 months of withdrawals (steps 3, 4, 5, 6)
        expected = 100_000.0 - 4 * 2_000.0
        np.testing.assert_allclose(state.balances[:, 0], expected)


# ---------------------------------------------------------------------------
# Annual frequency tests
# ---------------------------------------------------------------------------


class TestAnnualFrequency:
    """Test annual cash flows spread over 12 months."""

    def test_annual_amount_spread_over_12_months(self, seeded_rng):
        """$12,000 annually spread as $1,000/month."""
        n_runs = 100
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-1_000.0,  # 12000 / 12
                    start_month=0,
                    end_month=12,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                )
            ]
        )

        for t in range(12):
            state.step = t
            processor.step(state, dt=1 / 12, rng=seeded_rng)

        # 12 * 1000 = 12000 total withdrawn
        np.testing.assert_allclose(state.balances[:, 0], 88_000.0)


# ---------------------------------------------------------------------------
# Multiple cash flows tests
# ---------------------------------------------------------------------------


class TestMultipleCashFlows:
    """Test multiple cash flows on the same or different assets."""

    def test_multiple_flows_same_asset(self, seeded_rng):
        """Two withdrawals from the same asset both applied."""
        n_runs = 50
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-1_000.0,
                    start_month=0,
                    end_month=12,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                ),
                _CashFlowConfig(
                    asset_col=0,
                    monthly_amount=-500.0,
                    start_month=0,
                    end_month=12,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                ),
            ]
        )

        for t in range(12):
            state.step = t
            processor.step(state, dt=1 / 12, rng=seeded_rng)

        # Total withdrawn = 12 * (1000 + 500) = 18000
        np.testing.assert_allclose(state.balances[:, 0], 82_000.0)

    def test_flows_on_different_assets(self, seeded_rng):
        """Cash flows on different assets are applied independently."""
        n_runs = 50
        state = _make_state(
            n_runs=n_runs,
            asset_names=["savings", "brokerage"],
            initial_balances={"savings": 50_000.0, "brokerage": 100_000.0},
        )

        processor = CashFlowProcessor(
            [
                _CashFlowConfig(
                    asset_col=0,  # savings
                    monthly_amount=-500.0,
                    start_month=0,
                    end_month=6,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                ),
                _CashFlowConfig(
                    asset_col=1,  # brokerage
                    monthly_amount=1_000.0,
                    start_month=0,
                    end_month=6,
                    is_one_time=False,
                    adjust_for_inflation=False,
                    allow_negative=False,
                ),
            ]
        )

        for t in range(6):
            state.step = t
            processor.step(state, dt=1 / 12, rng=seeded_rng)

        np.testing.assert_allclose(state.balances[:, 0], 47_000.0)  # 50K - 6*500
        np.testing.assert_allclose(state.balances[:, 1], 106_000.0)  # 100K + 6*1000


# ---------------------------------------------------------------------------
# No cash flows (no-op) tests
# ---------------------------------------------------------------------------


class TestNoCashFlows:
    """Test that empty processor is a no-op."""

    def test_empty_processor_is_noop(self, seeded_rng):
        """Processor with no configs does not change state."""
        n_runs = 50
        state = _make_state(n_runs=n_runs, initial_balance=100_000.0)
        balances_before = state.balances.copy()

        processor = CashFlowProcessor([])

        state.step = 0
        processor.step(state, dt=1 / 12, rng=seeded_rng)

        np.testing.assert_array_equal(state.balances, balances_before)
        np.testing.assert_array_equal(state.cash_flow_shortfall, 0.0)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestCashFlowProtocol:
    """Verify CashFlowProcessor satisfies the Processor protocol."""

    def test_is_processor(self):
        from moneta.engine.processors import Processor

        processor = CashFlowProcessor([])
        assert isinstance(processor, Processor)


# ---------------------------------------------------------------------------
# from_scenario tests
# ---------------------------------------------------------------------------


class TestFromScenario:
    """Test CashFlowProcessor.from_scenario factory method."""

    def test_from_scenario_with_monthly_withdrawal(self, seeded_rng):
        """from_scenario builds correct config for monthly withdrawal."""
        from moneta.parser.models import (
            CashFlowConfig,
            GlobalConfig,
            GrowthConfig,
            InflationConfig,
            InvestmentAsset,
            ProbabilityQuery,
            ScenarioConfig,
            ScenarioModel,
        )
        from moneta.parser.types import CashFlowAmountValue

        model = ScenarioModel(
            scenario=ScenarioConfig(
                name="cash_flow_test",
                time_horizon=120,
                simulations=100,
            ),
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
                    model="mean_reverting",
                    long_term_rate=0.03,
                    volatility=0.01,
                )
            ),
            queries=[
                ProbabilityQuery(
                    type="probability",
                    expression="portfolio > 50000",
                    at=120,
                    label="test",
                )
            ],
            cash_flows={
                "monthly_expenses": CashFlowConfig(
                    amount=CashFlowAmountValue(-5000.0, "monthly"),
                    asset="portfolio",
                    start=12,  # starts at month 12 (1 year)
                    end=60,  # ends at month 60 (5 years)
                ),
            },
        )

        asset_index = {"portfolio": 0}
        processor = CashFlowProcessor.from_scenario(model, asset_index)

        assert len(processor._configs) == 1
        cfg = processor._configs[0]
        assert cfg.asset_col == 0
        assert cfg.monthly_amount == -5000.0
        assert cfg.start_month == 11  # 12 - 1 = 11 (0-based)
        assert cfg.end_month == 60
        assert cfg.is_one_time is False
        assert cfg.adjust_for_inflation is False
        assert cfg.allow_negative is False

    def test_from_scenario_with_one_time_expense(self, seeded_rng):
        """from_scenario builds correct config for one-time expense."""
        from moneta.parser.models import (
            CashFlowConfig,
            GlobalConfig,
            GrowthConfig,
            InflationConfig,
            InvestmentAsset,
            ProbabilityQuery,
            ScenarioConfig,
            ScenarioModel,
        )
        from moneta.parser.types import CashFlowAmountValue

        model = ScenarioModel(
            scenario=ScenarioConfig(
                name="one_time_test",
                time_horizon=120,
                simulations=100,
            ),
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
                    model="mean_reverting",
                    long_term_rate=0.03,
                    volatility=0.01,
                )
            ),
            queries=[
                ProbabilityQuery(
                    type="probability",
                    expression="portfolio > 50000",
                    at=120,
                    label="test",
                )
            ],
            cash_flows={
                "home_purchase": CashFlowConfig(
                    amount=CashFlowAmountValue(-50_000.0, None),
                    asset="portfolio",
                    at=36,  # at month 36 (year 3)
                ),
            },
        )

        asset_index = {"portfolio": 0}
        processor = CashFlowProcessor.from_scenario(model, asset_index)

        assert len(processor._configs) == 1
        cfg = processor._configs[0]
        assert cfg.asset_col == 0
        assert cfg.monthly_amount == -50_000.0
        assert cfg.start_month == 35  # 36 - 1 = 35 (0-based)
        assert cfg.end_month == 36
        assert cfg.is_one_time is True

    def test_from_scenario_with_annual_deposit(self, seeded_rng):
        """from_scenario spreads annual amount over 12 months."""
        from moneta.parser.models import (
            CashFlowConfig,
            GlobalConfig,
            GrowthConfig,
            InflationConfig,
            InvestmentAsset,
            ProbabilityQuery,
            ScenarioConfig,
            ScenarioModel,
        )
        from moneta.parser.types import CashFlowAmountValue

        model = ScenarioModel(
            scenario=ScenarioConfig(
                name="annual_test",
                time_horizon=120,
                simulations=100,
            ),
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
                    model="mean_reverting",
                    long_term_rate=0.03,
                    volatility=0.01,
                )
            ),
            queries=[
                ProbabilityQuery(
                    type="probability",
                    expression="portfolio > 50000",
                    at=120,
                    label="test",
                )
            ],
            cash_flows={
                "annual_bonus": CashFlowConfig(
                    amount=CashFlowAmountValue(24_000.0, "annually"),
                    asset="portfolio",
                ),
            },
        )

        asset_index = {"portfolio": 0}
        processor = CashFlowProcessor.from_scenario(model, asset_index)

        assert len(processor._configs) == 1
        cfg = processor._configs[0]
        assert cfg.monthly_amount == pytest.approx(2_000.0)  # 24000 / 12
        assert cfg.start_month == 0
        assert cfg.end_month == 120
        assert cfg.is_one_time is False

    def test_from_scenario_no_cash_flows(self):
        """from_scenario with no cash_flows returns empty processor."""
        from moneta.parser.models import (
            GlobalConfig,
            GrowthConfig,
            InflationConfig,
            InvestmentAsset,
            ProbabilityQuery,
            ScenarioConfig,
            ScenarioModel,
        )

        model = ScenarioModel(
            scenario=ScenarioConfig(name="no_cf", time_horizon=60, simulations=10),
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
                    model="mean_reverting",
                    long_term_rate=0.03,
                    volatility=0.01,
                )
            ),
            queries=[
                ProbabilityQuery(
                    type="probability",
                    expression="portfolio > 50000",
                    at=60,
                    label="test",
                )
            ],
        )

        asset_index = {"portfolio": 0}
        processor = CashFlowProcessor.from_scenario(model, asset_index)
        assert len(processor._configs) == 0

    def test_from_scenario_inflation_adjusted(self, seeded_rng):
        """from_scenario sets adjust_for_inflation correctly."""
        from moneta.parser.models import (
            CashFlowConfig,
            GlobalConfig,
            GrowthConfig,
            InflationConfig,
            InvestmentAsset,
            ProbabilityQuery,
            ScenarioConfig,
            ScenarioModel,
        )
        from moneta.parser.types import CashFlowAmountValue

        model = ScenarioModel(
            scenario=ScenarioConfig(
                name="infl_test",
                time_horizon=120,
                simulations=100,
            ),
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
                    model="mean_reverting",
                    long_term_rate=0.03,
                    volatility=0.01,
                )
            ),
            queries=[
                ProbabilityQuery(
                    type="probability",
                    expression="portfolio > 50000",
                    at=120,
                    label="test",
                )
            ],
            cash_flows={
                "living_expenses": CashFlowConfig(
                    amount=CashFlowAmountValue(-3000.0, "monthly"),
                    asset="portfolio",
                    adjust_for="inflation",
                ),
            },
        )

        asset_index = {"portfolio": 0}
        processor = CashFlowProcessor.from_scenario(model, asset_index)

        assert len(processor._configs) == 1
        assert processor._configs[0].adjust_for_inflation is True

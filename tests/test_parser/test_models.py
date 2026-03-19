"""Tests for Pydantic model validation."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from moneta.parser.models import (
    Asset,
    DistributionQuery,
    ExpectedQuery,
    GlobalConfig,
    GrowthConfig,
    IlliquidEquityAsset,
    InflationConfig,
    InvestmentAsset,
    LiquidityEvent,
    PercentilesQuery,
    PresetRef,
    ProbabilityQuery,
    Query,
    ScenarioConfig,
    ScenarioModel,
    SweepConfig,
    SweepScenario,
)

# ===================================================================
# Fixtures — reusable model data dicts
# ===================================================================


def _base_model_dict() -> dict:
    """Return a valid complete model dict for ScenarioModel."""
    return {
        "scenario": {
            "name": "Test scenario",
            "time_horizon": "10 years",
            "simulations": 1000,
            "time_step": "monthly",
            "seed": 42,
        },
        "assets": {
            "investment_portfolio": {
                "type": "investment",
                "initial_balance": "$850,000",
                "growth": {
                    "model": "gbm",
                    "expected_return": "7% annually",
                    "volatility": "15% annually",
                },
            },
            "startup_equity": {
                "type": "illiquid_equity",
                "current_valuation": "$500,000",
                "shares": 50000,
                "liquidity_events": [
                    {
                        "probability": "20% within 3 years",
                        "valuation_range": ["2x", "5x"],
                    },
                    {
                        "probability": "60% within 5-6 years",
                        "valuation_range": ["3x", "10x"],
                    },
                ],
                "on_liquidation": {
                    "transfer_to": "investment_portfolio",
                },
            },
        },
        "global": {
            "inflation": {
                "model": "mean_reverting",
                "long_term_rate": "3% annually",
                "volatility": "1% annually",
            },
        },
        "queries": [
            {
                "type": "probability",
                "expression": "investment_portfolio + startup_equity > 2000000",
                "at": "10 years",
                "label": "$2M net worth at year 10",
            },
            {
                "type": "percentiles",
                "values": [10, 25, 50, 75, 90],
                "of": "investment_portfolio",
                "at": ["5 years", "10 years"],
                "adjust_for": "inflation",
                "label": "Portfolio value distribution",
            },
        ],
    }


@pytest.fixture
def base_model_dict() -> dict:
    """Return a fresh copy of the base model dict for each test."""
    return _base_model_dict()


# ===================================================================
# ScenarioConfig tests
# ===================================================================


class TestScenarioConfig:
    """Tests for ScenarioConfig parsing."""

    def test_valid_config(self) -> None:
        cfg = ScenarioConfig(
            name="Test", time_horizon="30 years", simulations=10000, seed=42
        )
        assert cfg.name == "Test"
        assert cfg.time_horizon == 360
        assert cfg.simulations == 10000
        assert cfg.time_step == "monthly"
        assert cfg.seed == 42

    def test_defaults(self) -> None:
        cfg = ScenarioConfig(name="Test", time_horizon="5 years")
        assert cfg.simulations == 10000
        assert cfg.time_step == "monthly"
        assert cfg.seed is None

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioConfig(time_horizon="10 years")

    def test_missing_horizon_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioConfig(name="Test")


# ===================================================================
# Asset discriminated union tests
# ===================================================================


class TestAssetUnion:
    """Tests for Asset discriminated union routing."""

    def test_investment_asset(self) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(Asset)
        asset = ta.validate_python(
            {
                "type": "investment",
                "initial_balance": 100000,
                "growth": {
                    "model": "gbm",
                    "expected_return": "7% annually",
                    "volatility": "15% annually",
                },
            }
        )
        assert isinstance(asset, InvestmentAsset)
        assert asset.initial_balance == 100000.0

    def test_illiquid_equity_asset(self) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(Asset)
        asset = ta.validate_python(
            {
                "type": "illiquid_equity",
                "current_valuation": 500000,
                "liquidity_events": [
                    {
                        "probability": "20% within 3 years",
                        "valuation_range": ["2x", "5x"],
                    }
                ],
                "on_liquidation": {"transfer_to": "portfolio"},
            }
        )
        assert isinstance(asset, IlliquidEquityAsset)
        assert asset.current_valuation == 500000.0
        assert len(asset.liquidity_events) == 1

    def test_unknown_type_raises(self) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(Asset)
        with pytest.raises(ValidationError):
            ta.validate_python(
                {
                    "type": "savings_account",
                    "balance": 10000,
                }
            )

    def test_investment_with_preset(self) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(Asset)
        asset = ta.validate_python(
            {
                "type": "investment",
                "initial_balance": "$100,000",
                "growth": {"preset": "sp500"},
            }
        )
        assert isinstance(asset, InvestmentAsset)
        assert isinstance(asset.growth, PresetRef)
        assert asset.growth.preset == "sp500"


# ===================================================================
# GrowthConfig and PresetRef tests
# ===================================================================


class TestGrowthConfig:
    """Tests for GrowthConfig and PresetRef."""

    def test_valid_growth(self) -> None:
        gc = GrowthConfig(
            model="gbm",
            expected_return="7% annually",
            volatility="15% annually",
        )
        assert gc.model == "gbm"
        assert gc.expected_return == pytest.approx(0.07)
        assert gc.volatility == pytest.approx(0.15)

    def test_preset_ref(self) -> None:
        pr = PresetRef(preset="sp500")
        assert pr.preset == "sp500"

    def test_invalid_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            GrowthConfig(
                model="linear",
                expected_return="7% annually",
                volatility="15% annually",
            )


# ===================================================================
# InflationConfig tests
# ===================================================================


class TestInflationConfig:
    """Tests for InflationConfig."""

    def test_valid_inflation(self) -> None:
        ic = InflationConfig(
            model="mean_reverting",
            long_term_rate="3% annually",
            volatility="1% annually",
        )
        assert ic.long_term_rate == pytest.approx(0.03)
        assert ic.volatility == pytest.approx(0.01)
        assert ic.mean_reversion_speed == 0.5  # default

    def test_custom_mean_reversion(self) -> None:
        ic = InflationConfig(
            model="mean_reverting",
            long_term_rate="3% annually",
            volatility="1% annually",
            mean_reversion_speed=0.8,
        )
        assert ic.mean_reversion_speed == 0.8


# ===================================================================
# GlobalConfig tests
# ===================================================================


class TestGlobalConfig:
    """Tests for GlobalConfig."""

    def test_with_inflation_config(self) -> None:
        gc = GlobalConfig(
            inflation={
                "model": "mean_reverting",
                "long_term_rate": "3% annually",
                "volatility": "1% annually",
            }
        )
        assert isinstance(gc.inflation, InflationConfig)

    def test_with_preset(self) -> None:
        gc = GlobalConfig(inflation={"preset": "us_inflation"})
        assert isinstance(gc.inflation, PresetRef)


# ===================================================================
# Query discriminated union tests
# ===================================================================


class TestQueryUnion:
    """Tests for Query discriminated union routing."""

    def test_probability_query(self) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(Query)
        q = ta.validate_python(
            {
                "type": "probability",
                "expression": "portfolio > 1000000",
                "at": "10 years",
                "label": "Million at 10",
            }
        )
        assert isinstance(q, ProbabilityQuery)
        assert q.at == 120

    def test_percentiles_query(self) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(Query)
        q = ta.validate_python(
            {
                "type": "percentiles",
                "values": [10, 50, 90],
                "of": "portfolio",
                "at": ["5 years", "10 years"],
            }
        )
        assert isinstance(q, PercentilesQuery)
        assert q.at == [60, 120]

    def test_expected_query(self) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(Query)
        q = ta.validate_python(
            {
                "type": "expected",
                "of": "portfolio",
                "at": "10 years",
            }
        )
        assert isinstance(q, ExpectedQuery)

    def test_distribution_query(self) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(Query)
        q = ta.validate_python(
            {
                "type": "distribution",
                "of": "portfolio",
                "at": "10 years",
                "bins": 100,
            }
        )
        assert isinstance(q, DistributionQuery)
        assert q.bins == 100

    def test_unknown_query_type_raises(self) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(Query)
        with pytest.raises(ValidationError):
            ta.validate_python(
                {
                    "type": "correlation",
                    "assets": ["a", "b"],
                }
            )

    def test_percentiles_single_at(self) -> None:
        """Percentiles query can accept a single Duration for 'at'."""
        from pydantic import TypeAdapter

        ta = TypeAdapter(Query)
        q = ta.validate_python(
            {
                "type": "percentiles",
                "values": [50],
                "of": "portfolio",
                "at": "10 years",
            }
        )
        assert isinstance(q, PercentilesQuery)
        assert q.at == 120


# ===================================================================
# LiquidityEvent tests
# ===================================================================


class TestLiquidityEvent:
    """Tests for LiquidityEvent."""

    def test_valid_event(self) -> None:
        ev = LiquidityEvent(
            probability="20% within 3 years",
            valuation_range=["2x", "5x"],
        )
        assert ev.probability.probability == pytest.approx(0.20)
        assert ev.probability.start_month == 0
        assert ev.probability.end_month == 36
        assert ev.valuation_range == (2.0, 5.0)


# ===================================================================
# SweepConfig tests
# ===================================================================


class TestSweepConfig:
    """Tests for SweepConfig and SweepScenario."""

    def test_valid_sweep(self) -> None:
        sc = SweepConfig(
            scenarios=[
                SweepScenario(
                    label="conservative",
                    overrides={"assets": {"equity": {"valuation_range": ["1x", "3x"]}}},
                ),
                SweepScenario(
                    label="optimistic",
                    overrides={
                        "assets": {"equity": {"valuation_range": ["5x", "20x"]}}
                    },
                ),
            ]
        )
        assert len(sc.scenarios) == 2
        assert sc.scenarios[0].label == "conservative"


# ===================================================================
# ScenarioModel (top-level) tests
# ===================================================================


class TestScenarioModel:
    """Tests for the top-level ScenarioModel with cross-reference validation."""

    def test_valid_complete_model(self, base_model_dict: dict) -> None:
        model = ScenarioModel.model_validate(base_model_dict)
        assert model.scenario.name == "Test scenario"
        assert model.scenario.time_horizon == 120
        assert "investment_portfolio" in model.assets
        assert "startup_equity" in model.assets
        assert isinstance(model.assets["investment_portfolio"], InvestmentAsset)
        assert isinstance(model.assets["startup_equity"], IlliquidEquityAsset)
        assert isinstance(model.global_config.inflation, InflationConfig)
        assert len(model.queries) == 2

    def test_global_alias(self, base_model_dict: dict) -> None:
        """The 'global' YAML key maps to 'global_config' attribute."""
        model = ScenarioModel.model_validate(base_model_dict)
        assert model.global_config is not None
        assert isinstance(model.global_config.inflation, InflationConfig)

    def test_populate_by_name(self, base_model_dict: dict) -> None:
        """Can also use 'global_config' directly instead of 'global' alias."""
        data = copy.deepcopy(base_model_dict)
        data["global_config"] = data.pop("global")
        model = ScenarioModel.model_validate(data)
        assert model.global_config is not None

    def test_missing_scenario_raises(self, base_model_dict: dict) -> None:
        del base_model_dict["scenario"]
        with pytest.raises(ValidationError):
            ScenarioModel.model_validate(base_model_dict)

    def test_missing_assets_raises(self, base_model_dict: dict) -> None:
        del base_model_dict["assets"]
        with pytest.raises(ValidationError):
            ScenarioModel.model_validate(base_model_dict)

    def test_missing_global_raises(self, base_model_dict: dict) -> None:
        del base_model_dict["global"]
        with pytest.raises(ValidationError):
            ScenarioModel.model_validate(base_model_dict)

    def test_missing_queries_raises(self, base_model_dict: dict) -> None:
        del base_model_dict["queries"]
        with pytest.raises(ValidationError):
            ScenarioModel.model_validate(base_model_dict)

    def test_bad_transfer_to_reference(self, base_model_dict: dict) -> None:
        """transfer_to referencing a non-existent asset should fail."""
        base_model_dict["assets"]["startup_equity"]["on_liquidation"]["transfer_to"] = (
            "nonexistent_asset"
        )
        with pytest.raises(ValidationError, match="no asset named 'nonexistent_asset'"):
            ScenarioModel.model_validate(base_model_dict)

    def test_self_transfer_raises(self, base_model_dict: dict) -> None:
        """An asset transferring to itself should fail."""
        base_model_dict["assets"]["startup_equity"]["on_liquidation"]["transfer_to"] = (
            "startup_equity"
        )
        with pytest.raises(ValidationError, match="transfers to itself"):
            ScenarioModel.model_validate(base_model_dict)

    def test_query_time_exceeds_horizon(self, base_model_dict: dict) -> None:
        """Query asking about a time beyond the horizon should fail."""
        base_model_dict["queries"] = [
            {
                "type": "probability",
                "expression": "investment_portfolio > 1000000",
                "at": "15 years",  # horizon is 10 years
            }
        ]
        with pytest.raises(ValidationError, match="time_horizon"):
            ScenarioModel.model_validate(base_model_dict)

    def test_percentile_query_time_exceeds_horizon(self, base_model_dict: dict) -> None:
        """Percentile query with a time point beyond horizon should fail."""
        base_model_dict["queries"] = [
            {
                "type": "percentiles",
                "values": [50],
                "of": "investment_portfolio",
                "at": ["5 years", "15 years"],  # 15 years > 10 year horizon
            }
        ]
        with pytest.raises(ValidationError, match="time_horizon"):
            ScenarioModel.model_validate(base_model_dict)

    def test_query_references_nonexistent_asset(self, base_model_dict: dict) -> None:
        """Query 'of' referencing a non-existent asset should fail."""
        base_model_dict["queries"] = [
            {
                "type": "expected",
                "of": "savings_account",
                "at": "5 years",
            }
        ]
        with pytest.raises(ValidationError, match="no asset with that name"):
            ScenarioModel.model_validate(base_model_dict)

    def test_query_expression_not_validated_as_asset(
        self, base_model_dict: dict
    ) -> None:
        """Complex expression in 'of' should not be validated as asset name."""
        base_model_dict["queries"] = [
            {
                "type": "expected",
                "of": "investment_portfolio + startup_equity",
                "at": "5 years",
            }
        ]
        # Should not raise — expressions with operators are deferred to
        # the expression parser
        model = ScenarioModel.model_validate(base_model_dict)
        assert len(model.queries) == 1

    def test_probability_query_expression_stored_as_string(
        self, base_model_dict: dict
    ) -> None:
        """Expression strings are stored as-is for later parsing."""
        model = ScenarioModel.model_validate(base_model_dict)
        prob_query = model.queries[0]
        assert isinstance(prob_query, ProbabilityQuery)
        assert (
            prob_query.expression == "investment_portfolio + startup_equity > 2000000"
        )

    def test_with_sweep(self, base_model_dict: dict) -> None:
        base_model_dict["sweep"] = {
            "scenarios": [
                {
                    "label": "conservative",
                    "overrides": {
                        "assets": {
                            "startup_equity": {
                                "liquidity_events": [
                                    {
                                        "probability": "20% within 3 years",
                                        "valuation_range": ["1x", "3x"],
                                    }
                                ]
                            }
                        }
                    },
                },
                {
                    "label": "optimistic",
                    "overrides": {
                        "assets": {
                            "startup_equity": {
                                "liquidity_events": [
                                    {
                                        "probability": "20% within 3 years",
                                        "valuation_range": ["5x", "20x"],
                                    }
                                ]
                            }
                        }
                    },
                },
            ]
        }
        model = ScenarioModel.model_validate(base_model_dict)
        assert model.sweep is not None
        assert len(model.sweep.scenarios) == 2

    def test_sweep_is_optional(self, base_model_dict: dict) -> None:
        model = ScenarioModel.model_validate(base_model_dict)
        assert model.sweep is None

    def test_currency_parsing_in_model(self, base_model_dict: dict) -> None:
        model = ScenarioModel.model_validate(base_model_dict)
        inv = model.assets["investment_portfolio"]
        assert isinstance(inv, InvestmentAsset)
        assert inv.initial_balance == pytest.approx(850000.0)

    def test_growth_config_in_model(self, base_model_dict: dict) -> None:
        model = ScenarioModel.model_validate(base_model_dict)
        inv = model.assets["investment_portfolio"]
        assert isinstance(inv, InvestmentAsset)
        assert isinstance(inv.growth, GrowthConfig)
        assert inv.growth.expected_return == pytest.approx(0.07)
        assert inv.growth.volatility == pytest.approx(0.15)

    def test_liquidity_events_in_model(self, base_model_dict: dict) -> None:
        model = ScenarioModel.model_validate(base_model_dict)
        equity = model.assets["startup_equity"]
        assert isinstance(equity, IlliquidEquityAsset)
        assert len(equity.liquidity_events) == 2

        ev0 = equity.liquidity_events[0]
        assert ev0.probability.probability == pytest.approx(0.20)
        assert ev0.probability.start_month == 0
        assert ev0.probability.end_month == 36
        assert ev0.valuation_range == (2.0, 5.0)

        ev1 = equity.liquidity_events[1]
        assert ev1.probability.probability == pytest.approx(0.60)
        assert ev1.probability.start_month == 60
        assert ev1.probability.end_month == 72
        assert ev1.valuation_range == (3.0, 10.0)

    def test_inflation_config_in_model(self, base_model_dict: dict) -> None:
        model = ScenarioModel.model_validate(base_model_dict)
        infl = model.global_config.inflation
        assert isinstance(infl, InflationConfig)
        assert infl.long_term_rate == pytest.approx(0.03)
        assert infl.volatility == pytest.approx(0.01)
        assert infl.mean_reversion_speed == 0.5

    def test_global_with_preset(self, base_model_dict: dict) -> None:
        base_model_dict["global"] = {"inflation": {"preset": "us_inflation"}}
        model = ScenarioModel.model_validate(base_model_dict)
        assert isinstance(model.global_config.inflation, PresetRef)
        assert model.global_config.inflation.preset == "us_inflation"

    def test_query_at_boundary_of_horizon(self, base_model_dict: dict) -> None:
        """Query at exactly the horizon should be valid."""
        base_model_dict["queries"] = [
            {
                "type": "probability",
                "expression": "investment_portfolio > 1000000",
                "at": "10 years",  # exactly the horizon
            }
        ]
        model = ScenarioModel.model_validate(base_model_dict)
        assert len(model.queries) == 1

    def test_distribution_query_default_bins(self) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(Query)
        q = ta.validate_python(
            {
                "type": "distribution",
                "of": "portfolio",
                "at": "10 years",
            }
        )
        assert isinstance(q, DistributionQuery)
        assert q.bins == 50  # default

    def test_adjust_for_inflation(self) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(Query)
        q = ta.validate_python(
            {
                "type": "expected",
                "of": "portfolio",
                "at": "10 years",
                "adjust_for": "inflation",
            }
        )
        assert isinstance(q, ExpectedQuery)
        assert q.adjust_for == "inflation"

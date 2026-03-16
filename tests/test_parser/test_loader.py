"""Tests for the YAML loader, preset resolution, and deep_merge."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from moneta import MonetaError
from moneta.parser.loader import deep_merge, load_model, load_model_from_string
from moneta.parser.models import (
    GrowthConfig,
    IlliquidEquityAsset,
    InflationConfig,
    InvestmentAsset,
    ScenarioModel,
)
from moneta.presets import get_preset, list_presets

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


# ===================================================================
# load_model — file-based loading
# ===================================================================


class TestLoadModel:
    """Tests for load_model (file-based)."""

    def test_load_simple_model(self) -> None:
        """Load the simple fixture and verify parsed values."""
        model = load_model(FIXTURES / "simple_model.moneta.yaml")
        assert isinstance(model, ScenarioModel)
        assert model.scenario.name == "Simple investment model"
        assert model.scenario.time_horizon == 120  # 10 years in months
        assert model.scenario.simulations == 1000
        assert model.scenario.seed == 42
        assert "portfolio" in model.assets
        inv = model.assets["portfolio"]
        assert isinstance(inv, InvestmentAsset)
        assert inv.initial_balance == pytest.approx(100000.0)
        assert isinstance(inv.growth, GrowthConfig)
        assert inv.growth.expected_return == pytest.approx(0.07)
        assert inv.growth.volatility == pytest.approx(0.15)

    def test_load_equity_model(self) -> None:
        """Load the equity fixture and verify parsed values."""
        model = load_model(FIXTURES / "equity_model.moneta.yaml")
        assert model.scenario.name == "Investment + equity model"
        assert "investment_portfolio" in model.assets
        assert "startup_equity" in model.assets
        equity = model.assets["startup_equity"]
        assert isinstance(equity, IlliquidEquityAsset)
        assert equity.current_valuation == pytest.approx(500000.0)
        assert len(equity.liquidity_events) == 2
        assert equity.on_liquidation.transfer_to == "investment_portfolio"
        assert len(model.queries) == 2

    def test_file_not_found(self) -> None:
        """Missing file raises MonetaError with clear message."""
        with pytest.raises(MonetaError, match="not found"):
            load_model(Path("/nonexistent/path/model.yaml"))

    def test_malformed_yaml_file(self, tmp_path: Path) -> None:
        """Malformed YAML raises MonetaError."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{ unclosed: [bracket", encoding="utf-8")
        with pytest.raises(MonetaError, match="Malformed YAML"):
            load_model(bad_file)

    def test_non_dict_yaml_file(self, tmp_path: Path) -> None:
        """YAML that is not a mapping at top level raises MonetaError."""
        bad_file = tmp_path / "list.yaml"
        bad_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(MonetaError, match="YAML mapping"):
            load_model(bad_file)


# ===================================================================
# load_model_from_string — string-based loading
# ===================================================================


class TestLoadModelFromString:
    """Tests for load_model_from_string."""

    def test_valid_string(self) -> None:
        """Parsing a valid YAML string produces a ScenarioModel."""
        yaml_str = textwrap.dedent("""\
            scenario:
              name: "String test"
              time_horizon: 5 years
              simulations: 100

            assets:
              portfolio:
                type: investment
                initial_balance: 50000
                growth:
                  model: gbm
                  expected_return: 7% annually
                  volatility: 15% annually

            global:
              inflation:
                model: mean_reverting
                long_term_rate: 3% annually
                volatility: 1% annually

            queries:
              - type: percentiles
                values: [50]
                of: portfolio
                at: 5 years
        """)
        model = load_model_from_string(yaml_str)
        assert isinstance(model, ScenarioModel)
        assert model.scenario.name == "String test"
        assert model.scenario.time_horizon == 60
        assert model.assets["portfolio"].initial_balance == pytest.approx(50000.0)

    def test_malformed_string(self) -> None:
        """Malformed YAML string raises MonetaError."""
        with pytest.raises(MonetaError, match="Malformed YAML"):
            load_model_from_string("{ bad: [yaml")

    def test_non_dict_string(self) -> None:
        """Non-mapping YAML string raises MonetaError."""
        with pytest.raises(MonetaError, match="YAML mapping"):
            load_model_from_string("- list\n- items\n")

    def test_validation_error_string(self) -> None:
        """Invalid model content raises MonetaError."""
        with pytest.raises(MonetaError, match="Validation error"):
            load_model_from_string("scenario:\n  name: missing fields\n")


# ===================================================================
# Preset resolution
# ===================================================================


class TestPresetResolution:
    """Tests for preset references being resolved in model loading."""

    def test_growth_preset_resolved(self) -> None:
        """A growth preset: sp500 is expanded to the full GBM config."""
        yaml_str = textwrap.dedent("""\
            scenario:
              name: "Preset test"
              time_horizon: 10 years
              simulations: 100

            assets:
              portfolio:
                type: investment
                initial_balance: 100000
                growth:
                  preset: sp500

            global:
              inflation:
                model: mean_reverting
                long_term_rate: 3% annually
                volatility: 1% annually

            queries:
              - type: percentiles
                values: [50]
                of: portfolio
                at: 10 years
        """)
        model = load_model_from_string(yaml_str)
        inv = model.assets["portfolio"]
        assert isinstance(inv, InvestmentAsset)
        assert isinstance(inv.growth, GrowthConfig)
        assert inv.growth.model == "gbm"
        assert inv.growth.expected_return == pytest.approx(0.07)
        assert inv.growth.volatility == pytest.approx(0.15)

    def test_inflation_preset_resolved(self) -> None:
        """An inflation preset: us_inflation is expanded correctly."""
        yaml_str = textwrap.dedent("""\
            scenario:
              name: "Inflation preset test"
              time_horizon: 5 years
              simulations: 100

            assets:
              portfolio:
                type: investment
                initial_balance: 100000
                growth:
                  model: gbm
                  expected_return: 7% annually
                  volatility: 15% annually

            global:
              inflation:
                preset: us_inflation

            queries:
              - type: percentiles
                values: [50]
                of: portfolio
                at: 5 years
        """)
        model = load_model_from_string(yaml_str)
        infl = model.global_config.inflation
        assert isinstance(infl, InflationConfig)
        assert infl.model == "mean_reverting"
        assert infl.long_term_rate == pytest.approx(0.03)
        assert infl.volatility == pytest.approx(0.01)
        assert infl.mean_reversion_speed == 0.5

    def test_total_market_preset(self) -> None:
        """The total_market preset has the expected params."""
        yaml_str = textwrap.dedent("""\
            scenario:
              name: "Total market test"
              time_horizon: 5 years
              simulations: 100

            assets:
              portfolio:
                type: investment
                initial_balance: 100000
                growth:
                  preset: total_market

            global:
              inflation:
                model: mean_reverting
                long_term_rate: 3% annually
                volatility: 1% annually

            queries:
              - type: percentiles
                values: [50]
                of: portfolio
                at: 5 years
        """)
        model = load_model_from_string(yaml_str)
        inv = model.assets["portfolio"]
        assert isinstance(inv, InvestmentAsset)
        assert isinstance(inv.growth, GrowthConfig)
        assert inv.growth.expected_return == pytest.approx(0.07)
        assert inv.growth.volatility == pytest.approx(0.12)

    def test_unknown_preset_raises(self) -> None:
        """Referencing an unknown preset raises MonetaError listing available ones."""
        yaml_str = textwrap.dedent("""\
            scenario:
              name: "Bad preset"
              time_horizon: 5 years
              simulations: 100

            assets:
              portfolio:
                type: investment
                initial_balance: 100000
                growth:
                  preset: nonexistent_preset

            global:
              inflation:
                model: mean_reverting
                long_term_rate: 3% annually
                volatility: 1% annually

            queries:
              - type: percentiles
                values: [50]
                of: portfolio
                at: 5 years
        """)
        with pytest.raises(MonetaError, match="Unknown preset 'nonexistent_preset'"):
            load_model_from_string(yaml_str)

    def test_unknown_preset_lists_available(self) -> None:
        """The error for unknown presets lists available preset names."""
        with pytest.raises(MonetaError, match="sp500") as exc_info:
            get_preset("does_not_exist")
        error_msg = str(exc_info.value)
        assert "us_inflation" in error_msg
        assert "total_market" in error_msg


# ===================================================================
# Presets module
# ===================================================================


class TestPresetsModule:
    """Tests for the presets module directly."""

    def test_list_presets(self) -> None:
        """list_presets returns all bundled preset names."""
        presets = list_presets()
        assert "sp500" in presets
        assert "us_inflation" in presets
        assert "us_treasuries" in presets
        assert "tech_startup_equity" in presets
        assert "total_market" in presets
        assert len(presets) == 5

    def test_get_sp500_preset(self) -> None:
        data = get_preset("sp500")
        assert data["model"] == "gbm"
        assert data["expected_return"] == "7% annually"
        assert data["volatility"] == "15% annually"

    def test_get_us_inflation_preset(self) -> None:
        data = get_preset("us_inflation")
        assert data["model"] == "mean_reverting"
        assert data["long_term_rate"] == "3% annually"
        assert data["volatility"] == "1% annually"
        assert data["mean_reversion_speed"] == 0.5

    def test_get_us_treasuries_preset(self) -> None:
        data = get_preset("us_treasuries")
        assert data["model"] == "mean_reverting"
        assert data["long_term_rate"] == "4% annually"
        assert data["volatility"] == "3% annually"
        assert data["mean_reversion_speed"] == 0.3

    def test_get_tech_startup_equity_preset(self) -> None:
        data = get_preset("tech_startup_equity")
        assert data["model"] == "gbm"
        # Should have reasonable return and volatility
        assert "expected_return" in data
        assert "volatility" in data

    def test_get_total_market_preset(self) -> None:
        data = get_preset("total_market")
        assert data["model"] == "gbm"
        assert data["expected_return"] == "7% annually"
        assert data["volatility"] == "12% annually"

    def test_unknown_preset_raises_moneta_error(self) -> None:
        with pytest.raises(MonetaError, match="Unknown preset"):
            get_preset("nonexistent")


# ===================================================================
# deep_merge
# ===================================================================


class TestDeepMerge:
    """Tests for the deep_merge utility function."""

    def test_simple_override(self) -> None:
        base = {"a": 1, "b": 2}
        overrides = {"b": 99}
        result = deep_merge(base, overrides)
        assert result == {"a": 1, "b": 99}

    def test_non_overridden_fields_preserved(self) -> None:
        base = {"a": 1, "b": 2, "c": 3}
        overrides = {"b": 99}
        result = deep_merge(base, overrides)
        assert result["a"] == 1
        assert result["c"] == 3

    def test_nested_override(self) -> None:
        base = {
            "outer": {
                "inner1": "original",
                "inner2": "keep_me",
            }
        }
        overrides = {
            "outer": {
                "inner1": "replaced",
            }
        }
        result = deep_merge(base, overrides)
        assert result["outer"]["inner1"] == "replaced"
        assert result["outer"]["inner2"] == "keep_me"

    def test_deeply_nested_override(self) -> None:
        base = {
            "level1": {
                "level2": {
                    "level3": "old",
                    "keep": "yes",
                }
            }
        }
        overrides = {
            "level1": {
                "level2": {
                    "level3": "new",
                }
            }
        }
        result = deep_merge(base, overrides)
        assert result["level1"]["level2"]["level3"] == "new"
        assert result["level1"]["level2"]["keep"] == "yes"

    def test_list_replaced_entirely(self) -> None:
        """List values in overrides replace the base list, not append."""
        base = {"items": [1, 2, 3]}
        overrides = {"items": [99]}
        result = deep_merge(base, overrides)
        assert result["items"] == [99]

    def test_list_not_appended(self) -> None:
        """Verify lists do not get merged/appended."""
        base = {"events": [{"a": 1}, {"b": 2}]}
        overrides = {"events": [{"c": 3}]}
        result = deep_merge(base, overrides)
        assert len(result["events"]) == 1
        assert result["events"][0] == {"c": 3}

    def test_new_key_added(self) -> None:
        base = {"a": 1}
        overrides = {"b": 2}
        result = deep_merge(base, overrides)
        assert result == {"a": 1, "b": 2}

    def test_scalar_replaces_dict(self) -> None:
        """A scalar override replaces a dict value."""
        base = {"a": {"nested": True}}
        overrides = {"a": "flat"}
        result = deep_merge(base, overrides)
        assert result["a"] == "flat"

    def test_dict_replaces_scalar(self) -> None:
        """A dict override replaces a scalar value."""
        base = {"a": "flat"}
        overrides = {"a": {"nested": True}}
        result = deep_merge(base, overrides)
        assert result["a"] == {"nested": True}

    def test_does_not_mutate_base(self) -> None:
        """deep_merge should not modify the original base dict."""
        base = {"a": {"b": 1}}
        overrides = {"a": {"b": 2}}
        deep_merge(base, overrides)
        assert base["a"]["b"] == 1

    def test_does_not_mutate_overrides(self) -> None:
        """deep_merge should not modify the overrides dict."""
        base = {"a": 1}
        overrides = {"a": {"nested": [1, 2]}}
        result = deep_merge(base, overrides)
        # Mutating result should not affect overrides
        result["a"]["nested"].append(3)
        assert overrides["a"]["nested"] == [1, 2]

    def test_empty_overrides(self) -> None:
        base = {"a": 1, "b": 2}
        result = deep_merge(base, {})
        assert result == {"a": 1, "b": 2}

    def test_empty_base(self) -> None:
        overrides = {"a": 1, "b": 2}
        result = deep_merge({}, overrides)
        assert result == {"a": 1, "b": 2}

    def test_realistic_sweep_override(self) -> None:
        """Simulate a realistic sweep override for equity valuation ranges."""
        base = {
            "scenario": {"name": "base", "time_horizon": "10 years"},
            "assets": {
                "startup_equity": {
                    "type": "illiquid_equity",
                    "current_valuation": 500000,
                    "liquidity_events": [
                        {
                            "probability": "20% within 3 years",
                            "valuation_range": ["2x", "5x"],
                        }
                    ],
                }
            },
        }
        overrides = {
            "assets": {
                "startup_equity": {
                    "liquidity_events": [
                        {
                            "probability": "20% within 3 years",
                            "valuation_range": ["1x", "3x"],
                        }
                    ],
                }
            }
        }
        result = deep_merge(base, overrides)
        # Scenario should be preserved
        assert result["scenario"]["name"] == "base"
        # Asset type and valuation should be preserved
        assert result["assets"]["startup_equity"]["type"] == "illiquid_equity"
        assert result["assets"]["startup_equity"]["current_valuation"] == 500000
        # Events should be replaced (list replacement)
        events = result["assets"]["startup_equity"]["liquidity_events"]
        assert len(events) == 1
        assert events[0]["valuation_range"] == ["1x", "3x"]

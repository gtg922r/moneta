"""Tests for cash flow types, models, and scenario-level validation."""

from __future__ import annotations

import copy
import textwrap

import pytest
from pydantic import BaseModel, ValidationError

from moneta.parser.loader import load_model_from_string
from moneta.parser.models import CashFlowConfig, ScenarioModel
from moneta.parser.types import (
    CashFlowAmount,
    CashFlowAmountValue,
)


# ===================================================================
# Helper model — wrap the type so we can test via Pydantic parsing
# ===================================================================


class CashFlowAmountModel(BaseModel):
    amt: CashFlowAmount


# ===================================================================
# CashFlowAmount type tests
# ===================================================================


class TestCashFlowAmount:
    """Tests for CashFlowAmount parsing."""

    def test_monthly_positive(self) -> None:
        m = CashFlowAmountModel(amt="$5,000 monthly")
        assert m.amt.amount == pytest.approx(5000.0)
        assert m.amt.frequency == "monthly"

    def test_monthly_negative(self) -> None:
        m = CashFlowAmountModel(amt="-$5,000 monthly")
        assert m.amt.amount == pytest.approx(-5000.0)
        assert m.amt.frequency == "monthly"

    def test_annually_positive(self) -> None:
        m = CashFlowAmountModel(amt="$50,000 annually")
        assert m.amt.amount == pytest.approx(50000.0)
        assert m.amt.frequency == "annually"

    def test_one_time_positive(self) -> None:
        m = CashFlowAmountModel(amt="$100,000")
        assert m.amt.amount == pytest.approx(100000.0)
        assert m.amt.frequency is None

    def test_one_time_negative(self) -> None:
        m = CashFlowAmountModel(amt="-$100,000")
        assert m.amt.amount == pytest.approx(-100000.0)
        assert m.amt.frequency is None

    def test_numeric_positive(self) -> None:
        m = CashFlowAmountModel(amt=5000)
        assert m.amt.amount == pytest.approx(5000.0)
        assert m.amt.frequency is None

    def test_numeric_negative(self) -> None:
        m = CashFlowAmountModel(amt=-5000)
        assert m.amt.amount == pytest.approx(-5000.0)
        assert m.amt.frequency is None

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValidationError):
            CashFlowAmountModel(amt="not a dollar amount")

    def test_passthrough(self) -> None:
        val = CashFlowAmountValue(amount=1234.0, frequency="monthly")
        m = CashFlowAmountModel(amt=val)
        assert m.amt.amount == pytest.approx(1234.0)
        assert m.amt.frequency == "monthly"

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            CashFlowAmountModel(amt=[100])


# ===================================================================
# CashFlowConfig model tests
# ===================================================================


class TestCashFlowConfig:
    """Tests for CashFlowConfig model validation."""

    def test_valid_recurring(self) -> None:
        cfg = CashFlowConfig(
            amount="-$5,000 monthly",
            asset="portfolio",
            start="0 months",
            end="30 years",
        )
        assert cfg.amount.amount == pytest.approx(-5000.0)
        assert cfg.amount.frequency == "monthly"
        assert cfg.start == 0
        assert cfg.end == 360
        assert cfg.at is None

    def test_valid_one_time(self) -> None:
        cfg = CashFlowConfig(
            amount="$100,000",
            asset="portfolio",
            at="5 years",
        )
        assert cfg.amount.amount == pytest.approx(100000.0)
        assert cfg.amount.frequency is None
        assert cfg.at == 60
        assert cfg.start is None
        assert cfg.end is None

    def test_both_at_and_start_raises(self) -> None:
        with pytest.raises(ValidationError, match="both 'at' and 'start'/'end'"):
            CashFlowConfig(
                amount="$100,000",
                asset="portfolio",
                at="5 years",
                start="0 months",
            )

    def test_recurring_with_at_raises(self) -> None:
        with pytest.raises(ValidationError, match="should use 'start'/'end', not 'at'"):
            CashFlowConfig(
                amount="$5,000 monthly",
                asset="portfolio",
                at="5 years",
            )

    def test_one_time_without_at_raises(self) -> None:
        with pytest.raises(ValidationError, match="must specify 'at'"):
            CashFlowConfig(
                amount="$100,000",
                asset="portfolio",
            )

    def test_adjust_for_inflation(self) -> None:
        cfg = CashFlowConfig(
            amount="-$5,000 monthly",
            asset="portfolio",
            start="0 months",
            end="30 years",
            adjust_for="inflation",
        )
        assert cfg.adjust_for == "inflation"

    def test_allow_negative(self) -> None:
        cfg = CashFlowConfig(
            amount="-$5,000 monthly",
            asset="portfolio",
            start="0 months",
            end="30 years",
            allow_negative=True,
        )
        assert cfg.allow_negative is True

    def test_recurring_with_only_start(self) -> None:
        """Recurring flow with only start (end defaults to None = entire horizon)."""
        cfg = CashFlowConfig(
            amount="$3,000 monthly",
            asset="portfolio",
            start="0 months",
        )
        assert cfg.start == 0
        assert cfg.end is None


# ===================================================================
# ScenarioModel with cash_flows tests
# ===================================================================


def _base_model_dict_with_cash_flows() -> dict:
    """Return a valid model dict including cash_flows."""
    return {
        "scenario": {
            "name": "Cash flow test",
            "time_horizon": "30 years",
            "simulations": 1000,
        },
        "assets": {
            "portfolio": {
                "type": "investment",
                "initial_balance": "$500,000",
                "growth": {
                    "model": "gbm",
                    "expected_return": "7% annually",
                    "volatility": "15% annually",
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
                "expression": "portfolio > 0",
                "at": "30 years",
                "label": "Still solvent at year 30",
            },
        ],
        "cash_flows": {
            "retirement_spending": {
                "amount": "-$5,000 monthly",
                "asset": "portfolio",
                "start": "0 months",
                "end": "30 years",
                "adjust_for": "inflation",
            },
            "salary_contribution": {
                "amount": "$3,000 monthly",
                "asset": "portfolio",
                "start": "0 months",
                "end": "20 years",
            },
        },
    }


class TestScenarioModelWithCashFlows:
    """Tests for ScenarioModel with cash_flows section."""

    def test_valid_cash_flows(self) -> None:
        data = _base_model_dict_with_cash_flows()
        model = ScenarioModel.model_validate(data)
        assert model.cash_flows is not None
        assert "retirement_spending" in model.cash_flows
        assert "salary_contribution" in model.cash_flows
        cf = model.cash_flows["retirement_spending"]
        assert cf.amount.amount == pytest.approx(-5000.0)
        assert cf.amount.frequency == "monthly"
        assert cf.asset == "portfolio"
        assert cf.adjust_for == "inflation"

    def test_cash_flows_optional(self) -> None:
        data = _base_model_dict_with_cash_flows()
        del data["cash_flows"]
        model = ScenarioModel.model_validate(data)
        assert model.cash_flows is None

    def test_bad_asset_ref_raises(self) -> None:
        data = _base_model_dict_with_cash_flows()
        data["cash_flows"]["retirement_spending"]["asset"] = "nonexistent"
        with pytest.raises(ValidationError, match="no asset with that name"):
            ScenarioModel.model_validate(data)

    def test_at_exceeds_horizon_raises(self) -> None:
        data = _base_model_dict_with_cash_flows()
        data["cash_flows"] = {
            "big_expense": {
                "amount": "$100,000",
                "asset": "portfolio",
                "at": "35 years",  # horizon is 30 years
            },
        }
        with pytest.raises(ValidationError, match="time_horizon"):
            ScenarioModel.model_validate(data)

    def test_start_exceeds_horizon_raises(self) -> None:
        data = _base_model_dict_with_cash_flows()
        data["cash_flows"] = {
            "late_start": {
                "amount": "$1,000 monthly",
                "asset": "portfolio",
                "start": "35 years",
            },
        }
        with pytest.raises(ValidationError, match="time_horizon"):
            ScenarioModel.model_validate(data)

    def test_end_exceeds_horizon_raises(self) -> None:
        data = _base_model_dict_with_cash_flows()
        data["cash_flows"] = {
            "long_flow": {
                "amount": "$1,000 monthly",
                "asset": "portfolio",
                "start": "0 months",
                "end": "35 years",
            },
        }
        with pytest.raises(ValidationError, match="time_horizon"):
            ScenarioModel.model_validate(data)

    def test_start_after_end_raises(self) -> None:
        data = _base_model_dict_with_cash_flows()
        data["cash_flows"] = {
            "backwards": {
                "amount": "$1,000 monthly",
                "asset": "portfolio",
                "start": "20 years",
                "end": "10 years",
            },
        }
        with pytest.raises(ValidationError, match="start must be before end"):
            ScenarioModel.model_validate(data)


# ===================================================================
# End-to-end YAML fixture test
# ===================================================================


class TestCashFlowYAMLFixture:
    """Test loading the cash flow YAML fixture file."""

    def test_load_cash_flow_fixture(self) -> None:
        from pathlib import Path

        from moneta.parser.loader import load_model

        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        model = load_model(fixtures / "cash_flow_model.moneta.yaml")
        assert model.scenario.name == "Cash flow test"
        assert model.cash_flows is not None
        assert len(model.cash_flows) == 3
        assert "retirement_spending" in model.cash_flows
        assert "college_tuition" in model.cash_flows
        assert "salary_contribution" in model.cash_flows

        # Verify parsed amounts
        rs = model.cash_flows["retirement_spending"]
        assert rs.amount.amount == pytest.approx(-5000.0)
        assert rs.amount.frequency == "monthly"
        assert rs.adjust_for == "inflation"

        ct = model.cash_flows["college_tuition"]
        assert ct.amount.amount == pytest.approx(-50000.0)
        assert ct.amount.frequency == "annually"
        assert ct.start == 216  # 18 years
        assert ct.end == 264  # 22 years

        sc = model.cash_flows["salary_contribution"]
        assert sc.amount.amount == pytest.approx(3000.0)
        assert sc.amount.frequency == "monthly"

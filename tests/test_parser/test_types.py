"""Tests for Pydantic custom annotated types."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from moneta.parser.types import (
    AnnualRate,
    CurrencyAmount,
    Duration,
    MultiplierRange,
    ProbabilityWindow,
    ProbabilityWindowValue,
)

# ===================================================================
# Helper models — wrap each type so we can test via Pydantic parsing
# ===================================================================


class RateModel(BaseModel):
    rate: AnnualRate


class DurationModel(BaseModel):
    dur: Duration


class ProbWinModel(BaseModel):
    pw: ProbabilityWindow


class MultRangeModel(BaseModel):
    mr: MultiplierRange


class CurrModel(BaseModel):
    amt: CurrencyAmount


# ===================================================================
# AnnualRate tests
# ===================================================================


class TestAnnualRate:
    """Tests for AnnualRate parsing."""

    def test_annually_integer(self) -> None:
        m = RateModel(rate="7% annually")
        assert m.rate == pytest.approx(0.07)

    def test_annually_decimal(self) -> None:
        m = RateModel(rate="7.5% annually")
        assert m.rate == pytest.approx(0.075)

    def test_monthly(self) -> None:
        m = RateModel(rate="0.5% monthly")
        assert m.rate == pytest.approx(0.06)  # 0.005 * 12

    def test_monthly_3_percent(self) -> None:
        m = RateModel(rate="3% monthly")
        assert m.rate == pytest.approx(0.36)  # 0.03 * 12

    def test_zero_percent(self) -> None:
        m = RateModel(rate="0% annually")
        assert m.rate == pytest.approx(0.0)

    def test_negative_rate(self) -> None:
        m = RateModel(rate="-2% annually")
        assert m.rate == pytest.approx(-0.02)

    def test_float_passthrough(self) -> None:
        m = RateModel(rate=0.07)
        assert m.rate == pytest.approx(0.07)

    def test_int_passthrough(self) -> None:
        m = RateModel(rate=0)
        assert m.rate == pytest.approx(0.0)

    def test_large_rate(self) -> None:
        m = RateModel(rate="100% annually")
        assert m.rate == pytest.approx(1.0)

    def test_missing_period_raises(self) -> None:
        with pytest.raises(ValidationError):
            RateModel(rate="7%")

    def test_wrong_period_raises(self) -> None:
        with pytest.raises(ValidationError):
            RateModel(rate="7% weekly")

    def test_typo_raises(self) -> None:
        with pytest.raises(ValidationError):
            RateModel(rate="7% anually")

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(ValidationError):
            RateModel(rate="abc% annually")

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            RateModel(rate=[1, 2])


# ===================================================================
# Duration tests
# ===================================================================


class TestDuration:
    """Tests for Duration parsing."""

    def test_years(self) -> None:
        m = DurationModel(dur="10 years")
        assert m.dur == 120

    def test_single_year(self) -> None:
        m = DurationModel(dur="1 year")
        assert m.dur == 12

    def test_months(self) -> None:
        m = DurationModel(dur="6 months")
        assert m.dur == 6

    def test_single_month(self) -> None:
        m = DurationModel(dur="1 month")
        assert m.dur == 1

    def test_int_passthrough(self) -> None:
        m = DurationModel(dur=360)
        assert m.dur == 360

    def test_float_whole_passthrough(self) -> None:
        m = DurationModel(dur=120.0)
        assert m.dur == 120

    def test_thirty_years(self) -> None:
        m = DurationModel(dur="30 years")
        assert m.dur == 360

    def test_missing_unit_raises(self) -> None:
        with pytest.raises(ValidationError):
            DurationModel(dur="10")

    def test_wrong_unit_raises(self) -> None:
        with pytest.raises(ValidationError):
            DurationModel(dur="10 days")

    def test_negative_raises(self) -> None:
        """Negative durations are not supported (no negative number in pattern)."""
        with pytest.raises(ValidationError):
            DurationModel(dur="-5 years")

    def test_float_fractional_raises(self) -> None:
        with pytest.raises(ValidationError):
            DurationModel(dur=10.5)

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(ValidationError):
            DurationModel(dur="abc years")

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            DurationModel(dur=[10])


# ===================================================================
# ProbabilityWindow tests
# ===================================================================


class TestProbabilityWindow:
    """Tests for ProbabilityWindow parsing."""

    def test_within_n_years(self) -> None:
        m = ProbWinModel(pw="20% within 3 years")
        assert m.pw.probability == pytest.approx(0.20)
        assert m.pw.start_month == 0
        assert m.pw.end_month == 36

    def test_within_range_years(self) -> None:
        m = ProbWinModel(pw="60% within 5-6 years")
        assert m.pw.probability == pytest.approx(0.60)
        assert m.pw.start_month == 60
        assert m.pw.end_month == 72

    def test_within_n_months(self) -> None:
        m = ProbWinModel(pw="10% within 18 months")
        assert m.pw.probability == pytest.approx(0.10)
        assert m.pw.start_month == 0
        assert m.pw.end_month == 18

    def test_within_range_months(self) -> None:
        m = ProbWinModel(pw="50% within 12-24 months")
        assert m.pw.probability == pytest.approx(0.50)
        assert m.pw.start_month == 12
        assert m.pw.end_month == 24

    def test_100_percent(self) -> None:
        m = ProbWinModel(pw="100% within 1 year")
        assert m.pw.probability == pytest.approx(1.0)
        assert m.pw.end_month == 12

    def test_zero_percent(self) -> None:
        m = ProbWinModel(pw="0% within 5 years")
        assert m.pw.probability == pytest.approx(0.0)

    def test_passthrough(self) -> None:
        pw = ProbabilityWindowValue(probability=0.2, start_month=0, end_month=36)
        m = ProbWinModel(pw=pw)
        assert m.pw.probability == pytest.approx(0.2)

    def test_negative_probability_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProbWinModel(pw="-10% within 3 years")

    def test_probability_over_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProbWinModel(pw="150% within 3 years")

    def test_missing_within_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProbWinModel(pw="20% 3 years")

    def test_missing_unit_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProbWinModel(pw="20% within 3")

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProbWinModel(pw=42)

    def test_non_string_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProbWinModel(pw=[20, 3])


# ===================================================================
# MultiplierRange tests
# ===================================================================


class TestMultiplierRange:
    """Tests for MultiplierRange parsing."""

    def test_x_notation(self) -> None:
        m = MultRangeModel(mr=["2x", "5x"])
        assert m.mr == (2.0, 5.0)

    def test_decimal_x_notation(self) -> None:
        m = MultRangeModel(mr=["1.5x", "3.5x"])
        assert m.mr == (1.5, 3.5)

    def test_numeric_list(self) -> None:
        m = MultRangeModel(mr=[2.0, 5.0])
        assert m.mr == (2.0, 5.0)

    def test_int_list(self) -> None:
        m = MultRangeModel(mr=[2, 5])
        assert m.mr == (2.0, 5.0)

    def test_mixed_types(self) -> None:
        m = MultRangeModel(mr=["2x", 5.0])
        assert m.mr == (2.0, 5.0)

    def test_same_values(self) -> None:
        m = MultRangeModel(mr=["3x", "3x"])
        assert m.mr == (3.0, 3.0)

    def test_uppercase_x(self) -> None:
        m = MultRangeModel(mr=["2X", "5X"])
        assert m.mr == (2.0, 5.0)

    def test_tuple_passthrough(self) -> None:
        m = MultRangeModel(mr=(2.0, 5.0))
        assert m.mr == (2.0, 5.0)

    def test_inverted_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            MultRangeModel(mr=["5x", "2x"])

    def test_single_value_raises(self) -> None:
        with pytest.raises(ValidationError):
            MultRangeModel(mr=["2x"])

    def test_three_values_raises(self) -> None:
        with pytest.raises(ValidationError):
            MultRangeModel(mr=["1x", "2x", "3x"])

    def test_non_list_raises(self) -> None:
        with pytest.raises(ValidationError):
            MultRangeModel(mr="2x")

    def test_invalid_multiplier_raises(self) -> None:
        with pytest.raises(ValidationError):
            MultRangeModel(mr=["abc", "5x"])


# ===================================================================
# CurrencyAmount tests
# ===================================================================


class TestCurrencyAmount:
    """Tests for CurrencyAmount parsing."""

    def test_dollar_with_commas(self) -> None:
        m = CurrModel(amt="$850,000")
        assert m.amt == pytest.approx(850000.0)

    def test_dollar_no_commas(self) -> None:
        m = CurrModel(amt="$500000")
        assert m.amt == pytest.approx(500000.0)

    def test_dollar_decimal(self) -> None:
        m = CurrModel(amt="$1,234,567.89")
        assert m.amt == pytest.approx(1234567.89)

    def test_no_dollar_sign(self) -> None:
        m = CurrModel(amt="850000")
        assert m.amt == pytest.approx(850000.0)

    def test_int_passthrough(self) -> None:
        m = CurrModel(amt=850000)
        assert m.amt == pytest.approx(850000.0)

    def test_float_passthrough(self) -> None:
        m = CurrModel(amt=850000.50)
        assert m.amt == pytest.approx(850000.50)

    def test_zero(self) -> None:
        m = CurrModel(amt=0)
        assert m.amt == pytest.approx(0.0)

    def test_small_amount(self) -> None:
        m = CurrModel(amt="$100")
        assert m.amt == pytest.approx(100.0)

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValidationError):
            CurrModel(amt="")

    def test_dollar_only_raises(self) -> None:
        with pytest.raises(ValidationError):
            CurrModel(amt="$")

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(ValidationError):
            CurrModel(amt="abc")

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            CurrModel(amt=[100])

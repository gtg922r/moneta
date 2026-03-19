"""Pydantic custom annotated types for parsing YAML model values.

Each type uses Annotated[T, BeforeValidator(parse_fn)] to parse
human-readable strings like "7% annually" into numeric values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BeforeValidator

# ---------------------------------------------------------------------------
# AnnualRate
# ---------------------------------------------------------------------------
# Parses: "7% annually" → 0.07, "0.5% monthly" → 0.06, 0.07 → 0.07


def _parse_annual_rate(value: Any) -> float:
    """Parse an annual rate from a string or numeric value.

    Strings like "7% annually" → 0.07.
    Strings like "0.5% monthly" → 0.06 (monthly_rate * 12).
    Numeric passthrough: 0.07 → 0.07.
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise ValueError(
            f"AnnualRate expects a string or number, got {type(value).__name__}"
        )

    value = value.strip()

    # Pattern: "<number>% <period>"
    match = re.match(
        r"^(-?\d+(?:\.\d+)?)\s*%\s*(annually|monthly)$",
        value,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            f"Cannot parse AnnualRate from '{value}'. "
            "Expected format: '<number>% annually' or '<number>% monthly', "
            "e.g. '7% annually' or '0.5% monthly'"
        )

    rate_pct = float(match.group(1))
    period = match.group(2).lower()

    if period == "annually":
        return rate_pct / 100.0
    else:  # monthly
        return rate_pct / 100.0 * 12


AnnualRate = Annotated[float, BeforeValidator(_parse_annual_rate)]


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------
# Parses: "10 years" → 120, "6 months" → 6, 360 → 360


def _parse_duration(value: Any) -> int:
    """Parse a duration into months.

    "10 years" → 120, "6 months" → 6, 360 → 360 (passthrough).
    """
    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value == int(value):
            return int(value)
        raise ValueError(f"Duration must be a whole number of months, got {value}")

    if not isinstance(value, str):
        raise ValueError(
            f"Duration expects a string or int, got {type(value).__name__}"
        )

    value = value.strip()

    match = re.match(
        r"^(\d+(?:\.\d+)?)\s+(years?|months?)$",
        value,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            f"Cannot parse Duration from '{value}'. "
            "Expected format: '<number> years' or '<number> months', "
            "e.g. '10 years' or '6 months'"
        )

    amount = float(match.group(1))
    unit = match.group(2).lower()

    months = int(amount * 12) if unit.startswith("year") else int(amount)

    if months < 0:
        raise ValueError(f"Duration must be non-negative, got {months} months")

    return months


Duration = Annotated[int, BeforeValidator(_parse_duration)]


# ---------------------------------------------------------------------------
# ProbabilityWindow
# ---------------------------------------------------------------------------
# Parses: "20% within 3 years" → ProbWin(0.20, 0, 36)
#         "60% within 5-6 years" → ProbWin(0.60, 60, 72)


@dataclass(frozen=True)
class ProbabilityWindowValue:
    """Parsed probability window: probability within a time range."""

    probability: float  # 0.0 to 1.0
    start_month: int
    end_month: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                "Probability must be between 0% and 100%, "
                f"got {self.probability * 100}%"
            )
        if self.start_month < 0:
            raise ValueError(
                f"Start month must be non-negative, got {self.start_month}"
            )
        if self.end_month <= self.start_month:
            raise ValueError(
                f"End month ({self.end_month}) must be greater than "
                f"start month ({self.start_month})"
            )


def _parse_probability_window(value: Any) -> ProbabilityWindowValue:
    """Parse a probability window from a string.

    "20% within 3 years" → ProbabilityWindowValue(0.20, 0, 36)
    "60% within 5-6 years" → ProbabilityWindowValue(0.60, 60, 72)
    """
    if isinstance(value, ProbabilityWindowValue):
        return value

    # Handle dict from model_dump() during sweep re-validation
    if isinstance(value, dict):
        return ProbabilityWindowValue(
            probability=value["probability"],
            start_month=value["start_month"],
            end_month=value["end_month"],
        )

    if not isinstance(value, str):
        raise ValueError(
            f"ProbabilityWindow expects a string, got {type(value).__name__}"
        )

    value = value.strip()

    # Pattern: "<pct>% within <start>-<end> years" or "<pct>% within <end> years"
    # Also handle months
    match = re.match(
        r"^(-?\d+(?:\.\d+)?)\s*%\s+within\s+"
        r"(\d+(?:\.\d+)?)"
        r"(?:\s*-\s*(\d+(?:\.\d+)?))?"
        r"\s+(years?|months?)$",
        value,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            f"Cannot parse ProbabilityWindow from '{value}'. "
            "Expected format: '<pct>% within <N> years' or "
            "'<pct>% within <start>-<end> years', "
            "e.g. '20% within 3 years' or '60% within 5-6 years'"
        )

    probability = float(match.group(1)) / 100.0
    first_num = float(match.group(2))
    second_num = match.group(3)
    unit = match.group(4).lower()

    if unit.startswith("year"):
        if second_num is not None:
            # Range: "5-6 years" → start=60, end=72
            start_month = int(first_num * 12)
            end_month = int(float(second_num) * 12)
        else:
            # Single: "3 years" → start=0, end=36
            start_month = 0
            end_month = int(first_num * 12)
    else:
        # months
        if second_num is not None:
            start_month = int(first_num)
            end_month = int(float(second_num))
        else:
            start_month = 0
            end_month = int(first_num)

    return ProbabilityWindowValue(
        probability=probability,
        start_month=start_month,
        end_month=end_month,
    )


ProbabilityWindow = Annotated[
    ProbabilityWindowValue, BeforeValidator(_parse_probability_window)
]


# ---------------------------------------------------------------------------
# MultiplierRange
# ---------------------------------------------------------------------------
# Parses: ["2x", "5x"] → (2.0, 5.0), [2.0, 5.0] → (2.0, 5.0)


def _parse_multiplier_range(value: Any) -> tuple[float, float]:
    """Parse a multiplier range from a list.

    ["2x", "5x"] → (2.0, 5.0)
    [2.0, 5.0] → (2.0, 5.0)
    """
    if isinstance(value, tuple) and len(value) == 2:
        return (float(value[0]), float(value[1]))

    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"MultiplierRange expects a list of two values, got {type(value).__name__}"
        )

    if len(value) != 2:
        raise ValueError(f"MultiplierRange expects exactly 2 values, got {len(value)}")

    def _parse_multiplier(v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            v = v.strip()
            # Handle "2x" or "5x" format
            match = re.match(r"^(-?\d+(?:\.\d+)?)\s*[xX]$", v)
            if match:
                return float(match.group(1))
            # Try plain number
            try:
                return float(v)
            except ValueError as err:
                raise ValueError(
                    f"Cannot parse multiplier from '{v}'. "
                    "Expected format: '<number>x' or a number, e.g. '2x'"
                ) from err
        raise ValueError(
            f"Multiplier expects a string or number, got {type(v).__name__}"
        )

    low = _parse_multiplier(value[0])
    high = _parse_multiplier(value[1])

    if low > high:
        raise ValueError(f"MultiplierRange low ({low}) must not exceed high ({high})")

    return (low, high)


MultiplierRange = Annotated[
    tuple[float, float], BeforeValidator(_parse_multiplier_range)
]


# ---------------------------------------------------------------------------
# CurrencyAmount
# ---------------------------------------------------------------------------
# Parses: "$850,000" → 850000, 850000 → 850000


def _parse_currency_amount(value: Any) -> float:
    """Parse a currency amount from a string or number.

    "$850,000" → 850000.0, 850000 → 850000.0
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise ValueError(
            f"CurrencyAmount expects a string or number, got {type(value).__name__}"
        )

    value = value.strip()

    # Strip leading $ sign
    if value.startswith("$"):
        value = value[1:]

    # Remove commas
    value = value.replace(",", "")

    # Remove whitespace
    value = value.strip()

    if not value:
        raise ValueError("CurrencyAmount cannot be empty")

    try:
        result = float(value)
    except ValueError as err:
        raise ValueError(
            f"Cannot parse CurrencyAmount from '{value}'. "
            "Expected format: '$850,000' or a number"
        ) from err

    return result


CurrencyAmount = Annotated[float, BeforeValidator(_parse_currency_amount)]


# ---------------------------------------------------------------------------
# CashFlowAmount
# ---------------------------------------------------------------------------
# Parses: "$5,000 monthly" → CashFlowAmountValue(5000.0, "monthly")
#         "-$5,000 monthly" → CashFlowAmountValue(-5000.0, "monthly")
#         "$100,000" → CashFlowAmountValue(100000.0, None)
#         5000 → CashFlowAmountValue(5000.0, None)


@dataclass(frozen=True)
class CashFlowAmountValue:
    """Parsed cash flow amount with optional frequency."""

    amount: float  # positive = deposit, negative = withdrawal
    frequency: str | None  # "monthly", "annually", or None (one-time)


def _parse_cash_flow_amount(value: Any) -> CashFlowAmountValue:
    """Parse a cash flow amount from a string, number, or CashFlowAmountValue.

    "$5,000 monthly" → CashFlowAmountValue(5000.0, "monthly")
    "-$5,000 monthly" → CashFlowAmountValue(-5000.0, "monthly")
    "$50,000 annually" → CashFlowAmountValue(50000.0, "annually")
    "$100,000" → CashFlowAmountValue(100000.0, None)
    "-$100,000" → CashFlowAmountValue(-100000.0, None)
    5000 → CashFlowAmountValue(5000.0, None)
    """
    if isinstance(value, CashFlowAmountValue):
        return value

    # Handle dict from model_dump() during sweep re-validation
    if isinstance(value, dict):
        return CashFlowAmountValue(
            amount=value["amount"],
            frequency=value.get("frequency"),
        )

    if isinstance(value, (int, float)):
        return CashFlowAmountValue(amount=float(value), frequency=None)

    if not isinstance(value, str):
        raise ValueError(
            f"CashFlowAmount expects a string or number, got {type(value).__name__}"
        )

    raw = value.strip()

    # Detect leading sign
    sign = 1.0
    s = raw
    if s.startswith("-"):
        sign = -1.0
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]

    # Check for trailing frequency
    frequency: str | None = None
    freq_match = re.match(r"^(.+)\s+(monthly|annually)$", s, re.IGNORECASE)
    if freq_match:
        s = freq_match.group(1).strip()
        frequency = freq_match.group(2).lower()

    # Parse the currency portion (reuse _parse_currency_amount logic)
    s = s.strip()
    if s.startswith("$"):
        s = s[1:]
    s = s.replace(",", "").strip()

    if not s:
        raise ValueError(f"Cannot parse CashFlowAmount from '{raw}'")

    try:
        amount = float(s)
    except ValueError as err:
        raise ValueError(
            f"Cannot parse CashFlowAmount from '{raw}'. "
            "Expected format: '[-]$<amount> [monthly|annually]', "
            "e.g. '$5,000 monthly' or '-$100,000'"
        ) from err

    return CashFlowAmountValue(amount=sign * amount, frequency=frequency)


CashFlowAmount = Annotated[
    CashFlowAmountValue, BeforeValidator(_parse_cash_flow_amount)
]

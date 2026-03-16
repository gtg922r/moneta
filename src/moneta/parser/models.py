"""Pydantic model classes for Moneta scenario files.

Defines the complete model hierarchy for .moneta.yaml files including:
- ScenarioConfig, asset types (discriminated union), global config
- Query types (discriminated union)
- Sweep configuration
- Cross-reference validation
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from moneta.parser.types import (
    AnnualRate,
    CashFlowAmount,
    CurrencyAmount,
    Duration,
    MultiplierRange,
    ProbabilityWindow,
)


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------


class ScenarioConfig(BaseModel):
    """Top-level scenario settings."""

    name: str
    time_horizon: Duration
    simulations: int = 10000
    time_step: Literal["monthly"] = "monthly"
    seed: int | None = None


# ---------------------------------------------------------------------------
# Growth / Inflation configs and preset references
# ---------------------------------------------------------------------------


class GrowthConfig(BaseModel):
    """Geometric Brownian Motion growth configuration."""

    model: Literal["gbm"]
    expected_return: AnnualRate
    volatility: AnnualRate


class PresetRef(BaseModel):
    """Reference to a built-in preset by name."""

    preset: str


class InflationConfig(BaseModel):
    """Mean-reverting (Ornstein-Uhlenbeck) inflation configuration."""

    model: Literal["mean_reverting"]
    long_term_rate: AnnualRate
    volatility: AnnualRate
    mean_reversion_speed: float = 0.5


# ---------------------------------------------------------------------------
# Liquidity events and transfers
# ---------------------------------------------------------------------------


class LiquidityEvent(BaseModel):
    """A discrete probabilistic event that may trigger asset liquidation."""

    probability: ProbabilityWindow
    valuation_range: MultiplierRange


class TransferConfig(BaseModel):
    """Configuration for transferring an asset's value on liquidation."""

    transfer_to: str


# ---------------------------------------------------------------------------
# Asset types (discriminated union)
# ---------------------------------------------------------------------------


class InvestmentAsset(BaseModel):
    """A liquid investment asset with stochastic growth."""

    type: Literal["investment"]
    initial_balance: CurrencyAmount
    growth: GrowthConfig | PresetRef


class IlliquidEquityAsset(BaseModel):
    """An illiquid equity position with probabilistic liquidity events."""

    type: Literal["illiquid_equity"]
    current_valuation: CurrencyAmount
    shares: int | None = None
    liquidity_events: list[LiquidityEvent]
    on_liquidation: TransferConfig


Asset = Annotated[
    InvestmentAsset | IlliquidEquityAsset,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------


class GlobalConfig(BaseModel):
    """Global settings (inflation, etc.)."""

    inflation: InflationConfig | PresetRef


# ---------------------------------------------------------------------------
# Query types (discriminated union)
# ---------------------------------------------------------------------------


class ProbabilityQuery(BaseModel):
    """Query: what fraction of runs satisfy an expression at a given time?"""

    type: Literal["probability"]
    expression: str
    at: Duration
    label: str | None = None
    adjust_for: Literal["inflation"] | None = None


class PercentilesQuery(BaseModel):
    """Query: percentile values of an asset at given time points."""

    type: Literal["percentiles"]
    values: list[int]
    of: str
    at: list[Duration] | Duration
    label: str | None = None
    adjust_for: Literal["inflation"] | None = None


class ExpectedQuery(BaseModel):
    """Query: expected value (mean, median, std) of an expression at a time."""

    type: Literal["expected"]
    of: str
    at: Duration
    label: str | None = None
    adjust_for: Literal["inflation"] | None = None


class DistributionQuery(BaseModel):
    """Query: full histogram of an asset at a given time."""

    type: Literal["distribution"]
    of: str
    at: Duration
    bins: int = 50
    label: str | None = None
    adjust_for: Literal["inflation"] | None = None


Query = Annotated[
    ProbabilityQuery | PercentilesQuery | ExpectedQuery | DistributionQuery,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------


class SweepScenario(BaseModel):
    """A named scenario variant with overrides."""

    label: str
    overrides: dict[str, Any]


class SweepConfig(BaseModel):
    """Sweep mode: run multiple named scenario variants."""

    scenarios: list[SweepScenario]


# ---------------------------------------------------------------------------
# Cash flow configuration
# ---------------------------------------------------------------------------


class CashFlowConfig(BaseModel):
    """A scheduled cash flow (income, withdrawal, or one-time expense).

    Positive amounts are deposits, negative are withdrawals.
    One-time flows use 'at', recurring flows use 'start'/'end'.
    """

    amount: CashFlowAmount
    asset: str  # which asset this flows in/out of

    # For recurring flows
    start: Duration | None = None
    end: Duration | None = None

    # For one-time flows
    at: Duration | None = None

    # Optional inflation adjustment
    adjust_for: Literal["inflation"] | None = None

    # Balance behavior: clamp to zero or allow negative
    allow_negative: bool = False

    @model_validator(mode="after")
    def _validate_timing(self) -> CashFlowConfig:
        """Validate that either at (one-time) or start/end (recurring) is specified."""
        has_at = self.at is not None
        has_schedule = self.start is not None or self.end is not None
        amount_val = self.amount  # CashFlowAmountValue

        if has_at and has_schedule:
            raise ValueError(
                "Cash flow cannot have both 'at' and 'start'/'end' "
                "— use 'at' for one-time, 'start'/'end' for recurring"
            )

        # One-time: must have 'at' OR amount has no frequency
        if amount_val.frequency is None and not has_at and not has_schedule:
            raise ValueError("One-time cash flow (no frequency) must specify 'at'")

        # Recurring: must have frequency in amount
        if amount_val.frequency is not None and has_at:
            raise ValueError(
                f"Recurring cash flow ('{amount_val.frequency}') "
                "should use 'start'/'end', not 'at'"
            )

        return self


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class ScenarioModel(BaseModel):
    """Top-level model for a .moneta.yaml file.

    Contains the full scenario definition including assets, global config,
    queries, and optional sweep configuration.
    """

    model_config = ConfigDict(populate_by_name=True)

    scenario: ScenarioConfig
    assets: dict[str, Asset]
    global_config: GlobalConfig = Field(alias="global")
    queries: list[Query]
    cash_flows: dict[str, CashFlowConfig] | None = None
    sweep: SweepConfig | None = None

    @model_validator(mode="after")
    def _validate_cross_references(self) -> ScenarioModel:
        """Validate cross-references between assets, queries, and transfers."""
        asset_names = set(self.assets.keys())
        horizon_months = self.scenario.time_horizon

        # Validate transfer_to references
        for asset_name, asset in self.assets.items():
            if isinstance(asset, IlliquidEquityAsset):
                target = asset.on_liquidation.transfer_to
                if target not in asset_names:
                    raise ValueError(
                        f"transfer_to: '{target}' — no asset named '{target}'. "
                        f"Available: {', '.join(sorted(asset_names))}"
                    )
                if target == asset_name:
                    raise ValueError(
                        f"Asset '{asset_name}' transfers to itself — "
                        "this would zero the balance"
                    )

        # Validate query references
        for i, query in enumerate(self.queries):
            # Validate query time within horizon
            if isinstance(query, PercentilesQuery):
                at_values = query.at if isinstance(query.at, list) else [query.at]
                for at_val in at_values:
                    if at_val > horizon_months:
                        raise ValueError(
                            f"Query {i} asks about month {at_val} "
                            f"but time_horizon is {horizon_months} months "
                            f"({horizon_months // 12} years)"
                        )
            else:
                if query.at > horizon_months:
                    raise ValueError(
                        f"Query {i} asks about month {query.at} "
                        f"but time_horizon is {horizon_months} months "
                        f"({horizon_months // 12} years)"
                    )

        # Validate cash flow references
        if self.cash_flows:
            for cf_name, cf in self.cash_flows.items():
                # Asset must exist
                if cf.asset not in asset_names:
                    raise ValueError(
                        f"Cash flow '{cf_name}' references asset '{cf.asset}' "
                        f"but no asset with that name exists. "
                        f"Available: {', '.join(sorted(asset_names))}"
                    )

                # Validate 'at' within horizon
                if cf.at is not None and cf.at > horizon_months:
                    raise ValueError(
                        f"Cash flow '{cf_name}' has 'at' = {cf.at} months "
                        f"but time_horizon is {horizon_months} months "
                        f"({horizon_months // 12} years)"
                    )

                # Validate 'start'/'end' within horizon
                if cf.start is not None and cf.start > horizon_months:
                    raise ValueError(
                        f"Cash flow '{cf_name}' has 'start' = {cf.start} months "
                        f"but time_horizon is {horizon_months} months "
                        f"({horizon_months // 12} years)"
                    )
                if cf.end is not None and cf.end > horizon_months:
                    raise ValueError(
                        f"Cash flow '{cf_name}' has 'end' = {cf.end} months "
                        f"but time_horizon is {horizon_months} months "
                        f"({horizon_months // 12} years)"
                    )

                # Validate start < end when both specified
                if (
                    cf.start is not None
                    and cf.end is not None
                    and cf.start >= cf.end
                ):
                    raise ValueError(
                        f"Cash flow '{cf_name}' has 'start' ({cf.start}) "
                        f">= 'end' ({cf.end}) — start must be before end"
                    )

        # Validate asset references in 'of' field (for non-probability queries)
        for i, query in enumerate(self.queries):
            if isinstance(query, (PercentilesQuery, ExpectedQuery, DistributionQuery)):
                # The 'of' field can be a simple asset name or an expression.
                # For now, check simple asset names. Complex expressions
                # will be validated by the expression parser later.
                of_value = query.of
                # Check if it looks like a simple asset name (no operators)
                if not any(op in of_value for op in ["+", "-", "*", "/", ">", "<"]):
                    if of_value not in asset_names:
                        raise ValueError(
                            f"Query {i} references asset '{of_value}' "
                            f"but no asset with that name exists. "
                            f"Available: {', '.join(sorted(asset_names))}"
                        )

        return self

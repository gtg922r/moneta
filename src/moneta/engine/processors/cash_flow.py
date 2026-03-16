"""Cash flow processor -- applies scheduled income, withdrawals, and one-time expenses.

Cash flows are applied in pipeline position 3 (after events/transfers,
before growth). This means:
- Equity liquidation proceeds are in the account before withdrawals
- The post-cash-flow balance is what gets growth applied

For inflation-adjusted cash flows, the amount is multiplied by
state.cum_inflation (from the previous step's inflation update).

When allow_negative is False (default), balances are floored at zero.
Any unmet withdrawal is tracked in state.cash_flow_shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moneta.engine.state import SimulationState
from moneta.parser.models import ScenarioModel


@dataclass
class _CashFlowConfig:
    """Internal cash flow configuration, pre-computed from model."""

    asset_col: int  # column index in balances
    monthly_amount: float  # amount per month (adjusted for frequency)
    start_month: int  # first month this flow is active (inclusive, 0-based step)
    end_month: int  # last month this flow is active (exclusive, 0-based step)
    is_one_time: bool  # if True, only applies at start_month
    adjust_for_inflation: bool
    allow_negative: bool


class CashFlowProcessor:
    """Apply scheduled cash flows (income, withdrawals, expenses) to assets.

    Cash flows are applied in pipeline position 3 (after events/transfers,
    before growth). This means:
    - Equity liquidation proceeds are in the account before withdrawals
    - The post-cash-flow balance is what gets growth applied

    For inflation-adjusted cash flows, the amount is multiplied by
    state.cum_inflation (from the previous step's inflation update).

    When allow_negative is False (default), balances are floored at zero.
    Any unmet withdrawal is tracked in state.cash_flow_shortfall.
    """

    def __init__(self, configs: list[_CashFlowConfig]) -> None:
        self._configs = configs

    @classmethod
    def from_scenario(
        cls, model: ScenarioModel, asset_index: dict[str, int]
    ) -> CashFlowProcessor:
        """Build cash flow configs from the model's cash_flows section.

        Duration values from the model are already in months. We use them
        directly as 0-based step indices:
        - start=None -> step 0 (beginning of simulation)
        - end=None -> step = time_horizon (full duration)
        - at=N -> one-time at step N-1 (same convention as query _time_to_step)

        For annual frequency, the amount is spread evenly across 12 months
        (annual_amount / 12 applied each month).
        """
        configs: list[_CashFlowConfig] = []
        if not model.cash_flows:
            return cls([])

        time_horizon = model.scenario.time_horizon

        for _name, cf in model.cash_flows.items():
            col = asset_index[cf.asset]
            amount_val = cf.amount  # CashFlowAmountValue

            # Determine timing
            if cf.at is not None:
                # One-time cash flow: apply at step (at - 1) to match
                # query convention (1-based month -> 0-based step)
                start_month = cf.at - 1
                end_month = cf.at  # exclusive, so only one step
                is_one_time = True
                monthly_amount = amount_val.amount  # full amount at once
            else:
                # Recurring cash flow
                start_month = (cf.start - 1) if cf.start else 0
                end_month = cf.end if cf.end else time_horizon
                is_one_time = False

                if amount_val.frequency == "annually":
                    monthly_amount = amount_val.amount / 12.0
                else:
                    # "monthly" or None (shouldn't hit None for recurring due to validation)
                    monthly_amount = amount_val.amount

            configs.append(
                _CashFlowConfig(
                    asset_col=col,
                    monthly_amount=monthly_amount,
                    start_month=start_month,
                    end_month=end_month,
                    is_one_time=is_one_time,
                    adjust_for_inflation=cf.adjust_for == "inflation",
                    allow_negative=cf.allow_negative,
                )
            )

        return cls(configs)

    def step(
        self, state: SimulationState, dt: float, rng: np.random.Generator
    ) -> None:
        """Apply cash flows for the current time step.

        For each active cash flow config:
        1. Check if this step is within the cash flow's active window
        2. Compute the amount (optionally inflation-adjusted)
        3. Add the amount to the target asset column
        4. If allow_negative is False, clamp negative balances to zero
           and accumulate the shortfall in state.cash_flow_shortfall
        """
        t = state.step
        n_runs = state.balances.shape[0]

        for cfg in self._configs:
            # Check if this cash flow is active this month
            if cfg.is_one_time:
                if t != cfg.start_month:
                    continue
            else:
                if t < cfg.start_month or t >= cfg.end_month:
                    continue

            # Compute the amount (possibly inflation-adjusted)
            amount = cfg.monthly_amount
            if cfg.adjust_for_inflation:
                # Multiply by cumulative inflation to maintain real purchasing power
                amount_arr = amount * state.cum_inflation
            else:
                amount_arr = np.full(n_runs, amount)

            # Apply to balances
            state.balances[:, cfg.asset_col] += amount_arr

            # Handle negative balance clamping
            if not cfg.allow_negative:
                col = cfg.asset_col
                negative_mask = state.balances[:, col] < 0
                if np.any(negative_mask):
                    # Track shortfall (how much below zero)
                    shortfall = np.abs(state.balances[negative_mask, col])
                    state.cash_flow_shortfall[negative_mask] += shortfall
                    # Clamp to zero
                    state.balances[negative_mask, col] = 0.0

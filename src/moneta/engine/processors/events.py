"""Hazard-Rate Event Processor.

Fires discrete events based on hazard rates calibrated to user probabilities.

For "20% within 3 years", computes monthly hazard rate h such that
1 - (1-h)^36 = 0.20, then each month draws U ~ Uniform(0,1) and fires
if U < h and event hasn't fired yet.

On firing: draws liquidation value from the multiplier range applied to
the asset's current valuation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moneta.engine.state import SimulationState
from moneta.parser.models import IlliquidEquityAsset, ScenarioModel


@dataclass
class _EventConfig:
    """Internal config for a single hazard-rate event."""

    asset_name: str
    asset_col: int  # column index in balances
    event_idx: int  # index in events_fired
    hazard_rate: float  # monthly
    start_month: int  # when the window starts
    end_month: int  # when the window ends
    multiplier_low: float
    multiplier_high: float
    base_valuation: float  # current_valuation from the asset config


def _compute_hazard_rate(probability: float, window_months: int) -> float:
    """Compute monthly hazard rate from cumulative probability and window.

    Given: P(fire within N months) = p
    Solve: h = 1 - (1-p)^(1/N)

    This ensures 1 - (1-h)^N = p.
    """
    if window_months <= 0:
        return 0.0
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        # Can't have h=1.0 exactly (would fire instantly), use very high rate
        return 1.0
    return float(1.0 - (1.0 - probability) ** (1.0 / window_months))


class EventProcessor:
    """Fire discrete events based on hazard rates calibrated to user probabilities."""

    def __init__(self, event_configs: list[_EventConfig]) -> None:
        self._configs = event_configs

    @classmethod
    def from_scenario(cls, model: ScenarioModel) -> EventProcessor:
        """Build event configs from the model's illiquid equity assets."""
        asset_names = list(model.assets.keys())
        asset_index = {name: i for i, name in enumerate(asset_names)}

        configs: list[_EventConfig] = []
        event_counter = 0

        for name, asset in model.assets.items():
            if isinstance(asset, IlliquidEquityAsset):
                for _j, liquidity_event in enumerate(asset.liquidity_events):
                    prob_window = liquidity_event.probability
                    window_months = prob_window.end_month - prob_window.start_month
                    hazard_rate = _compute_hazard_rate(
                        prob_window.probability, window_months
                    )

                    mult_low, mult_high = liquidity_event.valuation_range

                    configs.append(
                        _EventConfig(
                            asset_name=name,
                            asset_col=asset_index[name],
                            event_idx=event_counter,
                            hazard_rate=hazard_rate,
                            start_month=prob_window.start_month,
                            end_month=prob_window.end_month,
                            multiplier_low=mult_low,
                            multiplier_high=mult_high,
                            base_valuation=asset.current_valuation,
                        )
                    )
                    event_counter += 1

        return cls(configs)

    def step(self, state: SimulationState, dt: float, rng: np.random.Generator) -> None:
        """Check and fire events for the current time step.

        For each event config within its active window:
        - Only consider runs where the event hasn't already fired
        - Draw U ~ Uniform(0,1) for each run
        - Fire where U < hazard_rate
        - Set liquidation value = base_valuation * uniform(mult_low, mult_high)
        - Mark events_fired = True
        """
        n_runs = state.balances.shape[0]

        for cfg in self._configs:
            # Only active within the event's window
            if state.step < cfg.start_month or state.step >= cfg.end_month:
                continue

            # Mask: only unfired events
            mask = ~state.events_fired[:, cfg.event_idx]
            n_eligible = mask.sum()
            if n_eligible == 0:
                continue

            # Draw uniform for ALL runs (to consume RNG state consistently)
            draws = rng.uniform(size=n_runs)

            # Determine which eligible runs fire
            fires = mask & (draws < cfg.hazard_rate)
            n_fires = fires.sum()

            if n_fires == 0:
                continue

            # Mark as fired
            state.events_fired[fires, cfg.event_idx] = True

            # Draw liquidation multipliers for fired runs
            multipliers = rng.uniform(
                cfg.multiplier_low, cfg.multiplier_high, size=int(n_fires)
            )

            # Set asset balance to base_valuation * multiplier
            state.balances[fires, cfg.asset_col] = cfg.base_valuation * multipliers

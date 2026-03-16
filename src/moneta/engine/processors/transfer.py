"""Transfer Processor.

Moves asset values on event triggers. When an event fires (tracked in
events_fired), transfers the source asset's balance to the destination
asset and zeros the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from moneta.engine.state import SimulationState
from moneta.parser.models import IlliquidEquityAsset, ScenarioModel


@dataclass
class _TransferConfig:
    """Internal config for a single event-triggered transfer."""

    source_col: int  # source asset column in balances
    dest_col: int  # destination asset column in balances
    event_indices: list[int]  # which events trigger this transfer
    transferred: np.ndarray = field(repr=False)  # bool[n_runs], tracks if done


class TransferProcessor:
    """Move value between assets on event triggers."""

    def __init__(self, transfer_configs: list[_TransferConfig]) -> None:
        self._configs = transfer_configs

    @classmethod
    def from_scenario(
        cls, model: ScenarioModel, n_runs: int
    ) -> TransferProcessor:
        """Build transfer configs from the model's illiquid equity assets."""
        asset_names = list(model.assets.keys())
        asset_index = {name: i for i, name in enumerate(asset_names)}

        # Build event_index: same logic as SimulationState.from_scenario
        event_index: dict[str, int] = {}
        event_counter = 0
        for name, asset in model.assets.items():
            if isinstance(asset, IlliquidEquityAsset):
                for j, _event in enumerate(asset.liquidity_events):
                    key = f"{name}:{j}"
                    event_index[key] = event_counter
                    event_counter += 1

        configs: list[_TransferConfig] = []

        for name, asset in model.assets.items():
            if isinstance(asset, IlliquidEquityAsset):
                dest_name = asset.on_liquidation.transfer_to
                source_col = asset_index[name]
                dest_col = asset_index[dest_name]

                # Collect all event indices for this asset
                event_indices = []
                for j in range(len(asset.liquidity_events)):
                    key = f"{name}:{j}"
                    event_indices.append(event_index[key])

                configs.append(
                    _TransferConfig(
                        source_col=source_col,
                        dest_col=dest_col,
                        event_indices=event_indices,
                        transferred=np.zeros(n_runs, dtype=bool),
                    )
                )

        return cls(configs)

    def step(
        self, state: SimulationState, dt: float, rng: np.random.Generator
    ) -> None:
        """Execute triggered transfers for the current time step.

        For each transfer config:
        - Check if ANY of its associated events have fired
        - For runs where events fired but transfer hasn't happened yet,
          move the source balance to the destination and zero the source
        - Mark those runs as transferred (idempotent)
        """
        for cfg in self._configs:
            # OR across all events that trigger this transfer
            any_event_fired = np.zeros(state.balances.shape[0], dtype=bool)
            for eidx in cfg.event_indices:
                any_event_fired |= state.events_fired[:, eidx]

            # Only transfer for newly triggered runs
            newly_triggered = any_event_fired & ~cfg.transferred

            if not newly_triggered.any():
                continue

            # Move balance: dest += source, source = 0
            state.balances[newly_triggered, cfg.dest_col] += state.balances[
                newly_triggered, cfg.source_col
            ]
            state.balances[newly_triggered, cfg.source_col] = 0.0

            # Mark as transferred
            cfg.transferred[newly_triggered] = True

"""SimulationState and ResultStore dataclasses for the Monte Carlo engine.

SimulationState holds the mutable working state for all simulation runs.
ResultStore holds pre-allocated storage for simulation results, filled
one time-slice at a time by the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moneta.parser.models import (
    IlliquidEquityAsset,
    InvestmentAsset,
    PresetRef,
    ScenarioModel,
)


@dataclass
class SimulationState:
    """Mutable working state for all simulation runs.

    All arrays have first dimension = n_runs. Processors mutate in-place.
    The orchestrator snapshots into ResultStore after each time step.
    """

    balances: np.ndarray  # float64[n_runs, n_assets]
    events_fired: np.ndarray  # bool[n_runs, n_events]
    inflation_rate: np.ndarray  # float64[n_runs] — current annual rate
    cum_inflation: np.ndarray  # float64[n_runs] — cumulative factor (starts at 1.0)
    cash_flow_shortfall: np.ndarray  # float64[n_runs] — cumulative unmet withdrawals
    step: int  # current time step index
    asset_names: list[str]  # ordered asset names (column mapping)
    asset_index: dict[str, int]  # name → column index
    event_index: dict[str, int]  # event key → column index in events_fired

    @classmethod
    def from_scenario(cls, model: ScenarioModel, n_runs: int) -> SimulationState:
        """Initialize state from a parsed scenario model.

        Builds asset_names/asset_index from model.assets, initializes
        balances from asset configs, counts events, and sets up inflation.
        """
        # Build asset_names and asset_index from model.assets dict
        asset_names = list(model.assets.keys())
        asset_index = {name: i for i, name in enumerate(asset_names)}
        n_assets = len(asset_names)

        # Initialize balances
        balances = np.zeros((n_runs, n_assets), dtype=np.float64)
        for name, asset in model.assets.items():
            col = asset_index[name]
            if isinstance(asset, InvestmentAsset):
                balances[:, col] = asset.initial_balance
            elif isinstance(asset, IlliquidEquityAsset):
                balances[:, col] = asset.current_valuation

        # Count events and build event_index
        event_index: dict[str, int] = {}
        event_counter = 0
        for name, asset in model.assets.items():
            if isinstance(asset, IlliquidEquityAsset):
                for j, _event in enumerate(asset.liquidity_events):
                    key = f"{name}:{j}"
                    event_index[key] = event_counter
                    event_counter += 1

        n_events = event_counter
        events_fired = np.zeros((n_runs, n_events), dtype=bool)

        # Initialize inflation_rate from the inflation config
        inflation_config = model.global_config.inflation
        if isinstance(inflation_config, PresetRef):
            # PresetRef: use a sensible default (3% annual)
            initial_rate = 0.03
        else:
            # InflationConfig: use long_term_rate
            initial_rate = inflation_config.long_term_rate

        inflation_rate = np.full(n_runs, initial_rate, dtype=np.float64)

        # Initialize cum_inflation to 1.0 for all runs
        cum_inflation = np.ones(n_runs, dtype=np.float64)

        # Initialize cash_flow_shortfall to 0.0 for all runs
        cash_flow_shortfall = np.zeros(n_runs, dtype=np.float64)

        return cls(
            balances=balances,
            events_fired=events_fired,
            inflation_rate=inflation_rate,
            cum_inflation=cum_inflation,
            cash_flow_shortfall=cash_flow_shortfall,
            step=0,
            asset_names=asset_names,
            asset_index=asset_index,
            event_index=event_index,
        )


@dataclass
class ResultStore:
    """Pre-allocated storage for simulation results."""

    balances: np.ndarray  # float64[n_runs, n_steps, n_assets]
    cum_inflation: np.ndarray  # float64[n_runs, n_steps]
    cash_flow_shortfall: np.ndarray  # float64[n_runs, n_steps] — shortfall over time
    event_fired_at: np.ndarray  # int32[n_runs, n_events] — month fired, -1 if never
    asset_names: list[str]
    asset_index: dict[str, int]
    n_runs: int
    n_steps: int
    n_assets: int

    @classmethod
    def allocate(cls, model: ScenarioModel, n_runs: int) -> ResultStore:
        """Pre-allocate all arrays with correct shapes."""
        n_steps = model.scenario.time_horizon  # already in months
        asset_names = list(model.assets.keys())
        asset_index = {name: i for i, name in enumerate(asset_names)}
        n_assets = len(asset_names)

        # Count events across all assets
        n_events = 0
        for asset in model.assets.values():
            if isinstance(asset, IlliquidEquityAsset):
                n_events += len(asset.liquidity_events)

        # Pre-allocate arrays
        balances = np.empty((n_runs, n_steps, n_assets), dtype=np.float64)
        cum_inflation = np.empty((n_runs, n_steps), dtype=np.float64)
        cash_flow_shortfall = np.zeros((n_runs, n_steps), dtype=np.float64)
        event_fired_at = np.full((n_runs, n_events), -1, dtype=np.int32)

        return cls(
            balances=balances,
            cum_inflation=cum_inflation,
            cash_flow_shortfall=cash_flow_shortfall,
            event_fired_at=event_fired_at,
            asset_names=asset_names,
            asset_index=asset_index,
            n_runs=n_runs,
            n_steps=n_steps,
            n_assets=n_assets,
        )

    def record(self, state: SimulationState, step: int) -> None:
        """Copy current state slice into result arrays at the given time step."""
        self.balances[:, step, :] = state.balances
        self.cum_inflation[:, step] = state.cum_inflation
        self.cash_flow_shortfall[:, step] = state.cash_flow_shortfall

        # Update event_fired_at: for any newly fired events, record the step.
        # Only update entries that are still -1 (unfired) where events_fired
        # is now True.
        if state.events_fired.shape[1] > 0:
            newly_fired = state.events_fired & (self.event_fired_at == -1)
            self.event_fired_at[newly_fired] = step

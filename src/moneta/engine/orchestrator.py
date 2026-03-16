"""Monte Carlo orchestrator — runs the simulation pipeline.

Builds a fixed processor pipeline from the scenario model and executes
the simulation loop. Each time step runs all processors in order, then
records the state into the ResultStore.

Pipeline order (invariant):
1. EventProcessor (if any events exist)
2. TransferProcessor (if any transfers exist)
3. GrowthProcessor (if any growth configs exist)
4. InflationProcessor (always)
"""

from __future__ import annotations

import numpy as np

from moneta.engine.processors import Processor
from moneta.engine.processors.events import EventProcessor
from moneta.engine.processors.growth import GrowthProcessor
from moneta.engine.processors.inflation import InflationProcessor
from moneta.engine.processors.transfer import TransferProcessor
from moneta.engine.state import ResultStore, SimulationState
from moneta.parser.models import (
    GrowthConfig,
    IlliquidEquityAsset,
    InflationConfig,
    InvestmentAsset,
    PresetRef,
    ScenarioModel,
)


def build_pipeline(
    model: ScenarioModel, state: SimulationState, n_runs: int
) -> list[Processor]:
    """Build the fixed processor pipeline from the scenario model.

    Pipeline order (invariant):
    1. EventProcessor (if any events exist)
    2. TransferProcessor (if any transfers exist)
    3. GrowthProcessor (if any growth configs exist)
    4. InflationProcessor (always)

    Args:
        model: The parsed scenario model.
        state: The initialized simulation state (used for asset_index).
        n_runs: Number of simulation runs.

    Returns:
        Ordered list of processors.
    """
    pipeline: list[Processor] = []

    # Check for illiquid equity assets with liquidity events
    has_events = False
    has_transfers = False
    for asset in model.assets.values():
        if isinstance(asset, IlliquidEquityAsset):
            if asset.liquidity_events:
                has_events = True
                has_transfers = True
            break  # only need to find one

    # Re-check more carefully: any illiquid equity asset with events
    has_events = any(
        isinstance(asset, IlliquidEquityAsset) and len(asset.liquidity_events) > 0
        for asset in model.assets.values()
    )
    has_transfers = any(
        isinstance(asset, IlliquidEquityAsset)
        for asset in model.assets.values()
    )

    # 1. EventProcessor (if any events defined)
    if has_events:
        pipeline.append(EventProcessor.from_scenario(model))

    # 2. TransferProcessor (if any transfers defined)
    if has_transfers:
        pipeline.append(TransferProcessor.from_scenario(model, n_runs))

    # 3. GrowthProcessor (if any growth configs)
    growth_configs: dict[str, GrowthConfig] = {}
    for name, asset in model.assets.items():
        if isinstance(asset, InvestmentAsset):
            growth = asset.growth
            if isinstance(growth, GrowthConfig):
                growth_configs[name] = growth
            elif isinstance(growth, PresetRef):
                # Presets should already be resolved by the loader.
                # If we still have a PresetRef here, skip it gracefully.
                pass

    if growth_configs:
        pipeline.append(
            GrowthProcessor(
                growth_configs=growth_configs,
                asset_index=state.asset_index,
            )
        )

    # 4. InflationProcessor (always present)
    inflation_config = model.global_config.inflation
    if isinstance(inflation_config, InflationConfig):
        pipeline.append(InflationProcessor(inflation_config))
    elif isinstance(inflation_config, PresetRef):
        # Preset not resolved — use a sensible default config
        default_inflation = InflationConfig(
            model="mean_reverting",
            long_term_rate=0.03,
            volatility=0.01,
        )
        pipeline.append(InflationProcessor(default_inflation))

    return pipeline


def run_simulation(
    model: ScenarioModel, seed: int | None = None
) -> ResultStore:
    """Run Monte Carlo simulation on the given model.

    Returns a ResultStore with pre-allocated arrays filled with
    simulation results.

    Args:
        model: The parsed and validated scenario model.
        seed: Optional random seed for reproducibility.

    Returns:
        ResultStore with balances, cum_inflation, and event_fired_at
        filled for every time step.
    """
    n_runs = model.scenario.simulations
    rng = np.random.default_rng(seed)
    state = SimulationState.from_scenario(model, n_runs)
    results = ResultStore.allocate(model, n_runs)
    pipeline = build_pipeline(model, state, n_runs)

    total_steps = model.scenario.time_horizon  # already in months from Duration type

    for t in range(total_steps):
        state.step = t
        for processor in pipeline:
            processor.step(state, dt=1 / 12, rng=rng)
        results.record(state, t)

    return results

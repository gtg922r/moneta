"""Geometric Brownian Motion (GBM) growth processor.

Applies stochastic growth to investment-type assets using the exact
solution of the GBM SDE:

    S(t+dt) = S(t) * exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z)

where Z ~ N(0,1). Vectorized across all runs simultaneously.
"""

from __future__ import annotations

import numpy as np

from moneta.engine.state import SimulationState
from moneta.parser.models import GrowthConfig


class GrowthProcessor:
    """Apply Geometric Brownian Motion growth to investment-type assets.

    Uses the exact solution:
      S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0,1). Vectorized across all runs simultaneously.

    Only applies growth to assets that have a resolved GrowthConfig (not PresetRef).
    The GrowthProcessor receives already-resolved configs.
    """

    def __init__(
        self,
        growth_configs: dict[str, GrowthConfig],
        asset_index: dict[str, int],
    ) -> None:
        """Initialize with growth configs and asset index mapping.

        Args:
            growth_configs: Mapping of asset name to its resolved GrowthConfig.
                Only assets with explicit GrowthConfig (not PresetRef) should
                be included.
            asset_index: Mapping of asset name to column index in state.balances.
        """
        # Build arrays of column indices, drifts, and vols for growth assets
        self._growth_indices: list[int] = []
        drifts: list[float] = []
        vols: list[float] = []

        for name, config in growth_configs.items():
            idx = asset_index[name]
            self._growth_indices.append(idx)
            mu = config.expected_return
            sigma = config.volatility
            # Pre-compute the drift-adjusted term: mu - sigma^2/2
            drifts.append(mu - 0.5 * sigma * sigma)
            vols.append(sigma)

        # Convert to numpy arrays for vectorized math
        self._drifts = np.array(drifts, dtype=np.float64)  # shape: (n_growth_assets,)
        self._vols = np.array(vols, dtype=np.float64)  # shape: (n_growth_assets,)
        self._n_growth_assets = len(self._growth_indices)

    def step(self, state: SimulationState, dt: float, rng: np.random.Generator) -> None:
        """Apply one step of GBM growth to all growth-asset columns.

        Mutates state.balances in-place for the columns corresponding
        to assets with growth configs.
        """
        if self._n_growth_assets == 0:
            return

        n_runs = state.balances.shape[0]

        # Draw standard normal samples: (n_runs, n_growth_assets)
        z = rng.standard_normal((n_runs, self._n_growth_assets))

        # Compute multipliers: exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        sqrt_dt = np.sqrt(dt)
        exponent = self._drifts * dt + self._vols * sqrt_dt * z
        multipliers = np.exp(exponent)

        # Apply growth to the relevant columns
        state.balances[:, self._growth_indices] *= multipliers

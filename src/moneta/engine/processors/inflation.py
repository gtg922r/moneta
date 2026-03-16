"""Ornstein-Uhlenbeck mean-reverting inflation processor.

Models inflation as a mean-reverting stochastic process:

    dx = theta * (mu - x) * dt + sigma * dW

where:
    x     = current inflation rate
    mu    = long-term mean rate
    theta = mean-reversion speed
    sigma = volatility
    dW    = Wiener process increment (sqrt(dt) * Z, Z ~ N(0,1))

Updates state.inflation_rate and state.cum_inflation each step.
"""

from __future__ import annotations

import numpy as np

from moneta.engine.state import SimulationState
from moneta.parser.models import InflationConfig


class InflationProcessor:
    """Track inflation using an Ornstein-Uhlenbeck mean-reverting process.

    dx = theta * (mu - x) * dt + sigma * dW
    Updates state.inflation_rate and state.cum_inflation each step.
    Inflation can go negative (deflation) -- this is realistic and allowed.
    """

    def __init__(self, config: InflationConfig) -> None:
        """Initialize from an InflationConfig.

        Args:
            config: Resolved InflationConfig with long_term_rate,
                mean_reversion_speed, and volatility.
        """
        self.mu = config.long_term_rate  # long-term mean (annual rate)
        self.theta = config.mean_reversion_speed  # mean-reversion speed
        self.sigma = config.volatility  # volatility (annual)

    def step(self, state: SimulationState, dt: float, rng: np.random.Generator) -> None:
        """Step the inflation process by dt and update cumulative inflation.

        Mutates state.inflation_rate and state.cum_inflation in-place.
        """
        n_runs = state.inflation_rate.shape[0]

        # Wiener increment: dW = sqrt(dt) * Z, Z ~ N(0,1)
        sqrt_dt = np.sqrt(dt)
        dw = rng.standard_normal(n_runs) * sqrt_dt

        # Ornstein-Uhlenbeck step: dx = theta * (mu - x) * dt + sigma * dW
        dx = self.theta * (self.mu - state.inflation_rate) * dt + self.sigma * dw

        # Update the instantaneous inflation rate
        state.inflation_rate += dx

        # Update cumulative inflation factor
        # For a monthly step, the inflation accrued is rate * dt
        state.cum_inflation *= 1.0 + state.inflation_rate * dt

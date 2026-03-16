"""Processor protocol and registry for the Monte Carlo engine.

Each processor implements the Processor protocol — a single `step` method
that mutates SimulationState in-place for one time step across all runs.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from moneta.engine.state import SimulationState


@runtime_checkable
class Processor(Protocol):
    """Protocol for simulation processors.

    Processors mutate SimulationState in-place and return nothing.
    All runs are processed simultaneously via vectorized NumPy operations.
    """

    def step(self, state: SimulationState, dt: float, rng: np.random.Generator) -> None:
        """Mutate state by one time step. All runs processed simultaneously."""
        ...

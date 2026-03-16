"""Shared test fixtures for Moneta."""

import numpy as np
import pytest


@pytest.fixture
def seeded_rng():
    """Return a seeded random number generator for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture
def large_seeded_rng():
    """Return a seeded RNG for statistical tests requiring many runs."""
    return np.random.default_rng(12345)

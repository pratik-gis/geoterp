"""Shared fixtures: deterministic synthetic data for the whole suite."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def constant_points(rng):
    """Scattered points all carrying the same value (15.0)."""
    x = rng.uniform(0, 100, 60)
    y = rng.uniform(0, 100, 60)
    z = np.full_like(x, 15.0)
    return x, y, z


@pytest.fixture
def linear_field():
    """A known linear surface z = 2x - 3y + 5 and 50 samples of it."""
    a, b, c = 2.0, -3.0, 5.0

    def f(x, y):
        return a * x + b * y + c

    rng = np.random.default_rng(99)
    x = rng.uniform(0, 50, 50)
    y = rng.uniform(0, 50, 50)
    z = f(x, y)
    return x, y, z, f


@pytest.fixture
def stations():
    from geoterp import datasets

    return datasets.load_stations()


@pytest.fixture
def zones():
    from geoterp import datasets

    return (
        datasets.load_source_zones(),
        datasets.load_target_zones(),
        datasets.load_landuse(),
    )

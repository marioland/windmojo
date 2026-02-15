"""Shared test fixtures for windmojo test suite."""

import datetime

import numpy as np
import pytest

# -- Coordinates --

LAKE_COMO_LAT = 46.083333
LAKE_COMO_LON = 9.3


@pytest.fixture
def lake_como_coords():
    """Return Lake Como center coordinates as (lat, lon)."""
    return (LAKE_COMO_LAT, LAKE_COMO_LON)


# -- Config fixtures --


@pytest.fixture
def valid_config():
    """Return a minimal valid windmojo config dict."""
    return {
        "spot": "dervio",
        "latitude": LAKE_COMO_LAT,
        "longitude": LAKE_COMO_LON,
    }


@pytest.fixture
def valid_config_with_training():
    """Return a valid config with training section."""
    return {
        "spot": "dervio",
        "latitude": LAKE_COMO_LAT,
        "longitude": LAKE_COMO_LON,
        "training": {
            "model": "spatial_heavy_unified",
            "test_dates": {
                "strategy": "first_n_days_every_nth_month",
                "n_days": 3,
                "nth_month": 2,
            },
        },
    }


# -- Data fixtures --


@pytest.fixture
def sample_wind_speeds():
    """Return realistic wind speed arrays (true, predicted)."""
    y_true = np.array([5.2, 8.1, 3.5, 12.0, 6.7, 4.3, 9.1, 7.5])
    y_pred = np.array([4.8, 7.5, 4.0, 11.2, 7.0, 4.8, 8.5, 7.8])
    return y_true, y_pred


@pytest.fixture
def sample_dates():
    """Return a list of dates spanning 6 months (28 days each)."""
    dates = []
    for month in range(1, 7):
        for day in range(1, 29):
            dates.append(datetime.date(2025, month, day))
    return sorted(dates)

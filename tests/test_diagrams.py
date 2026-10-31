"""Tests for windmojo.evaluation.diagrams (GFS baseline units)."""

import datetime
import logging

import pandas as pd

from windmojo.evaluation import create_validation_diagrams


def test_gfs_baseline_is_converted_from_kmh(tmp_path, caplog):
    """effective_wind_10m is in km/h; measurements and predictions in m/s."""
    times = pd.date_range("2025-05-01 06:00", periods=18, freq="h")
    predictions = pd.DataFrame({"time": times, "prediction": 3.0})
    measurements = pd.DataFrame({"time": times, "spot speed": 3.6})
    features = pd.DataFrame({"time": times, "effective_wind_10m": 3.6 * 3.6})

    with caplog.at_level(logging.INFO, logger="windmojo.evaluation.diagrams"):
        files = create_validation_diagrams(
            predictions=predictions,
            measurements=measurements,
            features=features,
            test_dates=[datetime.date(2025, 5, 1)],
            target_col="spot speed",
            output_dir=tmp_path,
        )

    assert len(files) == 1
    assert "GFS MAE: 0.00 m/s" in caplog.text
    assert "Our Model MAE: 0.60 m/s" in caplog.text

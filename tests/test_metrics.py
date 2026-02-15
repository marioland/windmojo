"""Tests for windmojo.evaluation.metrics module."""

import numpy as np
import pytest

# Import directly from module to avoid pulling in pandas via __init__.py
from windmojo.evaluation.metrics import calculate_metrics  # noqa: E402


class TestCalculateMetrics:
    """Tests for calculate_metrics."""

    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["mbe"] == 0.0
        assert metrics["samples"] == 5

    def test_constant_offset(self):
        """Predictions consistently 2.0 higher than truth."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["mae"] == 2.0
        assert metrics["rmse"] == 2.0
        assert metrics["mbe"] == 2.0  # positive bias

    def test_negative_bias(self):
        """Predictions consistently 1.0 lower than truth."""
        y_true = np.array([5.0, 6.0, 7.0])
        y_pred = np.array([4.0, 5.0, 6.0])
        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["mbe"] == -1.0  # negative bias

    def test_mae_is_mean_absolute_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 5.0])
        metrics = calculate_metrics(y_true, y_pred)

        # |2-1| + |2-2| + |5-3| = 1 + 0 + 2 = 3; mean = 1.0
        assert abs(metrics["mae"] - 1.0) < 1e-10

    def test_rmse_greater_than_or_equal_mae(self):
        """RMSE >= MAE always holds (penalizes large errors more)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 5.0])
        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["rmse"] >= metrics["mae"]

    def test_r2_range(self):
        """R2 should be between -inf and 1.0 for reasonable predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["r2"] <= 1.0
        assert metrics["r2"] > 0.9  # good predictions

    def test_r2_poor_predictions(self):
        """R2 should be low for poor predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 1.0, 4.0, 2.0, 3.0])  # scrambled
        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["r2"] < 0.5

    def test_samples_count(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["samples"] == 3

    def test_returns_float_types(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.5, 2.5])
        metrics = calculate_metrics(y_true, y_pred)

        assert isinstance(metrics["mae"], float)
        assert isinstance(metrics["rmse"], float)
        assert isinstance(metrics["r2"], float)
        assert isinstance(metrics["mbe"], float)

    def test_wind_speed_realistic_values(self):
        """Test with realistic wind speed values (m/s)."""
        y_true = np.array([5.2, 8.1, 3.5, 12.0, 6.7, 4.3])
        y_pred = np.array([4.8, 7.5, 4.0, 11.2, 7.0, 4.8])
        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["mae"] < 1.0  # good wind forecast
        assert metrics["rmse"] < 1.5
        assert metrics["r2"] > 0.9

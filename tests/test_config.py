"""Tests for windmojo.config module (simplified config)."""

import datetime
from pathlib import Path

import pytest

from windmojo.config import (
    ConfigurationError,
    get_forecast_config,
    get_grid_config,
    get_paths,
    get_spot_dir,
    get_test_dates,
    get_training_config,
    get_validation_dates,
    validate_config,
)


def make_valid_config(**overrides):
    """Create a minimal valid config dict."""
    config = {
        "spot": "dervio",
        "latitude": 46.083333,
        "longitude": 9.3,
    }
    config.update(overrides)
    return config


class TestValidateConfig:
    """Tests for validate_config."""

    def test_valid_minimal_config(self):
        config = make_valid_config()
        validate_config(config)  # should not raise

    def test_valid_measurement_unit(self):
        config = make_valid_config(target={"column": "x", "unit": "km/h"})
        validate_config(config)  # should not raise

    def test_unknown_measurement_unit_raises(self):
        config = make_valid_config(target={"column": "x", "unit": "knots"})
        with pytest.raises(ConfigurationError, match="knots"):
            validate_config(config)

    def test_missing_spot_raises(self):
        config = {"latitude": 46.0, "longitude": 9.0}
        with pytest.raises(ConfigurationError, match="spot"):
            validate_config(config)

    def test_missing_latitude_raises(self):
        config = {"spot": "test", "longitude": 9.0}
        with pytest.raises(ConfigurationError, match="latitude"):
            validate_config(config)

    def test_missing_longitude_raises(self):
        config = {"spot": "test", "latitude": 46.0}
        with pytest.raises(ConfigurationError, match="longitude"):
            validate_config(config)

    def test_invalid_latitude_raises(self):
        config = make_valid_config(latitude=91.0)
        with pytest.raises(ConfigurationError, match="coordinates"):
            validate_config(config)

    def test_invalid_longitude_raises(self):
        config = make_valid_config(longitude=200.0)
        with pytest.raises(ConfigurationError, match="coordinates"):
            validate_config(config)

    def test_valid_grid_config(self):
        config = make_valid_config(grid={"radius_km": 25.0})
        validate_config(config)  # should not raise

    def test_invalid_grid_radius_raises(self):
        config = make_valid_config(grid={"radius_km": -5})
        with pytest.raises(ConfigurationError, match="radius"):
            validate_config(config)


class TestGetSpotDir:
    """Tests for get_spot_dir."""

    def test_returns_path(self):
        result = get_spot_dir("dervio")
        assert isinstance(result, Path)

    def test_path_format(self):
        result = get_spot_dir("dervio")
        assert result == Path("data/spots/dervio")

    def test_different_spot(self):
        result = get_spot_dir("lecco")
        assert "lecco" in str(result)


class TestGetPaths:
    """Tests for get_paths."""

    def test_returns_all_required_keys(self):
        config = make_valid_config()
        paths = get_paths(config)
        expected_keys = {
            "measurements",
            "config",
            "raw_data",
            "features",
            "validation",
            "model",
            "metadata",
            "forecast",
        }
        assert set(paths.keys()) == expected_keys

    def test_paths_are_path_objects(self):
        config = make_valid_config()
        paths = get_paths(config)
        for key, path in paths.items():
            assert isinstance(path, Path), f"{key} is not a Path"

    def test_paths_contain_spot_name(self):
        config = make_valid_config(spot="testspot")
        paths = get_paths(config)
        for key, path in paths.items():
            assert "testspot" in str(path), f"{key} doesn't contain spot name"

    def test_custom_paths_override_defaults(self):
        config = make_valid_config(paths={"measurements": "custom/path.csv"})
        paths = get_paths(config)
        assert paths["measurements"] == Path("data/spots/dervio/custom/path.csv")



class TestGetGridConfig:
    """Tests for get_grid_config."""

    def test_default_radius(self):
        config = make_valid_config()
        grid = get_grid_config(config)
        assert grid["radius_km"] == 20.0

    def test_custom_radius(self):
        config = make_valid_config(grid={"radius_km": 30.0})
        grid = get_grid_config(config)
        assert grid["radius_km"] == 30.0


class TestGetTrainingConfig:
    """Tests for get_training_config."""

    def test_default_model(self):
        config = make_valid_config()
        training = get_training_config(config)
        assert training["model"] == "unified"

    def test_default_test_dates_strategy(self):
        config = make_valid_config()
        training = get_training_config(config)
        assert training["test_dates"]["strategy"] == "first_n_days_every_nth_month"

    def test_default_xgboost_params(self):
        config = make_valid_config()
        training = get_training_config(config)
        assert "xgboost" in training
        assert training["xgboost"]["max_depth"] == 6
        assert training["xgboost"]["learning_rate"] == 0.05

    def test_custom_xgboost_params_override(self):
        config = make_valid_config(training={"xgboost": {"max_depth": 8}})
        training = get_training_config(config)
        assert training["xgboost"]["max_depth"] == 8
        # Other defaults should still be present
        assert training["xgboost"]["learning_rate"] == 0.05


class TestGetForecastConfig:
    """Tests for get_forecast_config."""

    def test_defaults(self):
        config = make_valid_config()
        forecast = get_forecast_config(config)
        assert forecast["initialization_time"] == "06:00"
        assert forecast["horizon_hours"] == 18
        assert forecast["days_ahead"] == 7

    def test_custom_override(self):
        config = make_valid_config(forecast={"days_ahead": 3})
        forecast = get_forecast_config(config)
        assert forecast["days_ahead"] == 3
        # Other defaults preserved
        assert forecast["horizon_hours"] == 18


class TestGetTestDates:
    """Tests for get_test_dates."""

    def _make_dates(self, start_month=1, end_month=6, year=2025):
        """Create a list of dates spanning several months."""
        dates = []
        for month in range(start_month, end_month + 1):
            for day in range(1, 29):  # 28 days per month for simplicity
                dates.append(datetime.date(year, month, day))
        return sorted(dates)

    def test_first_n_days_strategy(self):
        config = make_valid_config(
            training={
                "test_dates": {
                    "strategy": "first_n_days_every_nth_month",
                    "n_days": 3,
                    "nth_month": 2,
                }
            }
        )
        dates = self._make_dates()
        test_dates = get_test_dates(config, dates)

        assert len(test_dates) > 0
        # Should pick from every 2nd month (0th, 2nd, 4th = Jan, Mar, May)
        months = {d.month for d in test_dates}
        assert 1 in months  # January (index 0)
        assert 3 in months  # March (index 2)
        assert 5 in months  # May (index 4)

    def test_last_n_days_strategy(self):
        config = make_valid_config(
            training={
                "test_dates": {
                    "strategy": "last_n_days_every_nth_month",
                    "n_days": 2,
                    "nth_month": 3,
                }
            }
        )
        dates = self._make_dates()
        test_dates = get_test_dates(config, dates)

        assert len(test_dates) > 0
        # Last days of selected months
        for td in test_dates:
            assert td.day >= 27  # should be near end of month

    def test_custom_dates_strategy(self):
        config = make_valid_config(
            training={
                "test_dates": {
                    "strategy": "custom",
                    "custom_dates": ["2025-03-01", "2025-06-15"],
                }
            }
        )
        dates = self._make_dates()
        test_dates = get_test_dates(config, dates)

        assert len(test_dates) == 2
        assert test_dates[0] == datetime.date(2025, 3, 1)
        assert test_dates[1] == datetime.date(2025, 6, 15)

    def test_returns_sorted_dates(self):
        config = make_valid_config(
            training={
                "test_dates": {
                    "strategy": "first_n_days_every_nth_month",
                    "n_days": 4,
                    "nth_month": 1,
                }
            }
        )
        dates = self._make_dates()
        test_dates = get_test_dates(config, dates)

        assert test_dates == sorted(test_dates)

    def test_n_days_limits_per_month(self):
        config = make_valid_config(
            training={
                "test_dates": {
                    "strategy": "first_n_days_every_nth_month",
                    "n_days": 2,
                    "nth_month": 1,  # every month
                }
            }
        )
        dates = self._make_dates(start_month=1, end_month=3)
        test_dates = get_test_dates(config, dates)

        # 3 months * 2 days = 6 test dates
        assert len(test_dates) == 6


class TestGetValidationDates:
    """Tests for get_validation_dates."""

    def test_every_nth_day(self):
        dates = [datetime.date(2025, 1, 1) + datetime.timedelta(days=i) for i in range(30)]
        assert get_validation_dates(dates, every_nth=10) == dates[::10]

    def test_disjoint_from_test_dates(self):
        config = make_valid_config()
        dates = [datetime.date(2025, 1, 1) + datetime.timedelta(days=i) for i in range(365)]
        test_dates = set(get_test_dates(config, dates))
        train_dates = [d for d in dates if d not in test_dates]
        assert not set(get_validation_dates(train_dates)) & test_dates

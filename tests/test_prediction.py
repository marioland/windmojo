"""Tests for windmojo.prediction (live forecast export)."""

import datetime
import json

import numpy as np
import pandas as pd
import pytest

from windmojo.data.grid_downloader import (
    CENTER_PRESSURE_LEVELS,
    CENTER_VARIABLES,
    CIRCLE_VARIABLES,
)
from windmojo.prediction import forecast as forecast_module
from windmojo.prediction.forecast import (
    apply_forecast_window,
    download_forecast,
    forecast_to_dict,
    save_forecast_json,
)
from windmojo.training import SpatialHeavyTrainer


def hourly_frame(start="2026-10-30", days=2, **columns):
    """Create an hourly frame over full days with constant columns."""
    times = pd.date_range(start, periods=24 * days, freq="h")
    return pd.DataFrame({"time": times, **columns})


class FakeClient:
    """Records fetch_forecast calls and returns constant hourly data."""

    def __init__(self):
        self.calls = []

    def fetch_forecast(
        self, latitude, longitude, start_date, end_date, variables, timezone
    ):
        self.calls.append(
            {"start": start_date, "end": end_date, "variables": variables}
        )
        days = (
            datetime.date.fromisoformat(end_date)
            - datetime.date.fromisoformat(start_date)
        ).days + 1
        return hourly_frame(start_date, days, **{v: 1.0 for v in variables})


class TestApplyForecastWindow:
    """Tests for apply_forecast_window."""

    def test_keeps_training_hours_only(self):
        raw = hourly_frame(location="center")
        result = apply_forecast_window(raw, "06:00", 18)
        assert sorted(result["time"].dt.hour.unique()) == list(range(6, 24))
        assert len(result) == 2 * 18

    def test_adds_raw_metadata_columns(self):
        raw = hourly_frame(location="center")
        result = apply_forecast_window(raw, "06:00", 18)
        first = result.iloc[0]
        assert first["forecast_made"] == pd.Timestamp("2026-10-30 06:00")
        assert first["forecast_date"] == "2026-10-30"
        assert first["month"] == 10

    def test_horizon_does_not_spill_into_next_day(self):
        raw = hourly_frame(location="center")
        result = apply_forecast_window(raw, "12:00", 18)
        assert result["time"].dt.hour.min() == 12
        assert len(result) == 2 * 12


class TestDownloadForecast:
    """Tests for download_forecast."""

    @pytest.fixture(autouse=True)
    def no_sleep(self, monkeypatch):
        monkeypatch.setattr(forecast_module.time, "sleep", lambda s: None)

    def test_one_call_per_grid_point(self, valid_config):
        client = FakeClient()
        download_forecast(valid_config, "2026-10-30", "2026-10-31", client)
        assert len(client.calls) == 9
        assert all(c["start"] == "2026-10-30" for c in client.calls)

    def test_center_gets_all_variables(self, valid_config):
        client = FakeClient()
        raw = download_forecast(valid_config, "2026-10-30", "2026-10-31", client)
        center_calls = [
            c for c in client.calls if c["variables"] is not CIRCLE_VARIABLES
        ]
        assert len(center_calls) == 1
        assert center_calls[0]["variables"] == (
            CENTER_VARIABLES + CENTER_PRESSURE_LEVELS
        )
        assert set(raw["location"]) == {
            "center",
            "n",
            "ne",
            "e",
            "se",
            "s",
            "sw",
            "w",
            "nw",
        }

    def test_applies_forecast_window(self, valid_config):
        raw = download_forecast(valid_config, "2026-10-30", "2026-10-30", FakeClient())
        assert raw["time"].dt.hour.min() == 6
        assert len(raw) == 9 * 18


class TestForecastToDict:
    """Tests for forecast_to_dict."""

    def make_trainer(self):
        trainer = SpatialHeavyTrainer({})
        trainer.training_history = {"test": {"mae": 0.74}}
        return trainer

    def test_hour_values(self, valid_config):
        result = pd.DataFrame(
            {
                "time": [pd.Timestamp("2026-10-31 14:00")],
                "wind_speed": [3.456],
                "openmeteo_wind_speed": [np.nan],
                "openmeteo_gusts": [4.0],
                "wind_direction": [229.6],
            }
        )
        data = forecast_to_dict(result, valid_config, self.make_trainer())
        assert data["hours"] == [
            {
                "time": "2026-10-31T14:00",
                "wind_speed": 3.5,
                "openmeteo_wind_speed": None,
                "openmeteo_gusts": 4.0,
                "wind_direction": 230,
            }
        ]
        assert data["model"]["test_mae"] == 0.74
        assert data["units"]["wind_speed"] == "m/s"

    def test_generated_at_has_timezone(self, valid_config):
        result = pd.DataFrame(
            columns=[
                "time",
                "wind_speed",
                "openmeteo_wind_speed",
                "openmeteo_gusts",
                "wind_direction",
            ]
        )
        config = {**valid_config, "timezone": "Europe/Zurich"}
        data = forecast_to_dict(result, config, self.make_trainer())
        parsed = datetime.datetime.fromisoformat(data["generated_at"])
        assert parsed.tzinfo is not None
        assert data["timezone"] == "Europe/Zurich"


class TestSaveForecastJson:
    """Tests for save_forecast_json."""

    def test_writes_json_without_temp_file(self, tmp_path):
        output = tmp_path / "web" / "spot.json"
        save_forecast_json({"spot": "dervio", "name": "Dervìo"}, output)
        assert json.loads(output.read_text()) == {"spot": "dervio", "name": "Dervìo"}
        assert list(output.parent.iterdir()) == [output]


class TestTrainerMedians:
    """predict() fills missing values with the training medians."""

    def train_tiny_model(self):
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"a": rng.normal(10, 1, 50), "b": rng.normal(size=50)})
        train["target"] = train["a"]
        trainer = SpatialHeavyTrainer({"xgboost": {"n_estimators": 5}})
        trainer.train(train, train.copy(), ["a", "b"])
        return trainer

    def test_medians_survive_save_and_load(self, tmp_path):
        trainer = self.train_tiny_model()
        trainer.save(tmp_path / "model.json", tmp_path / "metadata.json")

        loaded = SpatialHeavyTrainer({})
        loaded.load(tmp_path / "model.json", tmp_path / "metadata.json")

        pd.testing.assert_series_equal(
            loaded.feature_medians, trainer.feature_medians, check_names=False
        )

    def test_missing_values_use_training_medians(self):
        trainer = self.train_tiny_model()
        X = pd.DataFrame({"a": [np.nan, 100.0], "b": [0.0, 0.0]})
        filled = pd.DataFrame({"a": [trainer.feature_medians["a"], 100.0], "b": 0.0})
        np.testing.assert_allclose(trainer.predict(X), trainer.predict(filled))

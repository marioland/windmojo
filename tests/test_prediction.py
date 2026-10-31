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
    archive_forecast,
    archive_hindcast,
    archive_raw_forecast,
    download_forecast,
    forecast_to_dict,
    predict_forecast,
    save_forecast_json,
)
from windmojo.training import UnifiedTrainer


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
        trainer = UnifiedTrainer({})
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
        assert data["model"]["missing_features"] == 0
        assert data["units"]["wind_speed"] == "m/s"

    def test_reports_missing_features(self, valid_config):
        result = pd.DataFrame(
            columns=[
                "time",
                "wind_speed",
                "openmeteo_wind_speed",
                "openmeteo_gusts",
                "wind_direction",
            ]
        )
        result.attrs["missing_features"] = ["pressure_msl_grad_n_s", "cloud_cover_n"]
        data = forecast_to_dict(result, valid_config, self.make_trainer())
        assert data["model"]["missing_features"] == 2

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


class TestArchiveForecast:
    """Tests for archive_forecast."""

    def test_one_file_per_run_in_month_folder(self, tmp_path):
        data = {
            "spot": "greifensee",
            "generated_at": "2026-10-31T17:15:04+01:00",
            "hours": [],
        }
        path = archive_forecast(data, tmp_path / "archive")
        assert path == tmp_path / "archive" / "2026-10" / "2026-10-31T1715.json"
        assert json.loads(path.read_text()) == data

        later = {**data, "generated_at": "2026-10-31T23:15:02+01:00"}
        archive_forecast(later, tmp_path / "archive")
        assert len(list((tmp_path / "archive" / "2026-10").iterdir())) == 2


class TestArchiveRawForecast:
    """Tests for archive_raw_forecast."""

    def raw(self):
        times = pd.date_range("2026-10-24 06:00", "2026-11-06 23:00", freq="h")
        times = times[times.hour >= 6]
        return pd.DataFrame(
            {"time": times, "location": "center", "wind_speed_10m": 10.0}
        )

    def test_keeps_only_requested_lead_days_once_per_day(self, tmp_path):
        downloaded = datetime.datetime(
            2026, 10, 31, 5, 15, tzinfo=datetime.timezone.utc
        )
        path = archive_raw_forecast(
            self.raw(), datetime.date(2026, 10, 31), [1, 2], tmp_path, downloaded
        )
        assert path == tmp_path / "2026-10" / "2026-10-31.parquet"

        archived = pd.read_parquet(path)
        assert sorted(archived["time"].dt.date.unique()) == [
            datetime.date(2026, 11, 1),
            datetime.date(2026, 11, 2),
        ]
        assert sorted(archived["lead_days"].unique()) == [1, 2]
        assert (archived["downloaded_at"] == pd.Timestamp(downloaded)).all()

        # A later run on the same day keeps the first file
        later = downloaded + datetime.timedelta(hours=12)
        assert (
            archive_raw_forecast(
                self.raw(), datetime.date(2026, 10, 31), [1, 2], tmp_path, later
            )
            is None
        )
        assert (
            pd.read_parquet(path)["downloaded_at"] == pd.Timestamp(downloaded)
        ).all()


class HourTrainer:
    """Stand-in for UnifiedTrainer: predicts the hour of day."""

    feature_columns = ["hour"]
    training_history = {"data_until": "2026-10-25"}

    def predict(self, X):
        return X["hour"].to_numpy(dtype=float)


class TestHindcast:
    """predict_forecast also computes the past days with their weather data."""

    @pytest.fixture(autouse=True)
    def no_sleep(self, monkeypatch):
        monkeypatch.setattr(forecast_module.time, "sleep", lambda s: None)

    @pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
    def test_past_days_with_full_history(self, valid_config, tmp_path):
        config = {
            **valid_config,
            "timezone": "Europe/Zurich",
            "forecast": {"days_ahead": 2},
        }
        client = FakeClient()
        result = predict_forecast(
            config, HourTrainer(), today=datetime.date(2026, 10, 31), client=client
        )
        # 7 hindcast days need 7 more days of history
        assert client.calls[0]["start"] == "2026-10-17"
        assert result["time"].dt.date.min() == datetime.date(2026, 10, 31)

        hindcast = result.attrs["hindcast"]
        days = sorted(hindcast["time"].dt.date.unique())
        assert days[0] == datetime.date(2026, 10, 24) and days[-1] == datetime.date(
            2026, 10, 30
        )
        assert (hindcast["wind_speed"] == hindcast["time"].dt.hour).all()

        written = archive_hindcast(
            hindcast, HourTrainer(), tmp_path, "2026-10-31T17:00:00+01:00"
        )
        assert len(written) == 7
        stored = json.loads((tmp_path / "2026-10" / "2026-10-30.json").read_text())
        assert stored["model_data_until"] == "2026-10-25"
        assert stored["hours"][0] == {
            "time": "2026-10-30T06:00",
            "wind_speed": 6.0,
            "openmeteo_wind_speed": 0.28,
        }


class TestTrainerMedians:
    """predict() fills missing values with the training medians."""

    def train_tiny_model(self):
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"a": rng.normal(10, 1, 50), "b": rng.normal(size=50)})
        train["target"] = train["a"]
        trainer = UnifiedTrainer({"xgboost": {"n_estimators": 5}})
        trainer.train(train, train.copy(), train.copy(), ["a", "b"])
        return trainer

    def test_medians_survive_save_and_load(self, tmp_path):
        trainer = self.train_tiny_model()
        trainer.save(tmp_path / "model.json", tmp_path / "metadata.json")

        loaded = UnifiedTrainer({})
        loaded.load(tmp_path / "model.json", tmp_path / "metadata.json")

        # Written under temporary names and renamed: no leftovers
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "metadata.json",
            "model.json",
        ]
        pd.testing.assert_series_equal(
            loaded.feature_medians, trainer.feature_medians, check_names=False
        )

    def test_missing_values_use_training_medians(self):
        trainer = self.train_tiny_model()
        X = pd.DataFrame({"a": [np.nan, 100.0], "b": [0.0, 0.0]})
        filled = pd.DataFrame({"a": [trainer.feature_medians["a"], 100.0], "b": 0.0})
        np.testing.assert_allclose(trainer.predict(X), trainer.predict(filled))


class TestTrainerEarlyStopping:
    """Early stopping uses the validation days, the test days only give metrics."""

    def test_test_data_does_not_change_the_model(self):
        rng = np.random.default_rng(0)
        data = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
        data["target"] = data["a"] + rng.normal(scale=0.5, size=300)
        train, validation, test = data[:200], data[200:250], data[250:]
        noise = test.assign(target=rng.normal(scale=5, size=len(test)))

        params = {"xgboost": {"n_estimators": 200, "early_stopping_rounds": 5}}
        models = []
        for test_data in (test, noise):
            trainer = UnifiedTrainer(params)
            trainer.train(train, validation, test_data, ["a", "b"])
            models.append(trainer)

        assert models[0].model.best_iteration == models[1].model.best_iteration
        np.testing.assert_allclose(models[0].predict(test), models[1].predict(test))

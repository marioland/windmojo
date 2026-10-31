"""Tests for the forecast history and the local web server (cli.py serve)."""

import datetime
import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

from windmojo.prediction.forecast import (
    archive_forecast,
    save_forecast_json,
)
from windmojo.prediction.history import (
    forecast_for_day,
    history,
    load_archive,
    load_measurements,
    to_csv,
)
from windmojo.web import ForecastApp, serve


def forecast(generated_at, start, hours=3, speed=3.0):
    times = pd.date_range(start, periods=hours, freq="h")
    return {
        "spot": "testspot",
        "name": "Testspot",
        "timezone": "Europe/Zurich",
        "generated_at": generated_at,
        "model": {"type": "unified", "test_mae": 0.7, "missing_features": 0},
        "hours": [
            {
                "time": t.strftime("%Y-%m-%dT%H:%M"),
                "wind_speed": speed,
                "openmeteo_wind_speed": 2.0,
                "openmeteo_gusts": 4.0,
                "wind_direction": 270,
            }
            for t in times
        ],
    }


@pytest.fixture
def spot(tmp_path):
    """Spot with measurements (km/h) and two archived forecasts for 30 Oct."""
    measurements = tmp_path / "measurements.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-10-30_1000", "2026-10-30_1100", "2026-10-31_1000"],
            "test speed": [18.0, 36.0, 7.2],
            "test gust": [36.0, 54.0, 14.4],
            "test direction": [250, 260, 90],
        }
    ).to_csv(measurements, index=False)
    archive = tmp_path / "archive"
    archive_forecast(
        forecast("2026-10-29T17:15:00+01:00", "2026-10-30 10:00", speed=4.0), archive
    )
    archive_forecast(
        forecast("2026-10-30T05:15:00+01:00", "2026-10-30 10:00", speed=6.0), archive
    )
    config = {
        "spot": "testspot",
        "name": "Testspot",
        "latitude": 47.0,
        "longitude": 8.0,
        "timezone": "Europe/Zurich",
        "target": {
            "column": "test speed",
            "unit": "km/h",
            "gust_column": "test gust",
            "direction_column": "test direction",
        },
    }
    paths = {
        "measurements": measurements,
        "forecast": tmp_path / "forecast.json",
        "forecast_archive": archive,
        "forecast_raw_archive": tmp_path / "archive_raw",
        "model": tmp_path / "model.json",
        "metadata": tmp_path / "metadata.json",
    }
    return config, paths


class TestHistory:
    def test_forecast_shown_for_a_day_is_the_last_one_before_it(self, spot):
        config, paths = spot
        runs = load_archive(paths["forecast_archive"])
        shown = forecast_for_day(runs, datetime.date(2026, 10, 30))
        assert shown["generated_at"] == "2026-10-29T17:15:00+01:00"
        assert [h["wind_speed"] for h in shown["hours"]] == [4.0, 4.0, 4.0]

    def test_measurements_in_m_s(self, spot):
        config, paths = spot
        m = load_measurements(config, paths["measurements"])
        assert m.loc["2026-10-30 11:00", "speed"] == pytest.approx(10.0)
        assert m.loc["2026-10-30 11:00", "gust"] == pytest.approx(15.0)
        assert m.loc["2026-10-30 11:00", "direction"] == 260

    def test_days_with_forecast_and_measurement(self, spot):
        config, paths = spot
        days = history(
            load_archive(paths["forecast_archive"]),
            load_measurements(config, paths["measurements"]),
            datetime.date(2026, 10, 31),
            days=2,
        )
        assert [d["date"] for d in days] == ["2026-10-30", "2026-10-31"]
        first = days[0]["hours"]
        assert [h["time"] for h in first] == [
            "2026-10-30T10:00",
            "2026-10-30T11:00",
            "2026-10-30T12:00",
        ]
        assert first[0]["measured_speed"] == pytest.approx(5.0)
        assert first[2]["measured_speed"] is None
        assert days[1]["forecast_generated_at"] is None  # measured only

    def test_hindcast_and_training_period(self, spot, tmp_path):
        config, paths = spot
        for day, until in [("2026-10-30", "2026-10-30"), ("2026-10-31", "2026-10-30")]:
            save_forecast_json(
                {
                    "date": day,
                    "generated_at": "2026-10-31T17:00:00+01:00",
                    "model_data_until": until,
                    "hours": [
                        {
                            "time": f"{day}T10:00",
                            "wind_speed": 4.5,
                            "openmeteo_wind_speed": 2.0,
                        }
                    ],
                },
                tmp_path / "hindcast" / "2026-10" / f"{day}.json",
            )
        days = history(
            load_archive(paths["forecast_archive"]),
            load_measurements(config, paths["measurements"]),
            datetime.date(2026, 10, 31),
            days=1,
            hindcast_dir=tmp_path / "hindcast",
        )
        assert days[0]["hours"][0]["hindcast_speed"] == 4.5
        assert days[0]["hours"][0]["wind_speed"] == 4.0  # forecast shown then stays
        assert days[0]["hindcast_in_training"] is True
        assert days[1]["hindcast_in_training"] is False
        assert "windmojo_hindcast_m_s" in to_csv(days[0]["hours"]).splitlines()[0]

    def test_only_hours_of_the_forecast_window(self, spot):
        config, paths = spot
        days = history(
            [],
            load_measurements(config, paths["measurements"]),
            datetime.date(2026, 10, 30),
            days=0,
            hours_of_day=[11],
        )
        assert [h["time"] for h in days[0]["hours"]] == ["2026-10-30T11:00"]

    def test_csv(self):
        text = to_csv(
            [{"time": "2026-10-30T10:00", "wind_speed": 3.5, "measured_speed": None}],
            "2026-10-30T05:15",
        )
        header, row = text.strip().split("\n")
        assert header.split(",")[:2] == ["time", "windmojo_m_s"]
        assert row.split(",")[1] == "3.5"
        assert row.endswith("2026-10-30T05:15")


class TestServer:
    @pytest.fixture
    def server(self, spot):
        config, paths = spot
        calls = []

        def fake_forecast(config, paths):
            calls.append(1)
            data = forecast(
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "2026-10-31 09:00",
            )
            save_forecast_json(data, paths["forecast"])

        app = ForecastApp(config, paths, forecast_runner=fake_forecast)
        httpd = serve(app, port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}", calls
        httpd.shutdown()
        httpd.server_close()

    def get(self, url):
        with urllib.request.urlopen(url) as r:
            return r.status, r.headers, r.read().decode()

    def test_page_reads_json_from_this_server(self, server):
        base, _ = server
        status, _, body = self.get(base + "/")
        assert status == 200
        assert 'data-src="forecast.json"' in body

    def test_no_forecast_yet_then_update(self, server):
        base, calls = server
        with pytest.raises(urllib.error.HTTPError) as e:
            self.get(base + "/forecast.json")
        assert e.value.code == 404

        request = urllib.request.Request(base + "/api/update", method="POST")
        with urllib.request.urlopen(request) as r:
            result = json.loads(r.read())
        assert result["ok"] and calls == [1]

        # The forecast hours come with the measurements available so far
        _, _, body = self.get(base + "/forecast.json")
        hours = json.loads(body)["hours"]
        assert hours[1]["time"] == "2026-10-31T10:00"
        assert hours[1]["measured_speed"] == pytest.approx(2.0)

        # A second click within 10 minutes does not download again
        with urllib.request.urlopen(
            urllib.request.Request(base + "/api/update", method="POST")
        ) as r:
            assert "up to date" in " ".join(json.loads(r.read())["messages"])
        assert calls == [1]

    def test_status_history_and_csv(self, server):
        base, _ = server
        status = json.loads(self.get(base + "/api/status")[2])
        assert status["measurements_until"] == "2026-10-31T10:00"
        assert status["measurements_update"] is False

        days = json.loads(self.get(base + "/api/history?days=60")[2])["days"]
        assert "2026-10-30" in [d["date"] for d in days]

        _, headers, body = self.get(base + "/download/history.csv?days=60")
        assert headers["Content-Type"].startswith("text/csv")
        assert "attachment" in headers["Content-Disposition"]
        assert "2026-10-30T10:00" in body

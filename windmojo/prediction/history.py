"""
Forecast History

Past days as they were forecast (archived forecasts of `cli.py predict`)
next to what was measured, and CSV exports for the forecast page.
"""

import csv
import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from windmojo.units import measurement_unit, to_ms


def load_archive(archive_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all archived forecasts (archive_forecast), oldest first.

    Args:
        archive_dir: Archive root directory (paths["forecast_archive"])

    Returns:
        List of forecast dictionaries (JSON format of forecast_to_dict)
    """
    runs = []
    for path in sorted(Path(archive_dir).glob("*/*.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue  # Half-written or damaged file
    return sorted(runs, key=lambda r: r["generated_at"])


def forecast_for_day(runs: List[Dict[str, Any]], day: date) -> Optional[Dict[str, Any]]:
    """
    The forecast that was shown for a day: the last run before the day began,
    or the first run of the day if there was none before.

    Args:
        runs: Archived forecasts, oldest first (load_archive)
        day: Day of interest (spot local time)

    Returns:
        {"generated_at": ..., "hours": [hours of that day]} or None
    """
    prefix = day.isoformat()

    def generated(run):
        return datetime.fromisoformat(run["generated_at"]).replace(tzinfo=None)

    covering = [
        r for r in runs if any(h["time"].startswith(prefix) for h in r["hours"])
    ]
    if not covering:
        return None
    before = [r for r in covering if generated(r).date() < day]
    run = before[-1] if before else covering[0]
    return {
        "generated_at": run["generated_at"],
        "hours": [h for h in run["hours"] if h["time"].startswith(prefix)],
    }


def load_measurements(config: Dict[str, Any], path: Path) -> pd.DataFrame:
    """
    Load measurements as hourly m/s values.

    Args:
        config: Spot configuration (target.column, optional
            target.direction_column and target.gust_column)
        path: Measurement CSV (paths["measurements"])

    Returns:
        DataFrame indexed by local time with speed, gust (m/s) and direction
    """
    target = config["target"]
    columns = {"speed": target["column"]}
    if target.get("gust_column"):
        columns["gust"] = target["gust_column"]
    if target.get("direction_column"):
        columns["direction"] = target["direction_column"]

    raw = pd.read_csv(path, usecols=["timestamp", *columns.values()])
    raw.index = pd.to_datetime(raw["timestamp"], format="%Y-%m-%d_%H%M")
    unit = measurement_unit(config)
    data = pd.DataFrame(index=raw.index)
    for name, column in columns.items():
        data[name] = raw[column] if name == "direction" else to_ms(raw[column], unit)
    # Autumn clock change: the repeated hour appears twice, keep the first
    return data[~data.index.duplicated()].sort_index()


def _measured(measurements: pd.DataFrame, time: str) -> Dict[str, Optional[float]]:
    t = pd.Timestamp(time)
    row = measurements.loc[t] if t in measurements.index else None

    def value(name, digits):
        if row is None or name not in row or pd.isna(row[name]):
            return None
        return round(float(row[name]), digits)

    return {
        "measured_speed": value("speed", 2),
        "measured_gust": value("gust", 2),
        "measured_direction": value("direction", 0),
    }


def history(
    runs: List[Dict[str, Any]],
    measurements: pd.DataFrame,
    today: date,
    days: int = 7,
    hours_of_day: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Past days (and today so far) as forecast and as measured.

    Args:
        runs: Archived forecasts (load_archive)
        measurements: Output of load_measurements()
        today: Last day to include (usually today)
        days: Number of days before today
        hours_of_day: Hours with measurements to include (e.g. the forecast
            window 6..23); all if None

    Returns:
        One entry per day with a forecast or measurements, oldest first:
        {"date", "forecast_generated_at", "hours": [{time, wind_speed,
        openmeteo_wind_speed, openmeteo_gusts, wind_direction,
        measured_speed, measured_gust, measured_direction}]}
    """
    result = []
    for offset in range(days, -1, -1):
        day = today - timedelta(days=offset)
        forecast = forecast_for_day(runs, day)
        hours = {h["time"]: dict(h) for h in (forecast["hours"] if forecast else [])}
        on_day = measurements[measurements.index.normalize() == pd.Timestamp(day)]
        if hours_of_day is not None:
            on_day = on_day[on_day.index.hour.isin(hours_of_day)]
        for t in on_day.index:
            hours.setdefault(
                t.strftime("%Y-%m-%dT%H:%M"), {"time": t.strftime("%Y-%m-%dT%H:%M")}
            )
        if not hours:
            continue
        rows = [{**hours[t], **_measured(measurements, t)} for t in sorted(hours)]
        result.append(
            {
                "date": day.isoformat(),
                "forecast_generated_at": forecast["generated_at"] if forecast else None,
                "hours": rows,
            }
        )
    return result


def with_measurements(
    forecast: Dict[str, Any], measurements: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Hours of the current forecast with the measurements available so far.

    Args:
        forecast: Forecast JSON (forecast_to_dict)
        measurements: Output of load_measurements()

    Returns:
        Hours of the forecast, each with measured_speed/gust/direction (None
        where nothing was measured yet)
    """
    return [{**h, **_measured(measurements, h["time"])} for h in forecast["hours"]]


CSV_COLUMNS = [
    ("time", "time"),
    ("wind_speed", "windmojo_m_s"),
    ("openmeteo_wind_speed", "openmeteo_m_s"),
    ("openmeteo_gusts", "openmeteo_gusts_m_s"),
    ("wind_direction", "openmeteo_direction_deg"),
    ("measured_speed", "measured_m_s"),
    ("measured_gust", "measured_gust_m_s"),
    ("measured_direction", "measured_direction_deg"),
]


def to_csv(
    hours: List[Dict[str, Any]], forecast_generated_at: Optional[str] = None
) -> str:
    """
    Hours as CSV (comma-separated, m/s, local time of the spot).

    Args:
        hours: Rows with the keys of CSV_COLUMNS (missing keys stay empty);
            a "forecast_generated_at" key per row overrides the argument
        forecast_generated_at: Creation time of the forecast, as last column

    Returns:
        CSV text with header
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([name for _, name in CSV_COLUMNS] + ["forecast_generated_at"])
    for h in hours:
        values = ["" if h.get(key) is None else h[key] for key, _ in CSV_COLUMNS]
        writer.writerow(
            values + [h.get("forecast_generated_at", forecast_generated_at) or ""]
        )
    return out.getvalue()

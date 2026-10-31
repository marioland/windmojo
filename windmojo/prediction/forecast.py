"""
Live Forecast

Downloads the current Open-Meteo forecast for a spot's 9-point grid, runs it
through the same feature pipeline as the training data and applies the trained
model. The result is exported as JSON for websites (see DEPLOYMENT.md).
"""

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytz

from windmojo.config import get_forecast_config, get_grid_config
from windmojo.data.grid import calculate_all_grids
from windmojo.data.grid_downloader import (
    CENTER_PRESSURE_LEVELS,
    CENTER_VARIABLES,
    CIRCLE_VARIABLES,
)
from windmojo.data.open_meteo_client import OpenMeteoClient
from windmojo.features.pipeline import FeatureEngineeringPipeline
from windmojo.units import KMH_TO_MS

logger = logging.getLogger(__name__)

# Longest time window among the model features (precipitation_rolling_7d_sum,
# MAX_HOURS_SINCE_EVENT in features/snow_features.py)
DEFAULT_HISTORY_DAYS = 7


def apply_forecast_window(
    raw: pd.DataFrame, initialization_time: str, horizon_hours: int
) -> pd.DataFrame:
    """
    Keep only the hours of each day that the training data covers.

    Training data holds, per day, the hours from `initialization_time` to
    `initialization_time + horizon_hours` (default 06:00-23:00 local time).
    Adds the metadata columns of the raw Parquet files (forecast_made, month,
    forecast_date).

    Args:
        raw: Long-format data with 'time' (local, naive) and 'location'
        initialization_time: Window start per day (HH:MM, local time)
        horizon_hours: Window length in hours

    Returns:
        Filtered DataFrame in the raw Parquet format
    """
    hour, minute = map(int, initialization_time.split(":"))
    day = raw["time"].dt.normalize()
    forecast_made = day + pd.Timedelta(hours=hour, minutes=minute)
    lead = raw["time"] - forecast_made
    in_window = (lead >= pd.Timedelta(0)) & (lead < pd.Timedelta(hours=horizon_hours))

    raw = raw.assign(
        forecast_made=forecast_made,
        month=raw["time"].dt.month.astype("int32"),
        forecast_date=day.dt.strftime("%Y-%m-%d"),
    )
    return raw[in_window].reset_index(drop=True)


def download_forecast(
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
    client: Optional[OpenMeteoClient] = None,
) -> pd.DataFrame:
    """
    Download forecast data for all 9 grid points.

    Same variables and grid as the training download (one API call per grid
    point for the whole date range). The Open-Meteo forecast API also serves
    the past days needed for lag and rolling features.

    Args:
        config: Spot configuration (simplified format)
        start_date: First day (YYYY-MM-DD), including history days
        end_date: Last forecast day (YYYY-MM-DD)
        client: Optional OpenMeteoClient (creates a new one if None)

    Returns:
        Long-format DataFrame in the raw Parquet format
    """
    client = client or OpenMeteoClient()
    forecast_cfg = get_forecast_config(config)
    timezone = config.get("timezone", "UTC")
    grids = calculate_all_grids(
        config["latitude"], config["longitude"], get_grid_config(config)["radius_km"]
    )

    frames = []
    for location, (lat, lon) in grids["grid"].items():
        if location == "center":
            variables = CENTER_VARIABLES + CENTER_PRESSURE_LEVELS
        else:
            variables = CIRCLE_VARIABLES
        df = client.fetch_forecast(lat, lon, start_date, end_date, variables, timezone)
        df["location"] = location
        frames.append(df)
        time.sleep(0.3)  # Same rate limiting as the training download

    raw = pd.concat(frames, ignore_index=True)
    logger.info(f"Downloaded {len(raw)} rows for {len(frames)} grid points")

    return apply_forecast_window(
        raw, forecast_cfg["initialization_time"], forecast_cfg["horizon_hours"]
    )


def predict_forecast(
    config: Dict[str, Any],
    trainer,
    today: Optional[date] = None,
    history_days: int = DEFAULT_HISTORY_DAYS,
    client: Optional[OpenMeteoClient] = None,
) -> pd.DataFrame:
    """
    Create a wind forecast from today on.

    Args:
        config: Spot configuration (simplified format)
        trainer: Loaded UnifiedTrainer
        today: First forecast day (default: today in the spot's timezone)
        history_days: Past days downloaded for lag and rolling features
        client: Optional OpenMeteoClient

    Returns:
        DataFrame with time, wind_speed (model, m/s) and the Open-Meteo
        values at the center point (openmeteo_wind_speed, openmeteo_gusts in
        m/s, wind_direction in degrees). attrs["missing_features"] lists the
        model features missing in the forecast data (filled with medians).
    """
    timezone = pytz.timezone(config.get("timezone", "UTC"))
    today = today or datetime.now(timezone).date()
    days_ahead = get_forecast_config(config)["days_ahead"]

    start = today - timedelta(days=history_days)
    end = today + timedelta(days=days_ahead - 1)
    logger.info(f"Forecast {today} to {end} (history from {start})")

    raw = download_forecast(config, start.isoformat(), end.isoformat(), client)
    features = FeatureEngineeringPipeline().transform(raw)
    features = features[features["time"].dt.date >= today].reset_index(drop=True)

    missing = [f for f in trainer.feature_columns if f not in features.columns]
    if missing:
        logger.warning(
            f"{len(missing)} model features missing in forecast data, "
            f"filled with training medians: {', '.join(missing[:5])}"
            f"{', ...' if len(missing) > 5 else ''}"
        )
        features = features.assign(**{f: np.nan for f in missing})

    center = raw[raw["location"] == "center"].set_index("time")
    result = pd.DataFrame(
        {
            "time": features["time"],
            "wind_speed": np.clip(trainer.predict(features), 0, None),
        }
    )
    result["openmeteo_wind_speed"] = (
        result["time"].map(center["wind_speed_10m"]) * KMH_TO_MS
    )
    result["openmeteo_gusts"] = result["time"].map(center["wind_gusts_10m"]) * KMH_TO_MS
    result["wind_direction"] = result["time"].map(center["wind_direction_10m"])
    result.attrs["missing_features"] = missing

    return result


def forecast_to_dict(
    result: pd.DataFrame,
    config: Dict[str, Any],
    trainer,
    generated_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Convert a forecast to the JSON structure published for websites.

    Args:
        result: Output of predict_forecast()
        config: Spot configuration
        trainer: Loaded UnifiedTrainer (for model metadata)
        generated_at: Creation time (default: now in the spot's timezone)

    Returns:
        JSON-serializable dictionary (format documented in DEPLOYMENT.md)
    """
    timezone = pytz.timezone(config.get("timezone", "UTC"))
    generated_at = generated_at or datetime.now(timezone)

    def value(x, digits=1):
        return None if pd.isna(x) else round(float(x), digits or None)

    hours = [
        {
            "time": row.time.strftime("%Y-%m-%dT%H:%M"),
            "wind_speed": value(row.wind_speed),
            "openmeteo_wind_speed": value(row.openmeteo_wind_speed),
            "openmeteo_gusts": value(row.openmeteo_gusts),
            "wind_direction": value(row.wind_direction, 0),
        }
        for row in result.itertuples()
    ]

    return {
        "spot": config["spot"],
        "name": config.get("name", config["spot"]),
        "latitude": config["latitude"],
        "longitude": config["longitude"],
        "timezone": config.get("timezone", "UTC"),
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "units": {"wind_speed": "m/s", "wind_direction": "deg"},
        "model": {
            "type": "unified",
            "test_mae": trainer.training_history.get("test", {}).get("mae"),
            "missing_features": len(result.attrs.get("missing_features", [])),
        },
        "source": "Weather data by Open-Meteo.com (CC BY 4.0)",
        "hours": hours,
    }


def save_forecast_json(data: Dict[str, Any], output_path: Path) -> None:
    """
    Write forecast JSON atomically.

    Writes to a temporary file and renames it, so a web server never serves a
    half-written file.

    Args:
        data: Output of forecast_to_dict()
        output_path: Target file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    tmp_path.replace(output_path)

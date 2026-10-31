"""
Configuration Loading and Validation

Load YAML configuration files for windmojo data downloading and model training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from windmojo.data.grid import validate_coordinates

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigurationError: If config is invalid

    Example:
        >>> config = load_config('config/lake_como/dervio.yaml')
        >>> config['area']
        'lake_como'
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML: {e}")

    # Validate configuration
    validate_config(config)

    # Expand template strings in output paths
    if "output" in config:
        area = config.get("area", "")
        spot = config.get("spot", "")

        for key in ["raw_data_dir", "features_dir", "forecasts_dir"]:
            if key in config["output"]:
                config["output"][key] = config["output"][key].format(
                    area=area, spot=spot
                )

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Required top-level fields
    required_fields = ["area", "spot", "center"]

    for field in required_fields:
        if field not in config:
            raise ConfigurationError(f"Missing required field: '{field}'")

    # Validate center coordinates
    if "center" in config:
        center = config["center"]

        if "latitude" not in center:
            raise ConfigurationError("Missing 'center.latitude'")

        if "longitude" not in center:
            raise ConfigurationError("Missing 'center.longitude'")

        try:
            lat = float(center["latitude"])
            lon = float(center["longitude"])
            validate_coordinates(lat, lon)
        except (ValueError, TypeError) as e:
            raise ConfigurationError(f"Invalid center coordinates: {e}")

    # Validate grid distances if present
    if "grid" in config:
        grid = config["grid"]

        for key in ["near_distance_km", "far_distance_km"]:
            if key in grid:
                try:
                    distance = float(grid[key])
                    if distance <= 0:
                        raise ValueError("Distance must be positive")
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"Invalid {key}: {e}")

    # Validate timezone if present
    if "timezone" in config:
        import pytz

        try:
            pytz.timezone(config["timezone"])
        except pytz.exceptions.UnknownTimeZoneError:
            raise ConfigurationError(f"Unknown timezone: {config['timezone']}")

    # Validate forecast configuration if present
    if "forecast" in config:
        forecast = config["forecast"]

        if "time" in forecast:
            # Validate time format (HH:MM)
            import re

            if not re.match(r"^\d{2}:\d{2}$", forecast["time"]):
                raise ConfigurationError(
                    f"Invalid forecast time format: {forecast['time']} (expected HH:MM)"
                )

            hour, minute = map(int, forecast["time"].split(":"))
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ConfigurationError(
                    f"Invalid forecast time: {forecast['time']} (hour must be 0-23, minute 0-59)"
                )

        for key in ["horizon_hours", "forecast_days"]:
            if key in forecast:
                try:
                    value = int(forecast[key])
                    if value <= 0:
                        raise ValueError("Must be positive")
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"Invalid {key}: {e}")

    # Validate training configuration if present
    if "training" in config:
        training = config["training"]

        if "test_split" in training:
            try:
                split = float(training["test_split"])
                if not 0 < split < 1:
                    raise ValueError("Must be between 0 and 1")
            except (ValueError, TypeError) as e:
                raise ConfigurationError(f"Invalid test_split: {e}")


def get_grid_distances(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Get grid distances from configuration with defaults.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with keys: radius_km
    """
    defaults = {"radius_km": 20.0}

    if "grid" in config:
        if "radius_km" in config["grid"]:
            defaults["radius_km"] = float(config["grid"]["radius_km"])

    return defaults


def get_output_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Get output directory paths from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with keys: raw_data_dir, features_dir, forecasts_dir
    """
    if "output" not in config:
        # Default output structure
        area = config["area"]
        spot = config["spot"]
        return {
            "raw_data_dir": Path(f"data/{area}/{spot}/raw"),
            "features_dir": Path(f"data/{area}/{spot}/features"),
            "forecasts_dir": Path(f"data/{area}/{spot}/forecasts"),
        }

    output = config["output"]
    return {
        "raw_data_dir": Path(
            output.get("raw_data_dir", f'data/{config["area"]}/{config["spot"]}/raw')
        ),
        "features_dir": Path(
            output.get(
                "features_dir", f'data/{config["area"]}/{config["spot"]}/features'
            )
        ),
        "forecasts_dir": Path(
            output.get(
                "forecasts_dir", f'data/{config["area"]}/{config["spot"]}/forecasts'
            )
        ),
    }


def get_forecast_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get forecast configuration with defaults.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with forecast parameters
    """
    defaults = {"time": "06:00", "horizon_hours": 18, "forecast_days": 7}

    if "forecast" in config:
        defaults.update(config["forecast"])

    return defaults


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print human-readable configuration summary.

    Args:
        config: Configuration dictionary
    """
    logger.info("Configuration Summary:")
    logger.info("=" * 60)
    logger.info(f"  Area: {config['area']}")
    logger.info(f"  Spot: {config['spot']}")
    logger.info(
        f"  Center: {config['center']['latitude']}, {config['center']['longitude']}"
    )

    if "name" in config["center"]:
        logger.info(f"  Location name: {config['center']['name']}")

    if "grid" in config:
        grid = config["grid"]
        logger.info(f"  Grid distances:")
        logger.info(f"    Near: {grid.get('near_distance_km', 15)} km")
        logger.info(f"    Far: {grid.get('far_distance_km', 150)} km")

    if "timezone" in config:
        logger.info(f"  Timezone: {config['timezone']}")

    if "forecast" in config:
        forecast = config["forecast"]
        logger.info(f"  Forecast configuration:")
        logger.info(f"    Initialization time: {forecast.get('time', '06:00')}")
        logger.info(f"    Horizon: {forecast.get('horizon_hours', 18)} hours")
        logger.info(f"    Days ahead: {forecast.get('forecast_days', 7)}")

    if "measurement_file" in config:
        logger.info(f"  Measurement file: {config['measurement_file']}")

    if "training" in config:
        training = config["training"]
        logger.info(f"  Training configuration:")
        logger.info(f"    Test split: {training.get('test_split', 0.15)}")

    if "validation" in config:
        validation = config["validation"]
        logger.info(f"  Validation configuration:")
        logger.info(
            f"    Strategy: {validation.get('strategy', 'first_n_days_every_nth_month')}"
        )
        logger.info(f"    Days per month: {validation.get('n_days', 4)}")
        logger.info(f"    Every Nth month: {validation.get('nth_month', 2)}")

    logger.info("=" * 60)

"""
Configuration Loading and Validation

Simplified configuration for windmojo single-model approach.
No area/spot duality, no phase 1/2 separation, clean paths.
"""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigurationError: If config is invalid
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

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Required fields
    required_fields = ["spot", "latitude", "longitude"]

    for field in required_fields:
        if field not in config:
            raise ConfigurationError(f"Missing required field: '{field}'")

    # Validate coordinates
    try:
        lat = float(config["latitude"])
        lon = float(config["longitude"])

        if not -90 <= lat <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        if not -180 <= lon <= 180:
            raise ValueError("Longitude must be between -180 and 180")
    except (ValueError, TypeError) as e:
        raise ConfigurationError(f"Invalid coordinates: {e}")

    # Validate grid if present
    if "grid" in config:
        grid = config["grid"]
        if "radius_km" in grid:
            try:
                radius = float(grid["radius_km"])
                if radius <= 0:
                    raise ValueError("Radius must be positive")
            except (ValueError, TypeError) as e:
                raise ConfigurationError(f"Invalid grid radius: {e}")

    # Validate timezone if present
    if "timezone" in config:
        import pytz

        try:
            pytz.timezone(config["timezone"])
        except pytz.exceptions.UnknownTimeZoneError:
            raise ConfigurationError(f"Unknown timezone: {config['timezone']}")


def get_spot_dir(spot: str) -> Path:
    """
    Get spot directory path.

    Args:
        spot: Spot name

    Returns:
        Path to spot directory (data/spots/{spot}/)
    """
    return Path(f"data/spots/{spot}")


def get_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Get all file paths for a spot based on config.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with all relevant paths
    """
    spot = config["spot"]
    spot_dir = get_spot_dir(spot)

    # Default paths (can be overridden in config)
    defaults = {
        "measurements": "input/measurements.csv",
        "config": "input/config.yaml",
        "raw_data": "training/raw",
        "features": "training/features",
        "validation": "training/validation",
        "model": "models/model.json",
        "metadata": "models/metadata.json",
    }

    # Override with custom paths from config if present
    if "paths" in config:
        defaults.update(config["paths"])

    # Convert to absolute paths
    return {key: spot_dir / path for key, path in defaults.items()}


def get_grid_config(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Get grid configuration with defaults.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with grid parameters
    """
    defaults = {"radius_km": 20.0}

    if "grid" in config:
        defaults.update(config["grid"])

    return defaults


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get training configuration with defaults.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with training parameters
    """
    defaults = {
        "model": "spatial_heavy_unified",
        "test_dates": {
            "strategy": "first_n_days_every_nth_month",
            "n_days": 4,
            "nth_month": 2,
        },
        "xgboost": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "random_state": 42,
            "tree_method": "hist",
            "early_stopping_rounds": 50,
        },
    }

    if "training" in config:
        if "test_dates" in config["training"]:
            defaults["test_dates"].update(config["training"]["test_dates"])
        if "xgboost" in config["training"]:
            defaults["xgboost"].update(config["training"]["xgboost"])
        if "model" in config["training"]:
            defaults["model"] = config["training"]["model"]

    return defaults


def get_test_dates(
    config: Dict[str, Any], available_dates: List[datetime.date]
) -> List[datetime.date]:
    """
    Get test dates based on training configuration.

    Args:
        config: Configuration dictionary
        available_dates: List of available datetime.date objects

    Returns:
        List of datetime.date objects to use for testing/validation
    """
    training = get_training_config(config)
    test_config = training["test_dates"]
    strategy = test_config["strategy"]

    # Custom dates
    if strategy == "custom" and "custom_dates" in test_config:
        custom = test_config["custom_dates"]
        if isinstance(custom[0], str):
            return [datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in custom]
        return custom

    # Group dates by month
    dates_by_month = {}
    for date in available_dates:
        key = (date.year, date.month)
        if key not in dates_by_month:
            dates_by_month[key] = []
        dates_by_month[key].append(date)

    # Select months based on nth_month
    nth_month = test_config["nth_month"]
    selected_months = []
    for i, (key, dates) in enumerate(sorted(dates_by_month.items())):
        if i % nth_month == 0:  # Every nth month
            selected_months.append((key, sorted(dates)))

    # Select n_days from each selected month
    n_days = test_config["n_days"]
    test_dates = []

    for key, month_dates in selected_months:
        if strategy == "first_n_days_every_nth_month":
            test_dates.extend(month_dates[:n_days])
        elif strategy == "last_n_days_every_nth_month":
            test_dates.extend(month_dates[-n_days:])

    return sorted(test_dates)


def get_forecast_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get forecast configuration with defaults.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with forecast parameters
    """
    defaults = {"initialization_time": "06:00", "horizon_hours": 18, "days_ahead": 7}

    if "forecast" in config:
        defaults.update(config["forecast"])

    return defaults


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print human-readable configuration summary.

    Args:
        config: Configuration dictionary
    """
    logger.info("Windmojo Configuration Summary:")
    logger.info("=" * 60)
    logger.info(f"  Spot: {config['spot']}")
    if "name" in config:
        logger.info(f"  Name: {config['name']}")
    logger.info(f"  Location: {config['latitude']}, {config['longitude']}")

    if "timezone" in config:
        logger.info(f"  Timezone: {config['timezone']}")

    grid = get_grid_config(config)
    logger.info(f"  Grid:")
    logger.info(f"    Radius: {grid['radius_km']} km (9-point circular)")

    training = get_training_config(config)
    logger.info(f"  Training:")
    logger.info(f"    Model: {training['model']}")
    test_cfg = training["test_dates"]
    logger.info(f"    Test strategy: {test_cfg['strategy']}")
    if test_cfg["strategy"] != "custom":
        logger.info(
            f"    Test days: {test_cfg['n_days']} days every {test_cfg['nth_month']} months"
        )

    if "target" in config:
        target = config["target"]
        logger.info(f"  Target:")
        logger.info(f"    Column: {target.get('column', 'Not specified')}")
        if "direction_column" in target:
            logger.info(f"    Direction: {target['direction_column']}")

    forecast = get_forecast_config(config)
    logger.info(f"  Forecast:")
    logger.info(f"    Initialization: {forecast['initialization_time']}")
    logger.info(f"    Horizon: {forecast['horizon_hours']} hours")
    logger.info(f"    Days ahead: {forecast['days_ahead']}")

    logger.info("=" * 60)

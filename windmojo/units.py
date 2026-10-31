"""
Units

Raw weather data is stored in Open-Meteo's default units: wind speeds (and the
features derived from them, e.g. effective_wind_10m) in km/h. Measurements can be
in any unit of SPEED_TO_MS, set in the config (`target.unit`, legacy configs:
`measurement_unit`; default m/s). The simplified pipeline converts measurements
to m/s on loading, so models, metrics and forecasts are always in m/s.
"""

from typing import Any, Dict

KMH_TO_MS = 1 / 3.6

# Factor to convert a wind speed in the given unit to m/s
SPEED_TO_MS = {
    "m/s": 1.0,
    "km/h": KMH_TO_MS,
    "kn": 1852 / 3600,
    "mph": 0.44704,
}


def measurement_unit(config: Dict[str, Any]) -> str:
    """
    Wind speed unit of the measurements.

    Args:
        config: Simplified (`target.unit`) or legacy (`measurement_unit`) config

    Returns:
        Unit key of SPEED_TO_MS

    Raises:
        ValueError: If the unit is not supported
    """
    unit = (config.get("target") or {}).get("unit") or config.get(
        "measurement_unit", "m/s"
    )
    if unit not in SPEED_TO_MS:
        raise ValueError(
            f"Unknown wind speed unit '{unit}', expected one of: "
            f"{', '.join(SPEED_TO_MS)}"
        )
    return unit


def to_ms(values, unit: str):
    """Convert wind speeds from `unit` to m/s (scalars, arrays, Series)."""
    return values * SPEED_TO_MS[unit]


def kmh_to_unit(values, unit: str):
    """Convert wind speeds from km/h (raw weather data) to `unit`."""
    return values * KMH_TO_MS / SPEED_TO_MS[unit]

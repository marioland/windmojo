"""
Grid Calculation Utilities

Calculate GPS coordinates for spatial grid patterns used in windmojo.
Supports 9-point circular grid pattern (8 compass directions + center).
"""

import logging
from math import cos, radians, sin, sqrt
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def calculate_offset_coordinates(
    center_lat: float, center_lon: float, offset_lat_km: float, offset_lon_km: float
) -> Tuple[float, float]:
    """
    Calculate GPS coordinates with km offsets from center point.

    Uses approximate conversion: 1° latitude ≈ 111 km, 1° longitude ≈ 111 km × cos(latitude)
    Accurate to ~1% for mid-latitudes (30-60°).

    Args:
        center_lat: Center point latitude (-90 to 90)
        center_lon: Center point longitude (-180 to 180)
        offset_lat_km: North/south offset in km (positive = north, negative = south)
        offset_lon_km: East/west offset in km (positive = east, negative = west)

    Returns:
        (latitude, longitude) tuple

    Example:
        >>> calculate_offset_coordinates(46.0, 9.0, 15, 0)
        (46.135135..., 9.0)  # 15 km north
    """
    # 1 degree latitude ≈ 111 km
    delta_lat = offset_lat_km / 111.0

    # 1 degree longitude ≈ 111 km * cos(latitude)
    delta_lon = offset_lon_km / (111.0 * cos(radians(center_lat)))

    return (center_lat + delta_lat, center_lon + delta_lon)


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate approximate distance between two GPS coordinates.

    Uses simple Euclidean approximation in km. Accurate enough for
    distances < 200 km at mid-latitudes.

    Args:
        lat1, lon1: First coordinate
        lat2, lon2: Second coordinate

    Returns:
        Distance in kilometers

    Example:
        >>> calculate_distance_km(46.0, 9.0, 46.135, 9.0)
        15.0  # approximately
    """
    # Convert to km offsets
    lat_km = (lat2 - lat1) * 111.0
    lon_km = (lon2 - lon1) * 111.0 * cos(radians((lat1 + lat2) / 2))

    return sqrt(lat_km**2 + lon_km**2)


def calculate_9_point_circle(
    center_lat: float, center_lon: float, radius_km: float
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate 9-point circular grid coordinates around center point.

    Creates 9 points: 1 center + 8 surrounding points at equal distance (radius_km)
    positioned at the 8 compass directions: N, NE, E, SE, S, SW, W, NW

    All 8 perimeter points are equidistant from the center.

    Args:
        center_lat: Center point latitude
        center_lon: Center point longitude
        radius_km: Distance from center to all perimeter points (e.g., 20 km)

    Returns:
        Dictionary mapping location names to (lat, lon) tuples
        Keys: 'center', 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'

    Example:
        >>> grid = calculate_9_point_circle(46.083333, 9.3, 20)
        >>> grid['center']
        (46.083333, 9.3)
        >>> grid['n']  # 20 km north
        (46.263333..., 9.3)
    """
    # Calculate 8 compass directions + center
    # Using trigonometry to ensure equal distances from center
    # Angles: N=0°, NE=45°, E=90°, SE=135°, S=180°, SW=225°, W=270°, NW=315°

    compass_points = {
        "n": 0,  # North (0°)
        "ne": 45,  # Northeast (45°)
        "e": 90,  # East (90°)
        "se": 135,  # Southeast (135°)
        "s": 180,  # South (180°)
        "sw": 225,  # Southwest (225°)
        "w": 270,  # West (270°)
        "nw": 315,  # Northwest (315°)
    }

    grid = {"center": (center_lat, center_lon)}

    for name, angle_deg in compass_points.items():
        # Convert compass bearing to mathematical angle
        # Compass: N=0°, clockwise; Math: E=0°, counter-clockwise
        # Math angle = 90° - compass bearing
        math_angle_rad = radians(90 - angle_deg)

        # Calculate offset in km
        lat_offset = radius_km * sin(math_angle_rad)
        lon_offset = radius_km * cos(math_angle_rad)

        grid[name] = calculate_offset_coordinates(
            center_lat, center_lon, lat_offset, lon_offset
        )

    return grid


def calculate_all_grids(
    center_lat: float, center_lon: float, radius_km: float = 20
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Calculate all grid patterns needed for windmojo.

    Returns 9-point circular grid (8 compass directions + center) for all variables.

    Args:
        center_lat: Center point latitude
        center_lon: Center point longitude
        radius_km: Radius distance for circular grid (default: 20 km)

    Returns:
        Dictionary with keys:
            - 'grid': 9-point circular grid locations
            - 'center': Single center point (for backward compatibility)

    Example:
        >>> grids = calculate_all_grids(46.083333, 9.3)
        >>> len(grids['grid'])
        9
        >>> grids['center']
        {'center': (46.083333, 9.3)}
    """
    # Calculate 9-point circular grid for all variables
    circular_grid = calculate_9_point_circle(center_lat, center_lon, radius_km)

    # Center point only (for backward compatibility and single-point operations)
    center_grid = {"center": (center_lat, center_lon)}

    return {"grid": circular_grid, "center": center_grid}


def get_variable_locations(
    variable_name: str, grids: Dict[str, Dict[str, Tuple[float, float]]]
) -> Dict[str, Tuple[float, float]]:
    """
    Get the appropriate grid locations for a given variable.

    All variables in the 9-point grid list use the full circular grid.
    Other variables default to center-only.

    Args:
        variable_name: Variable name (e.g., 'pressure_msl', 'temperature_2m')
        grids: Output from calculate_all_grids()

    Returns:
        Dictionary of location name → (lat, lon) for this variable

    Example:
        >>> grids = calculate_all_grids(46.0, 9.0)
        >>> locs = get_variable_locations('temperature_2m', grids)
        >>> len(locs)
        9
        >>> locs = get_variable_locations('wind_speed_100m', grids)
        >>> len(locs)
        1
    """
    # Variables that use full 9-point circular grid
    grid_variables = [
        "temperature_2m",
        "pressure_msl",
        "cloud_cover",
        "precipitation",
        "solar_radiation",
        "wind_speed_10m",
        "wind_direction_10m",
        "temperature_850hPa",
        "geopotential_height_500hPa",
    ]

    if variable_name in grid_variables:
        return grids["grid"]
    else:
        # All other variables default to center-only
        return grids["center"]


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate GPS coordinates are in valid range.

    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)

    Returns:
        True if valid, raises ValueError if invalid
    """
    if not -90 <= lat <= 90:
        raise ValueError(f"Invalid latitude: {lat} (must be -90 to 90)")

    if not -180 <= lon <= 180:
        raise ValueError(f"Invalid longitude: {lon} (must be -180 to 180)")

    return True


def print_grid_summary(grids: Dict[str, Dict[str, Tuple[float, float]]]) -> None:
    """
    Print human-readable summary of calculated grids.

    Args:
        grids: Output from calculate_all_grids()
    """
    logger.info("Grid Summary:")
    logger.info("=" * 60)

    for grid_type, locations in grids.items():
        logger.info(f"{grid_type.upper()} Grid: {len(locations)} points")
        logger.info(f"{'Location':<20} {'Latitude':>12} {'Longitude':>12}")
        logger.info("-" * 60)

        for name, (lat, lon) in sorted(locations.items()):
            logger.info(f"{name:<20} {lat:>12.6f} {lon:>12.6f}")

    # Calculate total unique locations
    all_coords = set()
    for locations in grids.values():
        all_coords.update(locations.values())

    logger.info(f"Total unique GPS coordinates: {len(all_coords)}")
    logger.info("=" * 60)

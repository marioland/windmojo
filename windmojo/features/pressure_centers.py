"""
Pressure Center Detection and Parameterization

Detects high and low pressure systems in geopotential height fields
and extracts features describing their position, intensity, and geometry.

This implements Approach #1 from GENERAL_WEATHER_SITUATIONS_FOR_ML.md
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class PressureCenterDetector:
    """
    Detect and parameterize pressure centers in synoptic grid data.

    Uses 500 hPa geopotential height as primary variable (less affected
    by topography than surface pressure).
    """

    def __init__(
        self,
        n_highs: int = 3,
        n_lows: int = 3,
        smoothing_sigma: float = 2.0,
        min_distance: int = 5,
        sort_by: str = "strength",
    ):
        """
        Initialize pressure center detector.

        Args:
            n_highs: Number of high-pressure systems to detect
            n_lows: Number of low-pressure systems to detect
            smoothing_sigma: Gaussian filter sigma for noise reduction
            min_distance: Minimum distance between centers (grid points)
            sort_by: How to sort detected centers ('strength' or 'distance')
                - 'strength': Sort by intensity (z500) - h1=strongest, l1=strongest
                - 'distance': Sort by distance to target - h1=nearest, l1=nearest
        """
        self.n_highs = n_highs
        self.n_lows = n_lows
        self.smoothing_sigma = smoothing_sigma
        self.min_distance = min_distance
        self.sort_by = sort_by

    def detect_and_parameterize(
        self,
        grid_df: pd.DataFrame,
        target_lat: float = 46.0,
        target_lon: float = 9.3,
        return_centers: bool = False,
    ) -> pd.DataFrame:
        """
        Detect pressure centers and extract features.

        Args:
            grid_df: DataFrame with columns: lat, lon, geopotential_height_500hPa
            target_lat: Latitude of forecast location (for distance calculation)
            target_lon: Longitude of forecast location
            return_centers: If True, return (features_df, highs_list, lows_list) tuple

        Returns:
            DataFrame with 1 row containing all pressure center features
            OR tuple of (DataFrame, highs_list, lows_list) if return_centers=True
        """
        features = {}

        # Convert to grid
        grid, lats, lons = self._dataframe_to_grid(
            grid_df, "geopotential_height_500hPa"
        )

        if grid is None:
            logger.warning("Failed to convert grid data, returning empty features")
            if return_centers:
                return self._empty_features(), [], []
            return self._empty_features()

        # Smooth field
        grid_smooth = gaussian_filter(grid, sigma=self.smoothing_sigma)

        # Detect highs and lows
        highs = self._find_local_maxima(grid_smooth, lats, lons)
        lows = self._find_local_minima(grid_smooth, lats, lons)

        # Calculate distance to target for all centers
        for center in highs + lows:
            center["distance_to_target"] = self._haversine_distance(
                center["lat"], center["lon"], target_lat, target_lon
            )

        # Sort by chosen criterion and take top N
        if self.sort_by == "distance":
            # h1 = nearest high, l1 = nearest low
            highs = sorted(highs, key=lambda x: x["distance_to_target"])[: self.n_highs]
            lows = sorted(lows, key=lambda x: x["distance_to_target"])[: self.n_lows]
            logger.debug(
                f"Detected {len(highs)} highs, {len(lows)} lows (sorted by distance)"
            )
        else:  # 'strength' (default)
            # h1 = strongest high, l1 = strongest low
            highs = sorted(highs, key=lambda x: x["z500"], reverse=True)[: self.n_highs]
            lows = sorted(lows, key=lambda x: x["z500"])[: self.n_lows]
            logger.debug(
                f"Detected {len(highs)} highs, {len(lows)} lows (sorted by strength)"
            )

        # Parameterize each center
        for i, high in enumerate(highs, 1):
            prefix = f"high{i}"
            features.update(
                self._parameterize_center(
                    high, grid_smooth, lats, lons, target_lat, target_lon, prefix
                )
            )

        for i, low in enumerate(lows, 1):
            prefix = f"low{i}"
            features.update(
                self._parameterize_center(
                    low, grid_smooth, lats, lons, target_lat, target_lon, prefix
                )
            )

        # Fill missing features with zeros (if fewer centers detected than expected)
        for i in range(len(highs) + 1, self.n_highs + 1):
            features.update(self._empty_center_features(f"high{i}"))

        for i in range(len(lows) + 1, self.n_lows + 1):
            features.update(self._empty_center_features(f"low{i}"))

        features_df = pd.DataFrame([features])

        if return_centers:
            return features_df, highs, lows
        return features_df

    def _dataframe_to_grid(
        self, df: pd.DataFrame, variable: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Convert DataFrame to regular grid.

        Args:
            df: DataFrame with lat, lon, and variable columns
            variable: Variable name to extract

        Returns:
            Tuple of (grid, lats, lons) or (None, None, None) if failed
        """
        if variable not in df.columns:
            logger.error(f"Variable {variable} not in DataFrame")
            return None, None, None

        # Get unique sorted coordinates
        lats = np.sort(df["lat"].unique())
        lons = np.sort(df["lon"].unique())

        # Create grid
        grid = np.full((len(lats), len(lons)), np.nan)

        # Fill grid
        for _, row in df.iterrows():
            lat_idx = np.searchsorted(lats, row["lat"])
            lon_idx = np.searchsorted(lons, row["lon"])
            grid[lat_idx, lon_idx] = row[variable]

        # Check for missing values
        if np.isnan(grid).sum() > 0:
            logger.warning(f"Grid has {np.isnan(grid).sum()} missing values")

        return grid, lats, lons

    def _find_local_maxima(
        self, grid: np.ndarray, lats: np.ndarray, lons: np.ndarray
    ) -> List[Dict]:
        """
        Find local maxima (high-pressure centers) in grid.

        Args:
            grid: 2D array of values
            lats: 1D array of latitudes
            lons: 1D array of longitudes

        Returns:
            List of dicts with 'lat', 'lon', 'z500' keys
        """
        maxima = []

        # Simple local maximum detection: point higher than all 8 neighbors
        for i in range(1, grid.shape[0] - 1):
            for j in range(1, grid.shape[1] - 1):
                value = grid[i, j]

                # Skip NaN
                if np.isnan(value):
                    continue

                # Check if local maximum
                neighbors = grid[i - 1 : i + 2, j - 1 : j + 2]
                if value == np.nanmax(neighbors):
                    maxima.append(
                        {"lat": lats[i], "lon": lons[j], "z500": value, "i": i, "j": j}
                    )

        return maxima

    def _find_local_minima(
        self, grid: np.ndarray, lats: np.ndarray, lons: np.ndarray
    ) -> List[Dict]:
        """
        Find local minima (low-pressure centers) in grid.

        Args:
            grid: 2D array of values
            lats: 1D array of latitudes
            lons: 1D array of longitudes

        Returns:
            List of dicts with 'lat', 'lon', 'z500' keys
        """
        minima = []

        # Simple local minimum detection: point lower than all 8 neighbors
        for i in range(1, grid.shape[0] - 1):
            for j in range(1, grid.shape[1] - 1):
                value = grid[i, j]

                # Skip NaN
                if np.isnan(value):
                    continue

                # Check if local minimum
                neighbors = grid[i - 1 : i + 2, j - 1 : j + 2]
                if value == np.nanmin(neighbors):
                    minima.append(
                        {"lat": lats[i], "lon": lons[j], "z500": value, "i": i, "j": j}
                    )

        return minima

    def _parameterize_center(
        self,
        center: Dict,
        grid: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        target_lat: float,
        target_lon: float,
        prefix: str,
    ) -> Dict:
        """
        Extract features from a pressure center.

        Args:
            center: Dict with lat, lon, z500, i, j
            grid: Full grid array
            lats, lons: Coordinate arrays
            target_lat, target_lon: Forecast location
            prefix: Feature name prefix (e.g., 'high1', 'low2')

        Returns:
            Dict of features
        """
        features = {}

        # Position
        features[f"{prefix}_lat"] = center["lat"]
        features[f"{prefix}_lon"] = center["lon"]

        # Intensity
        features[f"{prefix}_z500"] = center["z500"]

        # Distance to target location (Lake Como)
        dist_km = self._haversine_distance(
            center["lat"], center["lon"], target_lat, target_lon
        )
        features[f"{prefix}_dist_km"] = dist_km

        # Bearing to target (in degrees, 0=North, 90=East)
        bearing = self._calculate_bearing(
            center["lat"], center["lon"], target_lat, target_lon
        )
        features[f"{prefix}_bearing"] = bearing

        # Gradient strength (average gradient in 5x5 window)
        i, j = center["i"], center["j"]
        window = grid[
            max(0, i - 2) : min(grid.shape[0], i + 3),
            max(0, j - 2) : min(grid.shape[1], j + 3),
        ]
        if window.size > 0:
            gradient = np.nanmax(window) - np.nanmin(window)
            features[f"{prefix}_gradient"] = gradient
        else:
            features[f"{prefix}_gradient"] = 0.0

        # Binary exists flag
        features[f"{prefix}_exists"] = 1

        return features

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate distance between two points on Earth.

        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in km
        r = 6371

        return c * r

    def _calculate_bearing(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate bearing from point 1 to point 2.

        Args:
            lat1, lon1: From point (degrees)
            lat2, lon2: To point (degrees)

        Returns:
            Bearing in degrees (0=North, 90=East, 180=South, 270=West)
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(x, y))

        # Normalize to 0-360
        bearing = (bearing + 360) % 360

        return bearing

    def _empty_center_features(self, prefix: str) -> Dict:
        """
        Return empty features for a non-existent center.

        Args:
            prefix: Feature name prefix

        Returns:
            Dict of features with zero/null values
        """
        return {
            f"{prefix}_lat": 0.0,
            f"{prefix}_lon": 0.0,
            f"{prefix}_z500": 0.0,
            f"{prefix}_dist_km": 0.0,
            f"{prefix}_bearing": 0.0,
            f"{prefix}_gradient": 0.0,
            f"{prefix}_exists": 0,
        }

    def _empty_features(self) -> pd.DataFrame:
        """
        Return empty feature DataFrame when detection fails.

        Returns:
            DataFrame with 1 row of empty features
        """
        features = {}

        for i in range(1, self.n_highs + 1):
            features.update(self._empty_center_features(f"high{i}"))

        for i in range(1, self.n_lows + 1):
            features.update(self._empty_center_features(f"low{i}"))

        return pd.DataFrame([features])

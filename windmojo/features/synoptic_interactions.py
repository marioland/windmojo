"""
Synoptic Pattern Interactions - Location-Agnostic Features

Calculates features describing relationships between pressure systems
without reference to specific geographic locations. These features capture
fundamental atmospheric patterns that work anywhere.

Examples:
- Distance between high and low pressure systems
- Pressure difference between systems
- Spatial dispersion of pressure centers
- High-low configuration patterns
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SynopticInteractionCalculator:
    """
    Calculate location-agnostic features from pressure center interactions.

    All features are generic atmospheric patterns that don't depend on
    knowing the forecast location or specific geographic features.
    """

    def __init__(self):
        """Initialize calculator."""
        pass

    def calculate_from_centers(
        self, highs: List[Dict], lows: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate interaction features from pressure centers.

        Args:
            highs: List of high pressure centers (dicts with 'lat', 'lon', 'z500')
            lows: List of low pressure centers (dicts with 'lat', 'lon', 'z500')

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # Calculate all feature groups
        features.update(self._high_low_interactions(highs, lows))
        features.update(self._strength_ratios(highs, lows))
        features.update(self._pattern_geometry(highs, lows))
        features.update(self._configuration_counts(highs, lows))

        return features

    def _high_low_interactions(
        self, highs: List[Dict], lows: List[Dict]
    ) -> Dict[str, float]:
        """
        Pairwise interactions between high and low pressure systems.

        Features (18 total):
        - h1_l1_distance, h1_l2_distance, h1_l3_distance
        - h2_l1_distance, h2_l2_distance, h2_l3_distance
        - h3_l1_distance, h3_l2_distance, h3_l3_distance
        - h1_l1_pressure_diff, h1_l2_pressure_diff, h1_l3_pressure_diff
        - h2_l1_pressure_diff, h2_l2_pressure_diff, h2_l3_pressure_diff
        - h3_l1_pressure_diff, h3_l2_pressure_diff, h3_l3_pressure_diff
        """
        features = {}

        # Calculate pairwise distances and pressure differences
        for i, high in enumerate(highs, 1):
            for j, low in enumerate(lows, 1):
                # Distance in km
                dist_km = self._haversine_distance(
                    high["lat"], high["lon"], low["lat"], low["lon"]
                )
                features[f"h{i}_l{j}_distance"] = dist_km

                # Pressure difference (geopotential height at 500 hPa)
                # Higher difference = stronger gradient = stronger winds
                pressure_diff = high["z500"] - low["z500"]
                features[f"h{i}_l{j}_pressure_diff"] = pressure_diff

        # Fill missing pairs with NaN if we don't have 3 highs and 3 lows
        for i in range(1, 4):
            for j in range(1, 4):
                if f"h{i}_l{j}_distance" not in features:
                    features[f"h{i}_l{j}_distance"] = np.nan
                    features[f"h{i}_l{j}_pressure_diff"] = np.nan

        return features

    def _strength_ratios(self, highs: List[Dict], lows: List[Dict]) -> Dict[str, float]:
        """
        Ratios comparing system strengths.

        Features (5 total):
        - high1_high2_ratio: Strength ratio of strongest to second strongest high
        - low1_low2_ratio: Strength ratio of strongest to second strongest low
        - high_mean_strength: Average geopotential height of high systems
        - low_mean_strength: Average geopotential height of low systems
        - high_low_balance: Ratio of mean high to mean low strength
        """
        features = {}

        # High strength ratios
        if len(highs) >= 2:
            features["high1_high2_ratio"] = highs[0]["z500"] / highs[1]["z500"]
        else:
            features["high1_high2_ratio"] = np.nan

        # Low strength ratios (for lows, lower is stronger, so we invert)
        if len(lows) >= 2:
            features["low1_low2_ratio"] = lows[0]["z500"] / lows[1]["z500"]
        else:
            features["low1_low2_ratio"] = np.nan

        # Mean strengths
        if len(highs) > 0:
            features["high_mean_strength"] = np.mean([h["z500"] for h in highs])
        else:
            features["high_mean_strength"] = np.nan

        if len(lows) > 0:
            features["low_mean_strength"] = np.mean([l["z500"] for l in lows])
        else:
            features["low_mean_strength"] = np.nan

        # Balance between highs and lows
        if len(highs) > 0 and len(lows) > 0:
            high_mean = features["high_mean_strength"]
            low_mean = features["low_mean_strength"]
            features["high_low_balance"] = high_mean / low_mean
        else:
            features["high_low_balance"] = np.nan

        return features

    def _pattern_geometry(
        self, highs: List[Dict], lows: List[Dict]
    ) -> Dict[str, float]:
        """
        Geometric properties of pressure pattern.

        Features (4 total):
        - high_dispersion: Spatial spread of high pressure systems (km)
        - low_dispersion: Spatial spread of low pressure systems (km)
        - pattern_lat_range: Latitude range of all systems (degrees)
        - pattern_lon_range: Longitude range of all systems (degrees)
        """
        features = {}

        # High dispersion (average distance from centroid)
        if len(highs) >= 2:
            high_lats = [h["lat"] for h in highs]
            high_lons = [h["lon"] for h in highs]
            high_centroid_lat = np.mean(high_lats)
            high_centroid_lon = np.mean(high_lons)

            high_distances = [
                self._haversine_distance(
                    h["lat"], h["lon"], high_centroid_lat, high_centroid_lon
                )
                for h in highs
            ]
            features["high_dispersion"] = np.mean(high_distances)
        else:
            features["high_dispersion"] = np.nan

        # Low dispersion
        if len(lows) >= 2:
            low_lats = [l["lat"] for l in lows]
            low_lons = [l["lon"] for l in lows]
            low_centroid_lat = np.mean(low_lats)
            low_centroid_lon = np.mean(low_lons)

            low_distances = [
                self._haversine_distance(
                    l["lat"], l["lon"], low_centroid_lat, low_centroid_lon
                )
                for l in lows
            ]
            features["low_dispersion"] = np.mean(low_distances)
        else:
            features["low_dispersion"] = np.nan

        # Overall pattern spatial extent
        all_systems = highs + lows
        if len(all_systems) >= 2:
            all_lats = [s["lat"] for s in all_systems]
            all_lons = [s["lon"] for s in all_systems]
            features["pattern_lat_range"] = max(all_lats) - min(all_lats)
            features["pattern_lon_range"] = max(all_lons) - min(all_lons)
        else:
            features["pattern_lat_range"] = np.nan
            features["pattern_lon_range"] = np.nan

        return features

    def _configuration_counts(
        self, highs: List[Dict], lows: List[Dict]
    ) -> Dict[str, float]:
        """
        Count-based configuration features.

        Features (3 total):
        - northern_highs: Number of high systems in northern half of domain (>50°N)
        - southern_lows: Number of low systems in southern half of domain (<50°N)
        - north_south_ratio: Ratio of northern highs to southern lows
        """
        features = {}

        # Northern highs (>50°N)
        northern_highs = sum(1 for h in highs if h["lat"] > 50.0)
        features["northern_highs"] = float(northern_highs)

        # Southern lows (<50°N)
        southern_lows = sum(1 for l in lows if l["lat"] < 50.0)
        features["southern_lows"] = float(southern_lows)

        # North-south ratio
        if southern_lows > 0:
            features["north_south_ratio"] = northern_highs / southern_lows
        else:
            features["north_south_ratio"] = np.nan

        return features

    @staticmethod
    def _haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate great circle distance between two points.

        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in km
        R = 6371.0

        return R * c

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names this calculator produces.

        Returns:
            List of 30 feature names
        """
        feature_names = []

        # High-low interactions (18)
        for i in range(1, 4):
            for j in range(1, 4):
                feature_names.append(f"h{i}_l{j}_distance")
        for i in range(1, 4):
            for j in range(1, 4):
                feature_names.append(f"h{i}_l{j}_pressure_diff")

        # Strength ratios (5)
        feature_names.extend(
            [
                "high1_high2_ratio",
                "low1_low2_ratio",
                "high_mean_strength",
                "low_mean_strength",
                "high_low_balance",
            ]
        )

        # Pattern geometry (4)
        feature_names.extend(
            [
                "high_dispersion",
                "low_dispersion",
                "pattern_lat_range",
                "pattern_lon_range",
            ]
        )

        # Configuration counts (3)
        feature_names.extend(["northern_highs", "southern_lows", "north_south_ratio"])

        return feature_names

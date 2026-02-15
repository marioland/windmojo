"""
Teleconnection Indices Calculator

Calculates standard meteorological teleconnection indices (NAO, EA, SCAND)
and Alpine-specific indices from synoptic grid data.

This implements Approach #4 from GENERAL_WEATHER_SITUATIONS_FOR_ML.md
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Standard teleconnection index points (approximate)
NAO_POINTS = {"azores": {"lat": 38, "lon": -26}, "iceland": {"lat": 65, "lon": -20}}

EA_POINTS = {
    "north": {"lat": 55, "lon": -10},  # UK/Ireland
    "south": {"lat": 40, "lon": -10},  # Portugal
}

SCAND_POINTS = {
    "scandinavia": {"lat": 65, "lon": 20},
    "central_europe": {"lat": 50, "lon": 10},
}

ALPINE_POINTS = {
    "genoa": {"lat": 44.4, "lon": 8.9},
    "milan": {"lat": 45.5, "lon": 9.2},
    "south_alps": {"lat": 44, "lon": 9},
    "north_alps": {"lat": 48, "lon": 9},
    "como": {"lat": 46, "lon": 9},
}


class TeleconnectionCalculator:
    """
    Calculate teleconnection indices from synoptic grid data.

    Computes standard meteorological indices (NAO, EA, SCAND) and
    Alpine-specific indices for Lake Como wind prediction.
    """

    def __init__(self):
        """Initialize teleconnection calculator."""
        pass

    def calculate(self, grid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all teleconnection indices.

        Args:
            grid_df: DataFrame with columns: lat, lon, pressure_msl,
                    geopotential_height_500hPa, geopotential_height_850hPa,
                    temperature_850hPa

        Returns:
            DataFrame with 1 row containing all teleconnection features
        """
        features = {}

        # NAO (North Atlantic Oscillation)
        nao = self._calculate_nao(grid_df)
        features.update(nao)

        # EA (East Atlantic Pattern)
        ea = self._calculate_ea(grid_df)
        features.update(ea)

        # SCAND (Scandinavian Pattern)
        scand = self._calculate_scand(grid_df)
        features.update(scand)

        # Alpine-specific indices
        alpine = self._calculate_alpine_indices(grid_df)
        features.update(alpine)

        return pd.DataFrame([features])

    def _calculate_nao(self, grid_df: pd.DataFrame) -> Dict:
        """
        Calculate NAO index from Azores-Iceland pressure difference.

        Args:
            grid_df: Synoptic grid DataFrame

        Returns:
            Dict with NAO features
        """
        # Get pressure at standard NAO points
        p_azores = self._get_nearest_value(
            grid_df,
            NAO_POINTS["azores"]["lat"],
            NAO_POINTS["azores"]["lon"],
            "pressure_msl",
        )

        p_iceland = self._get_nearest_value(
            grid_df,
            NAO_POINTS["iceland"]["lat"],
            NAO_POINTS["iceland"]["lon"],
            "pressure_msl",
        )

        # Calculate NAO index
        # Positive NAO = higher pressure in Azores, lower in Iceland
        nao_raw = (p_azores - p_iceland) / 100  # Scale to hPa, ~10 range

        return {
            "nao_index": nao_raw,
            "nao_positive": 1 if nao_raw > 0 else 0,  # Binary NAO phase
        }

    def _calculate_ea(self, grid_df: pd.DataFrame) -> Dict:
        """
        Calculate East Atlantic pattern index.

        Args:
            grid_df: Synoptic grid DataFrame

        Returns:
            Dict with EA features
        """
        # Get pressure at EA points
        p_north = self._get_nearest_value(
            grid_df,
            EA_POINTS["north"]["lat"],
            EA_POINTS["north"]["lon"],
            "pressure_msl",
        )

        p_south = self._get_nearest_value(
            grid_df,
            EA_POINTS["south"]["lat"],
            EA_POINTS["south"]["lon"],
            "pressure_msl",
        )

        # Calculate EA index
        ea_index = (p_north - p_south) / 100

        return {"ea_index": ea_index}

    def _calculate_scand(self, grid_df: pd.DataFrame) -> Dict:
        """
        Calculate Scandinavian blocking pattern index.

        Args:
            grid_df: Synoptic grid DataFrame

        Returns:
            Dict with SCAND features
        """
        # Get pressure at SCAND points
        p_scandi = self._get_nearest_value(
            grid_df,
            SCAND_POINTS["scandinavia"]["lat"],
            SCAND_POINTS["scandinavia"]["lon"],
            "pressure_msl",
        )

        p_central = self._get_nearest_value(
            grid_df,
            SCAND_POINTS["central_europe"]["lat"],
            SCAND_POINTS["central_europe"]["lon"],
            "pressure_msl",
        )

        # Calculate SCAND index
        scand_index = (p_scandi - p_central) / 100

        return {
            "scand_index": scand_index,
            "scandi_blocking": (
                1 if scand_index > 10 else 0
            ),  # >10 hPa difference = blocking
        }

    def _calculate_alpine_indices(self, grid_df: pd.DataFrame) -> Dict:
        """
        Calculate Alpine-specific indices for Lake Como wind prediction.

        Args:
            grid_df: Synoptic grid DataFrame

        Returns:
            Dict with Alpine features
        """
        features = {}

        # Genoa Low Index (Mediterranean cyclogenesis)
        # Positive = low pressure in Genoa Gulf (cyclone)
        p_genoa = self._get_nearest_value(
            grid_df,
            ALPINE_POINTS["genoa"]["lat"],
            ALPINE_POINTS["genoa"]["lon"],
            "pressure_msl",
        )

        p_milan = self._get_nearest_value(
            grid_df,
            ALPINE_POINTS["milan"]["lat"],
            ALPINE_POINTS["milan"]["lon"],
            "pressure_msl",
        )

        features["genoa_low_index"] = p_milan - p_genoa  # Positive = low in Genoa

        # Foehn Gradient: South-north pressure gradient over Alps
        # Positive = higher pressure south (Foehn precondition)
        p_south_alps = self._get_nearest_value(
            grid_df,
            ALPINE_POINTS["south_alps"]["lat"],
            ALPINE_POINTS["south_alps"]["lon"],
            "pressure_msl",
        )

        p_north_alps = self._get_nearest_value(
            grid_df,
            ALPINE_POINTS["north_alps"]["lat"],
            ALPINE_POINTS["north_alps"]["lon"],
            "pressure_msl",
        )

        foehn_gradient = p_south_alps - p_north_alps
        features["foehn_gradient"] = foehn_gradient

        # Alpine thermal stability (850 hPa - surface temperature)
        # Used for thermal wind / Breva prediction
        t_850 = self._get_nearest_value(
            grid_df,
            ALPINE_POINTS["como"]["lat"],
            ALPINE_POINTS["como"]["lon"],
            "temperature_850hPa",
            default=0.0,
        )

        # Assume surface temp ~15°C warmer than 850 hPa in unstable conditions
        # Negative lapse rate = unstable (good for thermal winds)
        # This is a simplified index without actual surface temp
        features["alpine_thermal_stability"] = t_850  # Lower = more unstable

        # Breva Favorable Conditions (binary)
        # Foehn gradient negative (low south) + unstable atmosphere
        breva_favorable = foehn_gradient < -2 and t_850 < 5
        features["breva_favorable"] = 1 if breva_favorable else 0

        return features

    def _get_nearest_value(
        self,
        grid_df: pd.DataFrame,
        target_lat: float,
        target_lon: float,
        variable: str,
        default: float = 0.0,
    ) -> float:
        """
        Get value at nearest grid point to target coordinates.

        Args:
            grid_df: Grid DataFrame
            target_lat: Target latitude
            target_lon: Target longitude
            variable: Variable name to extract
            default: Default value if not found

        Returns:
            Value at nearest grid point
        """
        if variable not in grid_df.columns:
            logger.warning(f"Variable {variable} not in grid")
            return default

        if grid_df.empty:
            logger.warning("Grid is empty")
            return default

        # Calculate distance to all grid points
        distances = np.sqrt(
            (grid_df["lat"] - target_lat) ** 2 + (grid_df["lon"] - target_lon) ** 2
        )

        # Find nearest point
        nearest_idx = distances.idxmin()

        value = grid_df.loc[nearest_idx, variable]

        # Handle NaN
        if pd.isna(value):
            logger.warning(f"NaN value at nearest point for {variable}")
            return default

        return float(value)

    def get_feature_names(self) -> list:
        """
        Get list of all feature names.

        Returns:
            List of feature names
        """
        return [
            "nao_index",
            "nao_positive",
            "ea_index",
            "scand_index",
            "scandi_blocking",
            "genoa_low_index",
            "foehn_gradient",
            "alpine_thermal_stability",
            "breva_favorable",
        ]

    def get_n_features(self) -> int:
        """
        Get number of features produced.

        Returns:
            Number of features
        """
        return len(self.get_feature_names())

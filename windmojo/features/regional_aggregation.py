"""
Regional Aggregation for Synoptic Features

Divides Europe into meteorologically meaningful regions and computes
statistics and cross-regional gradients.

This implements Approach #2 from GENERAL_WEATHER_SITUATIONS_FOR_ML.md
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default European regions (meteorologically significant)
DEFAULT_REGIONS = {
    "iceland": {"lat": (63, 67), "lon": (-25, -15)},  # Iceland Low region
    "azores": {"lat": (35, 40), "lon": (-30, -20)},  # Azores High region
    "atlantic": {"lat": (45, 55), "lon": (-20, -5)},  # Atlantic approach
    "scandinavia": {"lat": (60, 70), "lon": (5, 25)},  # Scandinavian High
    "central_europe": {"lat": (47, 53), "lon": (5, 20)},  # Central Europe
    "mediterranean": {"lat": (38, 45), "lon": (5, 20)},  # Mediterranean
    "alps_north": {"lat": (47, 49), "lon": (6, 14)},  # North of Alps
    "alps_south": {"lat": (44, 46), "lon": (7, 13)},  # South of Alps (includes Como)
}


class RegionalAggregator:
    """
    Aggregate synoptic grid data by meteorological regions.

    Computes regional means, standard deviations, and cross-regional
    gradients for teleconnection-like indices.
    """

    def __init__(self, regions: Dict = None):
        """
        Initialize regional aggregator.

        Args:
            regions: Dict of region definitions. If None, uses DEFAULT_REGIONS.
                    Format: {'name': {'lat': (min, max), 'lon': (min, max)}}
        """
        self.regions = regions or DEFAULT_REGIONS

    def calculate(self, grid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regional features from synoptic grid.

        Args:
            grid_df: DataFrame with columns: lat, lon, geopotential_height_500hPa,
                    geopotential_height_850hPa, temperature_850hPa, pressure_msl

        Returns:
            DataFrame with 1 row containing all regional features
        """
        features = {}

        # Variables to aggregate
        variables = {
            "pressure_msl": "pressure_msl",
            "z500": "geopotential_height_500hPa",
            "z850": "geopotential_height_850hPa",
            "temp_850": "temperature_850hPa",
        }

        # Compute regional statistics
        regional_stats = {}

        for region_name, bounds in self.regions.items():
            # Extract region data
            region_data = self._extract_region(grid_df, bounds)

            if region_data.empty:
                logger.warning(f"No data for region {region_name}")
                # Fill with zeros
                for var_short, var_long in variables.items():
                    if var_long in grid_df.columns:
                        features[f"{region_name}_{var_short}_mean"] = 0.0
                        features[f"{region_name}_{var_short}_std"] = 0.0
                regional_stats[region_name] = {}
                continue

            # Calculate statistics for each variable
            for var_short, var_long in variables.items():
                if var_long in region_data.columns:
                    mean = region_data[var_long].mean()
                    std = region_data[var_long].std()

                    features[f"{region_name}_{var_short}_mean"] = mean
                    features[f"{region_name}_{var_short}_std"] = std

                    # Store for cross-regional calculations
                    if region_name not in regional_stats:
                        regional_stats[region_name] = {}
                    regional_stats[region_name][var_short] = mean

        # Calculate cross-regional features (teleconnection-like indices)
        cross_regional = self._calculate_cross_regional(regional_stats)
        features.update(cross_regional)

        return pd.DataFrame([features])

    def _extract_region(self, grid_df: pd.DataFrame, bounds: Dict) -> pd.DataFrame:
        """
        Extract grid points within region bounds.

        Args:
            grid_df: Full grid DataFrame
            bounds: Dict with 'lat' and 'lon' tuples: {'lat': (min, max), 'lon': (min, max)}

        Returns:
            Filtered DataFrame
        """
        lat_min, lat_max = bounds["lat"]
        lon_min, lon_max = bounds["lon"]

        mask = (
            (grid_df["lat"] >= lat_min)
            & (grid_df["lat"] <= lat_max)
            & (grid_df["lon"] >= lon_min)
            & (grid_df["lon"] <= lon_max)
        )

        return grid_df[mask]

    def _calculate_cross_regional(self, regional_stats: Dict) -> Dict:
        """
        Calculate cross-regional gradients and indices.

        Args:
            regional_stats: Dict of {region_name: {variable: mean_value}}

        Returns:
            Dict of cross-regional features
        """
        features = {}

        # Helper function to safely get regional value
        def get_val(region, var, default=0.0):
            return regional_stats.get(region, {}).get(var, default)

        # NAO Proxy: Azores - Iceland pressure difference
        # Positive NAO = high in Azores, low in Iceland
        azores_p = get_val("azores", "pressure_msl")
        iceland_p = get_val("iceland", "pressure_msl")
        features["nao_proxy"] = (azores_p - iceland_p) / 100  # Scale to ~10

        # Foehn Index: South - North Alps pressure gradient
        # Positive = higher pressure south of Alps (Foehn conditions)
        alps_south_p = get_val("alps_south", "pressure_msl")
        alps_north_p = get_val("alps_north", "pressure_msl")
        features["foehn_index"] = alps_south_p - alps_north_p

        # Mediterranean-Central Europe gradient
        # Positive = higher pressure in Med (northerly flow)
        med_p = get_val("mediterranean", "pressure_msl")
        ce_p = get_val("central_europe", "pressure_msl")
        features["med_ce_gradient"] = med_p - ce_p

        # Scandinavian blocking strength
        # Positive = high over Scandinavia (blocking)
        scandi_p = get_val("scandinavia", "pressure_msl")
        features["scandi_blocking_strength"] = scandi_p - ce_p

        # Zonal flow strength (500 hPa geopotential gradient)
        # Atlantic - Central Europe
        atlantic_z500 = get_val("atlantic", "z500")
        ce_z500 = get_val("central_europe", "z500")
        features["zonal_flow_strength"] = atlantic_z500 - ce_z500

        # Thermal advection (850 hPa temperature difference)
        # Med - CE: positive = warm advection from south
        med_temp = get_val("mediterranean", "temp_850")
        ce_temp = get_val("central_europe", "temp_850")
        features["thermal_advection"] = med_temp - ce_temp

        # Alpine thickness (thermal wind relationship)
        # 500-850 hPa thickness over Alps
        alps_south_z500 = get_val("alps_south", "z500")
        alps_south_z850 = get_val("alps_south", "z850")
        features["alpine_thickness"] = alps_south_z500 - alps_south_z850

        # Atlantic Ridge strength
        # High Atlantic pressure = ridge extension towards Europe
        atlantic_p = get_val("atlantic", "pressure_msl")
        features["atlantic_ridge_strength"] = atlantic_p - ce_p

        # === Location-Agnostic Pressure Gradients ===
        # These capture large-scale patterns without reference to specific geography

        # North-South pressure gradient (Scandinavia - Mediterranean)
        # Positive = higher pressure in north (northerly flow tendency)
        scandi_p = get_val("scandinavia", "pressure_msl")
        med_p = get_val("mediterranean", "pressure_msl")
        features["pressure_gradient_ns"] = scandi_p - med_p

        # East-West pressure gradient (Central Europe - Atlantic)
        # Positive = higher pressure in east (easterly flow tendency)
        features["pressure_gradient_ew"] = ce_p - atlantic_p

        # Pressure gradient magnitude (combines N-S and E-W)
        grad_ns = features["pressure_gradient_ns"]
        grad_ew = features["pressure_gradient_ew"]
        features["pressure_gradient_magnitude"] = np.sqrt(grad_ns**2 + grad_ew**2)

        # Pressure gradient direction (in degrees, 0=North, 90=East)
        features["pressure_gradient_direction"] = (
            np.degrees(np.arctan2(grad_ew, grad_ns)) % 360
        )

        # Pressure field curvature (Laplacian approximation)
        # Uses Iceland-Azores (N-S) and Atlantic-CE (E-W) as proxy for 2nd derivatives
        # Positive = low pressure center (convergence), Negative = high pressure center
        iceland_p = get_val("iceland", "pressure_msl")
        azores_p = get_val("azores", "pressure_msl")
        features["pressure_laplacian"] = (
            iceland_p + azores_p + atlantic_p + ce_p
        ) / 4 - scandi_p

        # === Location-Agnostic Temperature Gradients ===

        # North-South temperature gradient (Scandinavia - Mediterranean)
        # Typically negative (warmer in south), large magnitude = strong baroclinicity
        scandi_temp = get_val("scandinavia", "temp_850")
        med_temp = get_val("mediterranean", "temp_850")
        features["temperature_gradient_ns"] = scandi_temp - med_temp

        # East-West temperature gradient (Central Europe - Atlantic)
        # Positive = warmer over land (continental influence)
        atlantic_temp = get_val("atlantic", "temp_850")
        ce_temp = get_val("central_europe", "temp_850")
        features["temperature_gradient_ew"] = ce_temp - atlantic_temp

        # Temperature gradient magnitude
        temp_grad_ns = features["temperature_gradient_ns"]
        temp_grad_ew = features["temperature_gradient_ew"]
        features["temperature_gradient_magnitude"] = np.sqrt(
            temp_grad_ns**2 + temp_grad_ew**2
        )

        # Thermal wind strength (relates temp gradient to wind shear)
        # Proportional to temperature gradient and thickness contrast
        # Large value = strong baroclinicity = strong jet stream
        features["thermal_wind_strength"] = features["temperature_gradient_magnitude"]

        # Baroclinicity index (combination of pressure and temperature gradients)
        # Measures how misaligned pressure and temperature gradients are
        # High = strong baroclinic zone (frontal activity)
        features["baroclinicity_index"] = (
            features["pressure_gradient_magnitude"]
            * features["temperature_gradient_magnitude"]
        ) / 1000  # Scale down for readability

        return features

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names this aggregator produces.

        Returns:
            List of feature names
        """
        features = []

        # Regional statistics
        variables = ["pressure_msl", "z500", "z850", "temp_850"]
        for region in self.regions.keys():
            for var in variables:
                features.append(f"{region}_{var}_mean")
                features.append(f"{region}_{var}_std")

        # Cross-regional indices
        cross_regional = [
            "nao_proxy",
            "foehn_index",
            "med_ce_gradient",
            "scandi_blocking_strength",
            "zonal_flow_strength",
            "thermal_advection",
            "alpine_thickness",
            "atlantic_ridge_strength",
            # Location-agnostic gradient features
            "pressure_gradient_ns",
            "pressure_gradient_ew",
            "pressure_gradient_magnitude",
            "pressure_gradient_direction",
            "pressure_laplacian",
            "temperature_gradient_ns",
            "temperature_gradient_ew",
            "temperature_gradient_magnitude",
            "thermal_wind_strength",
            "baroclinicity_index",
        ]
        features.extend(cross_regional)

        return features

    def get_n_features(self) -> int:
        """
        Get total number of features produced.

        Returns:
            Number of features
        """
        # (n_regions * n_variables * 2 stats) + n_cross_regional
        n_regional = len(self.regions) * 4 * 2  # 4 variables × 2 stats (mean, std)
        n_cross = 18  # Cross-regional indices (8 original + 10 new gradient features)
        return n_regional + n_cross

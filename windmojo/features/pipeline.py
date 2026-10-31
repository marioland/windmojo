"""
Feature Engineering Pipeline

Transforms raw GFS weather data into ML-ready features for wind forecasting.
The same pipeline serves training and live forecasts, so it must not depend
on the amount of data passed in (no outlier capping or other statistics over
the whole input). Features:
- 9-point circular grid (8 compass directions + center)
- 5 pressure levels (1000, 850, 700, 500, 250 hPa) with full variables
- Comprehensive atmospheric profiling with vertical gradients
- Snow cover and precipitation history indicators (Oct-April)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from windmojo.features.snow_features import SnowCoverFeatures

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline for windmojo.

    Transforms raw weather data through sequential stages to create
    300-400 ML features capturing spatial gradients, temporal patterns,
    atmospheric physics, and snow cover indicators.
    """

    def __init__(self, bias_correction: Optional[Dict[str, float]] = None):
        """
        Initialize feature engineering pipeline.

        Args:
            bias_correction: Optional dict of variable → bias corrections
                           (e.g., {'wind_speed_10m': 3.21})
        """
        self.bias_correction = bias_correction or {}

        # Initialize stage processors
        self.pivot_stage = PivotToWide()
        self.gradient_stage = SpatialGradients()
        self.vertical_gradient_stage = VerticalGradients()
        self.interaction_stage = (
            InteractionFeatures()
        )  # NEW: Priority engineered features
        self.derivative_stage = TimeDerivatives()
        self.temporal_stage = TemporalFeatures()
        self.rolling_stage = RollingStatistics()
        self.snow_stage = (
            SnowCoverFeatures()
        )  # NEW: Snow cover and precipitation history
        self.physics_stage = PhysicsFeatures()
        self.lag_stage = LagFeatures()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline.

        Args:
            df: Raw data in long format (from Parquet files)

        Returns:
            Wide-format DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline...")

        # Stage 1: Pivot to wide format
        logger.info("Stage 1: Pivoting to wide format...")
        df = self.pivot_stage.transform(df)
        logger.info(f"  → {len(df)} timestamps, {len(df.columns)} base columns")

        # Fill hours missing in the raw data (e.g. nights outside the forecast
        # window) with empty rows, so that shift/diff/rolling in the following
        # stages work on real hours instead of row counts
        original_times = df["time"]
        df = self._reindex_hourly(df)

        # Stage 2: Bias correction (if specified)
        if self.bias_correction:
            logger.info("Stage 2: Applying bias corrections...")
            df = self._apply_bias_correction(df)

        # Stage 3: Spatial gradient features
        logger.info("Stage 3: Calculating spatial gradients...")
        df = self.gradient_stage.transform(df)
        logger.info(f"  → {len(df.columns)} columns after spatial gradients")

        # Stage 3b: Vertical gradient features (NEW)
        logger.info("Stage 3b: Calculating vertical gradients...")
        df = self.vertical_gradient_stage.transform(df)
        logger.info(f"  → {len(df.columns)} columns after vertical gradients")

        # Stage 3c: Interaction features (gradient magnitudes, cloud interactions, cross-level ratios)
        logger.info("Stage 3c: Creating interaction features...")
        df = self.interaction_stage.transform(df)
        logger.info(f"  → {len(df.columns)} columns after interactions")

        # Stage 4: Time derivative features
        logger.info("Stage 4: Calculating time derivatives...")
        df = self.derivative_stage.transform(df)
        logger.info(f"  → {len(df.columns)} columns after derivatives")

        # Stage 5: Temporal features (cyclical encoding, binary flags)
        logger.info("Stage 5: Adding temporal features...")
        df = self.temporal_stage.transform(df)
        logger.info(f"  → {len(df.columns)} columns after temporal features")

        # Stage 6: Short-term lag features (1h, 3h, 6h)
        logger.info("Stage 6: Adding short-term lags...")
        df = self.lag_stage.transform_short_term(df)
        logger.info(f"  → {len(df.columns)} columns after short-term lags")

        # Stage 7: Multi-day lag features (1d, 2d, 3d)
        logger.info("Stage 7: Adding multi-day lags...")
        df = self.lag_stage.transform_multi_day(df)
        logger.info(f"  → {len(df.columns)} columns after multi-day lags")

        # Stage 8: Rolling statistics
        logger.info("Stage 8: Calculating rolling statistics...")
        df = self.rolling_stage.transform(df)
        logger.info(f"  → {len(df.columns)} columns after rolling stats")

        # Stage 8b: Snow cover and precipitation history features
        logger.info("Stage 8b: Creating snow cover features...")
        df = self.snow_stage.transform(df)
        logger.info(f"  → {len(df.columns)} columns after snow features")

        # Stage 9: Physics-based features
        logger.info("Stage 9: Computing physics features...")
        df = self.physics_stage.transform(df)
        logger.info(f"  → {len(df.columns)} columns after physics features")

        # Drop the gap-filling rows again
        df = df[df["time"].isin(original_times)].reset_index(drop=True)

        logger.info(f"✓ Feature engineering complete: {len(df.columns)} total features")

        return df

    def _reindex_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex to a continuous hourly time axis (missing hours become NaN rows)."""
        full_index = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
        n_missing = len(full_index) - len(df)
        if n_missing > 0:
            logger.info(f"  Filling {n_missing} missing hours for time-based features")
        return (
            df.set_index("time").reindex(full_index).rename_axis("time").reset_index()
        )

    def _apply_bias_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply bias corrections to specified variables."""
        for var, correction in self.bias_correction.items():
            # After pivoting, variables have location suffixes (e.g., wind_speed_10m_center)
            col_name = f"{var}_center" if f"{var}_center" in df.columns else var

            if col_name in df.columns:
                df[f"{col_name}_corrected"] = df[col_name] + correction
                logger.info(f"  Applied {correction:+.2f} correction to {col_name}")
        return df


class PivotToWide:
    """Stage 1: Pivot from long format to wide format with location-prefixed columns."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot raw data from long to wide format.

        Input:  time, location, forecast_made, variable1, variable2, ...
        Output: time, forecast_made, variable1_center, variable1_n, ...
        """
        # Ensure time is datetime and sorted
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        # Check for and remove duplicates
        n_before = len(df)
        df = df.drop_duplicates(subset=["time", "location"], keep="last")
        n_after = len(df)

        if n_before != n_after:
            logger.warning(
                f"  Removed {n_before - n_after} duplicate time-location pairs (kept last)"
            )

        # Identify value columns (exclude metadata)
        metadata_cols = ["time", "location", "forecast_made", "month", "forecast_date"]
        value_cols = [c for c in df.columns if c not in metadata_cols]

        # Get forecast_made as a separate mapping (one per time)
        forecast_made_map = (
            df[["time", "forecast_made"]].drop_duplicates().set_index("time")
        )

        # Pivot each variable by location
        pivoted_dfs = []

        for var in value_cols:
            pivot = df.pivot(index="time", columns="location", values=var)
            pivot.columns = [f"{var}_{loc}" for loc in pivot.columns]
            pivoted_dfs.append(pivot)

        # Combine all pivoted variables
        result = pd.concat(pivoted_dfs, axis=1)
        result = result.loc[:, ~result.columns.duplicated()]  # Remove duplicates

        # Drop columns that are entirely NULL (e.g., pressure level vars at circle points)
        # These are created by pivot but have no data
        null_cols = result.columns[result.isna().all()].tolist()
        if null_cols:
            logger.info(
                f"  Dropping {len(null_cols)} entirely NULL columns (no data in raw)"
            )
            logger.debug(
                f"  NULL columns: {null_cols[:10]}..."
                if len(null_cols) > 10
                else f"  NULL columns: {null_cols}"
            )
            result = result.drop(columns=null_cols)

        # Add forecast_made back
        result = result.join(forecast_made_map)

        return result.reset_index()


class SpatialGradients:
    """Stage 3: Calculate spatial gradients across grid points."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pressure, cloud, and other spatial gradients.

        New Architecture:
        - Pressure gradients (9-point circular grid): 16 gradients from 8 directions + center
        - Cloud gradients (4 layers × 16 gradients each): total, low, mid, high
        - Surface variable gradients (9-point grid)
        """
        # Pressure gradients from 9-point circular grid
        df = self._pressure_gradients(df)

        # Cloud gradients (4 layers: total, low, mid, high)
        df = self._cloud_gradients(df)

        # Other surface variable gradients (optional, for compatibility)
        df = self._surface_gradients(df)

        return df

    def _pressure_gradients(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive pressure gradients from 9-point circular grid.

        16 gradients total:
        - 4 cardinal pair gradients (N-S, E-W)
        - 2 diagonal pair gradients (NE-SW, NW-SE)
        - 8 radial gradients (center to each direction)
        - 4 cross gradients (N-E, N-W, S-E, S-W)
        """
        p = "pressure_msl"  # New data uses pressure_msl from circle points

        # === CARDINAL PAIR GRADIENTS (4 features) ===

        # North-South gradient
        if f"{p}_n" in df.columns and f"{p}_s" in df.columns:
            df["pressure_msl_grad_n_s"] = df[f"{p}_n"] - df[f"{p}_s"]

        # East-West gradient
        if f"{p}_e" in df.columns and f"{p}_w" in df.columns:
            df["pressure_msl_grad_e_w"] = df[f"{p}_e"] - df[f"{p}_w"]

        # === DIAGONAL PAIR GRADIENTS (2 features) ===

        # NE-SW diagonal
        if f"{p}_ne" in df.columns and f"{p}_sw" in df.columns:
            df["pressure_msl_grad_ne_sw"] = df[f"{p}_ne"] - df[f"{p}_sw"]

        # NW-SE diagonal
        if f"{p}_nw" in df.columns and f"{p}_se" in df.columns:
            df["pressure_msl_grad_nw_se"] = df[f"{p}_nw"] - df[f"{p}_se"]

        # === RADIAL GRADIENTS: Center to Each Direction (8 features) ===

        directions = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
        if f"{p}_center" in df.columns:
            for direction in directions:
                if f"{p}_{direction}" in df.columns:
                    df[f"pressure_msl_grad_center_{direction}"] = (
                        df[f"{p}_{direction}"] - df[f"{p}_center"]
                    )

        # === CROSS GRADIENTS (4 features) ===
        # These capture diagonal pressure patterns not aligned with grid

        # North to East
        if f"{p}_n" in df.columns and f"{p}_e" in df.columns:
            df["pressure_msl_grad_n_e"] = df[f"{p}_n"] - df[f"{p}_e"]

        # North to West
        if f"{p}_n" in df.columns and f"{p}_w" in df.columns:
            df["pressure_msl_grad_n_w"] = df[f"{p}_n"] - df[f"{p}_w"]

        # South to East
        if f"{p}_s" in df.columns and f"{p}_e" in df.columns:
            df["pressure_msl_grad_s_e"] = df[f"{p}_s"] - df[f"{p}_e"]

        # South to West
        if f"{p}_s" in df.columns and f"{p}_w" in df.columns:
            df["pressure_msl_grad_s_w"] = df[f"{p}_s"] - df[f"{p}_w"]

        return df

    def _cloud_gradients(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cloud cover gradients for all 4 layers.

        Applies same 16-gradient pattern as pressure to:
        - cloud_cover (total)
        - cloud_cover_low
        - cloud_cover_mid
        - cloud_cover_high

        Total: 4 layers × 16 gradients = 64 cloud gradient features
        """
        cloud_layers = [
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
        ]

        for cloud_var in cloud_layers:
            # Check if this cloud layer exists
            if f"{cloud_var}_center" not in df.columns:
                continue

            # === CARDINAL PAIR GRADIENTS ===

            # North-South
            if f"{cloud_var}_n" in df.columns and f"{cloud_var}_s" in df.columns:
                df[f"{cloud_var}_grad_n_s"] = (
                    df[f"{cloud_var}_n"] - df[f"{cloud_var}_s"]
                )

            # East-West
            if f"{cloud_var}_e" in df.columns and f"{cloud_var}_w" in df.columns:
                df[f"{cloud_var}_grad_e_w"] = (
                    df[f"{cloud_var}_e"] - df[f"{cloud_var}_w"]
                )

            # === DIAGONAL PAIR GRADIENTS ===

            # NE-SW
            if f"{cloud_var}_ne" in df.columns and f"{cloud_var}_sw" in df.columns:
                df[f"{cloud_var}_grad_ne_sw"] = (
                    df[f"{cloud_var}_ne"] - df[f"{cloud_var}_sw"]
                )

            # NW-SE
            if f"{cloud_var}_nw" in df.columns and f"{cloud_var}_se" in df.columns:
                df[f"{cloud_var}_grad_nw_se"] = (
                    df[f"{cloud_var}_nw"] - df[f"{cloud_var}_se"]
                )

            # === RADIAL GRADIENTS ===

            directions = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
            for direction in directions:
                if f"{cloud_var}_{direction}" in df.columns:
                    df[f"{cloud_var}_grad_center_{direction}"] = (
                        df[f"{cloud_var}_{direction}"] - df[f"{cloud_var}_center"]
                    )

            # === CROSS GRADIENTS ===

            if f"{cloud_var}_n" in df.columns and f"{cloud_var}_e" in df.columns:
                df[f"{cloud_var}_grad_n_e"] = (
                    df[f"{cloud_var}_n"] - df[f"{cloud_var}_e"]
                )

            if f"{cloud_var}_n" in df.columns and f"{cloud_var}_w" in df.columns:
                df[f"{cloud_var}_grad_n_w"] = (
                    df[f"{cloud_var}_n"] - df[f"{cloud_var}_w"]
                )

            if f"{cloud_var}_s" in df.columns and f"{cloud_var}_e" in df.columns:
                df[f"{cloud_var}_grad_s_e"] = (
                    df[f"{cloud_var}_s"] - df[f"{cloud_var}_e"]
                )

            if f"{cloud_var}_s" in df.columns and f"{cloud_var}_w" in df.columns:
                df[f"{cloud_var}_grad_s_w"] = (
                    df[f"{cloud_var}_s"] - df[f"{cloud_var}_w"]
                )

        return df

    def _surface_gradients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate gradients for surface variables on 9-point grid (compatibility)."""
        # Note: Most surface gradients now handled by cloud_gradients
        # This is kept for any additional surface variables if needed
        surface_vars = [
            "temperature_2m",
            "shortwave_radiation",
            "soil_temperature_0cm",
            "boundary_layer_height",
        ]

        for var in surface_vars:
            # Check if variable exists in grid
            if f"{var}_center" not in df.columns:
                continue

            # === CARDINAL DIRECTIONS (N, S, E, W) ===

            # North-South gradient
            if f"{var}_n" in df.columns and f"{var}_s" in df.columns:
                df[f"{var}_grad_ns"] = df[f"{var}_n"] - df[f"{var}_s"]

            # North gradient (from center)
            if f"{var}_n" in df.columns:
                df[f"{var}_grad_n"] = df[f"{var}_n"] - df[f"{var}_center"]

            # South gradient (from center)
            if f"{var}_s" in df.columns:
                df[f"{var}_grad_s"] = df[f"{var}_center"] - df[f"{var}_s"]

            # East-West gradient
            if f"{var}_e" in df.columns and f"{var}_w" in df.columns:
                df[f"{var}_grad_ew"] = df[f"{var}_e"] - df[f"{var}_w"]

            # East gradient (from center)
            if f"{var}_e" in df.columns:
                df[f"{var}_grad_e"] = df[f"{var}_e"] - df[f"{var}_center"]

            # West gradient (from center)
            if f"{var}_w" in df.columns:
                df[f"{var}_grad_w"] = df[f"{var}_center"] - df[f"{var}_w"]

            # === DIAGONAL DIRECTIONS (NE, SE, SW, NW) ===

            # NE-SW diagonal gradient
            if f"{var}_ne" in df.columns and f"{var}_sw" in df.columns:
                df[f"{var}_grad_ne_sw"] = df[f"{var}_ne"] - df[f"{var}_sw"]

            # Northeast gradient (from center)
            if f"{var}_ne" in df.columns:
                df[f"{var}_grad_ne"] = df[f"{var}_ne"] - df[f"{var}_center"]

            # Southwest gradient (from center)
            if f"{var}_sw" in df.columns:
                df[f"{var}_grad_sw"] = df[f"{var}_center"] - df[f"{var}_sw"]

            # NW-SE diagonal gradient
            if f"{var}_nw" in df.columns and f"{var}_se" in df.columns:
                df[f"{var}_grad_nw_se"] = df[f"{var}_nw"] - df[f"{var}_se"]

            # Northwest gradient (from center)
            if f"{var}_nw" in df.columns:
                df[f"{var}_grad_nw"] = df[f"{var}_nw"] - df[f"{var}_center"]

            # Southeast gradient (from center)
            if f"{var}_se" in df.columns:
                df[f"{var}_grad_se"] = df[f"{var}_center"] - df[f"{var}_se"]

        return df


class VerticalGradients:
    """
    Stage 3b: Calculate vertical gradients across 5 pressure levels.

    Features:
    - Wind shear (speed + directional) across 4 layers
    - Temperature lapse rates across 4 layers
    - Relative humidity gradients across 4 layers
    - Geopotential height changes across 4 layers
    - Atmospheric stability indices

    Total: ~40 vertical gradient features
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all vertical gradient features."""
        df = self._wind_shear(df)
        df = self._temperature_lapse_rates(df)
        df = self._humidity_gradients(df)
        df = self._geopotential_changes(df)
        df = self._stability_indices(df)
        return df

    def _wind_shear(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate vertical wind shear across 5 pressure levels.

        Wind shear features (8 total):
        - Speed shear across 4 layers (1000-850, 850-700, 700-500, 500-250 hPa)
        - Directional shear across 4 layers (veering/backing)
        """
        levels = ["1000hPa", "850hPa", "700hPa", "500hPa", "250hPa"]
        layer_pairs = [
            ("1000hPa", "850hPa"),
            ("850hPa", "700hPa"),
            ("700hPa", "500hPa"),
            ("500hPa", "250hPa"),
        ]

        for lower, upper in layer_pairs:
            lower_speed = f"wind_speed_{lower}_center"
            upper_speed = f"wind_speed_{upper}_center"
            lower_dir = f"wind_direction_{lower}_center"
            upper_dir = f"wind_direction_{upper}_center"

            # Speed shear
            if lower_speed in df.columns and upper_speed in df.columns:
                df[f"wind_shear_{lower}_{upper}"] = df[upper_speed] - df[lower_speed]

            # Directional shear (accounting for 360° wrap)
            if lower_dir in df.columns and upper_dir in df.columns:
                # Calculate difference with wrap-around handling
                dir_diff = (df[upper_dir] - df[lower_dir] + 180) % 360 - 180
                df[f"wind_dir_shear_{lower}_{upper}"] = dir_diff

        return df

    def _temperature_lapse_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate temperature lapse rates across pressure levels.

        Lapse rate = ΔT / Δz (°C per km)
        Indicates atmospheric stability.
        """
        # Approximate heights (meters above sea level)
        heights = {
            "1000hPa": 110,
            "850hPa": 1500,
            "700hPa": 3000,
            "500hPa": 5600,
            "250hPa": 10400,
        }

        layer_pairs = [
            ("1000hPa", "850hPa"),
            ("850hPa", "700hPa"),
            ("700hPa", "500hPa"),
            ("500hPa", "250hPa"),
        ]

        for lower, upper in layer_pairs:
            lower_temp = f"temperature_{lower}_center"
            upper_temp = f"temperature_{upper}_center"

            if lower_temp in df.columns and upper_temp in df.columns:
                # Temperature difference (°C)
                temp_diff = df[lower_temp] - df[upper_temp]  # Lower minus upper

                # Height difference (km)
                height_diff = (heights[upper] - heights[lower]) / 1000.0

                # Lapse rate (°C/km)
                df[f"temp_lapse_{lower}_{upper}"] = temp_diff / height_diff

        return df

    def _humidity_gradients(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate relative humidity gradients across pressure levels.

        RH gradient = ΔRH / Δz (% per km)
        """
        heights = {
            "1000hPa": 110,
            "850hPa": 1500,
            "700hPa": 3000,
            "500hPa": 5600,
            "250hPa": 10400,
        }

        layer_pairs = [
            ("1000hPa", "850hPa"),
            ("850hPa", "700hPa"),
            ("700hPa", "500hPa"),
            ("500hPa", "250hPa"),
        ]

        for lower, upper in layer_pairs:
            lower_rh = f"relative_humidity_{lower}_center"
            upper_rh = f"relative_humidity_{upper}_center"

            if lower_rh in df.columns and upper_rh in df.columns:
                # RH difference (%)
                rh_diff = df[upper_rh] - df[lower_rh]

                # Height difference (km)
                height_diff = (heights[upper] - heights[lower]) / 1000.0

                # RH gradient (%/km)
                df[f"rh_grad_{lower}_{upper}"] = rh_diff / height_diff

        return df

    def _geopotential_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate geopotential height changes (thickness) between levels.

        Thickness is proportional to mean temperature of the layer.
        """
        layer_pairs = [
            ("1000hPa", "850hPa"),
            ("850hPa", "700hPa"),
            ("700hPa", "500hPa"),
            ("500hPa", "250hPa"),
        ]

        for lower, upper in layer_pairs:
            lower_geop = f"geopotential_height_{lower}_center"
            upper_geop = f"geopotential_height_{upper}_center"

            if lower_geop in df.columns and upper_geop in df.columns:
                # Thickness (meters)
                df[f"thickness_{lower}_{upper}"] = df[upper_geop] - df[lower_geop]

        return df

    def _stability_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate atmospheric stability indices from vertical profiles.

        These are composite indices used in meteorology to assess
        atmospheric stability, convective potential, etc.
        """
        # Bulk Richardson Number (stability from temp lapse + wind shear)
        if all(
            col in df.columns
            for col in ["temp_lapse_1000hPa_850hPa", "wind_shear_1000hPa_850hPa"]
        ):
            # Simplified Ri = (g/T) * (dT/dz) / (dU/dz)²
            # Higher Ri = more stable
            g = 9.81  # m/s²
            T_mean = 273.15 + 15  # Approximate mean temp (K)

            # Note: lapse rate is already ΔT/Δz, shear is ΔU (not normalized by Δz yet)
            # For simplicity, use ratio of lapse to shear²
            lapse = df["temp_lapse_1000hPa_850hPa"]
            shear = df["wind_shear_1000hPa_850hPa"]

            # Avoid division by zero
            df["bulk_richardson_number"] = np.where(
                shear.abs() > 0.1,
                lapse / (shear**2 + 0.01),  # Add small constant to avoid inf
                np.nan,
            )

        # K-Index (thunderstorm potential)
        # K = (T850 - T500) + Td850 - (T700 - Td700)
        if all(
            col in df.columns
            for col in [
                "temperature_850hPa_center",
                "temperature_500hPa_center",
                "temperature_700hPa_center",
                "relative_humidity_850hPa_center",
                "relative_humidity_700hPa_center",
            ]
        ):
            # Simplified K-index (without full dewpoint calculation)
            # Using RH as proxy for moisture contribution
            df["k_index_simplified"] = (
                (df["temperature_850hPa_center"] - df["temperature_500hPa_center"])
                + (df["relative_humidity_850hPa_center"] / 100.0) * 10  # Moisture term
                - (
                    df["temperature_700hPa_center"]
                    - df["relative_humidity_700hPa_center"] / 10.0
                )
            )

        # Total Totals Index (TT = VT + CT)
        # VT (Vertical Totals) = T850 - T500
        # CT (Cross Totals) = Td850 - T500
        if all(
            col in df.columns
            for col in [
                "temperature_850hPa_center",
                "temperature_500hPa_center",
                "relative_humidity_850hPa_center",
            ]
        ):
            vt = df["temperature_850hPa_center"] - df["temperature_500hPa_center"]
            # Simplified CT using RH as moisture proxy
            ct = (df["relative_humidity_850hPa_center"] / 100.0) * 10
            df["total_totals_index"] = vt + ct

        return df


class InteractionFeatures:
    """Stage 3c: Create high-value interaction features from Phase 1 analysis."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ~60 interaction features based on Phase 1 Model 1 validation:
        1. Gradient magnitudes and directions
        2. Cloud multi-layer interactions
        3. Cross-level ratios and thermal wind
        4. Temporal consistency features
        5. Thermodynamic features
        6. 10m wind features (surface wind analysis)
        """
        df = self._gradient_magnitudes(df)
        df = self._cloud_interactions(df)
        df = self._cross_level_features(df)
        df = self._temporal_consistency(df)
        df = self._thermodynamic_features(df)
        df = self._wind_10m_features(df)
        return df

    def _gradient_magnitudes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create gradient magnitude and direction features.

        Phase 1 insight: Individual gradients (N-S, E-W) perform well,
        but magnitude combines them for stronger signal.
        """
        # Pressure gradient magnitude
        if all(
            col in df.columns
            for col in ["pressure_msl_grad_n_s", "pressure_msl_grad_e_w"]
        ):
            df["pressure_msl_grad_magnitude"] = np.sqrt(
                df["pressure_msl_grad_n_s"] ** 2 + df["pressure_msl_grad_e_w"] ** 2
            )
            # Gradient direction (in degrees, 0=N, 90=E, 180=S, 270=W)
            df["pressure_msl_grad_direction"] = (
                np.degrees(
                    np.arctan2(df["pressure_msl_grad_e_w"], df["pressure_msl_grad_n_s"])
                )
                % 360
            )

        # Cloud gradient magnitudes (all 4 layers)
        cloud_layers = [
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
        ]
        for cloud in cloud_layers:
            grad_ns = f"{cloud}_grad_n_s"
            grad_ew = f"{cloud}_grad_e_w"
            if all(col in df.columns for col in [grad_ns, grad_ew]):
                df[f"{cloud}_grad_magnitude"] = np.sqrt(
                    df[grad_ns] ** 2 + df[grad_ew] ** 2
                )
                df[f"{cloud}_grad_direction"] = (
                    np.degrees(np.arctan2(df[grad_ew], df[grad_ns])) % 360
                )

        # Gradient asymmetry (total gradient activity, all directions)
        for cloud in cloud_layers:
            grads = [
                f"{cloud}_grad_n_s",
                f"{cloud}_grad_e_w",
                f"{cloud}_grad_ne_sw",
                f"{cloud}_grad_nw_se",
            ]
            if all(col in df.columns for col in grads):
                df[f"{cloud}_grad_asymmetry"] = sum(abs(df[g]) for g in grads)

        # Gradient × raw value interaction (strength)
        if all(
            col in df.columns
            for col in ["cloud_cover_grad_magnitude", "cloud_cover_center"]
        ):
            df["cloud_cover_gradient_strength"] = (
                df["cloud_cover_grad_magnitude"] * df["cloud_cover_center"]
            )

        return df

    def _cloud_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cloud multi-layer and spatial interaction features.

        Phase 1 insight: Cloud gradients are #1 and #2 features.
        Multi-layer products may capture convection signals.
        """
        # Multi-layer vertical structure
        cloud_cols = [
            "cloud_cover_low_center",
            "cloud_cover_mid_center",
            "cloud_cover_high_center",
        ]
        if all(col in df.columns for col in cloud_cols):
            # Product (all layers present)
            df["cloud_vertical_structure"] = (
                df["cloud_cover_low_center"]
                * df["cloud_cover_mid_center"]
                * df["cloud_cover_high_center"]
            )

            # Layer differences (convection indicator)
            df["cloud_layer_diff_low_high"] = (
                df["cloud_cover_low_center"] - df["cloud_cover_high_center"]
            )

            # Convection index (high when low clouds dominate)
            df["cloud_convection_index"] = df["cloud_cover_low_center"] / (
                df["cloud_cover_high_center"] + 1.0
            )

        # Cloud contrast across circle (spatial variability)
        circle_locs = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
        for cloud in ["cloud_cover", "cloud_cover_low", "cloud_cover_mid"]:
            circle_cols = [f"{cloud}_{loc}" for loc in circle_locs]
            if all(col in df.columns for col in circle_cols):
                # Max - min across circle
                df[f"{cloud}_spatial_contrast"] = df[circle_cols].max(axis=1) - df[
                    circle_cols
                ].min(axis=1)

        # Cloud + radiation interaction (cloud effect on heating)
        if all(
            col in df.columns
            for col in ["cloud_cover_grad_magnitude", "shortwave_radiation_center"]
        ):
            df["cloud_radiation_interaction"] = (
                (100 - df["cloud_cover_center"])
                * df["shortwave_radiation_center"]
                / 100
            )

        return df

    def _cross_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cross-pressure-level ratios and thermal wind features.

        Phase 1 insight: wind_speed_1000hPa is #1 feature, but no ratios exist.
        """
        # Wind speed ratios between levels
        if all(
            col in df.columns
            for col in ["wind_speed_1000hPa_center", "wind_speed_700hPa_center"]
        ):
            df["wind_ratio_1000_700"] = df["wind_speed_1000hPa_center"] / (
                df["wind_speed_700hPa_center"] + 0.1
            )

        if all(
            col in df.columns
            for col in ["wind_speed_700hPa_center", "wind_speed_250hPa_center"]
        ):
            df["wind_ratio_700_250"] = df["wind_speed_700hPa_center"] / (
                df["wind_speed_250hPa_center"] + 0.1
            )

        # Thermal wind strength (combines shear + temperature lapse)
        if all(
            col in df.columns
            for col in ["wind_shear_850hPa_700hPa", "temp_lapse_850hPa_700hPa"]
        ):
            df["thermal_wind_strength_mid"] = df["wind_shear_850hPa_700hPa"] * abs(
                df["temp_lapse_850hPa_700hPa"]
            )

        if all(
            col in df.columns
            for col in ["wind_shear_1000hPa_850hPa", "temp_lapse_1000hPa_850hPa"]
        ):
            df["thermal_wind_strength_low"] = df["wind_shear_1000hPa_850hPa"] * abs(
                df["temp_lapse_1000hPa_850hPa"]
            )

        # Moisture flux at key levels (wind × humidity)
        levels = ["1000hPa", "700hPa", "500hPa"]
        for level in levels:
            wind_col = f"wind_speed_{level}_center"
            rh_col = f"relative_humidity_{level}_center"
            if all(col in df.columns for col in [wind_col, rh_col]):
                df[f"moisture_flux_{level}"] = df[wind_col] * (df[rh_col] / 100.0)

        return df

    def _temporal_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal consistency and persistence features.

        Phase 1 insight: No lag features in top 20, but trends may matter.
        """
        # Wind persistence (how much has wind changed recently?)
        if "wind_speed_1000hPa_center" in df.columns:
            lag_6h = df["wind_speed_1000hPa_center"].shift(6)
            if lag_6h is not None:
                df["wind_1000hPa_persistence"] = abs(
                    df["wind_speed_1000hPa_center"] - lag_6h
                )

        # Cloud trend (rate of cloud change)
        if "cloud_cover_center" in df.columns:
            lag_3h = df["cloud_cover_center"].shift(3)
            if lag_3h is not None:
                df["cloud_trend_3h"] = (df["cloud_cover_center"] - lag_3h) / 3.0

        # Pressure tendency stability (is pressure change accelerating?)
        if "surface_pressure_center" in df.columns:
            change_1h = df["surface_pressure_center"].diff(1)
            change_1h_prev = change_1h.shift(1)
            df["pressure_accel"] = change_1h - change_1h_prev

        return df

    def _thermodynamic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create thermodynamic stability features.

        Phase 1 insight: Dewpoint depression is missing but theoretically important.
        """
        # Dewpoint depression (larger = more stable, less cloud formation)
        if all(
            col in df.columns for col in ["temperature_2m_center", "dewpoint_2m_center"]
        ):
            df["dewpoint_depression_2m"] = (
                df["temperature_2m_center"] - df["dewpoint_2m_center"]
            )

        # Saturation deficit
        if "relative_humidity_2m_center" in df.columns:
            df["saturation_deficit_2m"] = 100 - df["relative_humidity_2m_center"]

        # Level of free convection proxy (850-700 hPa temp diff + moisture)
        if all(
            col in df.columns
            for col in [
                "temperature_850hPa_center",
                "temperature_700hPa_center",
                "relative_humidity_850hPa_center",
            ]
        ):
            temp_diff = (
                df["temperature_850hPa_center"] - df["temperature_700hPa_center"]
            )
            moisture = df["relative_humidity_850hPa_center"] / 100.0
            df["lfc_proxy"] = temp_diff * moisture

        return df

    def _wind_10m_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive 10m wind features.

        The 10m wind is the target we're predicting, so features derived from it
        capture important patterns:
        - Wind shear from 10m to upper levels
        - Gust characteristics (turbulence indicator)
        - Wind acceleration/deceleration
        - Directional persistence
        """
        # Wind shear: 10m to upper levels
        # (Important: measures boundary layer wind profile)
        if all(
            col in df.columns
            for col in ["wind_speed_10m_center", "wind_speed_80m_center"]
        ):
            df["wind_shear_10m_80m"] = (
                df["wind_speed_80m_center"] - df["wind_speed_10m_center"]
            )

        if all(
            col in df.columns
            for col in ["wind_speed_10m_center", "wind_speed_120m_center"]
        ):
            df["wind_shear_10m_120m"] = (
                df["wind_speed_120m_center"] - df["wind_speed_10m_center"]
            )

        if all(
            col in df.columns
            for col in ["wind_speed_10m_center", "wind_speed_1000hPa_center"]
        ):
            df["wind_shear_10m_1000hPa"] = (
                df["wind_speed_1000hPa_center"] - df["wind_speed_10m_center"]
            )

        # Wind direction shear (backing/veering)
        if all(
            col in df.columns
            for col in ["wind_direction_10m_center", "wind_direction_80m_center"]
        ):
            # Normalize to -180 to 180
            dir_diff = df["wind_direction_80m_center"] - df["wind_direction_10m_center"]
            dir_diff = ((dir_diff + 180) % 360) - 180
            df["wind_dir_shear_10m_80m"] = dir_diff

        # Gust intensity (already have gust_factor and gust_excess, add more)
        if all(
            col in df.columns
            for col in ["wind_gusts_10m_center", "wind_speed_10m_center"]
        ):
            # Gust intensity = gusts / mean wind (turbulence indicator)
            df["gust_intensity"] = df["wind_gusts_10m_center"] / (
                df["wind_speed_10m_center"] + 0.1
            )

            # Gustiness (how much gusts exceed mean wind, in m/s)
            # Already have gust_excess, so skip

        # Wind acceleration (6h and 12h)
        if "wind_speed_10m_center" in df.columns:
            # 6-hour wind change
            df["wind_10m_change_6h"] = df["wind_speed_10m_center"].diff(6)

            # 12-hour wind change
            df["wind_10m_change_12h"] = df["wind_speed_10m_center"].diff(12)

            # Wind acceleration (change in change)
            df["wind_10m_acceleration_3h"] = df["wind_speed_10m_center"].diff(3) - df[
                "wind_speed_10m_center"
            ].shift(3).diff(3)

        # Wind direction persistence (how much direction changed)
        if "wind_direction_10m_center" in df.columns:
            # 3-hour direction change
            dir_3h = df["wind_direction_10m_center"].diff(3)
            dir_3h = ((dir_3h + 180) % 360) - 180
            df["wind_dir_10m_change_3h"] = dir_3h

            # 6-hour direction change
            dir_6h = df["wind_direction_10m_center"].diff(6)
            dir_6h = ((dir_6h + 180) % 360) - 180
            df["wind_dir_10m_change_6h"] = dir_6h

            # Directional variability (absolute change)
            df["wind_dir_10m_variability_3h"] = abs(dir_3h)

        # Wind power (kinetic energy, proportional to speed^2)
        if "wind_speed_10m_center" in df.columns:
            df["wind_power_10m"] = df["wind_speed_10m_center"] ** 2

        # Effective wind (combines speed and gustiness)
        if all(
            col in df.columns
            for col in ["wind_speed_10m_center", "wind_gusts_10m_center"]
        ):
            # Weighted average: 70% mean + 30% gusts
            df["effective_wind_10m"] = (
                0.7 * df["wind_speed_10m_center"] + 0.3 * df["wind_gusts_10m_center"]
            )

        # Cross-level wind speed ratio (boundary layer profile shape)
        if all(
            col in df.columns
            for col in ["wind_speed_10m_center", "wind_speed_120m_center"]
        ):
            df["wind_ratio_10m_120m"] = df["wind_speed_10m_center"] / (
                df["wind_speed_120m_center"] + 0.1
            )

        # Wind-temperature coupling (thermal wind at surface)
        if all(
            col in df.columns
            for col in [
                "wind_speed_10m_center",
                "temperature_2m_center",
                "temperature_80m_center",
            ]
        ):
            temp_grad = df["temperature_80m_center"] - df["temperature_2m_center"]
            # Positive temp gradient = stable, negative = unstable
            df["wind_temp_coupling_10m"] = df["wind_speed_10m_center"] * temp_grad

        # Gust lag features (gust persistence)
        if "wind_gusts_10m_center" in df.columns:
            df["gust_10m_lag_1h"] = df["wind_gusts_10m_center"].shift(1)
            df["gust_10m_lag_3h"] = df["wind_gusts_10m_center"].shift(3)
            df["gust_10m_change_3h"] = df["wind_gusts_10m_center"].diff(3)

        return df


class TimeDerivatives:
    """Stage 4: Calculate time derivatives (rates of change)."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 1h and 3h changes for key variables.

        Derivatives capture:
        - Pressure tendency (approaching systems)
        - Temperature changes (heating/cooling rates)
        - Wind acceleration
        - Cloud development
        """
        # Key variables for derivatives
        vars_to_diff = [
            "surface_pressure_center",
            "temperature_2m_center",
            "cloud_cover_center",
            "wind_speed_10m_center",
            "shortwave_radiation_center",
            "boundary_layer_height_center",
        ]

        for var in vars_to_diff:
            if var in df.columns:
                # 1-hour change
                df[f"{var}_change_1h"] = df[var].diff(1)

                # 3-hour change
                df[f"{var}_change_3h"] = df[var].diff(3)

                # Acceleration (second derivative, 1h interval)
                df[f"{var}_accel_1h"] = df[f"{var}_change_1h"].diff(1)

        # Pressure gradient tendency (how fast is gradient changing)
        if "pressure_grad_ns_far" in df.columns:
            df["pressure_grad_ns_far_change_3h"] = df["pressure_grad_ns_far"].diff(3)

        return df


class TemporalFeatures:
    """Stage 5: Add cyclical and binary temporal features."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features:
        - Cyclical encoding (sin/cos for hour/month)
        - Binary indicators (is_afternoon, is_summer, etc.)
        - Wind direction encoding (sin/cos for circular variable)
        """
        # Extract time components
        df["hour"] = df["time"].dt.hour
        df["month"] = df["time"].dt.month
        df["day_of_week"] = df["time"].dt.dayofweek

        # Cyclical encoding: hour of day
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Cyclical encoding: month (seasonal cycle)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Binary indicators: time of day regimes
        df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
        df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
        df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)

        # Binary indicators: seasons
        df["is_summer"] = (df["month"].isin([6, 7, 8])).astype(int)
        df["is_winter"] = (df["month"].isin([12, 1, 2])).astype(int)
        df["is_shoulder"] = (df["month"].isin([3, 4, 5, 9, 10, 11])).astype(int)

        # Binary indicator: weekend
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Wind direction encoding (circular variable)
        if "wind_direction_10m_center" in df.columns:
            # Convert degrees to radians
            wind_dir_rad = np.deg2rad(df["wind_direction_10m_center"])
            df["wind_direction_sin"] = np.sin(wind_dir_rad)
            df["wind_direction_cos"] = np.cos(wind_dir_rad)

        # Precipitation indicator
        if "precipitation_center" in df.columns:
            df["is_raining"] = (df["precipitation_center"] > 0.1).astype(int)

        return df


class LagFeatures:
    """Stages 6-7: Add temporal lag features."""

    def transform_short_term(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 1h, 3h, 6h lags."""
        lag_vars = [
            "wind_speed_10m_center",
            "wind_gusts_10m_center",
            "temperature_2m_center",
            "precipitation_center",
            "cloud_cover_center",
            "relative_humidity_2m_center",
            "surface_pressure_center",
        ]

        for var in lag_vars:
            if var in df.columns:
                df[f"{var}_lag_1h"] = df[var].shift(1)
                df[f"{var}_lag_3h"] = df[var].shift(3)
                df[f"{var}_lag_6h"] = df[var].shift(6)

        return df

    def transform_multi_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 1d, 2d, 3d lags."""
        lag_vars = [
            "temperature_2m_center",
            "precipitation_center",
            "shortwave_radiation_center",
            "wind_speed_10m_center",
            "wind_gusts_10m_center",
        ]

        for var in lag_vars:
            if var in df.columns:
                df[f"{var}_lag_1d"] = df[var].shift(24)
                df[f"{var}_lag_2d"] = df[var].shift(48)
                df[f"{var}_lag_3d"] = df[var].shift(72)

        return df


class RollingStatistics:
    """Stage 8: Calculate rolling window statistics."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling statistics:
        - 3h/6h volatility (std, min, max)
        - Multi-day means (3d, 7d)
        - Precipitation accumulation
        """
        # 3-hour rolling statistics (volatility capture)
        rolling_vars_3h = [
            "wind_speed_10m_center",
            "wind_gusts_10m_center",
            "temperature_2m_center",
            "precipitation_center",
        ]

        for var in rolling_vars_3h:
            if var in df.columns:
                rolling = df[var].rolling(window=3, min_periods=1)
                df[f"{var}_rolling_3h_std"] = rolling.std()
                df[f"{var}_rolling_3h_max"] = rolling.max()
                df[f"{var}_rolling_3h_min"] = rolling.min()

        # 6-hour rolling statistics
        if "wind_speed_10m_center" in df.columns:
            rolling = df["wind_speed_10m_center"].rolling(window=6, min_periods=1)
            df["wind_speed_10m_center_rolling_6h_std"] = rolling.std()
            df["wind_speed_10m_center_rolling_6h_max"] = rolling.max()

        # 6-hour rolling statistics for gusts
        if "wind_gusts_10m_center" in df.columns:
            rolling = df["wind_gusts_10m_center"].rolling(window=6, min_periods=1)
            df["wind_gusts_10m_center_rolling_6h_std"] = rolling.std()
            df["wind_gusts_10m_center_rolling_6h_max"] = rolling.max()

        # Multi-day rolling means (smoothed trends)
        multi_day_vars = [
            "temperature_2m_center",
            "shortwave_radiation_center",
            "wind_speed_10m_center",
            "wind_gusts_10m_center",
        ]

        for var in multi_day_vars:
            if var in df.columns:
                df[f"{var}_rolling_3d_mean"] = (
                    df[var].rolling(window=72, min_periods=1).mean()
                )
                df[f"{var}_rolling_7d_mean"] = (
                    df[var].rolling(window=168, min_periods=1).mean()
                )

        # Precipitation accumulation
        if "precipitation_center" in df.columns:
            df["precipitation_rolling_3h_sum"] = (
                df["precipitation_center"].rolling(window=3, min_periods=1).sum()
            )
            df["precipitation_rolling_12h_sum"] = (
                df["precipitation_center"].rolling(window=12, min_periods=1).sum()
            )
            df["precipitation_rolling_3d_sum"] = (
                df["precipitation_center"].rolling(window=72, min_periods=1).sum()
            )
            df["precipitation_rolling_7d_sum"] = (
                df["precipitation_center"].rolling(window=168, min_periods=1).sum()
            )

        return df


class PhysicsFeatures:
    """Stage 9: Calculate physics-based features."""

    GRAVITY = 9.81  # m/s²

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate atmospheric physics features:
        - Thermal stability indicators
        - Solar heating effects
        - Lake thermal effects
        - Wind-temperature interactions
        """
        # Clear sky factor (inverse of cloud cover)
        if "cloud_cover_center" in df.columns:
            df["clear_sky_factor"] = 1 - (df["cloud_cover_center"] / 100)

        # Effective solar heating
        if all(
            col in df.columns
            for col in ["shortwave_radiation_center", "clear_sky_factor"]
        ):
            df["solar_heating_effective"] = (
                df["shortwave_radiation_center"] * df["clear_sky_factor"]
            )

        # AM/PM Sunshine Hours (cumulative daily)
        # Define "sunshine" as radiation > 120 W/m² (clear-sky threshold)
        if "shortwave_radiation_center" in df.columns and "hour" in df.columns:
            # Create sunshine indicator (1 if sunny, 0 if cloudy)
            sunshine_threshold = 120  # W/m²
            is_sunshine = (
                df["shortwave_radiation_center"] > sunshine_threshold
            ).astype(int)

            # Ensure we have date column for groupby
            if "time" in df.columns:
                df["_date_temp"] = pd.to_datetime(df["time"]).dt.date

                # Calculate cumulative sunshine hours within each day
                # AM: hours before noon (hour < 12)
                am_sunshine = is_sunshine * (df["hour"] < 12).astype(int)
                df["sunshine_hours_am"] = am_sunshine.groupby(df["_date_temp"]).cumsum()

                # PM: hours after noon (hour >= 12)
                pm_sunshine = is_sunshine * (df["hour"] >= 12).astype(int)
                df["sunshine_hours_pm"] = pm_sunshine.groupby(df["_date_temp"]).cumsum()

                # Total daily sunshine hours so far (running total)
                df["sunshine_hours_total"] = is_sunshine.groupby(
                    df["_date_temp"]
                ).cumsum()

                # Clean up temp column
                df.drop(columns=["_date_temp"], inplace=True)
            else:
                logger.warning(
                    "Cannot calculate sunshine hours: 'time' column not found"
                )

        # Daytime heating potential
        if all(
            col in df.columns
            for col in ["temperature_2m_center", "hour_sin", "clear_sky_factor"]
        ):
            df["daytime_heating_potential"] = (
                df["temperature_2m_center"]
                * np.maximum(0, df["hour_sin"])
                * df["clear_sky_factor"]
            )

        # Nighttime cooling potential
        if all(
            col in df.columns
            for col in ["temperature_2m_center", "hour_cos", "clear_sky_factor"]
        ):
            df["nighttime_cooling_potential"] = (
                df["temperature_2m_center"]
                * np.maximum(0, -df["hour_cos"])
                * df["clear_sky_factor"]
            )

        # Temperature × seasonal interaction
        if all(col in df.columns for col in ["temperature_2m_center", "month_cos"]):
            df["temp_seasonal_interaction"] = (
                df["temperature_2m_center"] * df["month_cos"]
            )

        # Lake thermal effect (water-air temperature difference proxy)
        if all(
            col in df.columns
            for col in [
                "temperature_2m_center_rolling_7d_mean",
                "temperature_2m_center",
            ]
        ):
            df["lake_thermal_effect"] = (
                df["temperature_2m_center_rolling_7d_mean"]
                - df["temperature_2m_center"]
            )

        # Thermal stability (temperature × wind shear interaction)
        if all(
            col in df.columns for col in ["temperature_2m_center", "wind_shear_total"]
        ):
            # Normalize temperature to 0-1 range for stability calculation
            temp_norm = (
                df["temperature_2m_center"] - df["temperature_2m_center"].min()
            ) / (
                df["temperature_2m_center"].max()
                - df["temperature_2m_center"].min()
                + 1e-10
            )
            df["thermal_stability"] = temp_norm * df["wind_shear_total"]

        # Buoyancy frequency (vertical stability indicator)
        if all(
            col in df.columns
            for col in ["temperature_2m_center", "temperature_2m_center_change_1h"]
        ):
            # Simplified buoyancy: (g/T) × dT/dt
            T_kelvin = df["temperature_2m_center"] + 273.15
            df["buoyancy_frequency"] = (self.GRAVITY / T_kelvin) * df[
                "temperature_2m_center_change_1h"
            ]

        # Richardson number (turbulence indicator)
        # Ri = (g/T × dT/dz) / (dU/dz)²
        # Using proxies: dT/dz ~ temp_gradient, dU/dz ~ wind_shear
        if all(
            col in df.columns
            for col in ["temperature_2m_grad_ns", "wind_shear_10m_100m"]
        ):
            T_kelvin = df["temperature_2m_center"] + 273.15
            temp_gradient = (
                df["temperature_2m_grad_ns"] / 30000
            )  # Convert to K/m (30km spacing)
            wind_gradient_sq = (
                df["wind_shear_10m_100m"] / 90
            ) ** 2  # (90m height diff)

            df["richardson_number"] = (
                (self.GRAVITY / T_kelvin)
                * temp_gradient
                / (wind_gradient_sq + 1e-10)  # Avoid division by zero
            )

        # Soil moisture proxy (exponentially weighted precipitation history)
        if "precipitation_center" in df.columns:
            weights = [1.0, 0.7, 0.5, 0.3, 0.15, 0.07, 0.03]
            soil_moisture = pd.Series(0, index=df.index, dtype=float)

            for i, weight in enumerate(weights, 1):
                lag_precip = df["precipitation_center"].shift(24 * i).fillna(0)
                soil_moisture += lag_precip * weight

            df["soil_moisture_proxy"] = soil_moisture

        # Wind gust features (turbulence and instability indicators)
        if all(
            col in df.columns
            for col in ["wind_gusts_10m_center", "wind_speed_10m_center"]
        ):
            # Gust factor (ratio): Shows atmospheric turbulence intensity
            # Typical values: 1.2-2.5 (calm to turbulent conditions)
            df["gust_factor"] = df["wind_gusts_10m_center"] / (
                df["wind_speed_10m_center"] + 0.1
            )

            # Gust excess (difference): Absolute gust magnitude above sustained wind
            df["gust_excess"] = (
                df["wind_gusts_10m_center"] - df["wind_speed_10m_center"]
            )

        return df


def load_raw_data(raw_data_dir: Path, spot: str) -> pd.DataFrame:
    """
    Load all raw Parquet files for a spot.

    Args:
        raw_data_dir: Directory containing raw Parquet files
        spot: Spot name (e.g., 'lecco')

    Returns:
        Combined DataFrame with all dates
    """
    parquet_files = sorted(raw_data_dir.glob(f"{spot}_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found for spot '{spot}' in {raw_data_dir}"
        )

    logger.info(f"Loading {len(parquet_files)} Parquet files...")

    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)

    logger.info(f"  Loaded {len(combined)} rows, {len(combined.columns)} columns")
    logger.info(f"  Date range: {combined['time'].min()} to {combined['time'].max()}")

    return combined

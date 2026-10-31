"""
Snow Cover Feature Engineering

Specialized features for detecting snow cover and recent precipitation history.
Critical for Oct-April period when mountain snow affects local wind patterns.

Physical impacts:
- Albedo: Fresh snow (0.8) vs bare ground (0.15) → 50-80% less heating
- Latent heat: Snow melt locks surface at 0°C → suppresses thermal winds
- Heat capacity: Wet ground dampens temperature swings
- Surface roughness: Snow changes boundary layer structure
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Longest look-back of the "hours since" features. A live forecast downloads
# 7 days of history (DEFAULT_HISTORY_DAYS in prediction/forecast.py), so longer
# spans would differ between training and live data.
MAX_HOURS_SINCE_EVENT = 168


class SnowCoverFeatures:
    """
    Calculate snow cover and precipitation history features.

    Uses existing downloaded variables (no additional API calls):
    - snowfall (cm per hour), snow_depth (m, Open-Meteo units)
    - precipitation, rain
    - temperature_2m, temperature_850hPa, temperature_700hPa

    Generates ~20 features:
    - Snow detection & classification
    - Temporal history (days since, rolling sums)
    - Snow age & freshness
    - Surface state classification
    - Mountain temperature indicators
    """

    def __init__(self):
        """Initialize snow feature calculator."""
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add snow cover and precipitation history features.

        Args:
            df: DataFrame with raw weather variables

        Returns:
            DataFrame with additional snow features
        """
        logger.info("  Calculating snow cover features...")

        # Check if required variables exist
        required = ["precipitation_center", "temperature_2m_center"]
        if not all(col in df.columns for col in required):
            logger.warning(f"  Missing required columns for snow features. Skipping.")
            return df

        # Apply feature calculations
        df = self._snow_detection(df)
        df = self._snow_temporal_history(df)
        df = self._snow_age_freshness(df)
        df = self._surface_state(df)
        df = self._mountain_temperature_indicators(df)

        return df

    def _snow_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect precipitation type: snow vs rain vs mixed.

        Uses 2°C threshold for rain/snow discrimination.
        """
        # Snow/rain discrimination based on temperature
        if all(
            col in df.columns
            for col in ["precipitation_center", "temperature_2m_center"]
        ):
            # Snowing: precipitation with temp < 2°C
            df["is_snowing"] = (
                (df["precipitation_center"] > 0.1) & (df["temperature_2m_center"] < 2.0)
            ).astype(int)

            # Raining: precipitation with temp >= 2°C
            df["is_raining"] = (
                (df["precipitation_center"] > 0.1)
                & (df["temperature_2m_center"] >= 2.0)
            ).astype(int)

        # Precipitation type classification
        if all(
            col in df.columns
            for col in ["snowfall_center", "rain_center", "precipitation_center"]
        ):

            def classify_precip_type(row):
                if row["precipitation_center"] < 0.1:
                    return "none"
                elif row["snowfall_center"] > 0.1 and row["rain_center"] < 0.1:
                    return "snow"
                elif row["rain_center"] > 0.1 and row["snowfall_center"] < 0.1:
                    return "rain"
                elif row["snowfall_center"] > 0.1 and row["rain_center"] > 0.1:
                    return "mixed"
                else:
                    return "none"

            df["precipitation_type"] = df.apply(classify_precip_type, axis=1)

            # One-hot encode precipitation type
            df["precip_is_snow"] = (df["precipitation_type"] == "snow").astype(int)
            df["precip_is_rain"] = (df["precipitation_type"] == "rain").astype(int)
            df["precip_is_mixed"] = (df["precipitation_type"] == "mixed").astype(int)

        return df

    def _snow_temporal_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate temporal features: rolling sums, days since events.
        """
        # Snowfall rolling accumulations
        if "snowfall_center" in df.columns:
            df["snowfall_rolling_24h"] = (
                df["snowfall_center"].rolling(24, min_periods=1).sum()
            )
            df["snowfall_rolling_72h"] = (
                df["snowfall_center"].rolling(72, min_periods=1).sum()
            )
            df["snowfall_rolling_7d"] = (
                df["snowfall_center"].rolling(168, min_periods=1).sum()
            )

        # Snow depth changes (melt rate indicator)
        if "snow_depth_center" in df.columns:
            df["snow_depth_change_24h"] = df["snow_depth_center"].diff(24)
            df["snow_depth_change_72h"] = df["snow_depth_center"].diff(72)

        # Days since last snowfall event
        if "snowfall_center" in df.columns:
            df["hours_since_snowfall"] = self._hours_since_event(
                df["snowfall_center"] > 0.1
            )

        # Days since last rain event
        if "is_raining" in df.columns:
            df["hours_since_rain"] = self._hours_since_event(df["is_raining"] == 1)

        return df

    def _snow_age_freshness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify snow as fresh vs old, calculate snow age.
        """
        # Snow on ground indicator
        if "snow_depth_center" in df.columns:
            df["is_snow_on_ground"] = (df["snow_depth_center"] > 0.01).astype(
                int
            )  # > 1 cm (snow_depth is in m)

        # Snow age (hours since last snowfall, only when snow present)
        if all(
            col in df.columns for col in ["hours_since_snowfall", "is_snow_on_ground"]
        ):
            df["snow_age_hours"] = np.where(
                df["is_snow_on_ground"] == 1, df["hours_since_snowfall"], 0
            )

            # Fresh snow: < 24 hours old
            df["is_fresh_snow"] = (
                (df["is_snow_on_ground"] == 1) & (df["snow_age_hours"] < 24)
            ).astype(int)

            # Old snow: > 7 days old
            df["is_old_snow"] = (
                (df["is_snow_on_ground"] == 1) & (df["snow_age_hours"] > 168)
            ).astype(int)

        # Active melting detection
        if all(
            col in df.columns
            for col in ["snow_depth_change_24h", "temperature_2m_center"]
        ):
            df["is_melting"] = (
                (df["snow_depth_change_24h"] < -0.005)  # Losing > 0.5 cm/day
                & (df["temperature_2m_center"] > 0)
            ).astype(int)

        return df

    def _surface_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify ground surface state and estimate albedo.
        """
        # Surface state classification
        if all(
            col in df.columns
            for col in [
                "is_fresh_snow",
                "is_old_snow",
                "hours_since_rain",
                "temperature_2m_center",
            ]
        ):

            def classify_surface(row):
                if row["is_fresh_snow"] == 1:
                    return "fresh_snow"
                elif row["is_old_snow"] == 1:
                    return "old_snow"
                elif row["hours_since_rain"] < 24:
                    return "wet_ground"
                elif row["temperature_2m_center"] < 0:
                    return "frozen_ground"
                else:
                    return "dry_ground"

            df["surface_state"] = df.apply(classify_surface, axis=1)

            # One-hot encode surface state
            df["surface_is_fresh_snow"] = (df["surface_state"] == "fresh_snow").astype(
                int
            )
            df["surface_is_old_snow"] = (df["surface_state"] == "old_snow").astype(int)
            df["surface_is_wet"] = (df["surface_state"] == "wet_ground").astype(int)
            df["surface_is_frozen"] = (df["surface_state"] == "frozen_ground").astype(
                int
            )
            df["surface_is_dry"] = (df["surface_state"] == "dry_ground").astype(int)

        # Albedo proxy (affects solar heating)
        if "snow_age_hours" in df.columns:
            df["albedo_proxy"] = self._estimate_albedo(
                df.get("is_snow_on_ground", pd.Series(0, index=df.index)),
                df["snow_age_hours"],
            )

        return df

    def _mountain_temperature_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use pressure-level temperatures to infer mountain snow conditions.

        850 hPa ~ 1500m elevation (typical mid-mountain)
        700 hPa ~ 3000m elevation (high peaks)
        """
        # Temperature at mountain level (850 hPa ~ 1500m)
        if "temperature_850hPa_center" in df.columns:
            # Below freezing at mountain level
            df["temp_850_subzero"] = (df["temperature_850hPa_center"] < 0).astype(int)

            # Very cold (heavy snow conditions)
            df["temp_850_very_cold"] = (df["temperature_850hPa_center"] < -5).astype(
                int
            )

            # Distance from freezing
            df["temp_850_freezing_margin"] = df["temperature_850hPa_center"]

        # High-altitude temperature (700 hPa ~ 3000m)
        if "temperature_700hPa_center" in df.columns:
            df["temp_700_subzero"] = (df["temperature_700hPa_center"] < 0).astype(int)

        # Mountain snow likely (cold + precipitation)
        if all(
            col in df.columns for col in ["temp_850_subzero", "precipitation_center"]
        ):
            df["mountain_snow_likely"] = (
                (df["temp_850_subzero"] == 1) & (df["precipitation_center"] > 0.1)
            ).astype(int)

        return df

    def _hours_since_event(self, condition: pd.Series) -> pd.Series:
        """
        Calculate hours since last time condition was True.

        Capped at MAX_HOURS_SINCE_EVENT, also for hours without any prior
        event in the data.

        Args:
            condition: Boolean series on a continuous hourly axis

        Returns:
            Series with hours since last True value (0 at the event itself)
        """
        position = pd.Series(
            np.arange(len(condition)), index=condition.index, dtype=float
        )
        last_event = position.where(condition.to_numpy()).ffill()
        return (
            (position - last_event)
            .fillna(MAX_HOURS_SINCE_EVENT)
            .clip(upper=MAX_HOURS_SINCE_EVENT)
        )

    def _estimate_albedo(
        self, is_snow_on_ground: pd.Series, snow_age_hours: pd.Series
    ) -> pd.Series:
        """
        Estimate surface albedo based on snow presence and age.

        Albedo values:
        - Fresh snow (< 24h): 0.8-0.9
        - Old snow (> 7d): 0.4-0.6
        - Bare ground: 0.15-0.20

        Args:
            is_snow_on_ground: Boolean series (1 if snow present)
            snow_age_hours: Hours since last snowfall (when snow present)

        Returns:
            Estimated albedo (0-1)
        """
        albedo = pd.Series(0.15, index=is_snow_on_ground.index)  # Default: bare ground

        # Snow present: albedo decreases with age
        snow_mask = is_snow_on_ground == 1
        if snow_mask.any():
            # Start at 0.85, decay to 0.40 over 14 days
            # albedo = 0.85 - 0.03 * (age_in_days)
            age_in_days = snow_age_hours / 24.0
            albedo[snow_mask] = np.clip(
                0.85 - 0.03 * age_in_days[snow_mask],
                0.40,  # Minimum (old snow)
                0.85,  # Maximum (fresh snow)
            )

        return albedo


def create_snow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add snow features to DataFrame.

    Args:
        df: DataFrame with raw weather data

    Returns:
        DataFrame with snow features added
    """
    snow_stage = SnowCoverFeatures()
    return snow_stage.transform(df)

"""
Synoptic Feature Engineering Pipeline

Transforms large-scale European weather grid into XGBoost-ready features.
Combines three approaches:
1. Pressure center parameterization (highs/lows detection)
2. Regional aggregation (meteorological regions)
3. Teleconnection indices (NAO, EA, SCAND, Alpine)

This is the main pipeline for Experiment8 (Model 1b).
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from windmojo.features.pressure_centers import PressureCenterDetector
from windmojo.features.regional_aggregation import RegionalAggregator
from windmojo.features.synoptic_interactions import SynopticInteractionCalculator
from windmojo.features.teleconnection import TeleconnectionCalculator

logger = logging.getLogger(__name__)


class SynopticFeaturePipeline:
    """
    Transform synoptic grid data into ML features.

    Takes 11,700-point European grid and produces ~193 features describing:
    - Pressure system positions, intensities, distances (42 features)
    - Regional pressure/temperature patterns (74 features)
    - Teleconnection indices and Alpine-specific patterns (9 features)
    - Location-agnostic synoptic interactions (30 features)
    - Temporal features (4 features - added during training)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize synoptic feature pipeline.

        Args:
            config: Optional configuration dict with synoptic_features section
        """
        self.config = config or {}

        # Extract configuration
        synoptic_config = self.config.get("synoptic_features", {})

        # Initialize components
        pressure_config = synoptic_config.get("pressure_centers", {})
        self.center_detector = PressureCenterDetector(
            n_highs=pressure_config.get("n_highs", 3),
            n_lows=pressure_config.get("n_lows", 3),
            smoothing_sigma=pressure_config.get("smoothing_sigma", 2.0),
            min_distance=pressure_config.get("min_distance", 5),
            sort_by=pressure_config.get("sort_by", "strength"),
        )

        regional_config = synoptic_config.get("regional_aggregation", {})
        custom_regions = regional_config.get("regions", None)
        self.regional_aggregator = RegionalAggregator(regions=custom_regions)

        self.teleconnection_calculator = TeleconnectionCalculator()

        self.interaction_calculator = SynopticInteractionCalculator()

        # Target location (for distance calculations)
        center = self.config.get("center", {})
        self.target_lat = center.get("latitude", 46.083333)
        self.target_lon = center.get("longitude", 9.3)

        logger.info("Synoptic feature pipeline initialized")
        logger.info(f"  Target location: {self.target_lat}°N, {self.target_lon}°E")
        logger.info(
            f"  Pressure centers: {pressure_config.get('n_highs', 3)} highs, "
            f"{pressure_config.get('n_lows', 3)} lows"
        )
        logger.info(
            f"  Regional aggregation: {len(self.regional_aggregator.regions)} regions"
        )

    def transform(self, grid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform synoptic grid into features.

        Args:
            grid_df: DataFrame with columns:
                    - lat, lon: Grid coordinates
                    - geopotential_height_500hPa: Upper-level geopotential
                    - geopotential_height_850hPa: Lower-level geopotential
                    - temperature_850hPa: 850 hPa temperature
                    - pressure_msl: Mean sea level pressure
                    Should contain ~11,700 rows (one per grid point) at 12:00 UTC

        Returns:
            DataFrame with 1 row containing ~83 synoptic features
        """
        if grid_df.empty:
            logger.error("Grid DataFrame is empty")
            return self._empty_features()

        logger.debug(f"Transforming grid with {len(grid_df)} points")

        features_list = []

        highs_list = []
        lows_list = []

        try:
            # Approach #1: Pressure center parameterization
            logger.debug("Detecting pressure centers...")
            centers, highs_list, lows_list = (
                self.center_detector.detect_and_parameterize(
                    grid_df,
                    target_lat=self.target_lat,
                    target_lon=self.target_lon,
                    return_centers=True,
                )
            )
            features_list.append(centers)
            logger.debug(f"  → {len(centers.columns)} pressure center features")

        except Exception as e:
            logger.error(f"Pressure center detection failed: {e}")
            features_list.append(self.center_detector._empty_features())

        try:
            # Approach #2: Regional aggregation
            logger.debug("Calculating regional aggregation...")
            regional = self.regional_aggregator.calculate(grid_df)
            features_list.append(regional)
            logger.debug(f"  → {len(regional.columns)} regional features")

        except Exception as e:
            logger.error(f"Regional aggregation failed: {e}")
            # Create empty regional features
            empty_regional = pd.DataFrame(
                [{f: 0.0 for f in self.regional_aggregator.get_feature_names()}]
            )
            features_list.append(empty_regional)

        try:
            # Approach #3: Teleconnection indices
            logger.debug("Calculating teleconnection indices...")
            teleconnection = self.teleconnection_calculator.calculate(grid_df)
            features_list.append(teleconnection)
            logger.debug(f"  → {len(teleconnection.columns)} teleconnection features")

        except Exception as e:
            logger.error(f"Teleconnection calculation failed: {e}")
            # Create empty teleconnection features
            empty_tele = pd.DataFrame(
                [{f: 0.0 for f in self.teleconnection_calculator.get_feature_names()}]
            )
            features_list.append(empty_tele)

        try:
            # Approach #4: Synoptic interactions (location-agnostic)
            logger.debug("Calculating synoptic interactions...")
            interaction_dict = self.interaction_calculator.calculate_from_centers(
                highs_list, lows_list
            )
            interaction_df = pd.DataFrame([interaction_dict])
            features_list.append(interaction_df)
            logger.debug(f"  → {len(interaction_df.columns)} interaction features")

        except Exception as e:
            logger.error(f"Synoptic interaction calculation failed: {e}")
            # Create empty interaction features
            empty_interaction = pd.DataFrame(
                [{f: 0.0 for f in self.interaction_calculator.get_feature_names()}]
            )
            features_list.append(empty_interaction)

        # Combine all features
        all_features = pd.concat(features_list, axis=1)

        logger.info(f"Generated {len(all_features.columns)} synoptic features")

        return all_features

    def transform_multiple_dates(self, grid_files: Dict[str, Path]) -> pd.DataFrame:
        """
        Transform multiple dates of synoptic grid data.

        Args:
            grid_files: Dict mapping date strings to Parquet file paths

        Returns:
            DataFrame with one row per date, containing all synoptic features
        """
        all_dates = []

        for date, filepath in sorted(grid_files.items()):
            logger.info(f"Processing {date}...")

            try:
                # Load grid
                grid_df = pd.read_parquet(filepath)

                # Transform
                features = self.transform(grid_df)

                # Add date/time
                features["date"] = date
                features["time"] = pd.to_datetime(f"{date} 12:00:00")

                all_dates.append(features)

            except Exception as e:
                logger.error(f"Failed to process {date}: {e}")
                continue

        if not all_dates:
            logger.error("No dates successfully processed")
            return pd.DataFrame()

        # Combine all dates
        combined = pd.concat(all_dates, ignore_index=True)

        logger.info(
            f"Processed {len(combined)} dates with {len(combined.columns)} features"
        )

        return combined

    def _empty_features(self) -> pd.DataFrame:
        """
        Return empty features when transformation fails.

        Returns:
            DataFrame with 1 row of zero-valued features
        """
        features = {}

        # Empty pressure center features
        empty_centers = self.center_detector._empty_features()
        features.update(empty_centers.iloc[0].to_dict())

        # Empty regional features
        for f in self.regional_aggregator.get_feature_names():
            features[f] = 0.0

        # Empty teleconnection features
        for f in self.teleconnection_calculator.get_feature_names():
            features[f] = 0.0

        # Empty interaction features
        for f in self.interaction_calculator.get_feature_names():
            features[f] = 0.0

        return pd.DataFrame([features])

    def get_feature_names(self) -> list:
        """
        Get list of all feature names produced by this pipeline.

        Returns:
            List of feature names
        """
        features = []

        # Pressure center features
        for i in range(1, self.center_detector.n_highs + 1):
            prefix = f"high{i}"
            features.extend(
                [
                    f"{prefix}_lat",
                    f"{prefix}_lon",
                    f"{prefix}_z500",
                    f"{prefix}_dist_km",
                    f"{prefix}_bearing",
                    f"{prefix}_gradient",
                    f"{prefix}_exists",
                ]
            )

        for i in range(1, self.center_detector.n_lows + 1):
            prefix = f"low{i}"
            features.extend(
                [
                    f"{prefix}_lat",
                    f"{prefix}_lon",
                    f"{prefix}_z500",
                    f"{prefix}_dist_km",
                    f"{prefix}_bearing",
                    f"{prefix}_gradient",
                    f"{prefix}_exists",
                ]
            )

        # Regional features
        features.extend(self.regional_aggregator.get_feature_names())

        # Teleconnection features
        features.extend(self.teleconnection_calculator.get_feature_names())

        # Interaction features
        features.extend(self.interaction_calculator.get_feature_names())

        return features

    def get_n_features(self) -> int:
        """
        Get total number of features produced.

        Returns:
            Number of features
        """
        return len(self.get_feature_names())

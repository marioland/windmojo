"""
Data Downloader

Main orchestration for downloading GFS forecast data from Open-Meteo API.
Coordinates grid calculation, API calls, and data storage.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from windmojo.core.config import (
    get_forecast_config,
    get_grid_distances,
    get_output_directories,
)
from windmojo.data.grid import calculate_all_grids, get_variable_locations
from windmojo.data.open_meteo_client import OpenMeteoClient, OpenMeteoError

logger = logging.getLogger(__name__)


# Variable groupings for efficient API calls
# Based on Open-Meteo Historical Forecast API documentation

# Circle/Grid variables: fetched at 8 circle points (compass directions, excluding center)
# These are the synoptic-scale variables for spatial patterns
CIRCLE_VARIABLES = [
    "pressure_msl",  # Sea Level Pressure
    "cloud_cover",  # Cloud Cover Total
    "cloud_cover_low",  # Cloud Cover Low
    "cloud_cover_mid",  # Cloud Cover Mid
    "cloud_cover_high",  # Cloud Cover High
]

# Center-only variables: ALL available variables from Open-Meteo API
# Fetched only at center point for comprehensive local conditions
CENTER_VARIABLES = [
    # Temperature variables
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    # Pressure variables
    "pressure_msl",
    "surface_pressure",
    # Cloud variables
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    # Precipitation variables
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "snow_depth",
    "precipitation_probability",
    # Wind variables at 10m
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    # Wind variables at different heights (for wind shear)
    "wind_speed_80m",
    "wind_direction_80m",
    "wind_speed_120m",
    "wind_direction_120m",
    "wind_speed_180m",
    "wind_direction_180m",
    # Temperature at heights
    "temperature_80m",
    "temperature_120m",
    "temperature_180m",
    # Solar radiation variables
    "shortwave_radiation",  # Total incoming solar radiation (W/m²)
    "direct_radiation",  # Direct beam radiation (W/m²)
    "diffuse_radiation",  # Scattered radiation (W/m²)
    "direct_normal_irradiance",  # Direct radiation normal to sun
    # Other atmospheric variables
    "visibility",
    "et0_fao_evapotranspiration",  # ("evapotranspiration" is always 0 in the API)
    "vapour_pressure_deficit",
    "weather_code",
    # Soil variables
    "soil_temperature_0cm",
    "soil_temperature_6cm",
    "soil_temperature_18cm",
    "soil_temperature_54cm",
    "soil_moisture_0_1cm",
    "soil_moisture_1_3cm",
    "soil_moisture_3_9cm",
    "soil_moisture_9_27cm",
    "soil_moisture_27_81cm",
]

# Pressure level variables (5 representative levels for complete atmospheric profile)
# Each level gets ALL 6 variables for comprehensive vertical structure
# This provides wind shear calculation across 5 atmospheric levels
CENTER_PRESSURE_LEVELS = [
    # 1000 hPa (~110m above sea level) - Surface boundary layer
    "temperature_1000hPa",
    "relative_humidity_1000hPa",
    "cloud_cover_1000hPa",
    "wind_speed_1000hPa",
    "wind_direction_1000hPa",
    "geopotential_height_1000hPa",
    # 850 hPa (~1500m above sea level) - Lower troposphere (critical for thermal winds)
    "temperature_850hPa",
    "relative_humidity_850hPa",
    "cloud_cover_850hPa",
    "wind_speed_850hPa",
    "wind_direction_850hPa",
    "geopotential_height_850hPa",
    # 700 hPa (~3000m above sea level) - Mid troposphere
    "temperature_700hPa",
    "relative_humidity_700hPa",
    "cloud_cover_700hPa",
    "wind_speed_700hPa",
    "wind_direction_700hPa",
    "geopotential_height_700hPa",
    # 500 hPa (~5600m above sea level) - Mid-upper troposphere (steering level)
    "temperature_500hPa",
    "relative_humidity_500hPa",
    "cloud_cover_500hPa",
    "wind_speed_500hPa",
    "wind_direction_500hPa",
    "geopotential_height_500hPa",
    # 250 hPa (~10400m above sea level) - Upper troposphere (jet stream level)
    "temperature_250hPa",
    "relative_humidity_250hPa",
    "cloud_cover_250hPa",
    "wind_speed_250hPa",
    "wind_direction_250hPa",
    "geopotential_height_250hPa",
]

ALL_VARIABLES = list(set(CIRCLE_VARIABLES + CENTER_VARIABLES + CENTER_PRESSURE_LEVELS))


class DataDownloader:
    """
    Coordinate data downloading from Open-Meteo API.

    Handles grid calculation, API requests, and Parquet file storage.
    """

    def __init__(self, config: Dict, client: Optional[OpenMeteoClient] = None):
        """
        Initialize data downloader.

        Args:
            config: Configuration dictionary (from load_config())
            client: Optional OpenMeteoClient instance (creates new if None)
        """
        self.config = config
        self.client = client or OpenMeteoClient()

        # Extract configuration
        self.area = config["area"]
        self.spot = config["spot"]
        self.center_lat = config["center"]["latitude"]
        self.center_lon = config["center"]["longitude"]
        self.timezone = config.get("timezone", "UTC")

        # Grid configuration
        grid_distances = get_grid_distances(config)
        self.radius_km = grid_distances["radius_km"]

        # Forecast configuration
        forecast_config = get_forecast_config(config)
        self.forecast_time = forecast_config["time"]
        self.horizon_hours = forecast_config["horizon_hours"]

        # Output directories
        output_dirs = get_output_directories(config)
        self.raw_data_dir = output_dirs["raw_data_dir"]

        # Calculate grids
        self.grids = calculate_all_grids(
            self.center_lat, self.center_lon, self.radius_km
        )

        # API call counter
        self.api_calls = 0

    def download_date(
        self, date: str, mode: str = "historical"
    ) -> Optional[pd.DataFrame]:
        """
        Download data for a single date across all grid locations.

        New Architecture:
        - Circle points (8 compass directions): Only synoptic variables (pressure, clouds)
        - Center point: ALL available variables from Open-Meteo API

        Args:
            date: Date string (YYYY-MM-DD)
            mode: 'historical' or 'forecast'

        Returns:
            Combined DataFrame with all locations, or None if download failed
        """
        logger.info(f"Downloading data for {date}...")

        all_data = []

        try:
            # 1. Fetch circle variables (8 compass points only, excluding center)
            logger.debug(
                "Fetching circle variables (8 points: N, NE, E, SE, S, SW, W, NW)..."
            )
            circle_data = self._fetch_circle_variables(CIRCLE_VARIABLES, date, mode)
            if circle_data is not None:
                all_data.append(circle_data)

            # 2. Fetch ALL variables at center point
            logger.debug("Fetching all variables at center (1 point)...")
            center_data = self._fetch_center_variables(
                CENTER_VARIABLES + CENTER_PRESSURE_LEVELS, date, mode
            )
            if center_data is not None:
                all_data.append(center_data)

            if not all_data:
                logger.error(f"No data collected for {date}")
                return None

            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)

            logger.info(
                f"  ✓ Downloaded {len(combined_df)} rows "
                f"({self.api_calls} API calls for this date)"
            )

            return combined_df

        except Exception as e:
            logger.error(f"Failed to download data for {date}: {e}")
            return None

    def _fetch_circle_variables(
        self, variables: List[str], date: str, mode: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch synoptic variables for 8 circle points (compass directions).

        Excludes center point - fetches only the 8 surrounding points.

        Args:
            variables: List of variable names (CIRCLE_VARIABLES)
            date: Date string (YYYY-MM-DD)
            mode: 'historical' or 'forecast'

        Returns:
            DataFrame with all 8 circle locations
        """
        # Get all 9 grid points, then exclude center
        all_grid_locations = self.grids["grid"]
        circle_locations = {
            k: v for k, v in all_grid_locations.items() if k != "center"
        }

        all_location_data = []

        for location_name, (lat, lon) in circle_locations.items():
            try:
                df = self.client.fetch_for_location(
                    latitude=lat,
                    longitude=lon,
                    location_name=location_name,
                    date=date,
                    variables=variables,
                    mode=mode,
                    timezone=self.timezone,
                    forecast_time=self.forecast_time,
                    horizon_hours=self.horizon_hours,
                )

                self.api_calls += 1

                if not df.empty:
                    all_location_data.append(df)

                # Add delay to avoid rate limiting (300ms between requests)
                time.sleep(0.3)

            except OpenMeteoError as e:
                logger.warning(f"  Failed to fetch {location_name}: {e}")
                continue

        if not all_location_data:
            return None

        return pd.concat(all_location_data, ignore_index=True)

    def _fetch_center_variables(
        self, variables: List[str], date: str, mode: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch ALL available variables at center point.

        This includes all surface, pressure level, and atmospheric variables
        from the Open-Meteo API.

        Args:
            variables: List of variable names (CENTER_VARIABLES + CENTER_PRESSURE_LEVELS)
            date: Date string (YYYY-MM-DD)
            mode: 'historical' or 'forecast'

        Returns:
            DataFrame with center location data
        """
        center_location = self.grids["center"]
        lat, lon = center_location["center"]

        try:
            df = self.client.fetch_for_location(
                latitude=lat,
                longitude=lon,
                location_name="center",
                date=date,
                variables=variables,
                mode=mode,
                timezone=self.timezone,
                forecast_time=self.forecast_time,
                horizon_hours=self.horizon_hours,
            )

            self.api_calls += 1

            # Add delay to avoid rate limiting (300ms)
            time.sleep(0.3)

            if df.empty:
                logger.warning("  Center location returned empty data")
                return None

            return df

        except OpenMeteoError as e:
            logger.warning(f"  Failed to fetch center location: {e}")
            return None

    def save_to_parquet(self, df: pd.DataFrame, date: str) -> Path:
        """
        Save DataFrame to Parquet file.

        Args:
            df: DataFrame to save
            date: Date string (YYYY-MM-DD) for filename

        Returns:
            Path to saved file
        """
        # Create output directory
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: {spot}_YYYYMMDD.parquet
        date_str = date.replace("-", "")
        filename = f"{self.spot}_{date_str}.parquet"
        filepath = self.raw_data_dir / filename

        # Save to Parquet
        df.to_parquet(filepath, index=False, compression="snappy")

        file_size_kb = filepath.stat().st_size / 1024

        logger.info(f"  ✓ Saved: {filename} ({file_size_kb:.1f} KB)")

        return filepath

    def download_date_range(
        self,
        start_date: str,
        end_date: str,
        mode: str = "historical",
        rerun: bool = False,
    ) -> Dict[str, any]:
        """
        Download data for a range of dates.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            mode: 'historical' or 'forecast'
            rerun: If True, delete existing files before downloading

        Returns:
            Dictionary with download statistics
        """
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate date list
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        logger.info(
            f"\nStarting download: {len(dates)} days ({start_date} to {end_date})"
        )

        # Handle rerun flag
        if rerun and self.raw_data_dir.exists():
            logger.info(f"🗑️  Deleting existing data in {self.raw_data_dir}...")
            for file in self.raw_data_dir.glob(f"{self.spot}_*.parquet"):
                file.unlink()
                logger.info(f"  Deleted: {file.name}")

        # Download each date
        successful = 0
        failed = 0
        total_size_kb = 0
        failed_dates = []

        for i, date in enumerate(dates, 1):
            logger.info(f"\n[{i}/{len(dates)}] Processing {date}...")

            # Check if file already exists (unless rerun mode)
            date_str = date.replace("-", "")
            filename = f"{self.spot}_{date_str}.parquet"
            filepath = self.raw_data_dir / filename

            if filepath.exists() and not rerun:
                logger.info(f"  ✓ Already exists: {filename} (skipping)")
                total_size_kb += filepath.stat().st_size / 1024
                successful += 1
                continue

            # Reset API call counter for this date
            self.api_calls = 0

            # Download data
            df = self.download_date(date, mode)

            if df is not None:
                # Save to Parquet
                filepath = self.save_to_parquet(df, date)
                total_size_kb += filepath.stat().st_size / 1024
                successful += 1
            else:
                failed += 1
                failed_dates.append(date)

        # Summary statistics
        stats = {
            "total_dates": len(dates),
            "successful": successful,
            "failed": failed,
            "failed_dates": failed_dates,
            "total_size_kb": total_size_kb,
            "mode": mode,
        }

        return stats

    def download_from_observation_file(
        self, observation_file: str, mode: str = "historical", rerun: bool = False
    ) -> Dict[str, any]:
        """
        Download data for dates present in observation file.

        Reads observation CSV to determine which dates to download.

        Args:
            observation_file: Path to observation CSV file
            mode: 'historical' or 'forecast'
            rerun: If True, delete existing files before downloading

        Returns:
            Dictionary with download statistics
        """
        # Load observation file
        logger.info(f"Loading observation file: {observation_file}")

        try:
            obs_df = pd.read_csv(observation_file)

            # Detect timestamp column
            timestamp_col = None
            for col in ["timestamp", "time", "datetime", "date"]:
                if col in obs_df.columns:
                    timestamp_col = col
                    break

            if timestamp_col is None:
                raise ValueError(
                    f"Could not find timestamp column in {observation_file}. "
                    f"Expected one of: timestamp, time, datetime, date"
                )

            # Parse timestamps - handle custom format like "2025-01-16_1900"
            try:
                obs_df[timestamp_col] = pd.to_datetime(obs_df[timestamp_col])
            except (ValueError, TypeError):
                # Try custom format: YYYY-MM-DD_HHMM
                obs_df[timestamp_col] = pd.to_datetime(
                    obs_df[timestamp_col], format="%Y-%m-%d_%H%M"
                )

            # Extract unique dates
            dates = obs_df[timestamp_col].dt.date.unique()
            dates = sorted([d.strftime("%Y-%m-%d") for d in dates])

            logger.info(f"  Found {len(dates)} unique dates in observations")
            logger.info(f"  Date range: {dates[0]} to {dates[-1]}")

            # Download using date range
            return self.download_date_range(
                start_date=dates[0], end_date=dates[-1], mode=mode, rerun=rerun
            )

        except Exception as e:
            logger.error(f"Failed to load observation file: {e}")
            raise


def print_download_summary(stats: Dict[str, any]):
    """
    Print human-readable download summary.

    Args:
        stats: Statistics dictionary from download_date_range()
    """
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total dates: {stats['total_dates']}")
    logger.info(f"  Successful: {stats['successful']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Total size: {stats['total_size_kb'] / 1024:.1f} MB")
    logger.info(f"  Mode: {stats['mode']}")

    if stats["failed"] > 0:
        logger.warning(f"  Failed dates: {', '.join(stats['failed_dates'])}")

    if stats["successful"] == stats["total_dates"]:
        logger.info("All downloads completed successfully!")
    else:
        logger.warning(f"{stats['failed']} date(s) failed to download")

    logger.info("=" * 60)

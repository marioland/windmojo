"""
Synoptic Grid Downloader - Experiment8

Downloads large-scale European weather grid for synoptic pattern analysis.
Creates 0.5° resolution grid (11,700 points) with geopotential heights,
temperature, and pressure data at 12:00 UTC.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from windmojo.data.open_meteo_client import OpenMeteoClient, OpenMeteoError

logger = logging.getLogger(__name__)


# Synoptic variables for large-scale pattern analysis
SYNOPTIC_VARIABLES = [
    "geopotential_height_500hPa",  # Upper-level patterns (ridges/troughs)
    "geopotential_height_850hPa",  # Lower-level synoptic features
    "temperature_850hPa",  # Thermal structure
    "pressure_msl",  # Mean sea level pressure
]


class SynopticGridDownloader:
    """
    Download European-scale weather grid for synoptic pattern analysis.

    Creates a regular lat/lon grid over Europe and downloads geopotential
    height, temperature, and pressure data at 12:00 UTC for each day.
    """

    def __init__(self, config: Dict, client: Optional[OpenMeteoClient] = None):
        """
        Initialize synoptic grid downloader.

        Args:
            config: Configuration dictionary with synoptic_grid section
            client: Optional OpenMeteoClient instance
        """
        self.config = config
        self.client = client or OpenMeteoClient()

        # Extract synoptic grid configuration
        synoptic_config = config.get("synoptic_grid", {})

        if not synoptic_config.get("enabled", False):
            raise ValueError("synoptic_grid.enabled must be true in config")

        # Grid coverage
        coverage = synoptic_config["coverage"]
        self.lat_min = coverage["lat_min"]
        self.lat_max = coverage["lat_max"]
        self.lon_min = coverage["lon_min"]
        self.lon_max = coverage["lon_max"]

        # Grid resolution
        self.resolution = synoptic_config["resolution"]

        # Time (12:00 UTC for lunch-time synoptic situation)
        self.time_utc = synoptic_config["time_utc"]

        # Variables to download
        self.variables = synoptic_config.get("variables", SYNOPTIC_VARIABLES)

        # Output directory
        output_template = synoptic_config.get(
            "output_dir", "data/{area}/{spot}/raw/synoptic"
        )
        self.output_dir = Path(
            output_template.format(area=config["area"], spot=config["spot"])
        )

        # Generate grid points
        self.grid_points = self._generate_grid()
        logger.info(
            f"Synoptic grid: {len(self.grid_points)} points "
            f"({self.lat_min}°N-{self.lat_max}°N, "
            f"{self.lon_min}°E-{self.lon_max}°E, "
            f"{self.resolution}° resolution)"
        )

        # API call counter
        self.api_calls = 0

    def _generate_grid(self) -> List[Dict]:
        """
        Generate regular lat/lon grid points.

        Returns:
            List of dicts with 'lat', 'lon', 'name' keys
        """
        # Generate grid coordinates
        lats = np.arange(self.lat_min, self.lat_max + 0.001, self.resolution)
        lons = np.arange(self.lon_min, self.lon_max + 0.001, self.resolution)

        grid_points = []
        for lat in lats:
            for lon in lons:
                grid_points.append(
                    {
                        "lat": round(lat, 2),
                        "lon": round(lon, 2),
                        "name": f"grid_{lat:.1f}_{lon:.1f}",
                    }
                )

        return grid_points

    def download_date(
        self, date: str, mode: str = "historical"
    ) -> Optional[pd.DataFrame]:
        """
        Download synoptic grid for a single date at 12:00 UTC.

        Args:
            date: Date string (YYYY-MM-DD)
            mode: 'historical' or 'forecast'

        Returns:
            DataFrame with all grid points, or None if download failed
        """
        logger.info(
            f"Downloading synoptic grid for {date} @ {self.time_utc} UTC "
            f"({len(self.grid_points)} points)..."
        )

        all_data = []
        failed_points = 0
        start_time = time.time()

        # Download data for each grid point
        for i, point in enumerate(self.grid_points):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(self.grid_points) - i - 1) / rate
                logger.info(
                    f"Progress: {i+1}/{len(self.grid_points)} points "
                    f"({i+1/len(self.grid_points)*100:.1f}%) "
                    f"- ETA: {remaining/60:.1f} min"
                )

            try:
                # Fetch data for this point
                data = self._fetch_point(point, date, mode)

                if data is not None:
                    # Add location info
                    data["location"] = point["name"]
                    data["lat"] = point["lat"]
                    data["lon"] = point["lon"]
                    all_data.append(data)
                else:
                    failed_points += 1

            except OpenMeteoError as e:
                logger.warning(f"Failed to download {point['name']}: {e}")
                failed_points += 1
                continue

            # Rate limiting (300ms delay to respect API limits)
            time.sleep(0.3)

        # Combine all data
        if not all_data:
            logger.error(f"No data downloaded for {date}")
            return None

        combined = pd.concat(all_data, ignore_index=True)

        elapsed = time.time() - start_time
        logger.info(
            f"Downloaded {len(all_data)} points in {elapsed/60:.1f} min "
            f"({failed_points} failed)"
        )
        logger.info(f"API calls: {self.api_calls}")

        return combined

    def _fetch_point(self, point: Dict, date: str, mode: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single grid point.

        Args:
            point: Dict with 'lat', 'lon' keys
            date: Date string (YYYY-MM-DD)
            mode: 'historical' or 'forecast'

        Returns:
            DataFrame with hourly data, or None if failed
        """
        # Parse date
        date_obj = datetime.strptime(date, "%Y-%m-%d")

        try:
            # Historical mode: Use historical forecast API
            if mode == "historical":
                data = self.client.fetch_historical_forecast(
                    latitude=point["lat"],
                    longitude=point["lon"],
                    start_date=date,
                    end_date=date,
                    variables=self.variables,
                )
                self.api_calls += 1

            # Forecast mode: Use current forecast API
            else:
                data = self.client.fetch_forecast(
                    latitude=point["lat"],
                    longitude=point["lon"],
                    start_date=date,
                    end_date=date,
                    variables=self.variables,
                )
                self.api_calls += 1

            if data is None or data.empty:
                return None

            # Filter to 12:00 UTC only (lunch-time synoptic situation)
            if "time" in data.columns:
                data["time"] = pd.to_datetime(data["time"])
                # Filter to the specific hour
                hour_target = int(self.time_utc.split(":")[0])
                data = data[data["time"].dt.hour == hour_target]

            if data.empty:
                logger.warning(f"No data at {self.time_utc} UTC for {point['name']}")
                return None

            return data

        except OpenMeteoError as e:
            logger.warning(f"API error for {point['name']}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {point['name']}: {e}")
            return None

    def download_date_range(
        self,
        start_date: str,
        end_date: str,
        mode: str = "historical",
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download synoptic grid for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            mode: 'historical' or 'forecast'
            save: Whether to save each day to Parquet

        Returns:
            Dict mapping date strings to DataFrames
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        results = {}
        current = start

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")

            # Download this date
            data = self.download_date(date_str, mode)

            if data is not None:
                results[date_str] = data

                # Save to Parquet if requested
                if save:
                    self.save_parquet(data, date_str)

            # Move to next day
            current += timedelta(days=1)

        logger.info(f"Downloaded {len(results)} days from {start_date} to {end_date}")
        return results

    def save_parquet(self, data: pd.DataFrame, date: str) -> Path:
        """
        Save synoptic grid data to Parquet file.

        Args:
            data: DataFrame with synoptic grid data
            date: Date string (YYYY-MM-DD) for filename

        Returns:
            Path to saved file
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"synoptic_grid_{date.replace('-', '')}.parquet"
        filepath = self.output_dir / filename

        # Save to Parquet with compression
        data.to_parquet(filepath, engine="pyarrow", compression="snappy")

        logger.info(
            f"Saved synoptic grid to {filepath} "
            f"({data.shape[0]} points, {data.shape[1]} variables)"
        )

        return filepath

    def load_parquet(self, date: str) -> Optional[pd.DataFrame]:
        """
        Load synoptic grid data from Parquet file.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            DataFrame, or None if file doesn't exist
        """
        filename = f"synoptic_grid_{date.replace('-', '')}.parquet"
        filepath = self.output_dir / filename

        if not filepath.exists():
            logger.warning(f"Synoptic grid file not found: {filepath}")
            return None

        data = pd.read_parquet(filepath)
        logger.debug(
            f"Loaded synoptic grid from {filepath} " f"({data.shape[0]} points)"
        )

        return data

    def get_grid_summary(self) -> Dict:
        """
        Get summary information about the synoptic grid.

        Returns:
            Dict with grid parameters and statistics
        """
        return {
            "n_points": len(self.grid_points),
            "lat_range": (self.lat_min, self.lat_max),
            "lon_range": (self.lon_min, self.lon_max),
            "resolution_deg": self.resolution,
            "resolution_km": self.resolution * 111,  # Approx km per degree
            "time_utc": self.time_utc,
            "variables": self.variables,
            "output_dir": str(self.output_dir),
        }

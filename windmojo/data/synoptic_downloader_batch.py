"""
Synoptic Grid Batch Downloader - Experiment8 (OPTIMIZED)

Downloads large-scale European weather grid using BATCH requests.
Uses Open-Meteo's comma-separated array notation to fetch up to 1000
locations per request, achieving 700x speedup over individual calls.

Key optimization: 11,921 points in 12 API calls instead of 11,921 calls.
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from windmojo.data.open_meteo_client import OpenMeteoError

logger = logging.getLogger(__name__)


# Synoptic variables for large-scale pattern analysis
SYNOPTIC_VARIABLES = [
    "geopotential_height_500hPa",
    "geopotential_height_850hPa",
    "temperature_850hPa",
    "pressure_msl",
]


class SynopticGridBatchDownloader:
    """
    Download European-scale weather grid using BATCH API requests.

    Optimized version that chunks grid into batches of 1000 points per request,
    reducing download time from 60+ hours to ~3 hours for 31 days.
    """

    # API endpoints
    HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    # Batch size (reduced to avoid HTTP 414 Request-URI Too Large)
    # 1000 points creates ~20k char URL which exceeds nginx limit
    # 100 points = ~2k chars = safe
    BATCH_SIZE = 100

    def __init__(self, config: Dict):
        """
        Initialize batch synoptic grid downloader.

        Args:
            config: Configuration dictionary with synoptic_grid section
        """
        self.config = config

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
        self.n_batches = (
            len(self.grid_points) + self.BATCH_SIZE - 1
        ) // self.BATCH_SIZE

        logger.info(f"Synoptic BATCH grid initialized:")
        logger.info(f"  Grid points: {len(self.grid_points):,}")
        logger.info(
            f"  Coverage: {self.lat_min}°N-{self.lat_max}°N, "
            f"{self.lon_min}°E-{self.lon_max}°E"
        )
        logger.info(f"  Resolution: {self.resolution}°")
        logger.info(f"  Batch size: {self.BATCH_SIZE}")
        logger.info(f"  Total batches: {self.n_batches}")
        logger.info(
            f"  Requests per day: {self.n_batches} (vs {len(self.grid_points):,} individual)"
        )

        # API call counter
        self.api_calls = 0

        # Load API key from .env file (optional, for commercial tier)
        load_dotenv()
        self.api_key = os.getenv("OPEN_METEO_API_KEY") or os.getenv("apikey")

        if self.api_key:
            logger.info("  API key loaded (commercial tier)")
        else:
            logger.debug("  No API key (using free tier)")

        # Parallel download configuration
        self.n_workers = 1  # Number of parallel workers

        # Adaptive rate limiter: Accelerate on success, slow down on 429 errors
        # With API key, can accelerate to zero delay
        if self.api_key:
            self.request_interval = 0.5  # Start fast with API key (0.5s)
            self.min_interval = 0.0  # Can go to zero delay with commercial tier
            self.max_interval = 60.0  # Fallback if something goes wrong
        else:
            self.request_interval = 6.0  # Start conservative without API key
            self.min_interval = 3.0  # Fastest for free tier
            self.max_interval = 60.0  # Slowest when hitting limits

        self.consecutive_successes = 0
        self.consecutive_failures = 0

        # Thread-safe lock for rate limit management in parallel mode
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0.0

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
        self, date: str, mode: str = "historical", n_workers: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Download synoptic grid for a single date using PARALLEL BATCH requests.

        Args:
            date: Date string (YYYY-MM-DD)
            mode: 'historical' or 'forecast'
            n_workers: Number of parallel workers (default: self.n_workers)

        Returns:
            DataFrame with all grid points, or None if download failed
        """
        workers = n_workers if n_workers is not None else self.n_workers

        logger.info(f"Downloading synoptic grid for {date} @ {self.time_utc} UTC")
        logger.info(
            f"  Method: PARALLEL BATCH ({workers} workers) with ADAPTIVE rate limiting"
        )
        logger.info(f"  Total points: {len(self.grid_points):,}")
        logger.info(f"  Total batches: {self.n_batches}")
        logger.info(
            f"  Rate: {self.request_interval}s (adaptive: {self.min_interval}s-{self.max_interval}s)"
        )
        estimated_time = self.n_batches * self.request_interval / (60 * workers)
        logger.info(f"  Estimated time: ~{estimated_time:.1f} minutes (optimistic)")

        start_time = time.time()

        # Download in parallel with thread pool
        all_data = []
        failed_batches = []

        # Prepare batch tasks
        batch_tasks = []
        for batch_idx in range(self.n_batches):
            batch_start = batch_idx * self.BATCH_SIZE
            batch_end = min((batch_idx + 1) * self.BATCH_SIZE, len(self.grid_points))
            batch = self.grid_points[batch_start:batch_end]
            batch_tasks.append((batch_idx, batch, date, mode))

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_batch = {
                executor.submit(self._download_batch_parallel, *task): task[0]
                for task in batch_tasks
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                completed += 1

                try:
                    batch_data, is_rate_limited = future.result()

                    with self.rate_limit_lock:
                        if batch_data is not None:
                            all_data.append(batch_data)
                            self.consecutive_successes += 1
                            self.consecutive_failures = 0

                            # Accelerate: Reduce interval after 5 consecutive successes
                            if self.consecutive_successes >= 5:
                                old_interval = self.request_interval
                                self.request_interval = max(
                                    self.min_interval,
                                    self.request_interval * 0.8,  # 20% faster
                                )
                                if old_interval != self.request_interval:
                                    logger.info(
                                        f"  ⚡ Accelerating: {old_interval:.1f}s → {self.request_interval:.1f}s"
                                    )
                                self.consecutive_successes = 0

                            logger.info(
                                f"  ✓ Batch {batch_idx + 1}/{self.n_batches} "
                                f"({len(batch_data)} points) [{completed}/{self.n_batches}]"
                            )

                        elif is_rate_limited:
                            # Rate limit hit - slow down significantly
                            failed_batches.append(batch_idx + 1)
                            self.consecutive_failures += 1
                            self.consecutive_successes = 0

                            old_interval = self.request_interval
                            self.request_interval = min(
                                self.max_interval,
                                self.request_interval * 2.0,  # Double the wait time
                            )
                            logger.warning(
                                f"  ✗ Batch {batch_idx + 1} RATE LIMITED - "
                                f"Slowing down: {old_interval:.1f}s → {self.request_interval:.1f}s"
                            )

                        else:
                            # Other failure (not rate limit)
                            failed_batches.append(batch_idx + 1)
                            logger.warning(
                                f"  ✗ Batch {batch_idx + 1} failed (non-rate-limit error)"
                            )

                except Exception as e:
                    logger.error(f"  ✗ Batch {batch_idx + 1} error: {e}")
                    failed_batches.append(batch_idx + 1)

        # Log summary
        elapsed = time.time() - start_time
        success_count = len(all_data)
        failed_count = len(failed_batches)

        logger.info(f"\nDownload complete:")
        logger.info(f"  Time: {elapsed:.1f}s")
        logger.info(f"  Success: {success_count}/{self.n_batches} batches")
        logger.info(f"  Failed: {failed_count} batches")
        logger.info(f"  API calls: {self.api_calls}")

        if failed_batches:
            logger.warning(
                f"  Failed batch IDs: {failed_batches[:10]}{'...' if len(failed_batches) > 10 else ''}"
            )

        # Combine all batches
        if not all_data:
            logger.error(f"No data downloaded for {date}")
            return None

        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"  Total points downloaded: {len(combined):,}")

        return combined

    def _download_batch_parallel(
        self, batch_idx: int, batch: List[Dict], date: str, mode: str
    ) -> tuple[Optional[pd.DataFrame], bool]:
        """
        Download a single batch with thread-safe adaptive rate limiting.

        Args:
            batch_idx: Batch index (for logging)
            batch: List of grid points
            date: Date string
            mode: 'historical' or 'forecast'

        Returns:
            Tuple of (DataFrame or None, is_rate_limited bool)
        """
        # Thread-safe adaptive rate limiting
        with self.rate_limit_lock:
            current_interval = self.request_interval

            # Calculate time to wait based on last request
            if self.last_request_time > 0:
                elapsed = time.time() - self.last_request_time
                wait_time = max(0, current_interval - elapsed)
            else:
                wait_time = 0

            # Update last request time
            self.last_request_time = time.time() + wait_time

        # Wait outside the lock to allow other threads to check
        if wait_time > 0:
            time.sleep(wait_time)

        try:
            batch_data, is_rate_limited = self._fetch_batch(batch, date, mode)
            return batch_data, is_rate_limited

        except Exception as e:
            logger.error(f"  Batch {batch_idx + 1} exception: {e}")
            return None, False

    def _fetch_batch(
        self, batch: List[Dict], date: str, mode: str
    ) -> tuple[Optional[pd.DataFrame], bool]:
        """
        Fetch data for a batch of grid points in ONE API call.

        Args:
            batch: List of grid point dicts with 'lat', 'lon', 'name'
            date: Date string (YYYY-MM-DD)
            mode: 'historical' or 'forecast'

        Returns:
            Tuple of (DataFrame or None, is_rate_limited bool)
        """
        # Build comma-separated lat/lon strings
        lats = ",".join(str(point["lat"]) for point in batch)
        lons = ",".join(str(point["lon"]) for point in batch)

        # Select API endpoint
        url = self.HISTORICAL_URL if mode == "historical" else self.FORECAST_URL

        # Build parameters
        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": date,
            "end_date": date,
            "hourly": ",".join(self.variables),
            "timezone": "UTC",
            "models": "best_match",  # Consistent model selection
        }

        # Add API key if available
        if self.api_key:
            params["apikey"] = self.api_key

        try:
            # Make API request
            response = requests.get(url, params=params, timeout=60)

            # Increment API call counter
            self.api_calls += 1

            # Check for rate limiting (HTTP 429)
            if response.status_code == 429:
                error_data = response.json() if response.text else {}
                reason = error_data.get("reason", "Rate limit exceeded")
                logger.error(f"API rate limit: {reason}")
                return None, True  # Return rate limited flag

            if response.status_code != 200:
                logger.error(
                    f"API returned status {response.status_code}: {response.text[:200]}"
                )
                return None, False

            # Parse JSON (array of location objects)
            data = response.json()

            # Handle error response
            if "error" in data:
                reason = data.get("reason", "Unknown error")
                logger.error(f"API error: {reason}")
                # Check if it's a rate limit error in the response body
                is_rate_limit = "limit" in reason.lower() or "429" in reason
                return None, is_rate_limit

            # Parse array response
            all_rows = []

            for i, point in enumerate(batch):
                # Get data for this location
                if isinstance(data, list):
                    location_data = data[i]
                else:
                    # Single location response (shouldn't happen with batch)
                    location_data = data

                # Extract hourly data
                if "hourly" not in location_data:
                    logger.warning(f"No hourly data for {point['name']}")
                    continue

                hourly = location_data["hourly"]

                # Convert to DataFrame
                df = pd.DataFrame(hourly)

                # Add location info
                df["location"] = point["name"]
                df["lat"] = point["lat"]
                df["lon"] = point["lon"]

                all_rows.append(df)

            if not all_rows:
                return None, False

            # Combine all locations
            batch_df = pd.concat(all_rows, ignore_index=True)

            # Convert time to datetime
            if "time" in batch_df.columns:
                batch_df["time"] = pd.to_datetime(batch_df["time"])

                # Filter to 12:00 UTC only
                hour_target = int(self.time_utc.split(":")[0])
                batch_df = batch_df[batch_df["time"].dt.hour == hour_target]

            return batch_df, False  # Success, not rate limited

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for batch")
            return None, False
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None, False
        except Exception as e:
            logger.error(f"Unexpected error parsing batch response: {e}")
            return None, False

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
        total_days = (end - start).days + 1

        logger.info(f"=" * 70)
        logger.info(f"BATCH DOWNLOAD: {start_date} to {end_date} ({total_days} days)")
        logger.info(f"=" * 70)

        day_num = 0
        while current <= end:
            day_num += 1
            date_str = current.strftime("%Y-%m-%d")

            logger.info(f"\n[Day {day_num}/{total_days}] {date_str}")

            # Download this date
            data = self.download_date(date_str, mode)

            if data is not None:
                results[date_str] = data

                # Save to Parquet if requested
                if save:
                    self.save_parquet(data, date_str)
            else:
                logger.warning(f"  Failed to download {date_str}")

            # Move to next day
            current += timedelta(days=1)

        logger.info(f"\n" + "=" * 70)
        logger.info(f"DOWNLOAD COMPLETE: {len(results)}/{total_days} days successful")
        logger.info(f"Total API calls: {self.api_calls}")
        logger.info(f"=" * 70)

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
            f"  Saved: {filepath.name} "
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
            return None

        data = pd.read_parquet(filepath)
        return data

    def get_grid_summary(self) -> Dict:
        """
        Get summary information about the synoptic grid.

        Returns:
            Dict with grid parameters and statistics
        """
        return {
            "n_points": len(self.grid_points),
            "n_batches": self.n_batches,
            "batch_size": self.BATCH_SIZE,
            "lat_range": (self.lat_min, self.lat_max),
            "lon_range": (self.lon_min, self.lon_max),
            "resolution_deg": self.resolution,
            "resolution_km": self.resolution * 111,
            "time_utc": self.time_utc,
            "variables": self.variables,
            "output_dir": str(self.output_dir),
            "speedup": f"{len(self.grid_points) / self.n_batches:.0f}x",
        }

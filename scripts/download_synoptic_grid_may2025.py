"""
Download synoptic grid data for May 2025 (31 days).

Downloads 0.5° European grid (11,700 points) at 12:00 UTC for:
- geopotential_height_500hPa
- geopotential_height_850hPa
- temperature_850hPa
- pressure_msl

Estimated time: ~30 hours for 31 days
This is Experiment8 Phase 1: Data Collection
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from windmojo.core.config import load_config
from windmojo.data.synoptic_downloader import SynopticGridDownloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Download synoptic grid for May 2025."""

    # Load experiment8 config
    config_path = project_root / "config" / "lake_como" / "dervio_experiment8.yaml"
    logger.info(f"Loading config from: {config_path}")

    config = load_config(str(config_path))

    # Create downloader
    downloader = SynopticGridDownloader(config)

    # Print grid summary
    summary = downloader.get_grid_summary()
    logger.info("=" * 70)
    logger.info("SYNOPTIC GRID CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Grid points:    {summary['n_points']:,}")
    logger.info(f"Lat range:      {summary['lat_range'][0]}°N - {summary['lat_range'][1]}°N")
    logger.info(f"Lon range:      {summary['lon_range'][0]}°E - {summary['lon_range'][1]}°E")
    logger.info(f"Resolution:     {summary['resolution_deg']}° (~{summary['resolution_km']:.0f} km)")
    logger.info(f"Time:           {summary['time_utc']} UTC")
    logger.info(f"Variables:      {', '.join(summary['variables'])}")
    logger.info(f"Output dir:     {summary['output_dir']}")
    logger.info("=" * 70)

    # Download dates: May 2025 (2025-05-01 to 2025-05-31)
    start_date = "2025-05-01"
    end_date = "2025-05-31"

    logger.info(f"\nDownloading synoptic grid for: {start_date} to {end_date}")
    logger.info(f"Total days: 31")

    # Estimate time
    api_calls_per_day = summary['n_points']
    delay_per_call = 0.3  # seconds (API rate limiting)
    time_per_day_min = (api_calls_per_day * delay_per_call) / 60
    total_time_hours = (31 * time_per_day_min) / 60

    logger.info(f"\nESTIMATED TIME:")
    logger.info(f"  Per day:  ~{time_per_day_min:.1f} minutes ({api_calls_per_day:,} API calls)")
    logger.info(f"  Total:    ~{total_time_hours:.1f} hours for 31 days")
    logger.info(f"\nThis will run in the background. You can monitor progress in the logs.")
    logger.info("=" * 70)

    # Confirm start
    response = input("\nStart download? This will take ~30 hours. (y/n): ")
    if response.lower() != 'y':
        logger.info("Download cancelled.")
        return

    # Start download
    logger.info("\n" + "=" * 70)
    logger.info("STARTING DOWNLOAD")
    logger.info("=" * 70)

    start_time = datetime.now()

    try:
        # Use historical forecast API (note: for 2025 we might need to use forecast API)
        # The downloader will handle this
        results = downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            mode='historical',  # Try historical first
            save=True           # Save each day to Parquet
        )

        elapsed = datetime.now() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("DOWNLOAD COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Downloaded: {len(results)} days")
        logger.info(f"Elapsed time: {elapsed}")
        logger.info(f"Files saved to: {downloader.output_dir}")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.info("\n\nDownload interrupted by user.")
        logger.info(f"Partial data saved to: {downloader.output_dir}")
    except Exception as e:
        logger.error(f"\n\nDownload failed with error: {e}")
        logger.error(f"Partial data may be in: {downloader.output_dir}")
        raise


if __name__ == "__main__":
    main()

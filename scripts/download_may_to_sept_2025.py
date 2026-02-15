"""
Download synoptic grid data from May to September 2025.

This script downloads 5 months (153 days) of synoptic weather data
for Experiment 8 training.

Usage:
    uv run python scripts/download_may_to_sept_2025.py

Features:
- Parallel downloads with 8 workers
- Adaptive rate limiting (accelerates to zero delay with API key)
- Skip-existing protection (resume capability)
- Progress tracking and time estimates
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from windmojo.core.config import load_config
from windmojo.data.synoptic_downloader_batch import SynopticGridBatchDownloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Download May-September 2025 synoptic data."""

    # Load experiment8 config
    config_path = project_root / "config" / "lake_como" / "dervio_experiment8.yaml"
    logger.info(f"Loading config from: {config_path}")

    config = load_config(str(config_path))

    # Create downloader
    downloader = SynopticGridBatchDownloader(config)

    # Generate all dates from May 1 to September 30, 2025
    start_date = datetime(2025, 5, 1)
    end_date = datetime(2025, 9, 30)

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    # Check which files already exist
    data_dir = project_root / "data" / "lake_como" / "dervio" / "raw" / "synoptic"
    data_dir.mkdir(parents=True, exist_ok=True)

    existing_files = []
    missing_dates = []

    for date_str in dates:
        filepath = data_dir / f"synoptic_grid_{date_str.replace('-', '')}.parquet"
        if filepath.exists():
            existing_files.append(date_str)
        else:
            missing_dates.append(date_str)

    logger.info("=" * 70)
    logger.info("MAY - SEPTEMBER 2025 DATASET DOWNLOAD")
    logger.info("=" * 70)
    logger.info(f"Date range:          May 1 - Sep 30, 2025")
    logger.info(f"Total dates:         {len(dates)} days")
    logger.info(f"Already downloaded:  {len(existing_files)} days")
    logger.info(f"Missing dates:       {len(missing_dates)} days")
    logger.info(f"")
    logger.info(f"Grid configuration:")
    logger.info(f"  Points per day:    {len(downloader.grid_points):,}")
    logger.info(f"  Batches per day:   {downloader.n_batches}")
    logger.info(f"  Workers:           {downloader.n_workers}")
    logger.info(f"  API key:           {'Yes' if downloader.api_key else 'No'}")

    if existing_files:
        logger.info(f"\nAlready have {len(existing_files)} days:")
        if len(existing_files) <= 10:
            logger.info(f"  {', '.join(existing_files)}")
        else:
            logger.info(f"  {', '.join(existing_files[:5])} ... {', '.join(existing_files[-5:])}")

    if not missing_dates:
        logger.info("\n✓ All May-September 2025 data already downloaded!")
        return True

    logger.info(f"\nMissing dates to download ({len(missing_dates)} days):")
    if len(missing_dates) <= 20:
        logger.info(f"  {', '.join(missing_dates)}")
    else:
        logger.info(f"  {missing_dates[0]} to {missing_dates[-1]}")

    # Calculate estimated time
    est_minutes = len(missing_dates) * downloader.n_batches * downloader.request_interval / (60 * downloader.n_workers)
    est_hours = est_minutes / 60

    logger.info(f"\nEstimated download time:")
    if est_hours < 1:
        logger.info(f"  ~{est_minutes:.0f} minutes")
    else:
        logger.info(f"  ~{est_hours:.1f} hours ({est_minutes:.0f} minutes)")
    logger.info(f"  ({len(missing_dates)} days × {downloader.n_batches} batches × {downloader.request_interval}s / {downloader.n_workers} workers)")

    # Download missing dates
    logger.info("\n" + "=" * 70)
    logger.info("STARTING DOWNLOAD")
    logger.info("=" * 70)

    start_time = datetime.now()
    successful_downloads = 0
    failed_downloads = []

    for i, date_str in enumerate(missing_dates, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"[{i}/{len(missing_dates)}] Downloading {date_str}...")
        logger.info(f"{'='*70}")

        try:
            # Download the grid
            grid_data = downloader.download_date(date_str, mode='historical')

            if grid_data is not None and len(grid_data) > 0:
                # Save to parquet
                filepath = downloader.save_parquet(grid_data, date_str)

                successful_downloads += 1
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                avg_time = elapsed / i
                remaining = avg_time * (len(missing_dates) - i)

                logger.info(f"\n✓ SAVED: {filepath.name}")
                logger.info(f"  Grid points: {len(grid_data):,}")
                logger.info(f"  Progress: {successful_downloads}/{len(missing_dates)} days ({successful_downloads/len(missing_dates)*100:.1f}%)")
                logger.info(f"  Time: Elapsed {elapsed:.1f}min | Remaining ~{remaining:.1f}min")
            else:
                logger.warning(f"\n✗ No data returned for {date_str}")
                failed_downloads.append(date_str)

        except Exception as e:
            logger.error(f"\n✗ Failed to download {date_str}: {e}")
            failed_downloads.append(date_str)
            import traceback
            traceback.print_exc()

    # Summary
    total_time = (datetime.now() - start_time).total_seconds() / 60

    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Successful:       {successful_downloads}/{len(missing_dates)} days")
    logger.info(f"Failed:           {len(failed_downloads)} days")
    logger.info(f"Total time:       {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    logger.info(f"Average per day:  {total_time / len(missing_dates):.1f} minutes")

    if failed_downloads:
        logger.warning(f"\nFailed downloads ({len(failed_downloads)}):")
        if len(failed_downloads) <= 20:
            logger.warning(f"  {', '.join(failed_downloads)}")
        else:
            logger.warning(f"  {failed_downloads[0]} to {failed_downloads[-1]}")
        logger.info("\nRe-run this script to retry failed downloads")

    # Final status check
    final_existing = [d for d in dates if (data_dir / f"synoptic_grid_{d.replace('-', '')}.parquet").exists()]

    logger.info(f"\nFinal status: {len(final_existing)}/{len(dates)} days downloaded ({len(final_existing)/len(dates)*100:.1f}%)")

    if len(final_existing) == len(dates):
        logger.info("\n✓ Full May-September 2025 dataset complete!")
        logger.info(f"  Total days: {len(dates)}")
        logger.info(f"  Data directory: {data_dir}")
        return True
    else:
        missing = len(dates) - len(final_existing)
        logger.warning(f"\n⚠ Still missing {missing} days ({missing/len(dates)*100:.1f}%)")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

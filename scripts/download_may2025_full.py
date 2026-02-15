"""
Download full May 2025 synoptic grid dataset (31 days).

Features:
- Uses batch downloader with adaptive rate limiting
- Skips already downloaded files (overwrite protection)
- Resume capability if interrupted
- Progress tracking and time estimation
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
    """Download May 2025 dataset with skip-existing logic."""

    # Load experiment8 config
    config_path = project_root / "config" / "lake_como" / "dervio_experiment8.yaml"
    logger.info(f"Loading config from: {config_path}")

    config = load_config(str(config_path))

    # Create downloader
    downloader = SynopticGridBatchDownloader(config)

    # Generate all dates in May 2025
    start_date = datetime(2025, 5, 1)
    end_date = datetime(2025, 5, 31)

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
    logger.info("MAY 2025 DATASET DOWNLOAD")
    logger.info("=" * 70)
    logger.info(f"Total dates:         {len(dates)}")
    logger.info(f"Already downloaded:  {len(existing_files)}")
    logger.info(f"Missing dates:       {len(missing_dates)}")

    if existing_files:
        logger.info(f"\nExisting files: {', '.join(existing_files)}")

    if not missing_dates:
        logger.info("\n✓ All May 2025 data already downloaded!")
        return True

    logger.info(f"\nMissing dates to download: {', '.join(missing_dates)}")
    logger.info(f"\nEstimated time: ~{len(missing_dates) * 36} minutes")
    logger.info(f"  ({len(missing_dates)} days × ~36 min/day with adaptive rate limiting)")

    # Download missing dates
    logger.info("\n" + "=" * 70)
    logger.info("STARTING DOWNLOAD")
    logger.info("=" * 70)

    start_time = datetime.now()
    successful_downloads = 0
    failed_downloads = []

    for i, date_str in enumerate(missing_dates, 1):
        logger.info(f"\n[{i}/{len(missing_dates)}] Downloading {date_str}...")

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

                logger.info(f"  ✓ Saved: {filepath.name}")
                logger.info(f"  Grid points: {len(grid_data):,}")
                logger.info(f"  Progress: {successful_downloads}/{len(missing_dates)} days")
                logger.info(f"  Elapsed: {elapsed:.1f} min | Remaining: {remaining:.1f} min")
            else:
                logger.warning(f"  ✗ No data returned for {date_str}")
                failed_downloads.append(date_str)

        except Exception as e:
            logger.error(f"  ✗ Failed to download {date_str}: {e}")
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
    logger.info(f"Total time:       {total_time:.1f} minutes")
    logger.info(f"Average per day:  {total_time / len(missing_dates):.1f} minutes")

    if failed_downloads:
        logger.warning(f"\nFailed downloads: {', '.join(failed_downloads)}")
        logger.info("Re-run this script to retry failed downloads")

    # Final status check
    final_existing = [d for d in dates if (data_dir / f"synoptic_grid_{d.replace('-', '')}.parquet").exists()]

    logger.info(f"\nFinal status: {len(final_existing)}/{len(dates)} days downloaded")

    if len(final_existing) == len(dates):
        logger.info("\n✓ Full May 2025 dataset complete!")
        return True
    else:
        logger.warning(f"\n⚠ Still missing {len(dates) - len(final_existing)} days")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
Test synoptic grid download for a single day.

Quick test to verify the downloader works before running the full 31-day download.
Downloads just 1 day (~30 minutes) to validate the approach.
"""

import sys
from pathlib import Path
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
    """Test synoptic grid download for May 1, 2025."""

    # Load experiment8 config
    config_path = project_root / "config" / "lake_como" / "dervio_experiment8.yaml"
    logger.info(f"Loading config from: {config_path}")

    config = load_config(str(config_path))

    # Create downloader
    downloader = SynopticGridDownloader(config)

    # Print grid summary
    summary = downloader.get_grid_summary()
    logger.info("=" * 70)
    logger.info("SYNOPTIC GRID TEST - SINGLE DAY")
    logger.info("=" * 70)
    logger.info(f"Grid points:    {summary['n_points']:,}")
    logger.info(f"Resolution:     {summary['resolution_deg']}°")
    logger.info(f"Variables:      {', '.join(summary['variables'])}")
    logger.info("=" * 70)

    # Test with a single day: May 1, 2025
    test_date = "2025-05-01"

    logger.info(f"\nTesting download for: {test_date}")
    logger.info(f"Estimated time: ~{(summary['n_points'] * 0.3 / 60):.1f} minutes")

    try:
        # Download single day
        data = downloader.download_date(test_date, mode='historical')

        if data is not None:
            logger.info("\n" + "=" * 70)
            logger.info("DOWNLOAD SUCCESSFUL!")
            logger.info("=" * 70)
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Columns: {list(data.columns)}")
            logger.info(f"\nSample data (first 5 rows):")
            print(data.head())

            # Save to Parquet
            filepath = downloader.save_parquet(data, test_date)
            logger.info(f"\nSaved to: {filepath}")

            # Test loading back
            loaded = downloader.load_parquet(test_date)
            if loaded is not None:
                logger.info(f"✓ Successfully loaded back from Parquet")
                logger.info(f"  Shape: {loaded.shape}")

            logger.info("\n" + "=" * 70)
            logger.info("TEST PASSED! Ready for full download.")
            logger.info("=" * 70)

        else:
            logger.error("\n✗ Download returned no data")
            return False

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

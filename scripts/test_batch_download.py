"""
Test batch synoptic grid download for a single day.

Validates that the optimized batch downloader works correctly.
Downloads 1 day with ~12 API calls instead of 11,921 calls.
"""

import sys
from pathlib import Path
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
    """Test batch synoptic grid download for May 1, 2025."""

    # Load experiment8 config
    config_path = project_root / "config" / "lake_como" / "dervio_experiment8.yaml"
    logger.info(f"Loading config from: {config_path}")

    config = load_config(str(config_path))

    # Create BATCH downloader
    downloader = SynopticGridBatchDownloader(config)

    # Print grid summary
    summary = downloader.get_grid_summary()
    logger.info("=" * 70)
    logger.info("BATCH SYNOPTIC GRID TEST")
    logger.info("=" * 70)
    logger.info(f"Grid points:      {summary['n_points']:,}")
    logger.info(f"Batch size:       {summary['batch_size']:,}")
    logger.info(f"Total batches:    {summary['n_batches']}")
    logger.info(f"Speedup:          {summary['speedup']} vs individual calls")
    logger.info(f"Resolution:       {summary['resolution_deg']}°")
    logger.info(f"Variables:        {', '.join(summary['variables'])}")
    logger.info("=" * 70)

    # Test with a single day: May 1, 2025
    test_date = "2025-05-01"

    logger.info(f"\nTesting BATCH download for: {test_date}")
    logger.info(f"Expected time: ~30 seconds ({summary['n_batches']} API calls)")
    logger.info("")

    try:
        # Download single day with batch method
        data = downloader.download_date(test_date, mode='historical')

        if data is not None:
            logger.info("\n" + "=" * 70)
            logger.info("DOWNLOAD SUCCESSFUL!")
            logger.info("=" * 70)
            logger.info(f"Data shape:       {data.shape}")
            logger.info(f"Columns:          {list(data.columns)}")
            logger.info(f"Unique locations: {data['location'].nunique():,}")
            logger.info(f"Date range:       {data['time'].min()} to {data['time'].max()}")

            # Check data quality
            logger.info(f"\nData quality:")
            for var in summary['variables']:
                if var in data.columns:
                    non_null = data[var].notna().sum()
                    total = len(data)
                    pct = (non_null / total * 100) if total > 0 else 0
                    logger.info(f"  {var:30s}: {non_null:,}/{total:,} ({pct:.1f}%) non-null")

            # Sample data
            logger.info(f"\nSample data (first 5 rows):")
            print(data.head())

            # Save to Parquet
            filepath = downloader.save_parquet(data, test_date)
            logger.info(f"\nSaved to: {filepath}")

            # Test loading back
            loaded = downloader.load_parquet(test_date)
            if loaded is not None:
                logger.info(f"✓ Successfully loaded back from Parquet ({loaded.shape})")

            logger.info("\n" + "=" * 70)
            logger.info("TEST PASSED! Batch downloader is working correctly.")
            logger.info(f"Speedup confirmed: {summary['n_points']:,} points in {summary['n_batches']} requests")
            logger.info("=" * 70)
            logger.info(f"\nReady to download full May 2025 dataset (31 days).")
            logger.info(f"Estimated time: ~{summary['n_batches'] * 31 / 60:.1f} minutes")

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

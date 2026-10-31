#!/usr/bin/env python3
"""
windmojo Data Fetcher

Command-line interface for downloading GFS forecast data from Open-Meteo API.

Usage:
    # Download historical data (for training)
    uv run python -m windmojo.data_fetcher --config config/lake_como/dervio.yaml --mode historical

    # Download forecast data (for production)
    uv run python -m windmojo.data_fetcher --config config/lake_como/dervio.yaml --mode forecast

    # Rerun (delete existing data first)
    uv run python -m windmojo.data_fetcher --config config/lake_como/dervio.yaml --mode historical --rerun

    # Specify custom date range
    uv run python -m windmojo.data_fetcher --config config/lake_como/dervio.yaml --mode historical \\
        --start-date 2024-01-01 --end-date 2024-12-31
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from windmojo.core.config import load_config, print_config_summary
from windmojo.data.grid import print_grid_summary
from windmojo.data.grid_downloader import DataDownloader, print_download_summary

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """
    Setup logging configuration.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download weather forecast data from Open-Meteo API for windmojo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download historical data for training
  %(prog)s --config config/lake_como/dervio.yaml --mode historical

  # Download next 7 days for forecasting
  %(prog)s --config config/lake_como/dervio.yaml --mode forecast

  # Download specific date range
  %(prog)s --config config/lake_como/dervio.yaml --mode historical \\
      --start-date 2024-01-01 --end-date 2024-12-31

  # Rerun (delete existing files first)
  %(prog)s --config config/lake_como/dervio.yaml --mode historical --rerun
        """,
    )

    parser.add_argument(
        "--config", required=True, help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["historical", "forecast"],
        help="Download mode: historical (past data for training) or forecast (future data for predictions)",
    )

    parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD). If not specified, uses observation file or forecast defaults.",
    )

    parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD). If not specified, uses observation file or forecast defaults.",
    )

    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Delete existing data files before downloading",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )

    return parser.parse_args()


def get_date_range(args, config):
    """
    Determine date range to download.

    Args:
        args: Parsed command line arguments
        config: Configuration dictionary

    Returns:
        (start_date, end_date) tuple as strings (YYYY-MM-DD)
    """
    # If dates explicitly provided, use them
    if args.start_date and args.end_date:
        return args.start_date, args.end_date

    # For historical mode, check if observation file exists
    if args.mode == "historical":
        if "measurement_file" in config:
            measurement_file = Path(config["measurement_file"])

            if measurement_file.exists():
                logger.info(
                    f"Using date range from observation file: {measurement_file}"
                )
                return None, None  # Will be handled by downloader

            else:
                logger.error(f"Measurement file not found: {measurement_file}")
                logger.error("  Please specify --start-date and --end-date explicitly")
                sys.exit(1)

        # No measurement file specified
        logger.error("For historical mode, either:")
        logger.error("  1. Specify measurement_file in config, or")
        logger.error("  2. Provide --start-date and --end-date arguments")
        sys.exit(1)

    # For forecast mode, use next N days
    else:
        forecast_config = config.get("forecast", {})
        forecast_days = forecast_config.get("forecast_days", 7)

        today = datetime.now()
        start_date = today.strftime("%Y-%m-%d")
        end_date = (today + timedelta(days=forecast_days - 1)).strftime("%Y-%m-%d")

        logger.info(f"Forecast mode: Downloading next {forecast_days} days")

        return start_date, end_date


def main():
    """Main execution function."""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Print header
    logger.info("=" * 60)
    logger.info("windmojo Data Fetcher")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")

    if args.rerun:
        logger.warning("RERUN mode: Existing data will be deleted")

    if args.dry_run:
        logger.info("DRY RUN mode: No data will be downloaded")

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)

        print_config_summary(config)

        # Get date range
        start_date, end_date = get_date_range(args, config)

        # Create downloader
        logger.info("Initializing downloader...")
        downloader = DataDownloader(config)

        # Print grid summary
        print_grid_summary(downloader.grids)

        # Calculate estimated API calls
        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            num_days = (end - start).days + 1

            # Each date requires ~27 API calls (17 pressure + 9 surface + 1 center)
            estimated_calls = num_days * 27

            logger.info(f"Estimated API calls: ~{estimated_calls:,}")
            logger.info(f"  (27 calls/day x {num_days} days)")

            if estimated_calls > 10000:
                logger.warning(
                    f"{estimated_calls:,} API calls exceeds free tier limit (10,000/day)"
                )
                logger.warning(
                    f"  This will require {estimated_calls // 10000 + 1} days to complete"
                )

        if args.dry_run:
            logger.info("Dry run complete. No data downloaded.")
            return

        # Perform download
        logger.info("Starting download...")

        if start_date and end_date:
            # Download specific date range
            stats = downloader.download_date_range(
                start_date=start_date,
                end_date=end_date,
                mode=args.mode,
                rerun=args.rerun,
            )
        else:
            # Download from observation file
            stats = downloader.download_from_observation_file(
                observation_file=config["measurement_file"],
                mode=args.mode,
                rerun=args.rerun,
            )

        # Print summary
        print_download_summary(stats)

        # Next steps
        if stats["successful"] > 0:
            logger.info("Next steps:")
            logger.info("  1. Run feature engineering:")
            logger.info(
                f"     uv run python -m windmojo.feature_engineer --config {args.config}"
            )
            logger.info("  2. Train model:")
            logger.info("     uv run python cli.py train --spot <spot> (see README)")

        sys.exit(0 if stats["failed"] == 0 else 1)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()

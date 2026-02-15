"""
Download synoptic data commands.

Download historical and forecast synoptic grid data from Open-Meteo API.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import click
import yaml

from windmojo.data.synoptic_downloader_batch import SynopticGridBatchDownloader

logger = logging.getLogger(__name__)


@click.group()
def download():
    """Download synoptic grid data."""
    pass


@download.command()
@click.option(
    "--config",
    default="config/lake_como/dervio_experiment8.yaml",
    help="Configuration file",
    type=click.Path(exists=True),
)
@click.option(
    "--start-date", help="Start date (YYYY-MM-DD). Overrides config.", type=str
)
@click.option("--end-date", help="End date (YYYY-MM-DD). Overrides config.", type=str)
@click.option(
    "--test-mode",
    is_flag=True,
    help="Test mode: download only 3 days from start date",
)
@click.option(
    "--mode",
    default="historical",
    type=click.Choice(["historical", "forecast", "both"]),
    help="Download mode",
)
@click.option(
    "--output-dir", help="Output directory. Overrides config.", type=click.Path()
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Skip cache and download fresh data",
)
@click.option(
    "--workers",
    type=int,
    help="Number of parallel download workers (default: 8)",
)
def synoptic(
    config, start_date, end_date, test_mode, mode, output_dir, no_cache, workers
):
    """Download synoptic grid data from Open-Meteo API.

    Examples:

        # Test mode (3 days only)
        windmojo download synoptic --test-mode

        # Custom date range
        windmojo download synoptic --start-date 2025-05-01 --end-date 2025-05-10

        # Full May-September 2025
        windmojo download synoptic --start-date 2025-05-01 --end-date 2025-09-30
    """
    # Load configuration
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    # Determine date range
    if start_date and end_date:
        # Use command-line dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    elif "download" in cfg and "start_date" in cfg["download"]:
        # Use config dates
        start = datetime.strptime(cfg["download"]["start_date"], "%Y-%m-%d")
        end = datetime.strptime(cfg["download"]["end_date"], "%Y-%m-%d")
    else:
        # Default to May-September 2025
        start = datetime(2025, 5, 1)
        end = datetime(2025, 9, 30)

    # Test mode: only 3 days
    if test_mode:
        end = start + timedelta(days=2)  # 3 days total (inclusive)
        click.echo(
            f"TEST MODE: Downloading only 3 days ({start.date()} to {end.date()})"
        )

    # Generate date list
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    click.echo(f"Downloading {len(dates)} days: {dates[0]} to {dates[-1]}")

    # Determine output directory
    if output_dir:
        data_dir = Path(output_dir)
    elif "data" in cfg and "raw_dir" in cfg["data"]:
        data_dir = Path(cfg["data"]["raw_dir"]) / "synoptic"
    else:
        area = cfg.get("area", "default")
        spot = cfg.get("spot", "default")
        data_dir = Path(f"data/{area}/{spot}/raw/synoptic")

    data_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Output directory: {data_dir}")

    # Initialize downloader
    downloader = SynopticGridBatchDownloader(config=cfg)

    # Check for existing files
    existing_files = []
    missing_dates = []

    for date_str in dates:
        filepath = data_dir / f"synoptic_grid_{date_str.replace('-', '')}.parquet"
        if filepath.exists() and not no_cache:
            existing_files.append(date_str)
        else:
            missing_dates.append(date_str)

    if existing_files:
        click.echo(f"Found {len(existing_files)} existing files (will skip)")

    if not missing_dates:
        click.echo("All files already downloaded!")
        return

    click.echo(f"Need to download {len(missing_dates)} files")

    # Download missing dates
    success_count = 0
    failed_dates = []

    with click.progressbar(
        missing_dates, label="Downloading", show_pos=True
    ) as dates_bar:
        for date_str in dates_bar:
            try:
                # Download date
                grid_data = downloader.download_date(
                    date_str, mode=mode, n_workers=workers
                )

                if grid_data is not None:
                    # Save to parquet
                    filepath = downloader.save_parquet(grid_data, date_str)
                    success_count += 1
                    logger.debug(f"Saved: {filepath}")
                else:
                    failed_dates.append(date_str)
                    logger.warning(f"Failed: {date_str}")

            except Exception as e:
                failed_dates.append(date_str)
                logger.error(f"Error downloading {date_str}: {e}")

    # Summary
    click.echo("\nDownload complete!")
    click.echo(f"Successfully downloaded: {success_count}/{len(missing_dates)} files")

    if failed_dates:
        click.echo(f"Failed dates ({len(failed_dates)}):")
        for date in failed_dates:
            click.echo(f"  {date}")
        click.echo("\nYou can re-run this command to retry failed dates.")

"""
Test script to verify Open-Meteo API supports synoptic variables.

Tests if we can download:
- geopotential_height_500hPa
- geopotential_height_850hPa
- temperature_850hPa
- pressure_msl
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from windmojo.data.open_meteo_client import OpenMeteoClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_synoptic_variables():
    """Test downloading synoptic variables from Open-Meteo API."""

    client = OpenMeteoClient()

    # Test location: Central Europe (near Lake Como)
    lat = 46.0
    lon = 9.0

    # Test date: Recent historical data
    start_date = "2024-01-15"
    end_date = "2024-01-15"

    # Test each variable individually
    test_variables = [
        'pressure_msl',
        'temperature_850hPa',
        'geopotential_height_850hPa',
        'geopotential_height_500hPa'
    ]

    logger.info("Testing Open-Meteo Historical Forecast API for synoptic variables...")
    logger.info(f"Location: {lat}°N, {lon}°E")
    logger.info(f"Date: {start_date}")
    logger.info("-" * 60)

    results = {}

    for var in test_variables:
        logger.info(f"\nTesting variable: {var}")
        try:
            data = client.fetch_historical_forecast(
                latitude=lat,
                longitude=lon,
                start_date=start_date,
                end_date=end_date,
                variables=[var]
            )

            if data is not None and not data.empty:
                logger.info(f"✓ SUCCESS: {var}")
                logger.info(f"  Rows: {len(data)}, Columns: {list(data.columns)}")
                if var in data.columns:
                    logger.info(f"  Sample values: {data[var].head(3).values}")
                    results[var] = 'SUCCESS'
                else:
                    logger.warning(f"  Variable not in columns!")
                    results[var] = 'MISSING_COLUMN'
            else:
                logger.error(f"✗ FAILED: {var} - No data returned")
                results[var] = 'NO_DATA'

        except Exception as e:
            logger.error(f"✗ ERROR: {var} - {e}")
            results[var] = f'ERROR: {e}'

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY:")
    logger.info("=" * 60)

    for var, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        logger.info(f"{symbol} {var}: {status}")

    success_count = sum(1 for s in results.values() if s == "SUCCESS")
    logger.info(f"\n{success_count}/{len(test_variables)} variables available")

    if success_count == len(test_variables):
        logger.info("\n🎉 All synoptic variables are available!")
        logger.info("Experiment8 can proceed with full feature set.")
        return True
    else:
        logger.warning("\n⚠️  Some variables are not available!")
        logger.warning("Experiment8 may need to use alternative variables.")
        return False


if __name__ == "__main__":
    success = test_synoptic_variables()
    sys.exit(0 if success else 1)

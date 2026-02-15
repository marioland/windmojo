"""
Feature Engineering CLI - Phase 1

Transform raw GFS data into ML-ready features for wind forecasting.
Phase 1: Comprehensive feature set (100-300 features) for initial model training.

Usage:
    python windmojo/feature_engineer.py --config config/lake_como/lecco.yaml
    python windmojo/feature_engineer.py --config config/lake_como/lecco.yaml --bias-correction wind_speed_10m:3.21
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

from windmojo.core.config import (
    get_output_directories,
    load_config,
    print_config_summary,
)
from windmojo.features.pipeline import FeatureEngineeringPipeline, load_raw_data
from windmojo.features.report import FeatureReport, save_feature_list

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_bias_corrections(bias_str: str) -> Dict[str, float]:
    """
    Parse bias correction string.

    Format: "variable1:correction1,variable2:correction2"
    Example: "wind_speed_10m:3.21,temperature_2m:-1.5"

    Args:
        bias_str: Comma-separated list of variable:correction pairs

    Returns:
        Dictionary mapping variable names to correction values
    """
    if not bias_str:
        return {}

    corrections = {}
    for pair in bias_str.split(","):
        var, correction = pair.split(":")
        corrections[var.strip()] = float(correction.strip())

    return corrections


def main():
    """Main feature engineering entry point - Phase 1."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Feature Engineering - Generate comprehensive feature set for windmojo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., config/lake_como/lecco.yaml)",
    )

    parser.add_argument(
        "--bias-correction",
        type=str,
        help="Bias corrections to apply (format: var1:correction1,var2:correction2)",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        help="Custom output filename (default: {spot}_features.parquet)",
    )

    parser.add_argument(
        "--no-report", action="store_true", help="Skip generating feature report"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}...")
        config = load_config(args.config)
        print_config_summary(config)

        # Extract configuration
        area = config["area"]
        spot = config["spot"]
        output_dirs = get_output_directories(config)
        raw_data_dir = output_dirs["raw_data_dir"]
        features_dir = output_dirs["features_dir"]

        # Parse bias corrections
        bias_correction = (
            parse_bias_corrections(args.bias_correction) if args.bias_correction else {}
        )
        if bias_correction:
            logger.info(f"Bias corrections: {bias_correction}")

        # Load raw data
        logger.info(f"\nLoading raw data from {raw_data_dir}...")
        raw_df = load_raw_data(raw_data_dir, spot)

        # Initialize pipeline
        logger.info("\nInitializing feature engineering pipeline...")
        pipeline = FeatureEngineeringPipeline(bias_correction=bias_correction)

        # Transform data
        logger.info("\nRunning feature engineering...")
        features_df = pipeline.transform(raw_df)

        logger.info(f"\n✓ Feature engineering complete!")
        logger.info(f"  Input: {len(raw_df)} rows × {len(raw_df.columns)} columns")
        logger.info(
            f"  Output: {len(features_df)} rows × {len(features_df.columns)} columns"
        )

        # Save features (Phase 1)
        features_dir.mkdir(parents=True, exist_ok=True)

        output_filename = args.output_name or f"{spot}_features_phase1.parquet"
        output_path = features_dir / output_filename

        logger.info(f"\nSaving Phase 1 features to {output_path}...")
        features_df.to_parquet(output_path, index=False, compression="snappy")

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Saved: {output_filename} ({file_size_mb:.1f} MB)")

        # Generate feature report
        if not args.no_report:
            logger.info("\nGenerating Phase 1 feature report...")

            # Print summary to console
            reporter = FeatureReport(features_df)
            reporter.print_summary()

            # Save detailed report to JSON
            report_path = features_dir / f"{spot}_feature_report_phase1.json"
            reporter.save_report(report_path)
            logger.info(f"  ✓ Saved detailed report (JSON): {report_path}")

            # Save markdown report (with optional wind direction analysis)
            md_report_path = features_dir / f"{spot}_feature_report_phase1.md"

            # Get measurement file and column names from config for direction analysis
            measurements_path = None
            target_column = None
            direction_column = None

            if "measurement_file" in config:
                from pathlib import Path

                measurements_path = Path(config["measurement_file"])
                target_column = config.get("target_column", None)
                direction_column = config.get("direction_column", None)

            reporter.save_markdown_report(
                md_report_path,
                measurements_path=measurements_path,
                target_column=target_column,
                direction_column=direction_column,
            )
            logger.info(f"  ✓ Saved detailed report (Markdown): {md_report_path}")

            # Save feature list
            feature_list_path = features_dir / f"{spot}_feature_list_phase1.txt"
            save_feature_list(features_df, feature_list_path)
            logger.info(f"  ✓ Saved feature list: {feature_list_path}")

        logger.info("=" * 70)
        logger.info("PHASE 1 FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Phase 1 features saved to: {output_path}")
        if not args.no_report:
            logger.info(f"Report saved to: {report_path}")
            logger.info(f"Feature list saved to: {feature_list_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)

    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

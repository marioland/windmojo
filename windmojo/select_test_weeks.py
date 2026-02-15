#!/usr/bin/env python3
"""
Select Test Weeks for Model 2 Evaluation

Selects 3 hold-out test weeks spread across seasons from the 270-day dataset.
These weeks are completely excluded from training/validation/test split.

Usage:
    python -m windmojo.select_test_weeks \
        --features-file data/lake_como/lecco/features/lecco_features_phase1.parquet \
        --measurement-file data/lake_como/lecco/measurements/wind_data.csv \
        --output data/lake_como/lecco/test_weeks.json
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_date_range(features_file: Path) -> tuple:
    """
    Load date range from features file.

    Returns:
        (start_date, end_date) as datetime objects
    """
    logger.info(f"Loading features from {features_file}...")
    df = pd.read_parquet(features_file)

    if "time" not in df.columns:
        raise ValueError("Features file must have 'time' column")

    df["time"] = pd.to_datetime(df["time"])
    start_date = df["time"].min()
    end_date = df["time"].max()

    logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"  Total days: {(end_date - start_date).days}")

    return start_date, end_date


def check_measurement_coverage(measurement_file: Path, test_weeks: list) -> dict:
    """
    Check if measurements exist for test weeks.

    Returns:
        Dict with coverage statistics
    """
    logger.info(f"\nChecking measurement coverage...")

    # Load measurements
    df = (
        pd.read_parquet(measurement_file)
        if measurement_file.suffix == ".parquet"
        else pd.read_csv(measurement_file)
    )

    # Parse timestamp
    timestamp_col = next(
        (
            c
            for c in df.columns
            if c.lower() in ["timestamp", "time", "datetime", "date"]
        ),
        None,
    )
    if timestamp_col is None:
        logger.warning("  Could not find timestamp column in measurements")
        return {}

    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except (ValueError, TypeError):
        # Try custom format
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="%Y-%m-%d_%H%M")

    # Check coverage for each week
    coverage = {}
    for week_name, week_info in test_weeks.items():
        start = pd.to_datetime(week_info["start_date"])
        end = pd.to_datetime(week_info["end_date"])

        week_data = df[(df[timestamp_col] >= start) & (df[timestamp_col] <= end)]
        expected_hours = 7 * 24  # 7 days × 24 hours
        actual_hours = len(week_data)
        coverage_pct = 100 * actual_hours / expected_hours

        coverage[week_name] = {
            "expected_hours": expected_hours,
            "actual_hours": actual_hours,
            "coverage_pct": coverage_pct,
        }

        logger.info(
            f"  {week_name}: {actual_hours}/{expected_hours} hours ({coverage_pct:.1f}% coverage)"
        )

    return coverage


def select_test_weeks(start_date: datetime, end_date: datetime) -> dict:
    """
    Select 3 test weeks spread across seasons.

    Strategy:
    - Early season: ~20% through dataset (week 6-7)
    - Mid season: ~50% through dataset (week 19-20)
    - Late season: ~80% through dataset (week 32-33)

    Returns:
        Dict with test week metadata
    """
    logger.info("\nSelecting 3 test weeks spread across seasons...")

    total_days = (end_date - start_date).days

    # Calculate week positions (20%, 50%, 80% through dataset)
    early_day = int(total_days * 0.20)
    mid_day = int(total_days * 0.50)
    late_day = int(total_days * 0.80)

    test_weeks = {}

    # Week 1: Early season
    week1_start = start_date + timedelta(days=early_day)
    week1_end = week1_start + timedelta(days=6, hours=23)  # 7 days

    test_weeks["week_1_early_season"] = {
        "start_date": week1_start.isoformat(),
        "end_date": week1_end.isoformat(),
        "season": "early",
        "position_pct": 20,
        "description": f"Early season test week ({week1_start.strftime('%b %d')} - {week1_end.strftime('%b %d')})",
    }

    # Week 2: Mid season
    week2_start = start_date + timedelta(days=mid_day)
    week2_end = week2_start + timedelta(days=6, hours=23)

    test_weeks["week_2_mid_season"] = {
        "start_date": week2_start.isoformat(),
        "end_date": week2_end.isoformat(),
        "season": "mid",
        "position_pct": 50,
        "description": f"Mid season test week ({week2_start.strftime('%b %d')} - {week2_end.strftime('%b %d')})",
    }

    # Week 3: Late season
    week3_start = start_date + timedelta(days=late_day)
    week3_end = week3_start + timedelta(days=6, hours=23)

    test_weeks["week_3_late_season"] = {
        "start_date": week3_start.isoformat(),
        "end_date": week3_end.isoformat(),
        "season": "late",
        "position_pct": 80,
        "description": f"Late season test week ({week3_start.strftime('%b %d')} - {week3_end.strftime('%b %d')})",
    }

    # Log selections
    for week_name, week_info in test_weeks.items():
        start = pd.to_datetime(week_info["start_date"])
        end = pd.to_datetime(week_info["end_date"])
        logger.info(f"  {week_name}:")
        logger.info(
            f"    {start.strftime('%Y-%m-%d %A')} to {end.strftime('%Y-%m-%d %A')}"
        )
        logger.info(
            f"    {week_info['season'].upper()} season ({week_info['position_pct']}% through dataset)"
        )

    return test_weeks


def save_test_weeks(test_weeks: dict, coverage: dict, output_file: Path):
    """Save test weeks metadata to JSON."""
    logger.info(f"\nSaving test weeks to {output_file}...")

    output = {
        "test_weeks": test_weeks,
        "measurement_coverage": coverage,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_weeks": len(test_weeks),
            "total_days": len(test_weeks) * 7,
            "strategy": "80/10/10 split + 3 hold-out weeks",
        },
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("✓ Test weeks saved successfully")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Select 3 hold-out test weeks spread across seasons"
    )
    parser.add_argument(
        "--features-file",
        required=True,
        type=Path,
        help="Path to features Parquet file",
    )
    parser.add_argument(
        "--measurement-file",
        required=True,
        type=Path,
        help="Path to measurements CSV file",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output JSON file for test weeks"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("TEST WEEK SELECTION")
    logger.info("=" * 70)

    # Load date range
    start_date, end_date = load_date_range(args.features_file)

    # Select test weeks
    test_weeks = select_test_weeks(start_date, end_date)

    # Check measurement coverage
    coverage = check_measurement_coverage(args.measurement_file, test_weeks)

    # Save output
    save_test_weeks(test_weeks, coverage, args.output)

    logger.info("\n" + "=" * 70)
    logger.info("✓ TEST WEEK SELECTION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

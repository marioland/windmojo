"""
Data Loading and Preparation for Model Training

Loads wind measurements and engineered features, joins them,
and prepares datasets for training.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_training_data(
    measurements_path: Path,
    features_path: Path,
    target_column: str = "lecco avgSpeed",
    target_hours: List[str] = None,
) -> pd.DataFrame:
    """
    Load and join measurement data with engineered features.

    Args:
        measurements_path: Path to wind_data.csv
        features_path: Path to features Parquet
        target_column: Column name for target variable
        target_hours: List of hours to keep (e.g., ['08:00', '13:00'])

    Returns:
        DataFrame with features and target, ready for training
    """
    logger.info("Loading training data...")

    # Load measurements
    logger.info(f"  Loading measurements from {measurements_path}")
    measurements = pd.read_csv(measurements_path)

    # Parse timestamp format: "2025-01-16_1900" → datetime
    measurements["timestamp"] = pd.to_datetime(
        measurements["timestamp"], format="%Y-%m-%d_%H%M"
    )

    # Rename for clarity
    measurements = measurements.rename(columns={"timestamp": "time"})

    logger.info(f"  Loaded {len(measurements)} measurement rows")
    logger.info(
        f"  Date range: {measurements['time'].min()} to {measurements['time'].max()}"
    )

    # Load Phase 1 features
    logger.info(f"  Loading Phase 1 features from {features_path}")
    features = pd.read_parquet(features_path)

    logger.info(
        f"  Loaded {len(features)} feature rows, {len(features.columns)} columns"
    )
    logger.info(f"  Date range: {features['time'].min()} to {features['time'].max()}")

    # Join on time
    logger.info("  Joining measurements with features...")
    joined = measurements.merge(features, on="time", how="inner")

    logger.info(f"  ✓ Joined dataset: {len(joined)} rows")

    # Extract target column
    if target_column not in joined.columns:
        raise ValueError(f"Target column '{target_column}' not found in measurements")

    joined["target"] = joined[target_column]

    # Filter to target hours if specified
    if target_hours:
        joined["hour"] = joined["time"].dt.strftime("%H:%M")
        mask = joined["hour"].isin(target_hours)
        joined = joined[mask].copy()
        logger.info(f"  Filtered to target hours {target_hours}: {len(joined)} rows")

    # Remove rows with missing target
    n_before = len(joined)
    joined = joined.dropna(subset=["target"])
    n_after = len(joined)

    if n_before != n_after:
        logger.warning(f"  Removed {n_before - n_after} rows with missing target")

    logger.info(f"  ✓ Final dataset: {n_after} samples ready for training")

    return joined


def prepare_datasets(
    data: pd.DataFrame,
    target_hours: List[str],
    validation_strategy: str = "weekly_walkforward",
    pilot_weeks: int = 8,
) -> Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Prepare train/test splits for each target hour.

    Args:
        data: Joined dataset from load_training_data()
        target_hours: List of hours (e.g., ['08:00', '13:00'])
        validation_strategy: 'weekly_walkforward' or 'simple_split'
        pilot_weeks: Number of weeks for pilot study

    Returns:
        Nested dict: {hour: {fold_name: (train_df, test_df)}}
    """
    logger.info(f"Preparing datasets with {validation_strategy} validation...")

    datasets = {}

    for hour in target_hours:
        logger.info(f"\n  Processing hour: {hour}")

        # Filter to this hour
        hour_data = data[data["time"].dt.strftime("%H:%M") == hour].copy()
        hour_data = hour_data.sort_values("time").reset_index(drop=True)

        logger.info(f"    {len(hour_data)} samples for {hour}")

        if len(hour_data) == 0:
            logger.warning(f"    No data for hour {hour}, skipping")
            continue

        # Limit to pilot weeks if specified
        if pilot_weeks:
            min_date = hour_data["time"].min()
            max_date = min_date + timedelta(weeks=pilot_weeks)
            hour_data = hour_data[hour_data["time"] <= max_date]
            logger.info(f"    Limited to {pilot_weeks} weeks: {len(hour_data)} samples")

        if validation_strategy == "weekly_walkforward":
            folds = _create_weekly_walkforward(hour_data)
        elif validation_strategy == "simple_split":
            folds = _create_simple_split(hour_data, test_ratio=0.2)
        else:
            raise ValueError(f"Unknown validation strategy: {validation_strategy}")

        datasets[hour] = folds
        logger.info(f"    Created {len(folds)} validation folds")

    return datasets


def _create_weekly_walkforward(
    data: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create weekly walk-forward validation folds.

    Train on weeks 1-4 → Test week 5
    Train on weeks 1-5 → Test week 6
    etc.
    """
    data = data.copy()
    data["week"] = ((data["time"] - data["time"].min()).dt.days // 7) + 1

    max_week = data["week"].max()
    min_train_weeks = 4

    folds = {}

    for test_week in range(min_train_weeks + 1, max_week + 1):
        train_data = data[data["week"] < test_week]
        test_data = data[data["week"] == test_week]

        if len(train_data) > 0 and len(test_data) > 0:
            fold_name = f"train_w1-{test_week-1}_test_w{test_week}"
            folds[fold_name] = (train_data, test_data)

    return folds


def _create_simple_split(
    data: pd.DataFrame, test_ratio: float = 0.2
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create simple train/test split.
    """
    split_idx = int(len(data) * (1 - test_ratio))

    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    return {f"train_{len(train_data)}_test_{len(test_data)}": (train_data, test_data)}


def get_feature_columns(
    data: pd.DataFrame, measurement_columns: List[str] = None
) -> List[str]:
    """
    Extract feature column names (exclude time, target, metadata).

    Args:
        data: Joined dataset
        measurement_columns: Columns of the measurement file (excluded to
            avoid leaking measured values into the features)

    Returns:
        List of feature column names
    """
    exclude_cols = [
        "time",
        "target",
        "hour",
        "week",
        "forecast_made",
        "month",
        "forecast_date",
        # Exclude all measurement columns
        "dervio direction",
        "gera direction",
        "lecco direction",
        "dervio speed",
        "gera speed",
        "lecco speed",
        "dervio avgSpeed",
        "gera avgSpeed",
        "lecco avgSpeed",
        "dervio purescore",
        "gera purescore",
        "lecco purescore",
    ]
    if measurement_columns is not None:
        exclude_cols += list(measurement_columns)

    # Get all columns that contain measurement-related keywords
    measurement_keywords = [
        "dervio",
        "gera",
        "lecco",
        "purescore",
        "score",
        "delta",
        "avg",
    ]

    feature_cols = []
    for col in data.columns:
        # Skip if in exclude list
        if col in exclude_cols:
            continue
        # Skip if contains measurement keywords
        if any(keyword in col.lower() for keyword in measurement_keywords):
            continue
        # Skip if datetime type (XGBoost can't handle)
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            continue
        # Skip text columns (e.g. precipitation_type; one-hot versions exist)
        if not (
            pd.api.types.is_numeric_dtype(data[col])
            or pd.api.types.is_bool_dtype(data[col])
        ):
            continue
        feature_cols.append(col)

    return feature_cols


def get_dataset_summary(data: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """
    Generate summary statistics for a dataset.

    Args:
        data: Dataset
        feature_cols: List of feature columns

    Returns:
        Dictionary with summary stats
    """
    return {
        "n_samples": len(data),
        "n_features": len(feature_cols),
        "date_range": {
            "start": str(data["time"].min()),
            "end": str(data["time"].max()),
            "days": (data["time"].max() - data["time"].min()).days,
        },
        "target_stats": {
            "mean": float(data["target"].mean()),
            "std": float(data["target"].std()),
            "min": float(data["target"].min()),
            "max": float(data["target"].max()),
            "median": float(data["target"].median()),
        },
        "missing_values": {
            "features": int(data[feature_cols].isna().sum().sum()),
            "pct": float(data[feature_cols].isna().mean().mean() * 100),
        },
    }

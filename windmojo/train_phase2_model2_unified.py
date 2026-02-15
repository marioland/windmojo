#!/usr/bin/env python3
"""
Phase 2: Model 2 Unified Training

Train ONE unified model per approach (not per-hour models).
Uses all hours 06:00-18:00 together, leveraging hour_sin/cos for time-of-day encoding.
This addresses overfitting in per-hour models by sharing data across all hours.

Key improvements vs per-hour training:
- 13× more training data (~2340 vs ~180 samples)
- Better sample-to-feature ratio (~30:1 vs ~2:1)
- Reduced overfitting through data sharing
- Single production model vs 13 files

Usage:
    python -m windmojo.train_phase2_model2_unified \
        --config config/lake_como/lecco.yaml \
        --feature-selection data/.../features_gfs_topn.json \
        --test-weeks data/.../test_weeks.json \
        --approach gfs_topn \
        --hours 06:00-18:00
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UnifiedModel2Trainer:
    """Unified Model 2 trainer - one model for all hours."""

    def __init__(
        self, config: dict, feature_selection: dict, test_weeks: dict, approach: str
    ):
        """
        Initialize unified trainer.

        Args:
            config: Configuration dict
            feature_selection: Selected features dict
            test_weeks: Test weeks dict
            approach: Approach name (e.g., 'gfs_topn')
        """
        self.config = config
        self.feature_selection = feature_selection
        self.test_weeks = test_weeks
        self.approach = approach

        # Selected features
        self.selected_features = [f["feature"] for f in feature_selection["features"]]

        # Ensure hour_sin and hour_cos are in features (required for time-of-day)
        if (
            "hour_sin" not in self.selected_features
            or "hour_cos" not in self.selected_features
        ):
            logger.warning(
                "hour_sin/hour_cos not in selected features - time-of-day encoding may be missing"
            )

        logger.info(f"Unified Model 2 Trainer initialized:")
        logger.info(f"  Approach: {approach}")
        logger.info(f"  Selected features: {len(self.selected_features)}")
        logger.info(f"  Test weeks excluded: {len(test_weeks['test_weeks'])}")
        logger.info("  Training mode: UNIFIED (one model for all hours)")

    def load_and_prepare_data(
        self,
        measurements_file: Path,
        features_file: Path,
        target_column: str,
        hours: list,
    ) -> pd.DataFrame:
        """
        Load and prepare training data (ALL hours together).

        Args:
            measurements_file: Path to measurements CSV
            features_file: Path to features Parquet
            target_column: Target column name
            hours: List of hours to include (e.g., ['06:00', '07:00', ..., '18:00'])

        Returns:
            DataFrame with measurements + features for all hours
        """
        logger.info("\nLoading and preparing data (unified mode)...")

        # Load measurements
        measurements = pd.read_csv(measurements_file)
        timestamp_col = next(
            (
                c
                for c in measurements.columns
                if c.lower() in ["timestamp", "time", "datetime"]
            ),
            None,
        )

        if timestamp_col is None:
            raise ValueError("Could not find timestamp column in measurements")

        # Parse timestamps
        try:
            measurements[timestamp_col] = pd.to_datetime(measurements[timestamp_col])
        except (ValueError, TypeError):
            measurements[timestamp_col] = pd.to_datetime(
                measurements[timestamp_col], format="%Y-%m-%d_%H%M"
            )

        measurements.rename(columns={timestamp_col: "time"}, inplace=True)
        measurements["target"] = measurements[target_column]

        logger.info(f"  Loaded {len(measurements)} measurements")
        logger.info(
            f"  Date range: {measurements['time'].min()} to {measurements['time'].max()}"
        )

        # Load features
        features = pd.read_parquet(features_file)
        features["time"] = pd.to_datetime(features["time"])

        logger.info(f"  Loaded {len(features)} feature rows")

        # Join
        df = pd.merge(measurements, features, on="time", how="inner")
        logger.info(f"  Joined: {len(df)} rows")

        # Exclude test weeks
        test_week_dates = []
        for week_name, week_info in self.test_weeks["test_weeks"].items():
            start = pd.to_datetime(week_info["start_date"])
            end = pd.to_datetime(week_info["end_date"])
            test_week_dates.append((start, end))

        before_exclusion = len(df)
        for start, end in test_week_dates:
            df = df[(df["time"] < start) | (df["time"] > end)]

        logger.info(
            f"  Excluded test weeks: {before_exclusion} → {len(df)} rows ({before_exclusion - len(df)} removed)"
        )

        # Filter to target hours (but keep all together)
        df["hour"] = df["time"].dt.strftime("%H:%M")
        df = df[df["hour"].isin(hours)]
        logger.info(f"  Filtered to {len(hours)} hours (unified): {len(df)} rows total")

        # Hour distribution
        hour_counts = df["hour"].value_counts().sort_index()
        logger.info(f"  Hour distribution:")
        for hour, count in hour_counts.items():
            logger.info(f"    {hour}: {count} samples")

        # Remove missing target
        df = df.dropna(subset=["target"])
        logger.info(f"  After removing missing target: {len(df)} rows")

        return df

    def split_data(self, df: pd.DataFrame) -> tuple:
        """
        Split data into train (80%), val (10%), test (10%).

        All hours mixed together for unified training.

        Args:
            df: DataFrame with all data (all hours)

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("\nSplitting data (unified 80/10/10)...")

        # Sort by time for temporal split
        df = df.sort_values("time").reset_index(drop=True)

        # Split 80/10/10
        n = len(df)
        train_idx = int(n * 0.80)
        val_idx = int(n * 0.90)

        train_data = df.iloc[:train_idx]
        val_data = df.iloc[train_idx:val_idx]
        test_data = df.iloc[val_idx:]

        logger.info(
            f"  Train: {len(train_data)} samples ({len(train_data) / n * 100:.1f}%)"
        )
        logger.info(f"  Val: {len(val_data)} samples ({len(val_data) / n * 100:.1f}%)")
        logger.info(
            f"  Test: {len(test_data)} samples ({len(test_data) / n * 100:.1f}%)"
        )

        # Verify features exist
        missing_features = [f for f in self.selected_features if f not in df.columns]
        if missing_features:
            logger.error(
                f"Missing {len(missing_features)} features: {missing_features[:5]}..."
            )
            raise ValueError(f"Missing features in dataset")

        # Prepare features
        X_train = train_data[self.selected_features]
        y_train = train_data["target"]

        X_val = val_data[self.selected_features]
        y_val = val_data["target"]

        X_test = test_data[self.selected_features]
        y_test = test_data["target"]

        logger.info(f"  Feature matrix shape: {X_train.shape}")
        logger.info(
            f"  Sample-to-feature ratio: {len(X_train) / len(self.selected_features):.1f}:1"
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_model(self, X_train, y_train, X_val, y_val) -> xgb.Booster:
        """
        Train unified XGBoost model with regularization.

        Returns:
            Trained model
        """
        logger.info("\nTraining unified model...")

        # Model 2 parameters (production settings with regularization)
        params = {
            "max_depth": 6,  # Reduced from 8
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,  # L1
            "reg_lambda": 1.0,  # L2
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "verbosity": 0,
        }

        logger.info(
            f"  XGBoost params: max_depth={params['max_depth']}, lr={params['learning_rate']}, "
            f"L1={params['reg_alpha']}, L2={params['reg_lambda']}"
        )

        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train with early stopping
        evals = [(dtrain, "train"), (dval, "val")]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        logger.info(f"  Training complete: {model.best_iteration} iterations")

        return model

    def evaluate_model(self, model: xgb.Booster, X, y) -> dict:
        """Evaluate model and return metrics."""
        dtest = xgb.DMatrix(X)
        y_pred = model.predict(dtest)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "n_samples": len(y),
        }

    def evaluate_by_hour(self, model: xgb.Booster, X, y, hours) -> dict:
        """
        Evaluate model performance broken down by hour.

        Args:
            model: Trained model
            X: Features (with 'hour' column in original df)
            y: Targets
            hours: Hour labels corresponding to each row

        Returns:
            Dict of {hour: metrics}
        """
        logger.info("\nEvaluating by hour...")

        dtest = xgb.DMatrix(X)
        y_pred = model.predict(dtest)

        hourly_results = {}
        unique_hours = sorted(set(hours))

        for hour in unique_hours:
            hour_mask = hours == hour
            y_hour = y[hour_mask]
            y_pred_hour = y_pred[hour_mask]

            if len(y_hour) > 0:
                mae = mean_absolute_error(y_hour, y_pred_hour)
                rmse = np.sqrt(mean_squared_error(y_hour, y_pred_hour))
                r2 = r2_score(y_hour, y_pred_hour)

                hourly_results[hour] = {
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "n_samples": int(len(y_hour)),
                }

                logger.info(
                    f"  {hour}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}, n={len(y_hour)}"
                )

        return hourly_results

    def train_unified(self, df: pd.DataFrame, output_dir: Path) -> dict:
        """
        Train unified model for all hours.

        Args:
            df: Full dataset (all hours)
            output_dir: Output directory

        Returns:
            Dict with results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING UNIFIED MODEL: {self.approach}")
        logger.info(f"{'='*70}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Split data (all hours together)
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(df)

        # Get hour labels for hourly evaluation
        train_hours = df.iloc[: len(X_train)]["hour"].values
        val_hours = df.iloc[len(X_train) : len(X_train) + len(X_val)]["hour"].values
        test_hours = df.iloc[len(X_train) + len(X_val) :]["hour"].values

        # Train
        model = self.train_model(X_train, y_train, X_val, y_val)

        # Evaluate overall
        logger.info("\nOverall performance:")
        train_metrics = self.evaluate_model(model, X_train, y_train)
        val_metrics = self.evaluate_model(model, X_val, y_val)
        test_metrics = self.evaluate_model(model, X_test, y_test)

        logger.info(
            f"  Train: MAE={train_metrics['mae']:.3f}, RMSE={train_metrics['rmse']:.3f}, R²={train_metrics['r2']:.3f}"
        )
        logger.info(
            f"  Val:   MAE={val_metrics['mae']:.3f}, RMSE={val_metrics['rmse']:.3f}, R²={val_metrics['r2']:.3f}"
        )
        logger.info(
            f"  Test:  MAE={test_metrics['mae']:.3f}, RMSE={test_metrics['rmse']:.3f}, R²={test_metrics['r2']:.3f}"
        )

        # Evaluate by hour (diagnostic)
        train_by_hour = self.evaluate_by_hour(model, X_train, y_train, train_hours)
        test_by_hour = self.evaluate_by_hour(model, X_test, y_test, test_hours)

        # Save model
        model_file = output_dir / f"model2_{self.approach}_unified.json"
        model.save_model(str(model_file))
        logger.info(f"\n✓ Model saved to {model_file}")

        # Build results
        results = {
            "overall": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
            "by_hour": {"train": train_by_hour, "test": test_by_hour},
            "model_file": str(model_file),
            "best_iteration": int(model.best_iteration),
            "training_mode": "unified",
            "total_samples": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            },
        }

        return results

    def save_results(self, results: dict, output_dir: Path):
        """Save training results to JSON."""
        results_file = output_dir / f"training_results_{self.approach}_unified.json"

        output = {
            "approach": self.approach,
            "feature_count": len(self.selected_features),
            "training_mode": "unified",
            "generated_at": datetime.now().isoformat(),
            "results": results,
        }

        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"✓ Results saved to {results_file}")


def parse_hours_arg(hours_str: str) -> list:
    """
    Parse hours argument.

    Examples:
        '06:00-18:00' → ['06:00', '07:00', ..., '18:00']
        '10:00,14:00,16:00' → ['10:00', '14:00', '16:00']
    """
    if "-" in hours_str:
        # Range
        start, end = hours_str.split("-")
        start_h = int(start.split(":")[0])
        end_h = int(end.split(":")[0])
        return [f"{h:02d}:00" for h in range(start_h, end_h + 1)]
    elif "," in hours_str:
        # List
        return [h.strip() for h in hours_str.split(",")]
    else:
        return [hours_str]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Phase 2 Model 2 (unified mode)")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument(
        "--feature-selection", required=True, help="Feature selection JSON"
    )
    parser.add_argument("--test-weeks", required=True, help="Test weeks JSON")
    parser.add_argument(
        "--approach", required=True, help="Approach name (e.g., gfs_topn)"
    )
    parser.add_argument(
        "--hours", default="06:00-18:00", help="Hours to train (e.g., 06:00-18:00)"
    )

    args = parser.parse_args()

    # Load config
    import yaml

    from windmojo.core.config import ensure_unified_config

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Ensure config has 'data' section for backward compatibility
    config = ensure_unified_config(config)

    # Load feature selection
    with open(args.feature_selection, "r") as f:
        feature_selection = json.load(f)

    # Load test weeks
    with open(args.test_weeks, "r") as f:
        test_weeks = json.load(f)

    # Parse hours
    hours = parse_hours_arg(args.hours)

    # Initialize trainer
    trainer = UnifiedModel2Trainer(config, feature_selection, test_weeks, args.approach)

    # Load data
    measurements_file = Path(config["data"]["measurements_file"])
    features_file = Path(config["data"]["features_file"])
    target_column = config["data"]["target_column"]

    df = trainer.load_and_prepare_data(
        measurements_file, features_file, target_column, hours
    )

    # Train
    output_dir = (
        Path(config["data"]["lake_dir"]) / f"models/phase2_model2/{args.approach}"
    )
    results = trainer.train_unified(df, output_dir)

    # Save results
    trainer.save_results(results, output_dir)

    logger.info(f"\n{'='*70}")
    logger.info("✓ UNIFIED TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"\nApproach: {args.approach}")
    logger.info(f"Test MAE: {results['overall']['test']['mae']:.3f} m/s")
    logger.info(f"Model file: {results['model_file']}")


if __name__ == "__main__":
    main()

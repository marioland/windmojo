#!/usr/bin/env python3
"""
Phase 2: Model 2 Training

Train Model 2 with selected features (from feature selection).
Uses 80/10/10 split excluding 3 hold-out test weeks.

Usage:
    python -m windmojo.train_phase2_model2 \
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


class Model2Trainer:
    """Model 2 trainer with feature selection and test week exclusion."""

    def __init__(
        self, config: dict, feature_selection: dict, test_weeks: dict, approach: str
    ):
        """
        Initialize trainer.

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

        logger.info(f"Model 2 Trainer initialized:")
        logger.info(f"  Approach: {approach}")
        logger.info(f"  Selected features: {len(self.selected_features)}")
        logger.info(f"  Test weeks excluded: {len(test_weeks['test_weeks'])}")

    def load_and_prepare_data(
        self,
        measurements_file: Path,
        features_file: Path,
        target_column: str,
        hours: list,
    ) -> pd.DataFrame:
        """
        Load and prepare training data.

        Args:
            measurements_file: Path to measurements CSV
            features_file: Path to features Parquet
            target_column: Target column name
            hours: List of hours to train (e.g., ['06:00', '07:00', ...])

        Returns:
            DataFrame with measurements + features
        """
        logger.info("\nLoading and preparing data...")

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

        # Filter to target hours
        df["hour"] = df["time"].dt.strftime("%H:%M")
        df = df[df["hour"].isin(hours)]
        logger.info(f"  Filtered to {len(hours)} hours: {len(df)} rows")

        # Remove missing target
        df = df.dropna(subset=["target"])
        logger.info(f"  After removing missing target: {len(df)} rows")

        return df

    def split_data(self, df: pd.DataFrame, hour: str) -> tuple:
        """
        Split data into train (80%), val (10%), test (10%).

        Args:
            df: DataFrame with all data
            hour: Hour to filter

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Filter to hour
        hour_data = df[df["hour"] == hour].copy()
        hour_data = hour_data.sort_values("time").reset_index(drop=True)

        logger.info(f"    {hour}: {len(hour_data)} samples")

        # Split 80/10/10
        n = len(hour_data)
        train_idx = int(n * 0.80)
        val_idx = int(n * 0.90)

        train_data = hour_data.iloc[:train_idx]
        val_data = hour_data.iloc[train_idx:val_idx]
        test_data = hour_data.iloc[val_idx:]

        logger.info(
            f"      Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )

        # Prepare features
        X_train = train_data[self.selected_features]
        y_train = train_data["target"]

        X_val = val_data[self.selected_features]
        y_val = val_data["target"]

        X_test = test_data[self.selected_features]
        y_test = test_data["target"]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_model(self, X_train, y_train, X_val, y_val) -> xgb.Booster:
        """
        Train XGBoost model with regularization.

        Returns:
            Trained model
        """
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

    def train_all_hours(self, df: pd.DataFrame, hours: list, output_dir: Path) -> dict:
        """
        Train models for all hours.

        Args:
            df: Full dataset
            hours: List of hours to train
            output_dir: Output directory

        Returns:
            Dict with results
        """
        logger.info(f"\nTraining {len(hours)} hours for approach: {self.approach}")

        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        for hour in hours:
            logger.info(f"\n  Training {hour}...")

            # Split data
            X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(df, hour)

            if len(X_train) < 10:
                logger.warning(f"    Insufficient data for {hour}, skipping")
                continue

            # Train
            model = self.train_model(X_train, y_train, X_val, y_val)

            # Evaluate
            train_metrics = self.evaluate_model(model, X_train, y_train)
            val_metrics = self.evaluate_model(model, X_val, y_val)
            test_metrics = self.evaluate_model(model, X_test, y_test)

            logger.info(f"    Train MAE: {train_metrics['mae']:.3f}")
            logger.info(f"    Val MAE: {val_metrics['mae']:.3f}")
            logger.info(f"    Test MAE: {test_metrics['mae']:.3f}")

            # Save model
            hour_str = hour.replace(":", "")
            model_file = output_dir / f"model2_{self.approach}_{hour_str}.json"
            model.save_model(str(model_file))

            # Store results
            results[hour] = {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "model_file": str(model_file),
                "best_iteration": int(model.best_iteration),
            }

        logger.info(f"\n✓ Trained {len(results)} models")

        return results

    def save_results(self, results: dict, output_dir: Path):
        """Save training results to JSON."""
        results_file = output_dir / f"training_results_{self.approach}.json"

        output = {
            "approach": self.approach,
            "feature_count": len(self.selected_features),
            "hours_trained": list(results.keys()),
            "generated_at": datetime.now().isoformat(),
            "results": results,
        }

        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\n✓ Results saved to {results_file}")


def parse_hours_arg(hours_str: str) -> list:
    """
    Parse hours argument.

    Examples:
        '06:00-18:00' → ['06:00', '07:00', ..., '18:00']
        '13:00,14:00,15:00' → ['13:00', '14:00', '15:00']
    """
    if "-" in hours_str:
        # Range format
        start, end = hours_str.split("-")
        start_hour = int(start.split(":")[0])
        end_hour = int(end.split(":")[0])

        return [f"{h:02d}:00" for h in range(start_hour, end_hour + 1)]
    else:
        # Comma-separated
        return [h.strip() for h in hours_str.split(",")]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Model 2 with feature selection")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--feature-selection", required=True, help="Path to feature selection JSON"
    )
    parser.add_argument("--test-weeks", required=True, help="Path to test weeks JSON")
    parser.add_argument(
        "--approach", required=True, help="Approach name (e.g., gfs_topn)"
    )
    parser.add_argument(
        "--hours",
        required=True,
        help="Hours to train (e.g., 06:00-18:00 or 13:00,14:00,15:00)",
    )
    parser.add_argument(
        "--target-column", default="lecco speed", help="Target column name"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"MODEL 2 TRAINING: {args.approach}")
    logger.info("=" * 70)

    # Load config
    import yaml

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load feature selection
    with open(args.feature_selection, "r") as f:
        feature_selection = json.load(f)

    # Load test weeks
    with open(args.test_weeks, "r") as f:
        test_weeks = json.load(f)

    # Parse hours
    hours = parse_hours_arg(args.hours)
    logger.info(f"\nTraining {len(hours)} hours: {hours[0]} to {hours[-1]}")

    # Initialize trainer
    trainer = Model2Trainer(config, feature_selection, test_weeks, args.approach)

    # Load data
    measurements_file = Path(config["measurement_file"])
    features_file = Path(
        f"data/{config['area']}/{config['spot']}/features/{config['spot']}_features_phase1.parquet"
    )

    df = trainer.load_and_prepare_data(
        measurements_file, features_file, args.target_column, hours
    )

    # Train all hours
    output_dir = Path(
        f"data/{config['area']}/{config['spot']}/models/phase2_model2/{args.approach}"
    )
    results = trainer.train_all_hours(df, hours, output_dir)

    # Save results
    trainer.save_results(results, output_dir)

    logger.info("\n" + "=" * 70)
    logger.info(f"✓ MODEL 2 TRAINING COMPLETE: {args.approach}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

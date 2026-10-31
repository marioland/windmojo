"""
Spatial Heavy Unified Trainer

Single unified model for all hours using Spatial Heavy feature selection.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


class SpatialHeavyTrainer:
    """
    Trainer for Spatial Heavy Unified model.

    Single XGBoost model trained on all hours with hour encoding.
    Uses selected Spatial Heavy features (no feature discovery phase needed).
    """

    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration from YAML
        """
        self.config = config
        self.params = config.get("xgboost", {})
        self.model = None
        self.feature_columns = None
        self.feature_medians = None
        self.training_history = {}

    def train(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_columns: List[str],
    ) -> Dict:
        """
        Train unified model on all hours.

        Args:
            train_data: Training data with target column
            test_data: Test data with target column
            feature_columns: List of feature column names (must include hour features)

        Returns:
            Dictionary with training results and metrics
        """
        self.feature_columns = feature_columns

        logger.info("Training Spatial Heavy Unified model...")
        logger.info(f"  Features: {len(feature_columns)}")
        logger.info(f"  Train samples: {len(train_data)}")
        logger.info(f"  Test samples: {len(test_data)}")

        # Verify hour features are present
        hour_features = ["hour", "hour_sin", "hour_cos"]
        missing_hour = [hf for hf in hour_features if hf not in feature_columns]
        if missing_hour:
            logger.warning(f"Missing hour features: {missing_hour}")

        # Prepare data
        X_train = train_data[feature_columns]
        y_train = train_data["target"]
        X_test = test_data[feature_columns]
        y_test = test_data["target"]

        # Log data quality
        train_missing = X_train.isna().sum().sum()
        test_missing = X_test.isna().sum().sum()
        logger.info(
            f"  Train missing: {train_missing} ({train_missing/(len(X_train)*len(feature_columns))*100:.1f}%)"
        )
        logger.info(
            f"  Test missing: {test_missing} ({test_missing/(len(X_test)*len(feature_columns))*100:.1f}%)"
        )

        # Handle missing values (fill with train median, also used by predict())
        self.feature_medians = X_train.median()
        if train_missing > 0:
            X_train = X_train.fillna(self.feature_medians)
        if test_missing > 0:
            X_test = X_test.fillna(self.feature_medians)

        # Train XGBoost
        logger.info("Training XGBoost model...")

        self.model = xgb.XGBRegressor(**self.params)

        # Fit with early stopping
        early_stopping = self.params.get("early_stopping_rounds", 50)

        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Get predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        # Calculate metrics
        train_mae = np.mean(np.abs(y_train - train_pred))
        test_mae = np.mean(np.abs(y_test - test_pred))
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

        results = {
            "train": {
                "mae": float(train_mae),
                "rmse": float(train_rmse),
                "samples": len(train_data),
            },
            "test": {
                "mae": float(test_mae),
                "rmse": float(test_rmse),
                "samples": len(test_data),
            },
            "features": len(feature_columns),
            "best_iteration": (
                self.model.best_iteration
                if hasattr(self.model, "best_iteration")
                else None
            ),
        }

        self.training_history = results

        logger.info(f"Training complete:")
        logger.info(f"  Train MAE: {train_mae:.3f} m/s")
        logger.info(f"  Test MAE: {test_mae:.3f} m/s")

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with trained model.

        Args:
            X: Features dataframe

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure correct feature order
        X_features = X[self.feature_columns]

        # Handle missing values (train medians; models saved before they were
        # stored fall back to the medians of X)
        if X_features.isna().sum().sum() > 0:
            medians = (
                self.feature_medians
                if self.feature_medians is not None
                else X_features.median()
            )
            X_features = X_features.fillna(medians)

        return self.model.predict(X_features)

    def save(self, model_path: Path, metadata_path: Path):
        """
        Save trained model and metadata.

        Args:
            model_path: Path to save model (JSON format)
            metadata_path: Path to save metadata (JSON format)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to: {model_path}")

        # Save metadata
        metadata = {
            "model_type": "spatial_heavy_unified",
            "features": self.feature_columns,
            "feature_medians": {
                f: None if pd.isna(v) else float(v)
                for f, v in self.feature_medians.items()
            },
            "training_results": self.training_history,
            "params": self.params,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to: {metadata_path}")

    def load(self, model_path: Path, metadata_path: Path):
        """
        Load trained model and metadata.

        Args:
            model_path: Path to model file
            metadata_path: Path to metadata file
        """
        # Load model
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))
        logger.info(f"Model loaded from: {model_path}")

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.feature_columns = metadata["features"]
        medians = metadata.get("feature_medians")
        self.feature_medians = (
            pd.Series(medians, dtype=float) if medians is not None else None
        )
        self.training_history = metadata.get("training_results", {})
        self.params = metadata.get("params", {})

        logger.info(f"Metadata loaded from: {metadata_path}")
        logger.info(f"  Features: {len(self.feature_columns)}")
        logger.info(
            f"  Test MAE: {self.training_history.get('test', {}).get('mae', 'N/A')}"
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings.

        Returns:
            DataFrame with features and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.feature_importances_

        df = pd.DataFrame({"feature": self.feature_columns, "importance": importance})

        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["importance_pct"] = 100 * df["importance"] / df["importance"].sum()

        return df

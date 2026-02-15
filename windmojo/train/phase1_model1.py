"""
Phase 1: Model 1 Training

Model 1 is the "kitchen sink" feature discovery model.
Uses ALL 433 features with intentional overfitting to find patterns.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class Model1Trainer:
    """
    Trainer for Phase 1 Model 1 (feature discovery).

    Trains XGBoost with all features and intentional overfitting
    to extract feature importance rankings.
    """

    # Phase 1 Model 1 default parameters (intentional overfitting)
    # Optimized for faster training while maintaining feature discovery quality
    DEFAULT_PARAMS = {
        "max_depth": 6,  # Reduced from 8 for faster training
        "learning_rate": 0.05,
        "n_estimators": 500,  # Reduced from 1000 for faster training
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "random_state": 42,
        "tree_method": "hist",  # Fast histogram-based method
        "n_jobs": -1,
    }

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize Model 1 trainer.

        Args:
            params: Optional dict to override default parameters
        """
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        self.model = None
        self.feature_columns = None
        self.training_history = []

    def train(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_columns: List[str],
        early_stopping_rounds: int = 25,  # Reduced from 50 for faster training
    ) -> Dict:
        """
        Train Model 1 on a single train/test fold.

        Args:
            train_data: Training data with 'target' column
            test_data: Test data with 'target' column
            feature_columns: List of feature column names
            early_stopping_rounds: Stop if no improvement for N rounds

        Returns:
            Dictionary with training results
        """
        self.feature_columns = feature_columns

        logger.info(f"Training Model 1...")
        logger.info(f"  Features: {len(feature_columns)}")
        logger.info(f"  Train samples: {len(train_data)}")
        logger.info(f"  Test samples: {len(test_data)}")

        # Prepare data
        X_train = train_data[feature_columns]
        y_train = train_data["target"]
        X_test = test_data[feature_columns]
        y_test = test_data["target"]

        # Log data quality
        train_missing = X_train.isna().sum().sum()
        test_missing = X_test.isna().sum().sum()
        logger.info(
            f"  Train missing values: {train_missing} ({train_missing/(len(X_train)*len(feature_columns))*100:.1f}%)"
        )
        logger.info(
            f"  Test missing values: {test_missing} ({test_missing/(len(X_test)*len(feature_columns))*100:.1f}%)"
        )

        # Train XGBoost
        # Update params for early stopping (newer XGBoost API)
        train_params = self.params.copy()
        train_params["early_stopping_rounds"] = early_stopping_rounds

        self.model = xgb.XGBRegressor(**train_params)

        eval_set = [(X_train, y_train), (X_test, y_test)]

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # Get results
        results = self._evaluate(X_train, y_train, X_test, y_test)

        # Store training history
        self.training_history.append(
            {
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                "n_features": len(feature_columns),
                "best_iteration": self.model.best_iteration,
                **results,
            }
        )

        logger.info(f"  ✓ Training complete")
        logger.info(f"    Best iteration: {self.model.best_iteration}")
        logger.info(f"    Train MAE: {results['train_mae']:.3f}")
        logger.info(f"    Test MAE: {results['test_mae']:.3f}")

        return results

    def _evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        """Calculate evaluation metrics."""

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)

        return {
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        }

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

    def save_model(self, output_path: Path):
        """
        Save trained Model 1 to disk.

        Args:
            output_path: Path to save model (JSON format)
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(output_path))
        logger.info(f"Model saved to {output_path}")

    def load_model(self, model_path: Path):
        """
        Load trained Model 1 from disk.

        Args:
            model_path: Path to saved model
        """
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))
        logger.info(f"Model loaded from {model_path}")


def _train_fold_parallel(
    hour: str,
    fold_name: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: List[str],
    params: Optional[Dict],
    early_stopping_rounds: int = 25,
) -> Tuple[str, str, Dict, "Model1Trainer"]:
    """
    Train a single fold in parallel.

    This function is designed to be called by joblib.Parallel() and trains
    one model for a specific (hour, fold) combination.

    Args:
        hour: Hour string (e.g., "08:00")
        fold_name: Fold identifier (e.g., "train_w1-4_test_w5")
        train_data: Training DataFrame with 'target' column
        test_data: Test DataFrame with 'target' column
        feature_columns: List of feature column names
        params: XGBoost parameters
        early_stopping_rounds: Early stopping patience

    Returns:
        Tuple of (hour, fold_name, results_dict, trained_model)
    """
    # Train model
    trainer = Model1Trainer(params)
    results = trainer.train(
        train_data=train_data,
        test_data=test_data,
        feature_columns=feature_columns,
        early_stopping_rounds=early_stopping_rounds,
    )

    return (hour, fold_name, results, trainer)


def train_model1(
    datasets: Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]],
    feature_columns: List[str],
    output_dir: Path,
    params: Optional[Dict] = None,
    n_jobs: int = 4,
) -> Dict[str, Dict]:
    """
    Train Model 1 for multiple hours with walk-forward validation.

    Supports parallel training of folds using joblib for significant speedup.

    Args:
        datasets: Nested dict {hour: {fold_name: (train_df, test_df)}}
        feature_columns: List of feature column names
        output_dir: Directory to save models and results
        params: Optional XGBoost parameters
        n_jobs: Number of parallel jobs (default: 4, use 1 for sequential, -1 for all cores)

    Returns:
        Nested dict with all results {hour: {fold: metrics}}

    Performance:
        - Sequential (n_jobs=1): ~60-76 minutes for typical dataset
        - Parallel (n_jobs=4): ~15-20 minutes (3-4× speedup)
        - Parallel (n_jobs=8): ~10-12 minutes (6-7× speedup)
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: MODEL 1 TRAINING (Feature Discovery)")
    logger.info(f"Parallel jobs: {n_jobs}")
    logger.info("=" * 70)

    all_results = {}

    # Flatten all (hour, fold) pairs into a list of training tasks
    all_tasks = []
    total_folds = 0
    for hour, folds in datasets.items():
        for fold_name, (train_data, test_data) in folds.items():
            all_tasks.append((hour, fold_name, train_data, test_data))
            total_folds += 1

    logger.info(
        f"\nTotal training tasks: {total_folds} ({len(datasets)} hours × ~{total_folds//len(datasets)} folds/hour)"
    )
    logger.info(
        f"Parallelization: {'Sequential' if n_jobs == 1 else f'{n_jobs} parallel jobs'}"
    )

    if n_jobs == 1:
        # Sequential execution (backward compatible)
        logger.info("\nRunning sequential training...")
        trained_models = []
        for idx, (hour, fold_name, train_data, test_data) in enumerate(all_tasks, 1):
            logger.info(f"Training fold {idx}/{total_folds}: {hour} - {fold_name}")
            result = _train_fold_parallel(
                hour, fold_name, train_data, test_data, feature_columns, params
            )
            trained_models.append(result)
    else:
        # Parallel execution with joblib
        logger.info(f"\nRunning parallel training with {n_jobs} workers...")
        trained_models = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_train_fold_parallel)(
                hour, fold_name, train_data, test_data, feature_columns, params
            )
            for hour, fold_name, train_data, test_data in all_tasks
        )

    # Post-processing: Save models and organize results
    logger.info("\nSaving models and organizing results...")
    for hour, fold_name, results, trainer in trained_models:
        # Ensure hour directory exists
        hour_dir = output_dir / hour.replace(":", "")
        hour_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_filename = f"model1_{hour.replace(':', '')}_{fold_name}.json"
        model_path = hour_dir / model_filename
        trainer.save_model(model_path)

        # Store results by hour
        if hour not in all_results:
            all_results[hour] = {}
        all_results[hour][fold_name] = results

    # Calculate summary statistics per hour
    for hour in datasets.keys():
        if hour in all_results:
            hour_results = all_results[hour]
            avg_metrics = _calculate_average_metrics(hour_results)
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Summary for {hour}:")
            logger.info(
                f"  Average Test MAE: {avg_metrics['avg_test_mae']:.3f} ± {avg_metrics['std_test_mae']:.3f}"
            )
            logger.info(
                f"  Average Test RMSE: {avg_metrics['avg_test_rmse']:.3f} ± {avg_metrics['std_test_rmse']:.3f}"
            )
            logger.info(
                f"  Average Test R²: {avg_metrics['avg_test_r2']:.3f} ± {avg_metrics['std_test_r2']:.3f}"
            )
            logger.info(f"{'=' * 70}")
            all_results[f"{hour}_summary"] = avg_metrics

    # Save all results
    results_path = output_dir / "phase1_model1_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n✓ All results saved to {results_path}")

    return all_results


def _calculate_average_metrics(fold_results: Dict[str, Dict]) -> Dict:
    """Calculate average metrics across folds."""
    metrics = [
        "test_mae",
        "test_rmse",
        "test_r2",
        "train_mae",
        "train_rmse",
        "train_r2",
    ]

    avg_metrics = {}
    for metric in metrics:
        values = [results[metric] for results in fold_results.values()]
        avg_metrics[f"avg_{metric}"] = float(np.mean(values))
        avg_metrics[f"std_{metric}"] = float(np.std(values))

    return avg_metrics

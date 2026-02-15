#!/usr/bin/env python3
"""
Evaluate Test Weeks (Unified Models)

Generate predictions for 3 hold-out test weeks using unified models
and create interactive comparison plots.

Usage:
    python -m windmojo.evaluate_test_weeks_unified \
        --config config/lake_como/lecco_training.yaml \
        --test-weeks data/.../test_weeks.json \
        --models-dir data/.../models/phase2_model2 \
        --output-dir data/.../test_week_evaluations_unified
"""

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UnifiedTestWeekEvaluator:
    """Evaluator for test weeks using unified models."""

    def __init__(self, config: dict, test_weeks: dict, models_dir: Path):
        """
        Initialize evaluator.

        Args:
            config: Configuration dict
            test_weeks: Test weeks dict
            models_dir: Directory with trained unified Model 2 models
        """
        self.config = config
        self.test_weeks = test_weeks
        self.models_dir = Path(models_dir)

        self.approaches = [
            "gfs_topn",
            "gfs_balanced",
            "gfs_hybrid",
            "gfs_no_wind",
            "gfs_regime_indicators",
        ]

        # Discover any xgboost*.pkl models in the models directory
        self.pkl_models = self._discover_pkl_models()

    def _discover_pkl_models(self) -> dict:
        """
        Discover any xgboost*.pkl files in the models directory.

        Scans the models directory for files matching the pattern 'xgboost*.pkl'
        and returns a dictionary mapping model names to file paths.

        Returns:
            Dictionary mapping model name (filename without extension) to Path object

        Example:
            If 'xgboost_speed_model_4wind.pkl' exists, returns:
            {'xgboost_speed_model_4wind': Path('data/.../xgboost_speed_model_4wind.pkl')}
        """
        pkl_models = {}

        # Search for xgboost*.pkl files in the models directory
        for pkl_file in self.models_dir.glob("xgboost*.pkl"):
            # Use filename without extension as the model name
            model_name = pkl_file.stem  # e.g., 'xgboost_speed_model_4wind'
            pkl_models[model_name] = pkl_file
            logger.info(f"  Discovered pkl model: {model_name} ({pkl_file.name})")

        if pkl_models:
            logger.info(f"  Found {len(pkl_models)} pkl model(s) to evaluate")
        else:
            logger.info("  No xgboost*.pkl models found in models directory")

        return pkl_models

    def load_unified_model(self, approach: str) -> xgb.Booster:
        """
        Load unified model for an approach.

        Returns:
            XGBoost Booster model
        """
        logger.info(f"  Loading unified model for {approach}...")

        approach_dir = self.models_dir / approach
        model_file = approach_dir / f"model2_{approach}_unified.json"

        if not model_file.exists():
            logger.warning(f"    Model not found: {model_file}")
            return None

        model = xgb.Booster()
        model.load_model(str(model_file))

        logger.info(f"    ✓ Loaded unified model")

        return model

    def load_pkl_model(self, model_name: str, pkl_path: Path) -> tuple:
        """
        Load XGBoost model from pickle file.

        The pickle file should contain a dict with:
        - 'model': XGBRegressor object
        - 'metadata': dict with 'feature_names' list

        Args:
            model_name: Name of the model (for logging)
            pkl_path: Path to pickle file

        Returns:
            Tuple of (model, feature_names) or (None, None) if loading fails

        Example:
            model, features = evaluator.load_pkl_model('xgboost_speed_model_4wind', pkl_path)
        """
        logger.info(f"  Loading pkl model: {model_name}...")

        try:
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)

            # Extract model and feature names
            model = pkl_data.get("model")
            metadata = pkl_data.get("metadata", {})
            feature_names = metadata.get("feature_names", [])

            if model is None:
                logger.warning(f"    No 'model' key found in pickle file")
                return None, None

            if not feature_names:
                logger.warning(f"    No feature names found in metadata")
                return None, None

            logger.info(f"    ✓ Loaded pkl model ({len(feature_names)} features)")

            return model, feature_names

        except Exception as e:
            logger.error(f"    Failed to load pkl model: {e}")
            return None, None

    def load_test_week_data(
        self,
        week_name: str,
        week_info: dict,
        features_file: Path,
        measurements_file: Path,
        target_column: str,
    ) -> tuple:
        """
        Load data for a test week.

        Returns:
            (features_df, measurements_df)
        """
        logger.info(f"  Loading data for {week_name}...")

        start = pd.to_datetime(week_info["start_date"])
        end = pd.to_datetime(week_info["end_date"])

        # Load features
        features = pd.read_parquet(features_file)
        features["time"] = pd.to_datetime(features["time"])
        features = features[(features["time"] >= start) & (features["time"] <= end)]

        logger.info(f"    Features: {len(features)} rows")

        # Load measurements
        measurements = pd.read_csv(measurements_file)
        timestamp_col = next(
            (c for c in measurements.columns if c.lower() in ["timestamp", "time"]),
            None,
        )

        try:
            measurements[timestamp_col] = pd.to_datetime(measurements[timestamp_col])
        except (ValueError, TypeError):
            measurements[timestamp_col] = pd.to_datetime(
                measurements[timestamp_col], format="%Y-%m-%d_%H%M"
            )

        measurements.rename(columns={timestamp_col: "time"}, inplace=True)
        measurements = measurements[
            (measurements["time"] >= start) & (measurements["time"] <= end)
        ]

        # Set target
        measurements["measured_speed"] = measurements[target_column]

        logger.info(f"    Measurements: {len(measurements)} rows")

        return features, measurements

    def generate_predictions_unified(
        self, model: xgb.Booster, features: pd.DataFrame, selected_features: list
    ) -> pd.DataFrame:
        """
        Generate predictions for all hours using unified model.

        Returns:
            DataFrame with time, prediction columns
        """
        # Filter to valid hours (06:00-18:00)
        features = features.copy()
        features["hour"] = features["time"].dt.strftime("%H:%M")
        valid_hours = [f"{h:02d}:00" for h in range(6, 19)]  # 06:00-18:00
        features = features[features["hour"].isin(valid_hours)]

        if len(features) == 0:
            return pd.DataFrame(columns=["time", "prediction"])

        # Verify features exist
        missing_features = [f for f in selected_features if f not in features.columns]
        if missing_features:
            logger.warning(
                f"    Missing {len(missing_features)} features, predictions may be inaccurate"
            )

        available_features = [f for f in selected_features if f in features.columns]

        # Prepare features
        X = features[available_features]

        # Predict
        dmatrix = xgb.DMatrix(X)
        preds = model.predict(dmatrix)

        # Build result
        result = pd.DataFrame({"time": features["time"].values, "prediction": preds})

        return result

    def generate_predictions_pkl(
        self, model, features: pd.DataFrame, feature_names: list
    ) -> pd.DataFrame:
        """
        Generate predictions using XGBRegressor from pickle file.

        Args:
            model: XGBRegressor model object (from pickle)
            features: Features DataFrame
            feature_names: List of feature names required by the model

        Returns:
            DataFrame with time, prediction columns
        """
        # Filter to valid hours (06:00-18:00)
        features = features.copy()
        features["hour"] = features["time"].dt.strftime("%H:%M")
        valid_hours = [f"{h:02d}:00" for h in range(6, 19)]  # 06:00-18:00
        features = features[features["hour"].isin(valid_hours)]

        if len(features) == 0:
            return pd.DataFrame(columns=["time", "prediction"])

        # Verify features exist
        missing_features = [f for f in feature_names if f not in features.columns]
        if missing_features:
            logger.warning(
                f"    Missing {len(missing_features)} features: {', '.join(missing_features[:5])}..."
            )
            logger.warning(
                f"    Skipping this model - feature set mismatch with current features file"
            )
            return pd.DataFrame(columns=["time", "prediction"])

        available_features = [f for f in feature_names if f in features.columns]

        # Prepare features (XGBRegressor uses DataFrame/array directly, not DMatrix)
        X = features[available_features]

        # Predict (use .predict() method of XGBRegressor)
        # Note: validate_features=False would bypass feature name checking but we already verified above
        preds = model.predict(X)

        # Build result
        result = pd.DataFrame({"time": features["time"].values, "prediction": preds})

        return result

    def get_gfs_wind(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Extract GFS raw wind from features.

        Returns:
            DataFrame with time, gfs_speed, gfs_direction
        """
        gfs_data = []

        for _, row in features.iterrows():
            gfs_data.append(
                {
                    "time": row["time"],
                    "gfs_speed": row.get("wind_speed_10m_center", np.nan),
                    "gfs_direction": row.get("wind_direction_10m_center", np.nan),
                }
            )

        return pd.DataFrame(gfs_data)

    def evaluate_week(self, week_name: str, week_info: dict, output_dir: Path) -> dict:
        """
        Evaluate all unified models on a test week.

        Returns:
            Dict with evaluation results
        """
        logger.info(f"\nEvaluating {week_name}...")

        features_file = Path(self.config["data"]["features_file"])
        measurements_file = Path(self.config["data"]["measurements_file"])
        target_column = self.config["data"]["target_column"]

        # Load data
        features, measurements = self.load_test_week_data(
            week_name, week_info, features_file, measurements_file, target_column
        )

        # Get GFS wind
        gfs = self.get_gfs_wind(features)

        # Collect predictions from all approaches
        all_predictions = {}

        for approach in self.approaches:
            # Load unified model
            model = self.load_unified_model(approach)

            if model is None:
                continue

            # Load feature selection
            feature_selection_file = (
                self.models_dir.parent
                / "phase1_model1"
                / "full"
                / "feature_selections"
                / f"features_{approach}.json"
            )

            if not feature_selection_file.exists():
                logger.warning(f"    No feature selection file for {approach}")
                continue

            with open(feature_selection_file, "r") as f:
                selection = json.load(f)

            selected_features = [f["feature"] for f in selection["features"]]

            # Generate predictions
            preds = self.generate_predictions_unified(
                model, features, selected_features
            )
            all_predictions[approach] = preds

            logger.info(f"    ✓ {approach}: {len(preds)} predictions")

        # Evaluate pickle models
        for pkl_name, pkl_path in self.pkl_models.items():
            # Load pkl model
            model, feature_names = self.load_pkl_model(pkl_name, pkl_path)

            if model is None or feature_names is None:
                continue

            # Generate predictions
            preds = self.generate_predictions_pkl(model, features, feature_names)
            all_predictions[pkl_name] = preds

            logger.info(f"    ✓ {pkl_name}: {len(preds)} predictions")

        # Merge all data
        merged = measurements[["time", "measured_speed"]].copy()
        merged = pd.merge(merged, gfs, on="time", how="left")

        for approach, preds in all_predictions.items():
            merged = pd.merge(
                merged, preds[["time", "prediction"]], on="time", how="left"
            )
            merged.rename(columns={"prediction": f"{approach}_pred"}, inplace=True)

        # Calculate metrics
        metrics = self.calculate_metrics(merged, all_predictions.keys())

        # Create visualization
        self.create_visualization(week_name, week_info, merged, metrics, output_dir)

        return {"week_name": week_name, "metrics": metrics, "data_points": len(merged)}

    def calculate_metrics(self, df: pd.DataFrame, approaches: list) -> dict:
        """Calculate MAE/RMSE for each approach + GFS."""
        metrics = {}

        # GFS metrics
        valid_gfs = df.dropna(subset=["measured_speed", "gfs_speed"])
        if len(valid_gfs) > 0:
            metrics["gfs"] = {
                "mae": float(
                    mean_absolute_error(
                        valid_gfs["measured_speed"], valid_gfs["gfs_speed"]
                    )
                ),
                "rmse": float(
                    np.sqrt(
                        mean_squared_error(
                            valid_gfs["measured_speed"], valid_gfs["gfs_speed"]
                        )
                    )
                ),
                "n_samples": len(valid_gfs),
            }

        # Model metrics
        for approach in approaches:
            pred_col = f"{approach}_pred"
            if pred_col not in df.columns:
                continue

            valid = df.dropna(subset=["measured_speed", pred_col])
            if len(valid) > 0:
                metrics[approach] = {
                    "mae": float(
                        mean_absolute_error(valid["measured_speed"], valid[pred_col])
                    ),
                    "rmse": float(
                        np.sqrt(
                            mean_squared_error(valid["measured_speed"], valid[pred_col])
                        )
                    ),
                    "n_samples": len(valid),
                }

        return metrics

    def create_visualization(
        self,
        week_name: str,
        week_info: dict,
        df: pd.DataFrame,
        metrics: dict,
        output_dir: Path,
    ):
        """Create interactive Plotly visualization (split into half-week views)."""
        logger.info(f"  Creating visualizations for {week_name}...")

        # Sort by time
        df = df.sort_values("time").reset_index(drop=True)

        # Split into two halves for better visibility
        midpoint = len(df) // 2
        df_first = df.iloc[:midpoint]
        df_second = df.iloc[midpoint:]

        for part_name, df_part in [
            ("first_half", df_first),
            ("second_half", df_second),
        ]:
            self._create_single_plot(
                week_name, week_info, df_part, metrics, output_dir, part_name
            )

    def _create_single_plot(
        self,
        week_name: str,
        week_info: dict,
        df: pd.DataFrame,
        metrics: dict,
        output_dir: Path,
        part_name: str,
    ):
        """Create a single plot for a portion of the week."""

        # Create figure
        fig = go.Figure()

        # Color scheme (expandable for >5 models)
        # Base colors for JSON models
        colors = {
            "measured": "black",
            "gfs": "gray",
            "gfs_topn": "blue",
            "gfs_balanced": "green",
            "gfs_hybrid": "red",
            "gfs_no_wind": "orange",
            "gfs_regime_indicators": "purple",
        }

        # Extended color palette for pkl models
        extra_colors = [
            "cyan",
            "magenta",
            "brown",
            "pink",
            "olive",
            "navy",
            "teal",
            "maroon",
        ]

        # Assign colors to pkl models dynamically
        for i, pkl_name in enumerate(self.pkl_models.keys()):
            if pkl_name not in colors:
                colors[pkl_name] = extra_colors[i % len(extra_colors)]

        # Add measured wind (thick black line)
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["measured_speed"],
                mode="lines+markers",
                name="Measured",
                line=dict(color=colors["measured"], width=3),
                marker=dict(size=6),
                hovertemplate="<b>Measured</b><br>Time: %{x}<br>Speed: %{y:.2f} m/s<extra></extra>",
            )
        )

        # Add GFS
        gfs_mae = metrics.get("gfs", {}).get("mae", 0)
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["gfs_speed"],
                mode="lines",
                name=f"GFS Raw (MAE: {gfs_mae:.2f})",
                line=dict(color=colors["gfs"], width=2, dash="dash"),
                hovertemplate="<b>GFS</b><br>Time: %{x}<br>Speed: %{y:.2f} m/s<extra></extra>",
            )
        )

        # Add model predictions (5 unified JSON models)
        for approach in self.approaches:
            pred_col = f"{approach}_pred"
            if pred_col in df.columns:
                mae = metrics.get(approach, {}).get("mae", 0)
                display_name = approach.replace("gfs_", "").replace("_", " ").title()

                fig.add_trace(
                    go.Scatter(
                        x=df["time"],
                        y=df[pred_col],
                        mode="lines",
                        name=f"{display_name} (MAE: {mae:.2f})",
                        line=dict(color=colors[approach], width=2),
                        hovertemplate=f"<b>{display_name}</b><br>Time: %{{x}}<br>Speed: %{{y:.2f}} m/s<extra></extra>",
                    )
                )

        # Add pkl model predictions
        for pkl_name in self.pkl_models.keys():
            pred_col = f"{pkl_name}_pred"
            if pred_col in df.columns:
                mae = metrics.get(pkl_name, {}).get("mae", 0)
                # Use filename as display name (without underscores)
                display_name = pkl_name.replace("_", " ").title()

                fig.add_trace(
                    go.Scatter(
                        x=df["time"],
                        y=df[pred_col],
                        mode="lines",
                        name=f"{display_name} (MAE: {mae:.2f})",
                        line=dict(color=colors[pkl_name], width=2),
                        hovertemplate=f"<b>{display_name}</b><br>Time: %{{x}}<br>Speed: %{{y:.2f}} m/s<extra></extra>",
                    )
                )

        # Part name for title
        part_display = part_name.replace("_", " ").title()

        # Update layout
        fig.update_layout(
            title=f"{week_info['description']} - {part_display}<br><sub>Wind Speed Comparison - Unified Models</sub>",
            xaxis_title="Time",
            yaxis_title="Wind Speed (m/s)",
            hovermode="x unified",
            template="plotly_white",
            height=700,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.18),
        )

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{week_name}_{part_name}.html"
        fig.write_html(str(output_file))

        logger.info(f"    ✓ Saved to {output_file}")

    def generate_summary_report(self, results: list, output_dir: Path):
        """Generate summary report."""
        logger.info("\nGenerating summary report...")

        # Markdown report
        lines = []
        lines.append("# Test Week Evaluation Report (Unified Models)")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("## Performance Summary")
        lines.append("")
        lines.append("| Week | Model | MAE (m/s) | RMSE (m/s) | Samples |")
        lines.append("|------|-------|-----------|------------|---------|")

        for result in results:
            week_name = result["week_name"]
            metrics = result["metrics"]

            # GFS
            if "gfs" in metrics:
                m = metrics["gfs"]
                lines.append(
                    f"| {week_name} | gfs | {m['mae']:.3f} | {m['rmse']:.3f} | {m['n_samples']} |"
                )

            # JSON Models
            for approach in self.approaches:
                if approach in metrics:
                    m = metrics[approach]
                    lines.append(
                        f"| {week_name} | {approach} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['n_samples']} |"
                    )

            # Pkl Models
            for pkl_name in self.pkl_models.keys():
                if pkl_name in metrics:
                    m = metrics[pkl_name]
                    lines.append(
                        f"| {week_name} | {pkl_name} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['n_samples']} |"
                    )

        lines.append("")
        lines.append("## Average Performance")
        lines.append("")

        # Calculate averages (include pkl models)
        avg_metrics = {}
        all_model_names = ["gfs"] + self.approaches + list(self.pkl_models.keys())

        for approach in all_model_names:
            maes = []
            rmses = []

            for result in results:
                if approach in result["metrics"]:
                    maes.append(result["metrics"][approach]["mae"])
                    rmses.append(result["metrics"][approach]["rmse"])

            if maes:
                avg_metrics[approach] = {"mae": np.mean(maes), "rmse": np.mean(rmses)}

        lines.append("| Model | Avg MAE (m/s) | Avg RMSE (m/s) |")
        lines.append("|-------|---------------|----------------|")

        for approach, m in sorted(avg_metrics.items(), key=lambda x: x[1]["mae"]):
            lines.append(f"| {approach} | {m['mae']:.3f} | {m['rmse']:.3f} |")

        lines.append("")
        lines.append("## Key Findings")
        lines.append("")

        # Find best model
        best_approach = min(
            [(k, v["mae"]) for k, v in avg_metrics.items() if k != "gfs"],
            key=lambda x: x[1],
        )
        gfs_mae = avg_metrics.get("gfs", {}).get("mae", 0)

        lines.append(
            f"- **Best Model:** {best_approach[0]} (MAE: {best_approach[1]:.3f} m/s)"
        )
        lines.append(f"- **GFS Baseline:** {gfs_mae:.3f} m/s")
        lines.append(
            f"- **Improvement:** {((gfs_mae - best_approach[1]) / gfs_mae * 100):.1f}%"
        )
        lines.append("")

        # Write report
        report_file = output_dir / "performance_summary.md"
        with open(report_file, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"✓ Report saved to {report_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate unified models on test weeks"
    )
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--test-weeks", required=True, help="Test weeks JSON")
    parser.add_argument("--models-dir", required=True, help="Models directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    # Load config
    import yaml

    from windmojo.core.config import ensure_unified_config

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Ensure config has 'data' section for backward compatibility
    config = ensure_unified_config(config)

    # Load test weeks
    with open(args.test_weeks, "r") as f:
        test_weeks = json.load(f)

    # Initialize evaluator
    evaluator = UnifiedTestWeekEvaluator(config, test_weeks, Path(args.models_dir))

    # Evaluate all test weeks
    results = []

    for week_name, week_info in test_weeks["test_weeks"].items():
        result = evaluator.evaluate_week(week_name, week_info, Path(args.output_dir))
        results.append(result)

    # Generate summary
    evaluator.generate_summary_report(results, Path(args.output_dir))

    logger.info(f"\n{'='*70}")
    logger.info("✓ EVALUATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"\nOutput directory: {args.output_dir}")
    logger.info(
        f"  - {len(results) * 2} interactive HTML plots ({len(results)} weeks × 2 halves)"
    )
    logger.info(f"  - 1 performance summary report")


if __name__ == "__main__":
    main()

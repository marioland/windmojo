#!/usr/bin/env python3
"""
Evaluate Test Weeks

Generate predictions for 3 hold-out test weeks and create interactive comparison plots.

Usage:
    python -m windmojo.evaluate_test_weeks \
        --config config/lake_como/lecco.yaml \
        --test-weeks data/.../test_weeks.json \
        --models-dir data/.../models/phase2_model2 \
        --output-dir data/.../test_week_evaluations
"""

import argparse
import json
import logging
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


class TestWeekEvaluator:
    """Evaluator for test weeks."""

    def __init__(self, config: dict, test_weeks: dict, models_dir: Path):
        """
        Initialize evaluator.

        Args:
            config: Configuration dict
            test_weeks: Test weeks dict
            models_dir: Directory with trained Model 2 models
        """
        self.config = config
        self.test_weeks = test_weeks
        self.models_dir = Path(models_dir)

        self.approaches = ["gfs_topn", "gfs_balanced", "gfs_hybrid", "gfs_no_wind"]

    def load_models(self, approach: str) -> dict:
        """
        Load all models for an approach.

        Returns:
            Dict mapping hour → model
        """
        logger.info(f"  Loading models for {approach}...")

        models = {}
        approach_dir = self.models_dir / approach

        if not approach_dir.exists():
            logger.warning(f"    No models found for {approach}")
            return models

        model_files = list(approach_dir.glob("model2_*.json"))

        for model_file in model_files:
            # Extract hour from filename: model2_gfs_topn_0600.json → 06:00
            hour_str = model_file.stem.split("_")[-1]
            hour = f"{hour_str[:2]}:{hour_str[2:]}"

            model = xgb.Booster()
            model.load_model(str(model_file))

            models[hour] = model

        logger.info(f"    Loaded {len(models)} models")

        return models

    def load_test_week_data(
        self,
        week_name: str,
        week_info: dict,
        features_file: Path,
        measurements_file: Path,
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

        logger.info(f"    Measurements: {len(measurements)} rows")

        return features, measurements

    def generate_predictions(
        self, models: dict, features: pd.DataFrame, feature_selection_file: Path
    ) -> pd.DataFrame:
        """
        Generate predictions for all hours.

        Returns:
            DataFrame with time, hour, prediction columns
        """
        # Load selected features
        with open(feature_selection_file, "r") as f:
            selection = json.load(f)

        selected_features = [f["feature"] for f in selection["features"]]

        predictions = []

        for hour, model in models.items():
            hour_data = features[features["time"].dt.strftime("%H:%M") == hour].copy()

            if len(hour_data) == 0:
                continue

            # Prepare features
            X = hour_data[selected_features]

            # Predict
            dmatrix = xgb.DMatrix(X)
            preds = model.predict(dmatrix)

            # Store
            for i, pred in enumerate(preds):
                predictions.append(
                    {
                        "time": hour_data.iloc[i]["time"],
                        "hour": hour,
                        "prediction": pred,
                    }
                )

        return pd.DataFrame(predictions)

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
        Evaluate all models on a test week.

        Returns:
            Dict with evaluation results
        """
        logger.info(f"\nEvaluating {week_name}...")

        features_file = Path(
            f"data/{self.config['area']}/{self.config['spot']}/features/{self.config['spot']}_features_phase1.parquet"
        )
        measurements_file = Path(self.config["measurement_file"])

        # Load data
        features, measurements = self.load_test_week_data(
            week_name, week_info, features_file, measurements_file
        )

        # Get GFS wind
        gfs = self.get_gfs_wind(features)

        # Get measured wind
        measurements["measured_speed"] = measurements["lecco speed"]
        measurements["measured_direction"] = measurements.get("lecco direction", np.nan)

        # Collect predictions from all approaches
        all_predictions = {}

        for approach in self.approaches:
            # Load models
            models = self.load_models(approach)

            if not models:
                continue

            # Generate predictions
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

            preds = self.generate_predictions(models, features, feature_selection_file)
            all_predictions[approach] = preds

        # Merge all data
        merged = measurements[["time", "measured_speed", "measured_direction"]].copy()
        merged = pd.merge(merged, gfs, on="time", how="left")

        for approach, preds in all_predictions.items():
            merged = pd.merge(
                merged, preds[["time", "prediction"]], on="time", how="left"
            )
            merged.rename(columns={"prediction": f"{approach}_pred"}, inplace=True)

        # Calculate metrics
        metrics = self.calculate_metrics(merged, all_predictions.keys())

        # Create visualization
        self.create_visualization(week_name, week_info, merged, output_dir)

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
        self, week_name: str, week_info: dict, df: pd.DataFrame, output_dir: Path
    ):
        """Create interactive Plotly visualization."""
        logger.info(f"  Creating visualization for {week_name}...")

        # Sort by time
        df = df.sort_values("time").reset_index(drop=True)

        # Create figure
        fig = go.Figure()

        # Color scheme
        colors = {
            "measured": "black",
            "gfs": "gray",
            "gfs_topn": "blue",
            "gfs_balanced": "green",
            "gfs_hybrid": "red",
            "gfs_no_wind": "orange",
        }

        # Add measured wind (thick black line)
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["measured_speed"],
                mode="lines+markers",
                name="Measured",
                line=dict(color=colors["measured"], width=3),
                marker=dict(size=4),
                hovertemplate="<b>Measured</b><br>Time: %{x}<br>Speed: %{y:.2f} m/s<extra></extra>",
            )
        )

        # Add GFS
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["gfs_speed"],
                mode="lines",
                name="GFS Raw",
                line=dict(color=colors["gfs"], width=2, dash="dash"),
                hovertemplate="<b>GFS</b><br>Time: %{x}<br>Speed: %{y:.2f} m/s<extra></extra>",
            )
        )

        # Add model predictions
        for approach in ["gfs_topn", "gfs_balanced", "gfs_hybrid", "gfs_no_wind"]:
            pred_col = f"{approach}_pred"
            if pred_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["time"],
                        y=df[pred_col],
                        mode="lines",
                        name=f"Model: {approach}",
                        line=dict(color=colors[approach], width=2),
                        hovertemplate=f"<b>{approach}</b><br>Time: %{{x}}<br>Speed: %{{y:.2f}} m/s<extra></extra>",
                    )
                )

        # Update layout
        fig.update_layout(
            title=f"{week_info['description']}<br><sub>Wind Speed Comparison</sub>",
            xaxis_title="Time",
            yaxis_title="Wind Speed (m/s)",
            hovermode="x unified",
            template="plotly_white",
            height=600,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.15),
        )

        # Save
        output_file = output_dir / f"{week_name}.html"
        fig.write_html(str(output_file))

        logger.info(f"    Saved to {output_file}")

    def generate_summary_report(self, results: list, output_dir: Path):
        """Generate summary report."""
        logger.info("\nGenerating summary report...")

        # Markdown report
        lines = []
        lines.append("# Test Week Evaluation Report")
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

            for model_name, model_metrics in metrics.items():
                mae = model_metrics["mae"]
                rmse = model_metrics["rmse"]
                n = model_metrics["n_samples"]

                lines.append(
                    f"| {week_name} | {model_name} | {mae:.3f} | {rmse:.3f} | {n} |"
                )

        lines.append("")

        # Save
        report_file = output_dir / "performance_summary.md"
        with open(report_file, "w") as f:
            f.write("\n".join(lines))

        # Save JSON
        json_file = output_dir / "performance_summary.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Reports saved to {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate test weeks")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--test-weeks", required=True, help="Path to test weeks JSON")
    parser.add_argument(
        "--models-dir", required=True, help="Path to Model 2 models directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for visualizations"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("TEST WEEK EVALUATION")
    logger.info("=" * 70)

    # Load config
    import yaml

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load test weeks
    with open(args.test_weeks, "r") as f:
        test_weeks = json.load(f)

    # Initialize evaluator
    evaluator = TestWeekEvaluator(config, test_weeks, Path(args.models_dir))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each test week
    results = []
    for week_name, week_info in test_weeks["test_weeks"].items():
        result = evaluator.evaluate_week(week_name, week_info, output_dir)
        results.append(result)

    # Generate summary
    evaluator.generate_summary_report(results, output_dir)

    logger.info("\n" + "=" * 70)
    logger.info("✓ TEST WEEK EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nOutputs:")
    logger.info(f"  - 3 interactive HTML plots in {output_dir}")
    logger.info(f"  - Performance summary (MD + JSON)")


if __name__ == "__main__":
    main()

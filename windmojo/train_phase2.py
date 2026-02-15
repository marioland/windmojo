"""
Phase 2 Training CLI - Feature Selection Comparison

Train Phase 2 models with different feature selection strategies.

Usage:
    # Train single selection
    uv run python windmojo/train_phase2.py \\
        --config config/lake_como/dervio.yaml \\
        --selection top_50 \\
        --hours 08:00,13:00,14:00,15:00,16:00

    # Train all 24 selections
    uv run python windmojo/train_phase2.py \\
        --config config/lake_como/dervio.yaml \\
        --selection all \\
        --hours 08:00,13:00,14:00,15:00,16:00

    # Train specific category
    uv run python windmojo/train_phase2.py \\
        --config config/lake_como/dervio.yaml \\
        --selection-category importance \\
        --hours 08:00,13:00,14:00,15:00,16:00
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from windmojo.core.config import load_config
from windmojo.train.importance_extractor import (
    categorize_features,
    extract_importance,
    save_importance,
)
from windmojo.train.phase1_model1 import Model1Trainer  # Re-use for Phase 2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_selection(area: str, spot_name: str, selection_name: str) -> Dict[str, Any]:
    """Load feature selection configuration."""
    selection_dir = Path(f"data/{area}/{spot_name}/models/phase2_selections")
    selection_file = selection_dir / f"{selection_name}.json"

    if not selection_file.exists():
        raise FileNotFoundError(f"Selection not found: {selection_file}")

    with open(selection_file) as f:
        return json.load(f)


def get_all_selections(area: str, spot_name: str) -> List[str]:
    """Get list of all available selections."""
    selection_dir = Path(f"data/{area}/{spot_name}/models/phase2_selections")
    return [f.stem for f in selection_dir.glob("*.json") if f.stem != "summary"]


def get_selections_by_category(area: str, spot_name: str, category: str) -> List[str]:
    """Get selections in a specific category."""
    all_selections = get_all_selections(area, spot_name)
    result = []

    for sel_name in all_selections:
        sel_data = load_selection(area, spot_name, sel_name)
        if sel_data.get("category") == category:
            result.append(sel_name)

    return result


def parse_hours(hours_str: str) -> list:
    """Parse comma-separated hours string."""
    return [h.strip() for h in hours_str.split(",")]


def train_selection(
    config: dict, selection_name: str, hours: List[str], target_column: str
) -> Dict[str, Any]:
    """Train a single feature selection."""
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 2 TRAINING: {selection_name}")
    logger.info(f"{'='*80}\n")

    # Load selection
    area = config["area"]
    spot_name = config["spot"]
    selection = load_selection(area, spot_name, selection_name)

    logger.info(f"Selection: {selection['name']}")
    logger.info(f"Description: {selection['description']}")
    logger.info(f"Category: {selection['category']}")
    logger.info(f"Features: {len(selection['features'])}")
    logger.info("")

    # Load training data
    logger.info("Loading training data...")
    from pathlib import Path

    measurements_path = Path(
        config.get("measurement_file", config["data"]["measurements_file"])
    )
    features_path = Path(config["data"]["features_file"])

    # Load features and measurements separately since we need to filter features
    import pandas as pd

    features_df = pd.read_parquet(features_path)
    measurements_df = pd.read_csv(measurements_path)
    measurements_df["time"] = pd.to_datetime(
        measurements_df["timestamp"], format="%Y-%m-%d_%H%M"
    )
    measurements_df = measurements_df.drop(columns=["timestamp"])

    # Filter features to selection
    available_features = set(features_df.columns)
    selected_features = [f for f in selection["features"] if f in available_features]

    if len(selected_features) < len(selection["features"]):
        missing = len(selection["features"]) - len(selected_features)
        logger.warning(f"  {missing} features not found in dataset")

    logger.info(f"  Using {len(selected_features)} features")

    # Keep only selected features + time column
    features_df = features_df[["time"] + selected_features]

    # Merge features with measurements
    merged_df = features_df.merge(
        measurements_df[["time", target_column]], on="time", how="inner"
    )
    logger.info(f"  Merged {len(merged_df)} rows")
    logger.info(f"  Features: {len(selected_features)}")

    # Output directory
    output_dir = Path(f"data/{area}/{spot_name}/models/phase2_results/{selection_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train for each hour
    results = {
        "selection_name": selection_name,
        "selection_info": {
            "name": selection["name"],
            "description": selection["description"],
            "category": selection["category"],
            "total_features": len(selection["features"]),
            "available_features": len(selected_features),
        },
        "hours": {},
        "overall": {},
    }

    all_maes = []
    all_rmses = []
    all_r2s = []

    for hour in hours:
        hour_key = hour.replace(":", "")
        logger.info(f"\n{'-'*80}")
        logger.info(f"Training for {hour}")
        logger.info(f"{'-'*80}\n")

        # Filter to this hour
        hour_int = int(hour.split(":")[0])
        hour_df = merged_df[merged_df["time"].dt.hour == hour_int].copy()

        if len(hour_df) == 0:
            logger.warning(f"  No data for {hour}, skipping")
            continue

        # Remove rows with NaN in target
        hour_df = hour_df.dropna(subset=[target_column])

        if len(hour_df) == 0:
            logger.warning(f"  No valid data for {hour} after removing NaN, skipping")
            continue

        # Train/test split (80/20)
        split_idx = int(len(hour_df) * 0.8)
        train_df = hour_df.iloc[:split_idx]
        test_df = hour_df.iloc[split_idx:]

        # Prepare train/test data with 'target' column
        train_subset = train_df[selected_features + [target_column]].copy()
        train_subset = train_subset.rename(columns={target_column: "target"})
        test_subset = test_df[selected_features + [target_column]].copy()
        test_subset = test_subset.rename(columns={target_column: "target"})

        # Train
        trainer = Model1Trainer()
        results_dict = trainer.train(
            train_data=train_subset,
            test_data=test_subset,
            feature_columns=selected_features,
        )

        # Get the trained model
        model = trainer.model

        # Generate predictions manually
        X_test = test_df[selected_features]
        y_test = test_df[target_column]
        y_pred = model.predict(X_test)

        predictions = test_df[["time"]].copy()
        predictions["actual"] = y_test.values
        predictions["predicted"] = y_pred
        predictions["error"] = predictions["predicted"] - predictions["actual"]
        predictions["abs_error"] = predictions["error"].abs()

        # Use test metrics from results_dict
        metrics = {
            "mae": results_dict["test_mae"],
            "rmse": results_dict["test_rmse"],
            "r2": results_dict["test_r2"],
            "max_error": predictions["abs_error"].max(),
        }

        # Extract importance
        importance = extract_importance(model, selected_features)
        feature_categories = categorize_features(importance)

        # Save model and results
        hour_dir = output_dir / hour_key
        hour_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save_model(str(hour_dir / "model.json"))

        # Save importance
        save_importance(
            importance_df=feature_categories,
            output_path=hour_dir / f"importance_{hour_key}.json",
        )

        # Save predictions
        pred_df = predictions[["time", "actual", "predicted", "error", "abs_error"]]
        pred_df.to_csv(hour_dir / "predictions.csv", index=False)

        # Save metrics
        with open(hour_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Store results
        results["hours"][hour_key] = {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "max_error": metrics["max_error"],
            "samples": len(y_test),
        }

        all_maes.append(metrics["mae"])
        all_rmses.append(metrics["rmse"])
        all_r2s.append(metrics["r2"])

        logger.info(f"  MAE:  {metrics['mae']:.2f} m/s")
        logger.info(f"  RMSE: {metrics['rmse']:.2f} m/s")
        logger.info(f"  R²:   {metrics['r2']:.3f}")

    # Overall results
    results["overall"] = {
        "avg_mae": sum(all_maes) / len(all_maes),
        "avg_rmse": sum(all_rmses) / len(all_rmses),
        "avg_r2": sum(all_r2s) / len(all_r2s),
        "min_mae": min(all_maes),
        "max_mae": max(all_maes),
    }

    # Save overall results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info(f"OVERALL RESULTS: {selection_name}")
    logger.info(f"{'='*80}")
    logger.info(f"  Average MAE:  {results['overall']['avg_mae']:.2f} m/s")
    logger.info(f"  Average RMSE: {results['overall']['avg_rmse']:.2f} m/s")
    logger.info(f"  Average R²:   {results['overall']['avg_r2']:.3f}")
    logger.info(
        f"  MAE range:    {results['overall']['min_mae']:.2f} - {results['overall']['max_mae']:.2f} m/s"
    )
    logger.info("")

    return results


def main():
    """Main entry point for Phase 2 training."""
    parser = argparse.ArgumentParser(
        description="Phase 2 Training - Feature Selection Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--selection",
        type=str,
        default="top_50",
        help='Feature selection name (or "all" for all selections)',
    )

    parser.add_argument(
        "--selection-category",
        type=str,
        choices=["importance", "category", "thematic", "dimensionality", "innovative"],
        help="Train all selections in this category",
    )

    parser.add_argument(
        "--hours",
        type=str,
        default="08:00,13:00,14:00,15:00,16:00",
        help="Comma-separated list of hours to predict",
    )

    parser.add_argument(
        "--target-column",
        type=str,
        default="dervio avgSpeed",
        help="Target column name in measurements CSV",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    area = config["area"]
    spot_name = config["spot"]
    hours = parse_hours(args.hours)

    # Determine which selections to train
    if args.selection == "all":
        selections = get_all_selections(area, spot_name)
        logger.info(f"Training ALL {len(selections)} selections")
    elif args.selection_category:
        selections = get_selections_by_category(
            area, spot_name, args.selection_category
        )
        logger.info(
            f"Training {len(selections)} selections in category: {args.selection_category}"
        )
    else:
        selections = [args.selection]
        logger.info(f"Training single selection: {args.selection}")

    logger.info(f"  Selections: {', '.join(selections)}")
    logger.info(f"  Hours: {', '.join(hours)}")
    logger.info("")

    # Train each selection
    all_results = {}

    for i, selection_name in enumerate(selections, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"SELECTION {i}/{len(selections)}: {selection_name}")
        logger.info(f"{'#'*80}\n")

        try:
            results = train_selection(
                config=config,
                selection_name=selection_name,
                hours=hours,
                target_column=args.target_column,
            )
            all_results[selection_name] = results
        except Exception as e:
            logger.error(f"Error training {selection_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save comparison summary
    if len(all_results) > 1:
        logger.info(f"\n{'='*80}")
        logger.info("COMPARISON SUMMARY")
        logger.info(f"{'='*80}\n")

        # Sort by average MAE
        sorted_results = sorted(
            all_results.items(), key=lambda x: x[1]["overall"]["avg_mae"]
        )

        logger.info(
            f"{'Rank':<6} {'Selection':<30} {'Features':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10}"
        )
        logger.info("-" * 80)

        for rank, (sel_name, res) in enumerate(sorted_results, 1):
            logger.info(
                f"{rank:<6} {sel_name:<30} "
                f"{res['selection_info']['available_features']:<10} "
                f"{res['overall']['avg_mae']:<10.2f} "
                f"{res['overall']['avg_rmse']:<10.2f} "
                f"{res['overall']['avg_r2']:<10.3f}"
            )

        # Save comparison
        comparison_dir = Path(f"data/{area}/{spot_name}/models/phase2_comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)

        with open(comparison_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\n✓ Comparison saved to: {comparison_dir}/all_results.json")

    logger.info("\n✓ Phase 2 training complete!")


if __name__ == "__main__":
    main()

"""
Phase 2: By-Hour vs Overall Training Comparison

Compare two training strategies:
- Strategy A: Separate model per hour (current approach)
- Strategy B: Single model for all hours (hour as feature)

Usage:
    uv run python windmojo/train_phase2_by_hour_vs_overall.py \
        --config config/lake_como/dervio.yaml \
        --selection top_20 \
        --hours 08:00,13:00,14:00,15:00,16:00
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from windmojo.core.config import load_config
from windmojo.train.data_loader import load_training_data
from windmojo.train.phase1_model1 import Model1Trainer
from windmojo.train_phase2 import load_selection, parse_hours

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_by_hour(
    config: dict, selection_name: str, hours: List[str], target_column: str
) -> Dict[str, Any]:
    """Train separate model for each hour (Strategy A)."""
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY A: BY-HOUR TRAINING (Separate model per hour)")
    logger.info("=" * 80 + "\n")

    # Load selection
    spot_name = config["spot"]
    selection = load_selection(spot_name, selection_name)

    logger.info(f"Selection: {selection['name']}")
    logger.info(f"Features: {len(selection['features'])}")
    logger.info("")

    # Load training data
    logger.info("Loading training data...")
    from pathlib import Path

    import pandas as pd

    measurements_path = Path(
        config.get("measurement_file", config["data"]["measurements_file"])
    )
    features_path = Path(config["data"]["features_file"])

    features_df = pd.read_parquet(features_path)
    measurements_df = pd.read_csv(measurements_path)
    measurements_df["time"] = pd.to_datetime(
        measurements_df["timestamp"], format="%Y-%m-%d_%H%M"
    )
    measurements_df = measurements_df.drop(columns=["timestamp"])

    # Filter features
    available_features = set(features_df.columns)
    selected_features = [f for f in selection["features"] if f in available_features]
    features_df = features_df[["time"] + selected_features]

    results = {"strategy": "by_hour", "hours": {}, "overall": {}}

    all_maes = []
    all_rmses = []
    all_r2s = []

    for hour in hours:
        hour_key = hour.replace(":", "")
        logger.info(f"\nTraining model for {hour}...")

        # Filter to this hour
        hour_int = int(hour.split(":")[0])
        hour_mask = features_df["time"].dt.hour == hour_int
        hour_features = features_df[hour_mask].copy()

        # Join with measurements
        merged = hour_features.merge(
            measurements_df[["time", target_column]], on="time", how="inner"
        )

        if len(merged) == 0:
            logger.warning(f"  No data for {hour}, skipping")
            continue

        # Remove rows with NaN in target
        merged = merged.dropna(subset=[target_column])

        if len(merged) == 0:
            logger.warning(f"  No valid data for {hour} after removing NaN, skipping")
            continue

        # Train/test split (80/20)
        split_idx = int(len(merged) * 0.8)
        train_df = merged.iloc[:split_idx]
        test_df = merged.iloc[split_idx:]

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

        results["hours"][hour_key] = {
            "mae": results_dict["test_mae"],
            "rmse": results_dict["test_rmse"],
            "r2": results_dict["test_r2"],
            "max_error": 0,  # Not available from train() method
            "samples_train": len(train_subset),
            "samples_test": len(test_subset),
        }

        all_maes.append(results_dict["test_mae"])
        all_rmses.append(results_dict["test_rmse"])
        all_r2s.append(results_dict["test_r2"])

        logger.info(f"  MAE:  {results_dict['test_mae']:.2f} m/s")
        logger.info(f"  RMSE: {results_dict['test_rmse']:.2f} m/s")
        logger.info(f"  R²:   {results_dict['test_r2']:.3f}")

    # Overall results
    results["overall"] = {
        "avg_mae": np.mean(all_maes),
        "avg_rmse": np.mean(all_rmses),
        "avg_r2": np.mean(all_r2s),
        "std_mae": np.std(all_maes),
        "std_rmse": np.std(all_rmses),
        "std_r2": np.std(all_r2s),
    }

    logger.info("\n" + "-" * 80)
    logger.info("STRATEGY A RESULTS:")
    logger.info(
        f"  Average MAE:  {results['overall']['avg_mae']:.2f} ± {results['overall']['std_mae']:.2f} m/s"
    )
    logger.info(
        f"  Average RMSE: {results['overall']['avg_rmse']:.2f} ± {results['overall']['std_rmse']:.2f} m/s"
    )
    logger.info(
        f"  Average R²:   {results['overall']['avg_r2']:.3f} ± {results['overall']['std_r2']:.3f}"
    )

    return results


def train_overall(
    config: dict, selection_name: str, hours: List[str], target_column: str
) -> Dict[str, Any]:
    """Train single model for all hours (Strategy B)."""
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY B: OVERALL TRAINING (Single model for all hours)")
    logger.info("=" * 80 + "\n")

    # Load selection
    spot_name = config["spot"]
    selection = load_selection(spot_name, selection_name)

    logger.info(f"Selection: {selection['name']}")
    logger.info(f"Features: {len(selection['features'])}")
    logger.info("")

    # Load training data
    logger.info("Loading training data...")
    from pathlib import Path

    import pandas as pd

    measurements_path = Path(
        config.get("measurement_file", config["data"]["measurements_file"])
    )
    features_path = Path(config["data"]["features_file"])

    features_df = pd.read_parquet(features_path)
    measurements_df = pd.read_csv(measurements_path)
    measurements_df["time"] = pd.to_datetime(
        measurements_df["timestamp"], format="%Y-%m-%d_%H%M"
    )
    measurements_df = measurements_df.drop(columns=["timestamp"])

    # Filter features
    available_features = set(features_df.columns)
    selected_features = [f for f in selection["features"] if f in available_features]

    # Add hour features if not already present
    hour_features = ["hour", "hour_sin", "hour_cos"]
    for hf in hour_features:
        if hf in features_df.columns and hf not in selected_features:
            selected_features.append(hf)

    features_df = features_df[["time"] + selected_features]

    # Filter to target hours
    hour_ints = [int(h.split(":")[0]) for h in hours]
    hour_mask = features_df["time"].dt.hour.isin(hour_ints)
    features_df = features_df[hour_mask].copy()

    # Join with measurements
    merged = features_df.merge(
        measurements_df[["time", target_column]], on="time", how="inner"
    )

    # Remove rows with NaN in target
    merged = merged.dropna(subset=[target_column])

    logger.info(f"Total samples: {len(merged)}")
    logger.info(f"Features used: {len(selected_features)}")

    # Train/test split (80/20)
    split_idx = int(len(merged) * 0.8)
    train_df = merged.iloc[:split_idx]
    test_df = merged.iloc[split_idx:]

    # Prepare train/test data with 'target' column
    train_subset = train_df[selected_features + [target_column]].copy()
    train_subset = train_subset.rename(columns={target_column: "target"})
    test_subset = test_df[selected_features + [target_column]].copy()
    test_subset = test_subset.rename(columns={target_column: "target"})

    logger.info(f"Train samples: {len(train_subset)}")
    logger.info(f"Test samples: {len(test_subset)}")

    # Train single model
    logger.info("\nTraining overall model...")
    trainer = Model1Trainer()
    results_dict = trainer.train(
        train_data=train_subset,
        test_data=test_subset,
        feature_columns=selected_features,
    )

    # Get the trained model for predictions
    model = trainer.model

    # Calculate per-hour metrics on test set
    X_test = test_df[selected_features]
    y_pred = model.predict(X_test)

    test_df_with_pred = test_df.copy()
    test_df_with_pred["predicted"] = y_pred
    test_df_with_pred["hour"] = test_df_with_pred["time"].dt.hour

    results = {
        "strategy": "overall",
        "overall": {
            "mae": results_dict["test_mae"],
            "rmse": results_dict["test_rmse"],
            "r2": results_dict["test_r2"],
            "max_error": 0,  # Not available from train() method
            "samples_train": len(train_subset),
            "samples_test": len(test_subset),
        },
        "hours": {},
    }

    # Break down by hour
    for hour in hours:
        hour_key = hour.replace(":", "")
        hour_int = int(hour.split(":")[0])

        hour_test = test_df_with_pred[test_df_with_pred["hour"] == hour_int]

        if len(hour_test) == 0:
            continue

        y_true = hour_test[target_column]
        y_pred = hour_test["predicted"]

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        results["hours"][hour_key] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "samples": len(hour_test),
        }

        logger.info(f"\n  Hour {hour}:")
        logger.info(f"    MAE:  {mae:.2f} m/s")
        logger.info(f"    RMSE: {rmse:.2f} m/s")
        logger.info(f"    R²:   {r2:.3f}")

    logger.info("\n" + "-" * 80)
    logger.info("STRATEGY B RESULTS:")
    logger.info(f"  Overall MAE:  {results['overall']['mae']:.2f} m/s")
    logger.info(f"  Overall RMSE: {results['overall']['rmse']:.2f} m/s")
    logger.info(f"  Overall R²:   {results['overall']['r2']:.3f}")

    return results


def compare_strategies(results_a: dict, results_b: dict) -> dict:
    """Compare the two strategies."""
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: BY-HOUR vs OVERALL")
    logger.info("=" * 80 + "\n")

    comparison = {"overall": {}, "by_hour": {}}

    # Overall comparison
    mae_diff = results_b["overall"]["mae"] - results_a["overall"]["avg_mae"]
    rmse_diff = results_b["overall"]["rmse"] - results_a["overall"]["avg_rmse"]
    r2_diff = results_b["overall"]["r2"] - results_a["overall"]["avg_r2"]

    comparison["overall"] = {
        "by_hour_mae": results_a["overall"]["avg_mae"],
        "overall_mae": results_b["overall"]["mae"],
        "mae_difference": mae_diff,
        "mae_pct_change": 100 * mae_diff / results_a["overall"]["avg_mae"],
        "by_hour_rmse": results_a["overall"]["avg_rmse"],
        "overall_rmse": results_b["overall"]["rmse"],
        "rmse_difference": rmse_diff,
        "by_hour_r2": results_a["overall"]["avg_r2"],
        "overall_r2": results_b["overall"]["r2"],
        "r2_difference": r2_diff,
    }

    logger.info("OVERALL METRICS:")
    logger.info(f"  MAE:")
    logger.info(f"    By-Hour:  {results_a['overall']['avg_mae']:.2f} m/s")
    logger.info(f"    Overall:  {results_b['overall']['mae']:.2f} m/s")
    logger.info(
        f"    Difference: {mae_diff:+.2f} m/s ({comparison['overall']['mae_pct_change']:+.1f}%)"
    )
    logger.info(f"")
    logger.info(f"  RMSE:")
    logger.info(f"    By-Hour:  {results_a['overall']['avg_rmse']:.2f} m/s")
    logger.info(f"    Overall:  {results_b['overall']['rmse']:.2f} m/s")
    logger.info(f"    Difference: {rmse_diff:+.2f} m/s")
    logger.info(f"")
    logger.info(f"  R²:")
    logger.info(f"    By-Hour:  {results_a['overall']['avg_r2']:.3f}")
    logger.info(f"    Overall:  {results_b['overall']['r2']:.3f}")
    logger.info(f"    Difference: {r2_diff:+.3f}")

    # Per-hour comparison
    logger.info("\nPER-HOUR COMPARISON:")
    logger.info(
        f"{'Hour':<10} {'By-Hour MAE':<15} {'Overall MAE':<15} {'Difference':<15}"
    )
    logger.info("-" * 60)

    for hour_key in results_a["hours"].keys():
        if hour_key in results_b["hours"]:
            mae_a = results_a["hours"][hour_key]["mae"]
            mae_b = results_b["hours"][hour_key]["mae"]
            diff = mae_b - mae_a
            pct = 100 * diff / mae_a

            comparison["by_hour"][hour_key] = {
                "by_hour_mae": mae_a,
                "overall_mae": mae_b,
                "difference": diff,
                "pct_change": pct,
            }

            logger.info(
                f"{hour_key:<10} {mae_a:<15.2f} {mae_b:<15.2f} {diff:+.2f} ({pct:+.1f}%)"
            )

    # Winner
    if mae_diff < 0:
        winner = "OVERALL"
        logger.info(f"\n✓ WINNER: {winner} strategy (better MAE)")
    else:
        winner = "BY-HOUR"
        logger.info(f"\n✓ WINNER: {winner} strategy (better MAE)")

    comparison["winner"] = winner
    comparison["recommendation"] = (
        f"Use {winner} strategy for production"
        if abs(mae_diff) > 0.2
        else "Performance is similar - use BY-HOUR for specialization or OVERALL for simplicity"
    )

    logger.info(f"\n💡 RECOMMENDATION: {comparison['recommendation']}")

    return comparison


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare by-hour vs overall training strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument(
        "--selection", type=str, default="top_20", help="Feature selection"
    )
    parser.add_argument(
        "--hours", type=str, default="08:00,13:00,14:00,15:00,16:00", help="Hours"
    )
    parser.add_argument(
        "--target-column", type=str, default="dervio avgSpeed", help="Target column"
    )

    args = parser.parse_args()

    config = load_config(args.config)
    hours = parse_hours(args.hours)
    area = config["area"]
    spot_name = config["spot"]

    logger.info("=" * 80)
    logger.info("BY-HOUR vs OVERALL TRAINING COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Selection: {args.selection}")
    logger.info(f"Hours: {', '.join(hours)}")
    logger.info("")

    # Train both strategies
    results_by_hour = train_by_hour(config, args.selection, hours, args.target_column)
    results_overall = train_overall(config, args.selection, hours, args.target_column)

    # Compare
    comparison = compare_strategies(results_by_hour, results_overall)

    # Save results
    output_dir = Path(f"data/{area}/{spot_name}/models/phase2_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "selection": args.selection,
        "hours": hours,
        "by_hour": results_by_hour,
        "overall": results_overall,
        "comparison": comparison,
    }

    output_file = output_dir / f"by_hour_vs_overall_{args.selection}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()

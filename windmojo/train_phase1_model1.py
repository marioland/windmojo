"""
Phase 1: Model 1 Training CLI

Train Model 1 (feature discovery) with ALL 433 features.

Usage:
    # Pilot training (Jan-Feb, ~60 days)
    uv run python windmojo/train_phase1_model1.py \\
        --config config/lake_como/lecco.yaml \\
        --scope pilot \\
        --hours 08:00,13:00,14:00,15:00,16:00

    # Full training (all 268 days)
    uv run python windmojo/train_phase1_model1.py \\
        --config config/lake_como/lecco.yaml \\
        --scope full \\
        --hours 08:00,13:00,14:00,15:00,16:00
"""

import argparse
import logging
import sys
from pathlib import Path

from windmojo.core.config import get_output_directories, load_config
from windmojo.train.data_loader import (
    get_dataset_summary,
    get_feature_columns,
    load_training_data,
    prepare_datasets,
)
from windmojo.train.importance_extractor import (
    calculate_shap_values,
    categorize_features,
    combine_importance_rankings,
    extract_importance,
    save_importance,
)
from windmojo.train.phase1_model1 import Model1Trainer, train_model1
from windmojo.train.phase1_report import generate_phase1_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_hours(hours_str: str) -> list:
    """Parse comma-separated hours string."""
    return [h.strip() for h in hours_str.split(",")]


def main():
    """Main entry point for Phase 1 Model 1 training."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Model 1 Training - Feature Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--scope",
        type=str,
        choices=["pilot", "full"],
        default="pilot",
        help="Training scope: pilot (Jan-Feb) or full (all data)",
    )

    parser.add_argument(
        "--hours",
        type=str,
        default="08:00,13:00,14:00,15:00,16:00",
        help='Comma-separated list of hours to predict (e.g., "08:00,13:00,15:00")',
    )

    parser.add_argument(
        "--target-column",
        type=str,
        default="lecco avgSpeed",
        help="Target column name in measurements CSV",
    )

    parser.add_argument(
        "--pilot-weeks",
        type=int,
        default=8,
        help="Number of weeks for pilot study (default: 8)",
    )

    parser.add_argument(
        "--no-shap", action="store_true", help="Skip SHAP value calculation (faster)"
    )

    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=4,
        help="Number of parallel training jobs (default: 4, use 1 for sequential, -1 for all cores)",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}...")
        config = load_config(args.config)

        area = config["area"]
        spot = config["spot"]
        output_dirs = get_output_directories(config)

        # Parse target hours
        target_hours = parse_hours(args.hours)
        logger.info(f"Target hours: {target_hours}")

        # Define paths
        measurements_path = Path(config["measurement_file"])
        features_path = output_dirs["features_dir"] / f"{spot}_features_phase1.parquet"
        output_dir = (
            output_dirs["features_dir"].parent / "models" / "phase1_model1" / args.scope
        )

        logger.info(f"\nPaths:")
        logger.info(f"  Measurements: {measurements_path}")
        logger.info(f"  Features: {features_path}")
        logger.info(f"  Output: {output_dir}")

        # Check if files exist
        if not measurements_path.exists():
            raise FileNotFoundError(f"Measurements file not found: {measurements_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        # Load training data
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Loading Training Data")
        logger.info("=" * 70)

        data = load_training_data(
            measurements_path=measurements_path,
            features_path=features_path,
            target_column=args.target_column,
            target_hours=target_hours,
        )

        # Get feature columns
        feature_columns = get_feature_columns(data)
        logger.info(f"\n✓ {len(feature_columns)} features available for training")

        # Prepare datasets
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Preparing Datasets")
        logger.info("=" * 70)

        pilot_weeks = args.pilot_weeks if args.scope == "pilot" else None

        datasets = prepare_datasets(
            data=data,
            target_hours=target_hours,
            validation_strategy="weekly_walkforward",
            pilot_weeks=pilot_weeks,
        )

        # Print dataset summary
        for hour, folds in datasets.items():
            logger.info(f"\n{hour}: {len(folds)} folds created")
            first_fold = list(folds.values())[0]
            train_df, test_df = first_fold
            logger.info(
                f"  Example fold: Train {len(train_df)} samples, Test {len(test_df)} samples"
            )

        # Train Model 1
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Training Model 1 (Feature Discovery)")
        logger.info("=" * 70)

        results = train_model1(
            datasets=datasets,
            feature_columns=feature_columns,
            output_dir=output_dir,
            params=None,  # Use defaults
            n_jobs=args.parallel_jobs,
        )

        # Extract feature importance
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: Extracting Feature Importance")
        logger.info("=" * 70)

        for hour, folds in datasets.items():
            logger.info(f"\nExtracting importance for {hour}...")

            hour_dir = output_dir / hour.replace(":", "")

            # Load last trained model (most data)
            model_files = sorted(hour_dir.glob("model1_*.json"))
            if not model_files:
                logger.warning(f"  No models found for {hour}, skipping")
                continue

            last_model_path = model_files[-1]
            logger.info(f"  Loading model from {last_model_path}")

            trainer = Model1Trainer()
            trainer.load_model(last_model_path)

            # Extract Gain importance
            importance_df = extract_importance(
                model=trainer.model,
                feature_columns=feature_columns,
                importance_type="gain",
            )

            # Add categories
            importance_df = categorize_features(importance_df)

            # Calculate SHAP (optional)
            shap_data = None
            if not args.no_shap:
                # Use last fold's test data
                _, test_data = list(folds.values())[-1]
                shap_data = calculate_shap_values(
                    model=trainer.model,
                    data=test_data,
                    feature_columns=feature_columns,
                    max_samples=500,
                )

            # Combine rankings (works with or without SHAP)
            importance_df = combine_importance_rankings(
                gain_df=importance_df,
                shap_df=shap_data["shap_ranking"] if shap_data else None,
            )

            # Save importance
            importance_path = hour_dir / f"importance_{hour.replace(':', '')}.json"
            save_importance(
                importance_df=importance_df,
                output_path=importance_path,
                shap_data=shap_data,
            )

        # Generate report
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: Generating Report")
        logger.info("=" * 70)

        report_path = output_dir / f"phase1_model1_{args.scope}_report.md"
        results_path = output_dir / "phase1_model1_results.json"

        generate_phase1_report(
            results_path=results_path,
            importance_dir=output_dir,
            output_path=report_path,
        )

        # Final summary
        logger.info("=" * 70)
        logger.info("PHASE 1 MODEL 1 TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Scope: {args.scope}")
        logger.info(f"Hours trained: {', '.join(target_hours)}")
        logger.info(f"Output:")
        logger.info(f"  Models: {output_dir}/")
        logger.info(f"  Results: {results_path}")
        logger.info(f"  Report: {report_path}")
        logger.info("Next steps:")
        if args.scope == "pilot":
            logger.info("  1. Review pilot report to validate approach")
            logger.info("  2. If successful, run full training:")
            logger.info(
                f"     uv run python {sys.argv[0]} --config {args.config} --scope full --hours {args.hours}"
            )
        else:
            logger.info("  1. Review feature importance rankings")
            logger.info("  2. Select top N features for Phase 2 Model 2")
            logger.info("  3. Engineer interaction features based on insights")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

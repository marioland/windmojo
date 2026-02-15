"""
Train Model 2 with 24 Synoptic-Focused Features

Uses atomic feature groups with distance-based pressure center sorting.
Focus on European synoptic weather patterns (37.5% of features).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from windmojo.features.feature_selection import MODEL2_SYNOPTIC_24

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_features_and_target():
    """Load synoptic features and merge with wind measurements."""
    logger.info("Loading features and target data...")

    # Load synoptic features (exp8 file from Model 1b)
    features_file = project_root / "data/lake_como/dervio/features/dervio_features_synoptic_exp8.parquet"
    if not features_file.exists():
        raise FileNotFoundError(
            f"Features not found: {features_file}\n"
            "Run train_model1b_synoptic.py first to extract features."
        )

    features_df = pd.read_parquet(features_file)
    logger.info(f"Loaded synoptic features: {len(features_df)} days, {len(features_df.columns)} columns")

    # Load wind measurements
    measurements_file = project_root / "data/lake_como/dervio/measurements/wind_data.csv"
    if not measurements_file.exists():
        raise FileNotFoundError(
            f"Measurements not found: {measurements_file}\n"
            "Ensure wind measurements are available."
        )

    wind_df = pd.read_csv(measurements_file)
    # Parse timestamp from wind_data.csv format (e.g., '2025-01-16_1900')
    wind_df['timestamp'] = pd.to_datetime(wind_df['timestamp'], format='%Y-%m-%d_%H%M')
    wind_df['date'] = wind_df['timestamp'].dt.date.astype(str)
    wind_df['hour'] = wind_df['timestamp'].dt.hour

    # Rename target column to standard 'avgspeed'
    if 'dervio avgSpeed' in wind_df.columns:
        wind_df['avgspeed'] = wind_df['dervio avgSpeed']
    elif 'dervio speed' in wind_df.columns:
        wind_df['avgspeed'] = wind_df['dervio speed']

    logger.info(f"Loaded wind measurements: {len(wind_df):,} hourly samples")

    # Merge features with measurements
    merged_df = wind_df.merge(features_df, on='date', how='inner')
    logger.info(f"Merged data: {len(merged_df):,} samples from {merged_df['date'].nunique()} unique dates")

    # Rename timestamp to valid_time for consistency
    merged_df['valid_time'] = merged_df['timestamp']

    return merged_df


def prepare_data(df):
    """Add temporal features if not present."""
    logger.info("Preparing data...")

    # Ensure date column is datetime
    if 'date' in df.columns:
        df['valid_time'] = pd.to_datetime(df['date'])
    elif 'valid_time' not in df.columns:
        raise ValueError("No date/valid_time column found")
    else:
        df['valid_time'] = pd.to_datetime(df['valid_time'])

    # Add temporal features if not present
    if 'hour_sin' not in df.columns:
        df['hour'] = df['valid_time'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    if 'day_of_year_sin' not in df.columns:
        df['day_of_year'] = df['valid_time'].dt.dayofyear
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    logger.info(f"Prepared: {len(df)} samples")
    return df


def train_test_split(df, target_col='avgspeed'):
    """Split data: train on all except May 1-4."""
    df['month'] = df['valid_time'].dt.month
    df['day'] = df['valid_time'].dt.day

    # Test set: May 1-4
    test_mask = (df['month'] == 5) & (df['day'] >= 1) & (df['day'] <= 4)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    return train_df, test_df


def clean_data(df, feature_cols, target_col):
    """Remove rows with NaN/inf in target, keep NaN in features for XGBoost."""
    initial_len = len(df)

    valid_mask = ~(
        df[target_col].isna() |
        np.isinf(df[target_col].values) |
        np.isinf(df[feature_cols].values).any(axis=1)
    )

    df_clean = df[valid_mask].copy()
    dropped = initial_len - len(df_clean)

    if dropped > 0:
        logger.info(f"Dropped {dropped} rows ({100*dropped/initial_len:.1f}%) - target NaN/inf or feature inf")

    # Report NaN in features (XGBoost handles these)
    remaining_nan = df_clean[feature_cols].isna().sum().sum()
    if remaining_nan > 0:
        logger.info(f"Keeping {remaining_nan:,} NaN values in features (XGBoost handles NaN)")

    return df_clean


def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model."""
    logger.info("Training XGBoost model...")

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        enable_categorical=False
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model on train and test sets."""
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Metrics
    train_metrics = {
        'mae': mean_absolute_error(y_train, train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'r2': r2_score(y_train, train_pred)
    }

    test_metrics = {
        'mae': mean_absolute_error(y_test, test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'r2': r2_score(y_test, test_pred)
    }

    logger.info("=" * 70)
    logger.info("MODEL PERFORMANCE")
    logger.info("=" * 70)
    logger.info(f"Training ({len(y_train)} samples):")
    logger.info(f"  MAE:  {train_metrics['mae']:.3f} kt")
    logger.info(f"  RMSE: {train_metrics['rmse']:.3f} kt")
    logger.info(f"  R²:   {train_metrics['r2']:.3f}")
    logger.info("")
    logger.info(f"Test ({len(y_test)} samples, May 1-4):")
    logger.info(f"  MAE:  {test_metrics['mae']:.3f} kt")
    logger.info(f"  RMSE: {test_metrics['rmse']:.3f} kt")
    logger.info(f"  R²:   {test_metrics['r2']:.3f}")

    return train_metrics, test_metrics, train_pred, test_pred


def save_model_and_report(model, feature_cols, train_metrics, test_metrics, train_df, test_df, train_pred, test_pred):
    """Save model and generate report."""
    # Create output directories
    models_dir = project_root / "data/lake_como/dervio/models/model2_synoptic24"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # Save model
    model_file = models_dir / "model2_synoptic24.json"
    model.save_model(model_file)
    logger.info(f"Model saved: {model_file}")

    # Feature importance
    importance_dict = dict(zip(feature_cols, model.feature_importances_))
    importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Generate report
    report = []
    report.append("# Model 2 - Synoptic 24 Features")
    report.append("")
    report.append("**European Synoptic Weather Focus (37.5% of features)**")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## Feature Selection")
    report.append("")
    report.append("Using `MODEL2_SYNOPTIC_24` from `feature_selection.py`:")
    report.append("")
    report.append("- **Temporal:** 4 features (16.7%)")
    report.append("- **Pressure centers:** 8 features (33.3%)")
    report.append("- **Interactions:** 2 features (8.3%)")
    report.append("- **European synoptic:** 9 features (37.5%)")
    report.append("- **Local topographic:** 1 feature (4.2%)")
    report.append("")
    report.append("**Key improvement:** Distance-based pressure center sorting (`sort_by='distance'`)")
    report.append("- `h1` = nearest high (not strongest)")
    report.append("- `l1` = nearest low (not strongest)")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## Test Results")
    report.append("")
    report.append("**Overall Test Performance:**")
    report.append(f"- MAE: {test_metrics['mae']:.3f} kt")
    report.append(f"- RMSE: {test_metrics['rmse']:.3f} kt")
    report.append(f"- R²: {test_metrics['r2']:.3f}")
    report.append("")
    report.append("**Training Performance:**")
    report.append(f"- MAE: {train_metrics['mae']:.3f} kt")
    report.append(f"- RMSE: {train_metrics['rmse']:.3f} kt")
    report.append(f"- R²: {train_metrics['r2']:.3f}")
    report.append(f"- Samples: {len(train_df)}")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## Feature Importance (Top 24)")
    report.append("")
    report.append("| Rank | Feature | Importance | Category |")
    report.append("|------|---------|------------|----------|")

    # Categorize features - showing atomic groups
    temporal = ['hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos']
    high1_group = ['high1_bearing', 'high1_dist_km', 'high1_z500', 'high1_gradient']
    low1_group = ['low1_bearing', 'low1_dist_km', 'low1_z500', 'low1_gradient']
    h1_l1_interaction = ['h1_l1_distance', 'h1_l1_pressure_diff']
    synoptic = ['zonal_flow_strength', 'pressure_gradient_magnitude', 'pressure_gradient_direction',
                'nao_index', 'scand_index', 'temperature_gradient_ew', 'temperature_gradient_ns',
                'baroclinicity_index', 'mediterranean_pressure_msl_mean']
    local = ['alpine_thermal_stability']

    def get_category(feat):
        if feat in temporal:
            return 'Temporal'
        elif feat in high1_group:
            return 'High1 (Nearest High)'
        elif feat in low1_group:
            return 'Low1 (Nearest Low)'
        elif feat in h1_l1_interaction:
            return 'H1-L1 Interaction'
        elif feat in synoptic:
            return 'European Synoptic'
        elif feat in local:
            return 'Local'
        else:
            return 'Other'

    for i, (feat, imp) in enumerate(importance_sorted, 1):
        category = get_category(feat)
        report.append(f"| {i} | `{feat}` | {imp:.1f} | {category} |")

    report.append("")
    report.append("---")
    report.append("")
    report.append("## Selected Features (Atomic Groups)")
    report.append("")
    report.append("**Temporal (4 features):**")
    for feat in temporal:
        if feat in feature_cols:
            report.append(f"- `{feat}`")
    report.append("")
    report.append("**High1 - Nearest High (4 features):**")
    for feat in high1_group:
        if feat in feature_cols:
            report.append(f"- `{feat}`")
    report.append("")
    report.append("**Low1 - Nearest Low (4 features):**")
    for feat in low1_group:
        if feat in feature_cols:
            report.append(f"- `{feat}`")
    report.append("")
    report.append("**H1-L1 Interaction (2 features):**")
    for feat in h1_l1_interaction:
        if feat in feature_cols:
            report.append(f"- `{feat}`")
    report.append("")
    report.append("**European Synoptic (9 features):**")
    for feat in synoptic:
        if feat in feature_cols:
            report.append(f"- `{feat}`")
    report.append("")
    report.append("**Local (1 feature):**")
    for feat in local:
        if feat in feature_cols:
            report.append(f"- `{feat}`")

    report.append("")
    report.append("---")
    report.append("")
    report.append("## Comparison with Model 1b")
    report.append("")
    report.append("| Metric | Model 1b (167 features) | Model 2 (24 features) | Change |")
    report.append("|--------|-------------------------|----------------------|--------|")
    report.append("| Features | 167 | 24 | -85.6% |")
    report.append("| Test MAE | 5.500 kt | TBD | TBD |")
    report.append("| Test R² | -0.424 | TBD | TBD |")
    report.append("")
    report.append("*Model 1b results from `model1b_synoptic_report.md`*")

    # Save report
    report_file = reports_dir / "model2_synoptic24_report.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    logger.info(f"Report saved: {report_file}")

    # Save predictions
    pred_df = pd.DataFrame({
        'valid_time': pd.concat([train_df['valid_time'], test_df['valid_time']]),
        'actual': pd.concat([train_df['avgspeed'], test_df['avgspeed']]),
        'predicted': np.concatenate([train_pred, test_pred]),
        'dataset': ['train'] * len(train_df) + ['test'] * len(test_df)
    })
    pred_file = models_dir / "predictions.parquet"
    pred_df.to_parquet(pred_file, index=False)
    logger.info(f"Predictions saved: {pred_file}")

    # Generate diagram
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import timedelta

        logger.info("Generating prediction diagram...")

        # Sort by time
        pred_df_sorted = pred_df.sort_values('valid_time').copy()
        train_pred_df = pred_df_sorted[pred_df_sorted['dataset'] == 'train']
        test_pred_df = pred_df_sorted[pred_df_sorted['dataset'] == 'test']

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot training
        ax.plot(train_pred_df['valid_time'], train_pred_df['actual'],
                'o', color='gray', alpha=0.3, markersize=3, label='Training (actual)')
        ax.plot(train_pred_df['valid_time'], train_pred_df['predicted'],
                '-', color='blue', alpha=0.5, linewidth=1, label='Training (predicted)')

        # Plot test
        if len(test_pred_df) > 0:
            ax.plot(test_pred_df['valid_time'], test_pred_df['actual'],
                    'o', color='red', markersize=5, label='Test (actual)')
            ax.plot(test_pred_df['valid_time'], test_pred_df['predicted'],
                    's', color='orange', markersize=4, label='Test (predicted)')

        # Add day separator lines
        all_times = pred_df_sorted['valid_time']
        if len(all_times) > 0:
            start_date = all_times.min().floor('D')
            end_date = all_times.max().ceil('D')
            current_date = start_date
            while current_date <= end_date:
                ax.axvline(current_date, color='black', linestyle='--', alpha=0.2, linewidth=0.5)
                current_date += timedelta(days=1)

        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Wind Speed (kt)', fontsize=12)
        ax.set_title('Model 2 - Synoptic 24 Features (Predictions)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        diagram_file = reports_dir / "model2_synoptic24_predictions.png"
        plt.savefig(diagram_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Diagram saved: {diagram_file}")
    except Exception as e:
        logger.warning(f"Failed to generate diagram: {e}")


def main():
    """Main training workflow."""
    logger.info("=" * 70)
    logger.info("MODEL 2: SYNOPTIC 24 FEATURES")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Configuration:")
    logger.info("  - Features: 24 (from MODEL2_SYNOPTIC_24)")
    logger.info("  - European synoptic focus: 37.5%")
    logger.info("  - Atomic feature groups: Yes")
    logger.info("  - Distance-based sorting: Yes (h1=nearest)")
    logger.info("")

    # Load data
    df = load_features_and_target()

    # Prepare
    merged_df = prepare_data(df)

    # Check if all features are available
    missing_features = [f for f in MODEL2_SYNOPTIC_24 if f not in merged_df.columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        logger.error("You need to retrain Model 1b with distance-based sorting first!")
        logger.error("Modify config/lake_como/dervio.yaml:")
        logger.error("  synoptic_features:")
        logger.error("    pressure_centers:")
        logger.error("      sort_by: 'distance'")
        logger.error("Then run: uv run python scripts/train_model1b_synoptic.py")
        sys.exit(1)

    # Split
    train_df, test_df = train_test_split(merged_df)

    # Clean
    train_df_clean = clean_data(train_df, MODEL2_SYNOPTIC_24, 'avgspeed')
    test_df_clean = clean_data(test_df, MODEL2_SYNOPTIC_24, 'avgspeed')

    logger.info(f"Clean train: {len(train_df_clean)} samples")
    logger.info(f"Clean test: {len(test_df_clean)} samples")

    # Prepare features and target
    X_train = train_df_clean[MODEL2_SYNOPTIC_24]
    y_train = train_df_clean['avgspeed']
    X_test = test_df_clean[MODEL2_SYNOPTIC_24]
    y_test = test_df_clean['avgspeed']

    logger.info("")
    logger.info("Selected features:")
    for i, feat in enumerate(MODEL2_SYNOPTIC_24, 1):
        logger.info(f"  {i:2d}. {feat}")
    logger.info("")

    # Train
    model = train_model(X_train, y_train, X_test, y_test)

    # Evaluate
    train_metrics, test_metrics, train_pred, test_pred = evaluate_model(
        model, X_train, y_train, X_test, y_test
    )

    # Save
    save_model_and_report(
        model, MODEL2_SYNOPTIC_24, train_metrics, test_metrics,
        train_df_clean, test_df_clean, train_pred, test_pred
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

"""
Compare Pressure Center Sorting Methods

Trains Model 1b with two different pressure center sorting strategies:
1. Strength-based: h1=strongest high, l1=strongest low (current default)
2. Distance-based: h1=nearest high, l1=nearest low (new approach)

Compares:
- Feature importance rankings
- Model performance
- Consistency of selected features across time periods
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from windmojo.features.synoptic_pipeline import SynopticFeaturePipeline
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(sort_by: str = 'strength') -> pd.DataFrame:
    """
    Load and engineer features with specified sorting method.

    Args:
        sort_by: 'strength' or 'distance'

    Returns:
        DataFrame with features
    """
    logger.info(f"Loading training data with sort_by='{sort_by}'")

    # Load synoptic data
    data_dir = project_root / "data" / "v2_synoptic"
    synoptic_file = data_dir / "synoptic_data.parquet"

    if not synoptic_file.exists():
        raise FileNotFoundError(f"Synoptic data not found: {synoptic_file}")

    df = pd.read_parquet(synoptic_file)
    logger.info(f"Loaded {len(df)} raw observations")

    # Configure pipeline with sorting method
    pipeline_config = {
        'center': {
            'latitude': 46.083333,
            'longitude': 9.3
        },
        'synoptic_features': {
            'pressure_centers': {
                'n_highs': 3,
                'n_lows': 3,
                'smoothing_sigma': 2.0,
                'min_distance': 5,
                'sort_by': sort_by  # KEY DIFFERENCE
            }
        }
    }

    # Initialize pipeline
    pipeline = SynopticFeaturePipeline(config=pipeline_config)

    # Transform synoptic data
    logger.info("Transforming synoptic data to features...")
    feature_rows = []

    for idx, (timestamp, group) in enumerate(df.groupby('valid_time')):
        if idx % 100 == 0:
            logger.info(f"Processing {idx}/{df['valid_time'].nunique()} timestamps...")

        try:
            features = pipeline.transform(group)
            features['valid_time'] = timestamp
            feature_rows.append(features)
        except Exception as e:
            logger.warning(f"Failed to process {timestamp}: {e}")
            continue

    features_df = pd.concat(feature_rows, ignore_index=True)
    logger.info(f"Generated {len(features_df)} feature vectors with {len(features_df.columns)} features")

    return features_df


def train_model(
    features_df: pd.DataFrame,
    target_df: pd.DataFrame,
    sort_by: str
) -> Dict:
    """
    Train XGBoost model and return metrics + feature importance.

    Args:
        features_df: Features DataFrame
        target_df: Target DataFrame
        sort_by: Sorting method used

    Returns:
        Dict with metrics and feature importance
    """
    logger.info(f"Training model with sort_by='{sort_by}'")

    # Merge features with target
    merged = features_df.merge(target_df, on='valid_time', how='inner')
    logger.info(f"Merged dataset: {len(merged)} samples")

    # Add temporal features
    merged['valid_time'] = pd.to_datetime(merged['valid_time'])
    merged['hour'] = merged['valid_time'].dt.hour
    merged['day_of_year'] = merged['valid_time'].dt.dayofyear

    merged['hour_sin'] = np.sin(2 * np.pi * merged['hour'] / 24)
    merged['hour_cos'] = np.cos(2 * np.pi * merged['hour'] / 24)
    merged['day_of_year_sin'] = np.sin(2 * np.pi * merged['day_of_year'] / 365.25)
    merged['day_of_year_cos'] = np.cos(2 * np.pi * merged['day_of_year'] / 365.25)

    # Define feature and target columns
    exclude_cols = ['valid_time', 'hour', 'day_of_year', 'station', 'avgspeed']
    feature_cols = [c for c in merged.columns if c not in exclude_cols]
    target_col = 'avgspeed'

    # Split train/test (May 1-4 as test)
    merged['month'] = merged['valid_time'].dt.month
    merged['day'] = merged['valid_time'].dt.day

    test_mask = (
        (merged['month'] == 5) & (merged['day'] >= 1) & (merged['day'] <= 4)
    )

    train_df = merged[~test_mask].copy()
    test_df = merged[test_mask].copy()

    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    # Clean data (drop target NaN/inf only)
    valid_train = ~(
        train_df[target_col].isna() |
        np.isinf(train_df[target_col].values) |
        np.isinf(train_df[feature_cols].values).any(axis=1)
    )
    train_df_clean = train_df[valid_train].copy()

    valid_test = ~(
        test_df[target_col].isna() |
        np.isinf(test_df[target_col].values) |
        np.isinf(test_df[feature_cols].values).any(axis=1)
    )
    test_df_clean = test_df[valid_test].copy()

    logger.info(f"Clean training samples: {len(train_df_clean)}")
    logger.info(f"Clean test samples: {len(test_df_clean)}")

    # Train XGBoost
    X_train = train_df_clean[feature_cols]
    y_train = train_df_clean[target_col]
    X_test = test_df_clean[feature_cols]
    y_test = test_df_clean[target_col]

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist'
    )

    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    # Feature importance
    importance_dict = dict(zip(feature_cols, model.feature_importances_))

    # Sort by importance
    importance_sorted = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = {
        'sort_by': sort_by,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'feature_importance': importance_sorted[:50],  # Top 50
    }

    logger.info(f"Results for sort_by='{sort_by}':")
    logger.info(f"  Train: MAE={train_mae:.3f}, RMSE={train_rmse:.3f}, R²={train_r2:.3f}")
    logger.info(f"  Test: MAE={test_mae:.3f}, RMSE={test_rmse:.3f}, R²={test_r2:.3f}")

    return results


def compare_feature_rankings(
    results_strength: Dict,
    results_distance: Dict
) -> pd.DataFrame:
    """
    Compare top feature rankings between sorting methods.

    Args:
        results_strength: Results with strength-based sorting
        results_distance: Results with distance-based sorting

    Returns:
        DataFrame comparing rankings
    """
    # Extract top 20 from each
    strength_top20 = {f: i for i, (f, _) in enumerate(results_strength['feature_importance'][:20], 1)}
    distance_top20 = {f: i for i, (f, _) in enumerate(results_distance['feature_importance'][:20], 1)}

    # All features in either top 20
    all_features = set(strength_top20.keys()) | set(distance_top20.keys())

    comparison = []
    for feat in all_features:
        comparison.append({
            'feature': feat,
            'rank_strength': strength_top20.get(feat, '>20'),
            'rank_distance': distance_top20.get(feat, '>20'),
            'importance_strength': dict(results_strength['feature_importance']).get(feat, 0),
            'importance_distance': dict(results_distance['feature_importance']).get(feat, 0),
        })

    df = pd.DataFrame(comparison)

    # Sort by distance ranking (new method)
    df['rank_distance_num'] = df['rank_distance'].apply(lambda x: 999 if x == '>20' else x)
    df = df.sort_values('rank_distance_num').drop('rank_distance_num', axis=1)

    return df


def main():
    """Main comparison workflow."""
    logger.info("=" * 80)
    logger.info("Comparing Pressure Center Sorting Methods")
    logger.info("=" * 80)

    # Load target data
    target_file = project_root / "data" / "v2_synoptic" / "target_dervio_gera.parquet"
    if not target_file.exists():
        raise FileNotFoundError(f"Target data not found: {target_file}")

    target_df = pd.read_parquet(target_file)
    logger.info(f"Loaded {len(target_df)} target observations")

    # 1. Train with strength-based sorting (current default)
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 1: Strength-based sorting (h1=strongest)")
    logger.info("=" * 80)
    features_strength = load_training_data(sort_by='strength')
    results_strength = train_model(features_strength, target_df, sort_by='strength')

    # 2. Train with distance-based sorting (new approach)
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 2: Distance-based sorting (h1=nearest)")
    logger.info("=" * 80)
    features_distance = load_training_data(sort_by='distance')
    results_distance = train_model(features_distance, target_df, sort_by='distance')

    # 3. Compare results
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON")
    logger.info("=" * 80)

    logger.info("\nPerformance Comparison:")
    logger.info(f"                    Strength-based    Distance-based")
    logger.info(f"Train MAE:          {results_strength['train_mae']:6.3f}           {results_distance['train_mae']:6.3f}")
    logger.info(f"Train R²:           {results_strength['train_r2']:6.3f}           {results_distance['train_r2']:6.3f}")
    logger.info(f"Test MAE:           {results_strength['test_mae']:6.3f}           {results_distance['test_mae']:6.3f}")
    logger.info(f"Test R²:            {results_strength['test_r2']:6.3f}           {results_distance['test_r2']:6.3f}")

    # Feature ranking comparison
    comparison_df = compare_feature_rankings(results_strength, results_distance)

    logger.info("\nTop 20 Feature Ranking Comparison:")
    logger.info(comparison_df.to_string(index=False))

    # Save detailed comparison
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)

    comparison_file = output_dir / "sorting_method_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"\nDetailed comparison saved to: {comparison_file}")

    # Save results JSON
    results_file = output_dir / "sorting_method_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'strength': {k: v for k, v in results_strength.items() if k != 'feature_importance'},
            'distance': {k: v for k, v in results_distance.items() if k != 'feature_importance'},
        }, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

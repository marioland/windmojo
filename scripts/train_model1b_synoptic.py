"""
Train Model 1b - Synoptic Features Only (Experiment 8)

Full pipeline:
1. Extract synoptic features for all May-Sep 2025 days
2. Merge with wind measurements
3. Train XGBoost with test sets (May 1-4, June 1-4, July 1-4)
4. Analyze feature importance, correlations, duplicates
5. Generate visualizations and markdown report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import yaml
import json

# ML imports
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Windmojo imports
from windmojo.features.synoptic_pipeline import SynopticFeaturePipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Model1bTrainer:
    """Train Model 1b with synoptic features only."""

    def __init__(self, config_path: str):
        """Initialize trainer with config."""
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Paths
        self.synoptic_dir = Path("data/lake_como/dervio/raw/synoptic")
        self.measurements_file = Path(self.config['measurement_file'])
        self.features_dir = Path("data/lake_como/dervio/features")
        self.models_dir = Path("data/lake_como/dervio/models/experiment8_model1b")
        self.reports_dir = Path("reports")

        # Create directories
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize feature pipeline
        self.feature_pipeline = SynopticFeaturePipeline(self.config)

        logger.info(f"Model 1b Trainer initialized")
        logger.info(f"Config: {config_path}")
        logger.info(f"Output: {self.models_dir}")

    def extract_all_features(self, start_date: str, end_date: str):
        """Extract synoptic features for all days in range."""
        logger.info("="*70)
        logger.info("STEP 1: EXTRACTING SYNOPTIC FEATURES")
        logger.info("="*70)

        # Generate date range
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        logger.info(f"Extracting features for {len(dates)} days ({dates[0]} to {dates[-1]})")

        # Extract features for each day
        all_features = []
        failed_dates = []

        for i, date_str in enumerate(dates, 1):
            try:
                # Load synoptic grid
                date_compact = date_str.replace('-', '')
                grid_file = self.synoptic_dir / f"synoptic_grid_{date_compact}.parquet"

                if not grid_file.exists():
                    logger.warning(f"Missing grid file: {grid_file.name}")
                    failed_dates.append(date_str)
                    continue

                grid_df = pd.read_parquet(grid_file)

                # Extract features
                features = self.feature_pipeline.transform(grid_df)
                features['date'] = date_str

                all_features.append(features)

                if i % 20 == 0:
                    logger.info(f"Progress: {i}/{len(dates)} days processed")

            except Exception as e:
                logger.error(f"Failed to extract features for {date_str}: {e}")
                failed_dates.append(date_str)

        # Combine all features
        if not all_features:
            raise ValueError("No features extracted!")

        features_df = pd.concat(all_features, ignore_index=True)

        logger.info(f"Features extracted: {len(features_df)} days")
        logger.info(f"Feature columns: {len(features_df.columns) - 1}")  # -1 for date column
        logger.info(f"Failed dates: {len(failed_dates)}")

        # Save features
        features_file = self.features_dir / "dervio_features_synoptic_exp8.parquet"
        features_df.to_parquet(features_file, index=False)
        logger.info(f"Features saved: {features_file}")

        return features_df, failed_dates

    def load_and_merge_data(self, features_df: pd.DataFrame):
        """Load wind measurements and merge with synoptic features."""
        logger.info("="*70)
        logger.info("STEP 2: LOADING & MERGING DATA")
        logger.info("="*70)

        # Load wind measurements
        logger.info(f"Loading measurements: {self.measurements_file}")
        wind_df = pd.read_csv(self.measurements_file)
        wind_df['timestamp'] = pd.to_datetime(wind_df['timestamp'], format='%Y-%m-%d_%H%M')

        # Extract date and hour
        wind_df['date'] = wind_df['timestamp'].dt.date.astype(str)
        wind_df['hour'] = wind_df['timestamp'].dt.hour

        # Add temporal features (sin/cos encoding for cyclical patterns)
        # Hour of day (0-23) - captures diurnal cycle
        wind_df['hour_sin'] = np.sin(2 * np.pi * wind_df['hour'] / 24)
        wind_df['hour_cos'] = np.cos(2 * np.pi * wind_df['hour'] / 24)

        # Day of year (1-365) - captures seasonal cycle
        wind_df['day_of_year'] = wind_df['timestamp'].dt.dayofyear
        wind_df['day_of_year_sin'] = np.sin(2 * np.pi * wind_df['day_of_year'] / 365.25)
        wind_df['day_of_year_cos'] = np.cos(2 * np.pi * wind_df['day_of_year'] / 365.25)

        logger.info(f"Wind measurements: {len(wind_df):,} rows (all 24 hours)")
        logger.info(f"Date range: {wind_df['timestamp'].min()} to {wind_df['timestamp'].max()}")
        logger.info(f"Hours range: {wind_df['hour'].min()}-{wind_df['hour'].max()}")
        logger.info(f"Temporal features: hour_sin, hour_cos, day_of_year_sin, day_of_year_cos")

        # Merge
        logger.info("Merging features with measurements...")
        merged_df = wind_df.merge(features_df, on='date', how='inner')

        logger.info(f"Merged data: {len(merged_df):,} rows")
        logger.info(f"Unique dates: {merged_df['date'].nunique()}")

        # Select target and features
        target_col = self.config['target_column']

        # Get synoptic feature names from features_df (before merge with wind data)
        synoptic_cols = [col for col in features_df.columns if col != 'date']

        # Add temporal features (sin/cos encodings)
        temporal_cols = ['hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos']

        # Combine synoptic + temporal features (positive filtering)
        feature_cols = [col for col in synoptic_cols if col in merged_df.columns]
        feature_cols.extend([col for col in temporal_cols if col in merged_df.columns])

        # Validation: Ensure target is NOT in features
        if target_col in feature_cols:
            logger.error(f"CRITICAL ERROR: Target '{target_col}' found in features!")
            raise ValueError(f"Data leakage: target column '{target_col}' is in feature set")

        # Validation: Ensure no local measurements leaked in
        # More specific patterns to avoid false positives (e.g., "pressure_gradient_direction" is OK)
        local_patterns = [
            'dervio',        # Dervio station
            'gera',          # Gera Lario station
            'lecco',         # Lecco station
            '_direction',    # Wind direction (but not "gradient_direction")
            '_speed',        # Wind speed (but not in compound words)
            'avgspeed',      # Average speed
            '_score',        # Wind score
            'purescore'      # Pure score
        ]
        leaked_features = [
            col for col in feature_cols
            if any(pattern in col.lower() for pattern in local_patterns)
            and not any(ok_pattern in col.lower() for ok_pattern in ['gradient_direction'])
        ]
        if leaked_features:
            logger.error(f"CRITICAL ERROR: Local measurements in features: {leaked_features}")
            raise ValueError(f"Data leakage: local measurements found in features: {leaked_features}")

        logger.info(f"Target: {target_col}")
        logger.info(f"Synoptic features: {len(synoptic_cols)}")
        logger.info(f"Temporal features: {len(temporal_cols)}")
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info(f"First 10 features: {feature_cols[:10]}")
        logger.info(f"Last 10 features: {feature_cols[-10:]}")

        return merged_df, target_col, feature_cols

    def create_train_test_split(self, df: pd.DataFrame):
        """Create train/test split with specific test dates."""
        logger.info("="*70)
        logger.info("STEP 3: TRAIN/TEST SPLIT")
        logger.info("="*70)

        # Test dates: May 1-4, June 1-4, July 1-4
        test_dates = (
            [f'2025-05-{d:02d}' for d in range(1, 5)] +
            [f'2025-06-{d:02d}' for d in range(1, 5)] +
            [f'2025-07-{d:02d}' for d in range(1, 5)]
        )

        logger.info(f"Test dates ({len(test_dates)} days):")
        for date in test_dates:
            logger.info(f"  {date}")

        # Split
        test_df = df[df['date'].isin(test_dates)].copy()
        train_df = df[~df['date'].isin(test_dates)].copy()

        logger.info(f"Training set: {len(train_df):,} rows ({train_df['date'].nunique()} days)")
        logger.info(f"Test set: {len(test_df):,} rows ({test_df['date'].nunique()} days)")

        return train_df, test_df, test_dates

    def train_model(self, train_df: pd.DataFrame, feature_cols: list, target_col: str):
        """Train XGBoost model."""
        logger.info("="*70)
        logger.info("STEP 4: TRAINING MODEL")
        logger.info("="*70)

        # Check for NaN/inf in features and target
        logger.info("Checking data quality...")

        # Check features
        feature_data = train_df[feature_cols]
        nan_counts = feature_data.isna().sum()
        inf_counts = np.isinf(feature_data.values).sum(axis=0)

        if nan_counts.sum() > 0:
            logger.warning(f"Found {nan_counts.sum()} NaN values in features")
            logger.warning(f"Features with NaN: {nan_counts[nan_counts > 0].to_dict()}")

        if inf_counts.sum() > 0:
            logger.warning(f"Found {inf_counts.sum()} infinity values in features")

        # Check target
        target_nan = train_df[target_col].isna().sum()
        target_inf = np.isinf(train_df[target_col].values).sum()

        if target_nan > 0:
            logger.warning(f"Found {target_nan} NaN values in target '{target_col}'")
        if target_inf > 0:
            logger.warning(f"Found {target_inf} infinity values in target '{target_col}'")

        # Clean data: XGBoost can handle NaN in features, but not in target
        # Only drop rows where target is NaN or inf, or where features have inf
        logger.info("Cleaning data (dropping rows with bad target or inf in features)...")
        initial_len = len(train_df)

        # Create mask for valid rows (no NaN/inf in target, no inf in features)
        # XGBoost handles NaN in features natively, so we keep those rows
        valid_mask = ~(
            train_df[target_col].isna() |
            np.isinf(train_df[target_col].values) |
            np.isinf(train_df[feature_cols].values).any(axis=1)
        )

        train_df_clean = train_df[valid_mask].copy()
        dropped = initial_len - len(train_df_clean)

        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows (target NaN/inf or feature inf) ({dropped/initial_len*100:.1f}%)")

        # Report how many NaN values remain in features (XGBoost will handle them)
        remaining_nan = train_df_clean[feature_cols].isna().sum().sum()
        if remaining_nan > 0:
            logger.info(f"Keeping {remaining_nan:,} NaN values in features (XGBoost handles NaN natively)")

        logger.info(f"Clean training data: {len(train_df_clean):,} rows")

        # Prepare data
        X_train = train_df_clean[feature_cols].values
        y_train = train_df_clean[target_col].values

        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")

        # Model parameters from config
        params = self.config['training']['model']
        logger.info(f"Model parameters: {params}")

        # Train model
        logger.info("Training XGBoost...")
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train,
                 eval_set=[(X_train, y_train)],
                 verbose=False)

        # Training metrics
        y_train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        logger.info(f"Training metrics:")
        logger.info(f"  MAE: {train_mae:.3f}")
        logger.info(f"  RMSE: {train_rmse:.3f}")
        logger.info(f"  R²: {train_r2:.3f}")

        # Save model
        model_file = self.models_dir / "model.json"
        model.save_model(str(model_file))
        logger.info(f"Model saved: {model_file}")

        return model

    def evaluate_model(self, model, test_df: pd.DataFrame, feature_cols: list, target_col: str, test_dates: list):
        """Evaluate model on test set."""
        logger.info("="*70)
        logger.info("STEP 5: MODEL EVALUATION")
        logger.info("="*70)

        # Clean test data (same as training)
        logger.info("Checking test data quality...")
        initial_len = len(test_df)

        # Check for NaN/inf
        target_nan = test_df[target_col].isna().sum()
        if target_nan > 0:
            logger.warning(f"Found {target_nan} NaN values in test target")

        # Create mask for valid rows
        valid_mask = ~(
            test_df[feature_cols].isna().any(axis=1) |
            test_df[target_col].isna() |
            np.isinf(test_df[feature_cols].values).any(axis=1) |
            np.isinf(test_df[target_col].values)
        )

        test_df_clean = test_df[valid_mask].copy()
        dropped = initial_len - len(test_df_clean)

        if dropped > 0:
            logger.warning(f"Dropped {dropped} test rows with NaN/inf ({dropped/initial_len*100:.1f}%)")

        logger.info(f"Clean test data: {len(test_df_clean):,} rows")

        # Prepare test data
        X_test = test_df_clean[feature_cols].values
        y_test = test_df_clean[target_col].values

        # Predictions
        y_pred = model.predict(X_test)
        test_df_clean['predicted'] = y_pred

        # Overall metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Test metrics (overall):")
        logger.info(f"  MAE: {mae:.3f}")
        logger.info(f"  RMSE: {rmse:.3f}")
        logger.info(f"  R²: {r2:.3f}")

        # Per-period metrics
        results = {'overall': {'mae': mae, 'rmse': rmse, 'r2': r2}}

        for period_name, period_dates in [
            ('May 1-4', [f'2025-05-{d:02d}' for d in range(1, 5)]),
            ('June 1-4', [f'2025-06-{d:02d}' for d in range(1, 5)]),
            ('July 1-4', [f'2025-07-{d:02d}' for d in range(1, 5)])
        ]:
            period_df = test_df_clean[test_df_clean['date'].isin(period_dates)]
            if len(period_df) > 0:
                y_true_period = period_df[target_col].values
                y_pred_period = period_df['predicted'].values

                period_mae = mean_absolute_error(y_true_period, y_pred_period)
                period_rmse = np.sqrt(mean_squared_error(y_true_period, y_pred_period))
                period_r2 = r2_score(y_true_period, y_pred_period)

                results[period_name] = {
                    'mae': period_mae,
                    'rmse': period_rmse,
                    'r2': period_r2,
                    'n_samples': len(period_df)
                }

                logger.info(f"{period_name}: MAE={period_mae:.3f}, RMSE={period_rmse:.3f}, R²={period_r2:.3f}, N={len(period_df)}")

        # Save results
        results_file = self.models_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {results_file}")

        return test_df_clean, results

    def analyze_features(self, model, train_df: pd.DataFrame, feature_cols: list):
        """Analyze feature importance and correlations."""
        logger.info("="*70)
        logger.info("STEP 6: FEATURE ANALYSIS")
        logger.info("="*70)

        # Feature importance (gain-based)
        importance_gain = model.get_booster().get_score(importance_type='gain')

        # Map to feature names
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': [importance_gain.get(f'f{i}', 0) for i in range(len(feature_cols))]
        }).sort_values('importance', ascending=False)

        logger.info(f"Top 20 features by importance:")
        for idx, row in feature_importance.head(20).iterrows():
            logger.info(f"  {row['feature']:40s} = {row['importance']:.1f}")

        # SHAP values (sample 500 rows)
        logger.info("Calculating SHAP values...")
        sample_size = min(500, len(train_df))
        sample_idx = np.random.choice(len(train_df), sample_size, replace=False)
        X_sample = train_df[feature_cols].iloc[sample_idx]

        try:
            # Try using model.get_booster() for better compatibility
            explainer = shap.TreeExplainer(model.get_booster())
            shap_values = explainer.shap_values(X_sample)
        except Exception as e:
            logger.warning(f"SHAP calculation with booster failed: {e}")
            logger.warning("Skipping SHAP values, using gain importance only")
            # Use gain importance as proxy for SHAP importance
            shap_importance = feature_importance.copy()
            shap_importance = shap_importance.rename(columns={'importance': 'shap_importance'})
            feature_analysis = shap_importance
            corr_matrix = None
            high_corr_df = None
            return feature_analysis, high_corr_df, corr_matrix

        shap_importance = pd.DataFrame({
            'feature': feature_cols,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)

        # Merge importance metrics
        feature_analysis = feature_importance.merge(shap_importance, on='feature')

        # Correlation analysis
        logger.info("Analyzing feature correlations...")
        X_train = train_df[feature_cols]
        corr_matrix = X_train.corr()

        # Find high correlations (>0.90)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.90:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False, key=abs)

        logger.info(f"High correlation pairs (>0.90): {len(high_corr_df)}")
        for idx, row in high_corr_df.head(10).iterrows():
            logger.info(f"  {row['feature1']} <-> {row['feature2']}: {row['correlation']:.3f}")

        # Save analysis
        feature_analysis.to_csv(self.models_dir / "feature_importance.csv", index=False)
        high_corr_df.to_csv(self.models_dir / "high_correlations.csv", index=False)
        corr_matrix.to_csv(self.models_dir / "correlation_matrix.csv")

        return feature_analysis, high_corr_df, corr_matrix

    def create_visualizations(self, test_df: pd.DataFrame, target_col: str,
                            feature_analysis: pd.DataFrame, corr_matrix: pd.DataFrame):
        """Create all visualizations."""
        logger.info("="*70)
        logger.info("STEP 7: CREATING VISUALIZATIONS")
        logger.info("="*70)

        # 1. Prediction plots for each test period
        for period_name, period_dates in [
            ('may', [f'2025-05-{d:02d}' for d in range(1, 5)]),
            ('june', [f'2025-06-{d:02d}' for d in range(1, 5)]),
            ('july', [f'2025-07-{d:02d}' for d in range(1, 5)])
        ]:
            period_df = test_df[test_df['date'].isin(period_dates)].copy()
            if len(period_df) == 0:
                continue

            period_df = period_df.sort_values('timestamp')

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(period_df['timestamp'], period_df[target_col],
                   'o-', label='Measured', linewidth=2, markersize=4)
            ax.plot(period_df['timestamp'], period_df['predicted'],
                   's-', label='Predicted', linewidth=2, markersize=4, alpha=0.7)

            # Add vertical lines to separate days (at midnight)
            unique_dates = period_df['date'].unique()
            for i, date in enumerate(unique_dates[1:]):  # Skip first date
                # Find the first timestamp of this date (start of day)
                day_start = period_df[period_df['date'] == date]['timestamp'].min()
                ax.axvline(x=day_start, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Wind Speed (kt)', fontsize=12)
            ax.set_title(f'Model 1b - {period_name.capitalize()} 1-4 Predictions',
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            fig.savefig(self.reports_dir / f"model1b_{period_name}_predictions.png", dpi=300)
            plt.close()
            logger.info(f"Saved: model1b_{period_name}_predictions.png")

        # 2. Feature importance (top 20)
        top20 = feature_analysis.head(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        # Use shap_importance if importance column doesn't exist (SHAP fallback case)
        importance_col = 'importance' if 'importance' in top20.columns else 'shap_importance'
        ax.barh(range(len(top20)), top20[importance_col].values)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20['feature'].values, fontsize=9)
        ax.set_xlabel('Importance (Gain)', fontsize=12)
        ax.set_title('Top 20 Feature Importance - Model 1b', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(self.reports_dir / "model1b_feature_importance_top20.png", dpi=300)
        plt.close()
        logger.info("Saved: model1b_feature_importance_top20.png")

        # 3. Correlation heatmap (top 30 features only) - skip if no corr_matrix
        if corr_matrix is not None:
            top30_features = feature_analysis.head(30)['feature'].tolist()
            corr_subset = corr_matrix.loc[top30_features, top30_features]

            fig, ax = plt.subplots(figsize=(16, 14))
            sns.heatmap(corr_subset, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            ax.set_title('Feature Correlations - Top 30 Features', fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig.savefig(self.reports_dir / "model1b_correlation_heatmap.png", dpi=300)
            plt.close()
            logger.info("Saved: model1b_correlation_heatmap.png")
        else:
            logger.warning("Skipping correlation heatmap (no correlation matrix)")

    def generate_report(self, results: dict, feature_analysis: pd.DataFrame,
                       high_corr_df: pd.DataFrame, failed_dates: list):
        """Generate markdown report."""
        logger.info("="*70)
        logger.info("STEP 8: GENERATING REPORT")
        logger.info("="*70)

        report_file = self.reports_dir / "model1b_synoptic_report.md"

        with open(report_file, 'w') as f:
            f.write("# Model 1b - Synoptic Features Only\n\n")
            f.write("**Experiment 8: Learning from Large-Scale Weather Patterns**\n\n")
            f.write("---\n\n")

            # Test Results
            f.write("## Test Results\n\n")
            f.write(f"**Overall Test Performance:**\n")
            f.write(f"- MAE: {results['overall']['mae']:.3f} kt\n")
            f.write(f"- RMSE: {results['overall']['rmse']:.3f} kt\n")
            f.write(f"- R²: {results['overall']['r2']:.3f}\n\n")

            f.write("**Per-Period Performance:**\n\n")
            for period in ['May 1-4', 'June 1-4', 'July 1-4']:
                if period in results:
                    r = results[period]
                    f.write(f"**{period}**\n")
                    f.write(f"- MAE: {r['mae']:.3f} kt\n")
                    f.write(f"- RMSE: {r['rmse']:.3f} kt\n")
                    f.write(f"- R²: {r['r2']:.3f}\n")
                    f.write(f"- Samples: {r['n_samples']}\n\n")

            # Top 20 Features
            f.write("---\n\n## Top 20 Most Important Features\n\n")

            # Check which columns are available
            has_importance = 'importance' in feature_analysis.columns
            has_shap = 'shap_importance' in feature_analysis.columns

            if has_importance and has_shap:
                f.write("| Rank | Feature | Importance | SHAP |\n")
                f.write("|------|---------|------------|------|\n")
                for i, row in feature_analysis.head(20).iterrows():
                    f.write(f"| {i+1} | `{row['feature']}` | {row['importance']:.1f} | {row['shap_importance']:.4f} |\n")
            elif has_shap:
                # SHAP fallback case
                f.write("| Rank | Feature | Importance (Gain) |\n")
                f.write("|------|---------|-------------------|\n")
                for i, row in feature_analysis.head(20).iterrows():
                    f.write(f"| {i+1} | `{row['feature']}` | {row['shap_importance']:.1f} |\n")
            else:
                # Importance only
                f.write("| Rank | Feature | Importance (Gain) |\n")
                f.write("|------|---------|-------------------|\n")
                for i, row in feature_analysis.head(20).iterrows():
                    f.write(f"| {i+1} | `{row['feature']}` | {row['importance']:.1f} |\n")

            # High Correlations
            f.write("\n---\n\n## High Correlations (>0.90)\n\n")
            if high_corr_df is not None and len(high_corr_df) > 0:
                f.write("**Potential duplicate or highly redundant features:**\n\n")
                f.write("| Feature 1 | Feature 2 | Correlation |\n")
                f.write("|-----------|-----------|-------------|\n")
                for idx, row in high_corr_df.head(30).iterrows():
                    f.write(f"| `{row['feature1']}` | `{row['feature2']}` | {row['correlation']:.3f} |\n")
            else:
                f.write("No feature pairs with correlation >0.90 found.\n")

            # Full Feature List
            f.write("\n---\n\n## Complete Feature Importance List\n\n")

            # Use same logic as top 20
            if has_importance and has_shap:
                f.write("| Rank | Feature | Importance | SHAP |\n")
                f.write("|------|---------|------------|------|\n")
                for i, row in feature_analysis.iterrows():
                    f.write(f"| {i+1} | `{row['feature']}` | {row['importance']:.1f} | {row['shap_importance']:.4f} |\n")
            elif has_shap:
                f.write("| Rank | Feature | Importance (Gain) |\n")
                f.write("|------|---------|-------------------|\n")
                for i, row in feature_analysis.iterrows():
                    f.write(f"| {i+1} | `{row['feature']}` | {row['shap_importance']:.1f} |\n")
            else:
                f.write("| Rank | Feature | Importance (Gain) |\n")
                f.write("|------|---------|-------------------|\n")
                for i, row in feature_analysis.iterrows():
                    f.write(f"| {i+1} | `{row['feature']}` | {row['importance']:.1f} |\n")

            # Failed dates
            if failed_dates:
                f.write("\n---\n\n## Failed Extractions\n\n")
                f.write(f"Dates with missing or failed feature extraction: {len(failed_dates)}\n\n")
                for date in failed_dates:
                    f.write(f"- {date}\n")

        logger.info(f"Report saved: {report_file}")

    def run(self):
        """Run full pipeline."""
        logger.info("\n" + "="*70)
        logger.info("MODEL 1B - SYNOPTIC FEATURES ONLY")
        logger.info("="*70 + "\n")

        # Step 1: Extract features
        # Use full overlap range: Jan 16 (wind data starts) to Sep 30 (synoptic data ends)
        features_df, failed_dates = self.extract_all_features('2025-01-16', '2025-09-30')

        # Step 2: Load and merge
        merged_df, target_col, feature_cols = self.load_and_merge_data(features_df)

        # Step 3: Train/test split
        train_df, test_df, test_dates = self.create_train_test_split(merged_df)

        # Step 4: Train model
        model = self.train_model(train_df, feature_cols, target_col)

        # Step 5: Evaluate
        test_df, results = self.evaluate_model(model, test_df, feature_cols, target_col, test_dates)

        # Step 6: Feature analysis
        feature_analysis, high_corr_df, corr_matrix = self.analyze_features(model, train_df, feature_cols)

        # Step 7: Visualizations
        self.create_visualizations(test_df, target_col, feature_analysis, corr_matrix)

        # Step 8: Report
        self.generate_report(results, feature_analysis, high_corr_df, failed_dates)

        logger.info("\n" + "="*70)
        logger.info("COMPLETE!")
        logger.info("="*70)
        logger.info(f"Models saved to: {self.models_dir}")
        logger.info(f"Reports saved to: {self.reports_dir}")


if __name__ == "__main__":
    trainer = Model1bTrainer("config/lake_como/dervio_experiment8.yaml")
    trainer.run()

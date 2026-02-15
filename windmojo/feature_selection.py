#!/usr/bin/env python3
"""
Feature Selection for Phase 2 Model 2

Implements 5 selection approaches on Phase 1 Model 1 feature importance results:
1. model2_gfs_topn - Pure data-driven top-N
2. model2_gfs_balanced - Balanced category selection
3. model2_gfs_hybrid - Correlation-pruned hybrid
4. model2_gfs_no_wind - No GFS wind features
5. model2_gfs_regime_indicators - Physics-driven regime discovery

Usage:
    python -m windmojo.feature_selection \
        --model1-dir data/lake_como/lecco/models/phase1_model1/full \
        --features-file data/lake_como/lecco/features/lecco_features_phase1.parquet \
        --output-dir data/lake_como/lecco/models/phase1_model1/full/feature_selections
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Feature selection for Phase 2 Model 2."""

    def __init__(self, model1_dir: Path, features_file: Path):
        """
        Initialize feature selector.

        Args:
            model1_dir: Directory with Phase 1 Model 1 results
            features_file: Parquet file with engineered features (for correlation)
        """
        self.model1_dir = Path(model1_dir)
        self.features_file = Path(features_file)

        # Storage
        self.importance_by_hour = {}
        self.hour_maes = {}
        self.correlation_matrix = None
        self.features_df = None

    def load_importance_data(self):
        """Load feature importance from all trained hours."""
        logger.info("Loading feature importance from all hours...")

        # Find all importance files
        importance_files = list(self.model1_dir.glob("*/importance_*.json"))

        if not importance_files:
            raise FileNotFoundError(f"No importance files found in {self.model1_dir}")

        for file in importance_files:
            # Extract hour from filename (e.g., importance_1300.json -> 13:00)
            hour_str = file.stem.split("_")[1]
            hour = f"{hour_str[:2]}:{hour_str[2:]}"

            with open(file, "r") as f:
                data = json.load(f)

            self.importance_by_hour[hour] = data["feature_importance"]
            logger.info(
                f"  Loaded {len(data['feature_importance'])} features for {hour}"
            )

        logger.info(f"✓ Loaded importance for {len(self.importance_by_hour)} hours")

    def load_performance_metrics(self):
        """Load test MAE for each hour (for weighted aggregation)."""
        logger.info("Loading performance metrics...")

        results_file = self.model1_dir / "phase1_model1_results.json"

        if results_file.exists():
            with open(results_file, "r") as f:
                results = json.load(f)

            for hour, metrics in results.items():
                if "test_mae_mean" in metrics:
                    self.hour_maes[hour] = metrics["test_mae_mean"]
                    logger.info(f"  {hour}: MAE = {metrics['test_mae_mean']:.3f}")
        else:
            logger.warning(f"Results file not found: {results_file}")
            # Use equal weights if no MAE data
            for hour in self.importance_by_hour.keys():
                self.hour_maes[hour] = 1.0

        logger.info(f"✓ Loaded MAE for {len(self.hour_maes)} hours")

    def load_correlation_matrix(self):
        """Load features and compute correlation matrix."""
        logger.info("Loading features and computing correlations...")

        if not self.features_file.exists():
            logger.warning(f"Features file not found: {self.features_file}")
            self.correlation_matrix = None
            return

        # Load features
        self.features_df = pd.read_parquet(self.features_file)
        logger.info(f"  Loaded features: {self.features_df.shape}")

        # Compute correlation matrix for numeric columns only
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [
            col for col in numeric_cols if col not in ["time", "forecast_made"]
        ]

        logger.info(
            f"  Computing correlations for {len(numeric_cols)} numeric features..."
        )
        self.correlation_matrix = self.features_df[numeric_cols].corr()

        logger.info(f"✓ Correlation matrix computed: {self.correlation_matrix.shape}")

    def aggregate_importance(self) -> pd.DataFrame:
        """
        Aggregate feature importance across all hours.

        Returns:
            DataFrame with aggregated importance (weighted by inverse MAE)
        """
        logger.info("Aggregating feature importance across hours...")

        # Collect all features
        all_features = set()
        for hour_data in self.importance_by_hour.values():
            all_features.update([f["feature"] for f in hour_data])

        # Build aggregated importance
        agg_data = []

        for feature in all_features:
            gains = []
            weights = []
            categories = []

            for hour, hour_data in self.importance_by_hour.items():
                # Find feature in this hour
                feat_data = next(
                    (f for f in hour_data if f["feature"] == feature), None
                )

                if feat_data and feat_data["gain_importance"] > 0:
                    gain = feat_data["gain_importance"]
                    weight = 1.0 / self.hour_maes.get(
                        hour, 1.0
                    )  # Inverse MAE weighting

                    gains.append(gain)
                    weights.append(weight)
                    categories.append(feat_data.get("category", "UNKNOWN"))

            if gains:
                # Weighted average
                weighted_gain = np.average(gains, weights=weights)

                agg_data.append(
                    {
                        "feature": feature,
                        "weighted_gain": weighted_gain,
                        "mean_gain": np.mean(gains),
                        "max_gain": np.max(gains),
                        "hours_used": len(gains),
                        "category": categories[0],  # Take first non-empty category
                    }
                )

        df = pd.DataFrame(agg_data)
        df = df.sort_values("weighted_gain", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        logger.info(f"✓ Aggregated {len(df)} features")
        logger.info(
            f"  Features used in all hours: {len(df[df['hours_used'] == len(self.importance_by_hour)])}"
        )

        return df

    def approach1_topn(self, agg_importance: pd.DataFrame, n: int = 75) -> List[str]:
        """
        Approach 1: Top-N pure data-driven selection.

        Args:
            agg_importance: Aggregated importance DataFrame
            n: Number of features to select

        Returns:
            List of selected feature names
        """
        logger.info(f"\nApproach 1: Top-N (n={n})")

        selected = agg_importance.head(n)["feature"].tolist()

        logger.info(f"  Selected {len(selected)} features")
        logger.info(
            f"  Categories: {agg_importance.head(n)['category'].value_counts().to_dict()}"
        )

        return selected

    def approach2_balanced(self, agg_importance: pd.DataFrame) -> List[str]:
        """
        Approach 2: Balanced category selection.

        Target distribution:
        - RAW: 40% (20 features)
        - SPATIAL_GRADIENT: 20% (10 features)
        - ROLLING_STATS: 20% (10 features)
        - TEMPORAL_LAG: 16% (8 features)
        - PHYSICS: 4% (2 features)
        - TIME_DERIVATIVE: 4% (2 features)
        - CYCLICAL: 2% (1 feature)
        - BINARY: 2% (1 feature)

        Returns:
            List of selected feature names
        """
        logger.info(f"\nApproach 2: Balanced Category Selection")

        targets = {
            "RAW": 20,
            "SPATIAL_GRADIENT": 10,
            "ROLLING_STATS": 10,
            "TEMPORAL_LAG": 8,
            "PHYSICS": 2,
            "TIME_DERIVATIVE": 2,
            "CYCLICAL": 1,
            "BINARY": 1,
        }

        selected = []

        for category, target_count in targets.items():
            cat_features = agg_importance[agg_importance["category"] == category]

            if len(cat_features) < target_count:
                logger.warning(
                    f"  {category}: Only {len(cat_features)} available (target: {target_count})"
                )
                n_select = len(cat_features)
            else:
                n_select = target_count

            cat_selected = cat_features.head(n_select)["feature"].tolist()
            selected.extend(cat_selected)

            logger.info(f"  {category}: Selected {len(cat_selected)}/{target_count}")

        logger.info(f"✓ Total selected: {len(selected)} features")

        return selected

    def approach3_hybrid(
        self,
        agg_importance: pd.DataFrame,
        correlation_threshold: float = 0.85,
        top_n_start: int = 150,
        min_per_category: int = 3,
    ) -> List[str]:
        """
        Approach 3: Correlation-pruned hybrid.

        Args:
            agg_importance: Aggregated importance DataFrame
            correlation_threshold: Remove pairs above this correlation
            top_n_start: Start with top N features
            min_per_category: Minimum features per category

        Returns:
            List of selected feature names
        """
        logger.info(f"\nApproach 3: Correlation-Pruned Hybrid")
        logger.info(f"  Starting with top {top_n_start} features")
        logger.info(f"  Correlation threshold: {correlation_threshold}")

        # Start with top N
        candidates = agg_importance.head(top_n_start)["feature"].tolist()

        if self.correlation_matrix is None:
            logger.warning("  No correlation matrix available - skipping pruning")
            return candidates[:75]

        # Prune correlated pairs
        removed = set()

        for i, feat_i in enumerate(candidates):
            if feat_i in removed:
                continue

            for feat_j in candidates[i + 1 :]:
                if feat_j in removed:
                    continue

                # Check correlation
                if (
                    feat_i in self.correlation_matrix.columns
                    and feat_j in self.correlation_matrix.columns
                ):
                    corr = abs(self.correlation_matrix.loc[feat_i, feat_j])

                    if corr >= correlation_threshold:
                        # Remove feature with lower importance
                        rank_i = agg_importance[
                            agg_importance["feature"] == feat_i
                        ].index[0]
                        rank_j = agg_importance[
                            agg_importance["feature"] == feat_j
                        ].index[0]

                        to_remove = feat_j if rank_j > rank_i else feat_i
                        removed.add(to_remove)

        logger.info(f"  Removed {len(removed)} correlated features")

        # Keep uncorrelated features
        selected = [f for f in candidates if f not in removed]

        # Ensure minimum per category
        category_counts = {}
        for feat in selected:
            cat = agg_importance[agg_importance["feature"] == feat]["category"].iloc[0]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        for category in agg_importance["category"].unique():
            if category_counts.get(category, 0) < min_per_category:
                # Add top features from this category
                cat_features = agg_importance[agg_importance["category"] == category]
                cat_features = cat_features[~cat_features["feature"].isin(selected)]

                needed = min_per_category - category_counts.get(category, 0)
                add_features = cat_features.head(needed)["feature"].tolist()
                selected.extend(add_features)

                logger.info(
                    f"  Added {len(add_features)} features to meet minimum for {category}"
                )

        logger.info(f"✓ Total selected: {len(selected)} features")
        logger.info(
            f"  Categories: {agg_importance[agg_importance['feature'].isin(selected)]['category'].value_counts().to_dict()}"
        )

        return selected

    def approach4_no_wind(self, agg_importance: pd.DataFrame, n: int = 75) -> List[str]:
        """
        Approach 4: No GFS wind features (test atmospheric conditions only).

        Remove all wind_speed, wind_gusts, wind_direction features + all derived features.
        This includes gust_factor, gust_excess, and any lag/rolling/gradient versions.

        Args:
            agg_importance: Aggregated importance DataFrame
            n: Target number of features

        Returns:
            List of selected feature names
        """
        logger.info(f"\nApproach 4: No GFS Wind Features")

        # Identify wind-related features to remove
        wind_patterns = [
            "wind_speed_10m",
            "wind_speed_100m",
            "wind_speed_850hPa",
            "wind_speed_250hPa",
            "wind_direction",  # All wind direction features
            "wind_gusts_10m",  # Raw gust measurements
            "wind_gusts_100m",  # Raw gust measurements at 100m
            "gust_factor",  # Derived gust feature
            "gust_excess",  # Derived gust feature
            "gust",  # Catch any other gust-related features
        ]

        # Filter out wind features
        def is_wind_feature(feat: str) -> bool:
            return any(pattern in feat for pattern in wind_patterns)

        non_wind = agg_importance[~agg_importance["feature"].apply(is_wind_feature)]

        removed_count = len(agg_importance) - len(non_wind)
        logger.info(f"  Removed {removed_count} wind/gust-related features")
        logger.info(f"  Remaining: {len(non_wind)} features")

        # Select top N from non-wind features
        selected = non_wind.head(n)["feature"].tolist()

        logger.info(
            f"✓ Selected {len(selected)} features (atmospheric conditions only)"
        )
        logger.info(
            f"  Categories: {non_wind.head(n)['category'].value_counts().to_dict()}"
        )

        return selected

    def approach5_regime_indicators(
        self, agg_importance: pd.DataFrame, target_n: int = 60
    ) -> List[str]:
        """
        Approach 5: Physics-driven regime discovery.

        Keep only atmospheric condition indicators that DRIVE wind regimes:
        - Pressure gradients (synoptic forcing)
        - Temperature × season (thermal regime timing)
        - Solar/cloud gradients (daytime heating)
        - Temperature gradients (thermal wind)
        - Precipitation (weather systems/föhn)
        - Cyclical time (regime timing)
        - Multi-day lags (pattern persistence)

        Remove:
        - All GFS wind features (test if wind can be learned from conditions alone)
        - Raw atmospheric values without spatial context
        - Short-term reactive lags (1-3hr)

        Args:
            agg_importance: Aggregated importance DataFrame
            target_n: Target number of features (~60)

        Returns:
            List of selected feature names
        """
        logger.info(f"\nApproach 5: Physics-Driven Regime Indicators")

        # Define keep patterns (atmospheric drivers)
        keep_patterns = [
            "pressure_grad",  # All pressure gradients
            "pressure_curvature",  # Pressure curvature
            "temp_seasonal",  # Temperature × season interaction
            "shortwave_radiation",  # Solar heating
            "direct_radiation",  # Direct solar
            "diffuse_radiation",  # Diffuse solar
            "cloud_cover",  # Cloud patterns
            "temperature_2m",  # Temperature patterns
            "precipitation",  # Precipitation indicators
            "hour_sin",
            "hour_cos",  # Time-of-day (regime timing)
            "month_sin",
            "month_cos",  # Season (regime timing)
            "is_daytime",  # Daytime indicator
        ]

        # Define remove patterns (avoid direct wind and reactive features)
        remove_patterns = [
            "wind_speed",
            "wind_gusts",
            "wind_direction",
            "_lag_1h",  # Too reactive
            "_lag_3h",  # Too reactive
            "_center",  # Raw values without gradients
            "_n_near",
            "_s_near",  # Keep gradients only
            "_ne",
            "_nw",
            "_se",
            "_sw",  # Keep gradients only (unless they're gradients themselves)
        ]

        def should_keep(feat: str) -> bool:
            """Check if feature matches regime indicator criteria."""
            # Must match at least one keep pattern
            has_keep = any(pattern in feat for pattern in keep_patterns)

            # Must not match any remove pattern
            has_remove = any(pattern in feat for pattern in remove_patterns)

            # Special handling for gradients (they contain directional terms but are actually gradients)
            is_gradient = "_grad_" in feat or "gradient_" in feat or "_diff_" in feat

            if is_gradient:
                # Gradients are good even if they contain directional terms
                has_remove = any(
                    pattern in feat for pattern in remove_patterns[:3]
                )  # Only check wind patterns

            return has_keep and not has_remove

        # Filter features
        regime_indicators = agg_importance[agg_importance["feature"].apply(should_keep)]

        removed_count = len(agg_importance) - len(regime_indicators)
        logger.info(f"  Removed {removed_count} non-regime features")
        logger.info(f"  Remaining regime indicators: {len(regime_indicators)}")

        # Select top N from regime indicators
        selected = regime_indicators.head(target_n)["feature"].tolist()

        # Log feature breakdown
        logger.info(f"✓ Selected {len(selected)} regime indicator features")
        logger.info(
            f"  Categories: {regime_indicators.head(target_n)['category'].value_counts().to_dict()}"
        )

        # Log feature type breakdown
        feature_types = {
            "Pressure": sum(1 for f in selected if "pressure" in f),
            "Temperature": sum(1 for f in selected if "temp" in f),
            "Solar/Cloud": sum(
                1
                for f in selected
                if any(x in f for x in ["solar", "radiation", "cloud"])
            ),
            "Precipitation": sum(1 for f in selected if "precip" in f),
            "Cyclical": sum(
                1 for f in selected if any(x in f for x in ["sin", "cos", "daytime"])
            ),
        }
        logger.info(f"  Feature types: {feature_types}")

        return selected

    def save_feature_list(
        self,
        features: List[str],
        approach_name: str,
        output_dir: Path,
        agg_importance: pd.DataFrame,
    ):
        """Save feature list to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build detailed feature data
        feature_data = []
        for i, feat in enumerate(features):
            feat_info = (
                agg_importance[agg_importance["feature"] == feat].iloc[0].to_dict()
            )
            feat_info["selection_rank"] = i + 1
            feature_data.append(feat_info)

        output = {
            "approach": approach_name,
            "feature_count": len(features),
            "features": feature_data,
        }

        output_file = output_dir / f"features_{approach_name}.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"  Saved to {output_file}")

    def generate_comparison_report(
        self,
        selections: Dict[str, List[str]],
        agg_importance: pd.DataFrame,
        output_dir: Path,
    ):
        """Generate markdown comparison report."""
        logger.info("\nGenerating comparison report...")

        report_lines = []
        report_lines.append("# Feature Selection Comparison Report")
        report_lines.append("")
        report_lines.append(
            f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("")
        report_lines.append("## Overview")
        report_lines.append("")
        report_lines.append(
            "This report compares 5 feature selection approaches for Phase 2 Model 2."
        )
        report_lines.append("")

        # Summary table
        report_lines.append("## Summary Table")
        report_lines.append("")
        report_lines.append("| Approach | Feature Count | Purpose |")
        report_lines.append("|----------|--------------|---------|")
        report_lines.append(
            f"| `model2_gfs_topn` | {len(selections['gfs_topn'])} | Pure data-driven, highest MAE performance |"
        )
        report_lines.append(
            f"| `model2_gfs_balanced` | {len(selections['gfs_balanced'])} | Balanced categories, better interpretability |"
        )
        report_lines.append(
            f"| `model2_gfs_hybrid` | {len(selections['gfs_hybrid'])} | Correlation-pruned, optimal balance |"
        )
        report_lines.append(
            f"| `model2_gfs_no_wind` | {len(selections['gfs_no_wind'])} | No GFS wind, atmospheric conditions only |"
        )
        report_lines.append(
            f"| `model2_gfs_regime_indicators` | {len(selections['gfs_regime_indicators'])} | Physics-driven regime discovery |"
        )
        report_lines.append("")

        # Feature overlap analysis
        report_lines.append("## Feature Overlap Analysis")
        report_lines.append("")

        all_approaches = list(selections.keys())
        for i, app1 in enumerate(all_approaches):
            for app2 in all_approaches[i + 1 :]:
                set1 = set(selections[app1])
                set2 = set(selections[app2])
                overlap = set1 & set2
                overlap_pct = 100 * len(overlap) / min(len(set1), len(set2))

                report_lines.append(
                    f"- **{app1} ∩ {app2}**: {len(overlap)} features ({overlap_pct:.1f}% overlap)"
                )

        report_lines.append("")

        # Top 20 features per approach
        for approach_name, features in selections.items():
            report_lines.append(f"## {approach_name.upper()}: Top 20 Features")
            report_lines.append("")
            report_lines.append("| Rank | Feature | Weighted Gain | Category |")
            report_lines.append("|------|---------|---------------|----------|")

            for i, feat in enumerate(features[:20]):
                feat_info = agg_importance[agg_importance["feature"] == feat].iloc[0]
                report_lines.append(
                    f"| {i+1} | `{feat}` | {feat_info['weighted_gain']:.2f} | {feat_info['category']} |"
                )

            report_lines.append("")

        # Category distribution
        report_lines.append("## Category Distribution")
        report_lines.append("")

        for approach_name, features in selections.items():
            report_lines.append(f"### {approach_name.upper()}")
            report_lines.append("")

            categories = []
            for feat in features:
                cat = agg_importance[agg_importance["feature"] == feat][
                    "category"
                ].iloc[0]
                categories.append(cat)

            cat_counts = pd.Series(categories).value_counts()

            report_lines.append("| Category | Count | % |")
            report_lines.append("|----------|-------|---|")
            for cat, count in cat_counts.items():
                pct = 100 * count / len(features)
                report_lines.append(f"| {cat} | {count} | {pct:.1f}% |")

            report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        report_lines.append("### For Model 2 Training:")
        report_lines.append("")
        report_lines.append(
            "1. **Start with `model2_gfs_hybrid`** (best balance of performance + diversity)"
        )
        report_lines.append(
            "2. **Benchmark against `model2_gfs_topn`** (expected best MAE)"
        )
        report_lines.append(
            "3. **Test `model2_gfs_no_wind`** (validate atmospheric learning)"
        )
        report_lines.append(
            "4. **Consider `model2_gfs_balanced`** if interpretability is priority"
        )
        report_lines.append("")
        report_lines.append("### Expected Performance:")
        report_lines.append("")
        report_lines.append(
            "- **model2_gfs_topn**: Best MAE (~10-20% improvement over Model 1)"
        )
        report_lines.append(
            "- **model2_gfs_hybrid**: Similar MAE to topn, better generalization"
        )
        report_lines.append("- **model2_gfs_balanced**: ~5-10% higher MAE, more stable")
        report_lines.append(
            "- **model2_gfs_no_wind**: Educational - tests if wind can be learned from conditions"
        )
        report_lines.append("")

        # Write report
        report_file = output_dir / "feature_selection_comparison.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))

        logger.info(f"✓ Report saved to {report_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Feature selection for Phase 2 Model 2"
    )
    parser.add_argument(
        "--model1-dir", required=True, help="Directory with Phase 1 Model 1 results"
    )
    parser.add_argument(
        "--features-file",
        required=True,
        help="Parquet file with engineered features (for correlation)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for feature selections"
    )

    args = parser.parse_args()

    # Initialize selector
    selector = FeatureSelector(Path(args.model1_dir), Path(args.features_file))

    # Load data
    selector.load_importance_data()
    selector.load_performance_metrics()
    selector.load_correlation_matrix()

    # Aggregate importance
    agg_importance = selector.aggregate_importance()

    # Run all 5 approaches
    selections = {}

    logger.info("\n" + "=" * 70)
    logger.info("RUNNING FEATURE SELECTION APPROACHES")
    logger.info("=" * 70)

    selections["gfs_topn"] = selector.approach1_topn(agg_importance, n=75)
    selections["gfs_balanced"] = selector.approach2_balanced(agg_importance)
    selections["gfs_hybrid"] = selector.approach3_hybrid(agg_importance)
    selections["gfs_no_wind"] = selector.approach4_no_wind(agg_importance, n=75)
    selections["gfs_regime_indicators"] = selector.approach5_regime_indicators(
        agg_importance, target_n=60
    )

    # Save all feature lists
    logger.info("\n" + "=" * 70)
    logger.info("SAVING FEATURE LISTS")
    logger.info("=" * 70)

    output_dir = Path(args.output_dir)

    for approach_name, features in selections.items():
        selector.save_feature_list(features, approach_name, output_dir, agg_importance)

    # Generate comparison report
    selector.generate_comparison_report(selections, agg_importance, output_dir)

    logger.info("\n" + "=" * 70)
    logger.info("✓ FEATURE SELECTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"  - 5 JSON feature lists")
    logger.info(f"  - 1 comparison report (markdown)")
    logger.info("\nApproaches:")
    logger.info("  1. gfs_topn (75 features) - Pure data-driven")
    logger.info("  2. gfs_balanced (53 features) - Balanced categories")
    logger.info("  3. gfs_hybrid (84 features) - Correlation-pruned")
    logger.info("  4. gfs_no_wind (75 features) - No GFS wind features")
    logger.info(
        "  5. gfs_regime_indicators (~60 features) - Physics-driven regime discovery"
    )


if __name__ == "__main__":
    main()

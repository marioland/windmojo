"""
Feature Report Generation

Generate documentation and analysis of engineered features.
Helps track feature counts, categories, and data quality.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureReport:
    """Generate comprehensive report on engineered features."""

    # Feature category patterns (for classification)
    CATEGORY_PATTERNS = {
        "SPATIAL_GRADIENT": ["grad", "diff", "curvature", "shear"],
        "TEMPORAL_LAG": ["lag_1h", "lag_3h", "lag_6h", "lag_1d", "lag_2d", "lag_3d"],
        "TIME_DERIVATIVE": ["change_1h", "change_3h", "accel"],
        "ROLLING_STATS": ["rolling", "_std", "_max", "_min", "_sum", "_mean"],
        "PHYSICS": [
            "stability",
            "richardson",
            "buoyancy",
            "heating",
            "cooling",
            "thermal",
            "soil_moisture",
            "gust_factor",
            "gust_excess",
        ],
        "CYCLICAL": ["_sin", "_cos"],
        "BINARY": ["is_"],
        "RAW": [
            # Identified by absence of engineering suffixes
        ],
    }

    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature report generator.

        Args:
            df: DataFrame with engineered features
        """
        self.df = df
        self.feature_columns = [
            c for c in df.columns if c not in ["time", "forecast_made"]
        ]

    def generate_report(self) -> Dict:
        """
        Generate comprehensive feature report.

        Returns:
            Dictionary with report sections
        """
        report = {
            "summary": self._summary_stats(),
            "categories": self._categorize_features(),
            "quality": self._data_quality(),
            "correlations": self._high_correlations(),
            "distributions": self._distribution_stats(),
        }

        return report

    def _summary_stats(self) -> Dict:
        """Generate summary statistics."""
        return {
            "total_features": len(self.feature_columns),
            "total_rows": len(self.df),
            "date_range": {
                "start": str(self.df["time"].min()),
                "end": str(self.df["time"].max()),
                "days": (self.df["time"].max() - self.df["time"].min()).days,
            },
            "missing_values": {
                "total_cells": len(self.df) * len(self.feature_columns),
                "missing_cells": self.df[self.feature_columns].isna().sum().sum(),
                "missing_pct": (
                    self.df[self.feature_columns].isna().sum().sum()
                    / (len(self.df) * len(self.feature_columns))
                    * 100
                ),
            },
        }

    def _categorize_features(self) -> Dict[str, List[str]]:
        """Categorize features by type."""
        categories = defaultdict(list)

        for feature in self.feature_columns:
            categorized = False

            # Check each category pattern
            for category, patterns in self.CATEGORY_PATTERNS.items():
                if category == "RAW":
                    continue

                if any(pattern in feature for pattern in patterns):
                    categories[category].append(feature)
                    categorized = True
                    break

            # If not categorized, assume it's a raw feature
            if not categorized:
                categories["RAW"].append(feature)

        # Add counts
        result = {cat: features for cat, features in categories.items()}
        result["COUNTS"] = {cat: len(features) for cat, features in categories.items()}

        return result

    def _data_quality(self) -> Dict:
        """Analyze data quality metrics."""
        quality = {}

        for col in self.feature_columns:
            if self.df[col].dtype in [np.float64, np.int64]:
                quality[col] = {
                    "missing_count": int(self.df[col].isna().sum()),
                    "missing_pct": float(self.df[col].isna().mean() * 100),
                    "zero_count": int((self.df[col] == 0).sum()),
                    "zero_pct": float((self.df[col] == 0).mean() * 100),
                    "inf_count": int(np.isinf(self.df[col]).sum()),
                    "unique_values": int(self.df[col].nunique()),
                }

        return quality

    def _high_correlations(
        self, threshold: float = 0.95
    ) -> List[Tuple[str, str, float]]:
        """Find highly correlated feature pairs."""
        # Calculate correlation matrix (only numeric features)
        numeric_features = [
            c
            for c in self.feature_columns
            if self.df[c].dtype in [np.float64, np.int64]
        ]

        if len(numeric_features) < 2:
            return []

        corr_matrix = self.df[numeric_features].corr()

        # Find high correlations (excluding self-correlations)
        high_corr = []
        for i, col1 in enumerate(numeric_features):
            for j, col2 in enumerate(numeric_features):
                if i < j:  # Only upper triangle
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) >= threshold:
                        high_corr.append((col1, col2, corr_val))

        # Sort by absolute correlation
        high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

        return high_corr

    def _distribution_stats(self) -> Dict:
        """Calculate distribution statistics for each feature."""
        stats = {}

        for col in self.feature_columns:
            if self.df[col].dtype in [np.float64, np.int64]:
                stats[col] = {
                    "mean": float(self.df[col].mean()),
                    "std": float(self.df[col].std()),
                    "min": float(self.df[col].min()),
                    "p25": float(self.df[col].quantile(0.25)),
                    "median": float(self.df[col].median()),
                    "p75": float(self.df[col].quantile(0.75)),
                    "max": float(self.df[col].max()),
                    "skew": float(self.df[col].skew()),
                    "kurtosis": float(self.df[col].kurtosis()),
                }

        return stats

    def print_summary(self):
        """Print human-readable summary report."""
        report = self.generate_report()

        logger.info("=" * 70)
        logger.info("FEATURE ENGINEERING REPORT")
        logger.info("=" * 70)

        # Summary
        summary = report["summary"]
        logger.info("SUMMARY")
        logger.info(f"  Total features: {summary['total_features']}")
        logger.info(f"  Total rows: {summary['total_rows']:,}")
        logger.info(
            f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}"
        )
        logger.info(f"  Coverage: {summary['date_range']['days']} days")
        logger.info(
            f"  Missing values: {summary['missing_values']['missing_pct']:.2f}%"
        )

        # Categories
        categories = report["categories"]
        logger.info("FEATURE CATEGORIES")
        for category, count in sorted(
            categories["COUNTS"].items(), key=lambda x: -x[1]
        ):
            pct = count / summary["total_features"] * 100
            logger.info(f"  {category:20s}: {count:3d} features ({pct:5.1f}%)")

        # High correlations
        high_corr = report["correlations"]
        if high_corr:
            logger.warning("HIGH CORRELATIONS (|r| >= 0.95)")
            for col1, col2, corr in high_corr[:10]:  # Show top 10
                logger.warning(f"  {corr:+.3f}  {col1} <-> {col2}")
            if len(high_corr) > 10:
                logger.warning(f"  ... and {len(high_corr) - 10} more")
        else:
            logger.info("No highly correlated features (|r| >= 0.95)")

        # Data quality issues
        quality = report["quality"]
        issues = []
        for feature, metrics in quality.items():
            if metrics["missing_pct"] > 10:
                issues.append((feature, f"{metrics['missing_pct']:.1f}% missing"))
            if metrics["inf_count"] > 0:
                issues.append((feature, f"{metrics['inf_count']} inf values"))

        if issues:
            logger.warning("DATA QUALITY ISSUES")
            for feature, issue in issues[:10]:
                logger.warning(f"  {feature}: {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... and {len(issues) - 10} more")
        else:
            logger.info("No major data quality issues detected")

        logger.info("=" * 70)

    def save_report(self, output_path: Path):
        """
        Save detailed report to JSON file.

        Args:
            output_path: Path to save report (JSON format)
        """
        import json

        report = self.generate_report()

        # Convert tuples to lists for JSON serialization
        if "correlations" in report:
            report["correlations"] = [
                {"feature1": c1, "feature2": c2, "correlation": float(corr)}
                for c1, c2, corr in report["correlations"]
            ]

        # Convert numpy/pandas types to native Python types
        report = self._convert_to_json_serializable(report)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Feature report saved to {output_path}")

    @staticmethod
    def _analyze_measurement_directions(
        measurements_path: Path,
        target_column: str,
        direction_column: str = None,
        speed_threshold: float = 10.0,
    ) -> str:
        """
        Analyze wind direction distribution from measurement data.

        Args:
            measurements_path: Path to measurements CSV file
            target_column: Wind speed column name
            direction_column: Wind direction column name (None = infer from target_column)
            speed_threshold: Minimum wind speed to count (m/s), default 10.0

        Returns:
            Markdown string with direction analysis
        """
        md_lines = []
        md_lines.append("## Input Data: Wind Direction Analysis\n")

        try:
            # Load measurements
            df = pd.read_csv(measurements_path)

            # Infer direction column if not provided
            if direction_column is None:
                # Try to infer from target column (e.g., "dervio speed" → "dervio direction")
                base_name = (
                    target_column.replace(" speed", "").replace("speed", "").strip()
                )
                direction_column = f"{base_name} direction" if base_name else None

            # Check if columns exist
            if target_column not in df.columns:
                md_lines.append(
                    f"⚠️ **Warning**: Target column '{target_column}' not found in measurements\n"
                )
                return "\n".join(md_lines)

            if direction_column and direction_column not in df.columns:
                md_lines.append(
                    f"⚠️ **Warning**: Direction column '{direction_column}' not found in measurements\n"
                )
                md_lines.append(f"Available columns: {', '.join(df.columns)}\n")
                return "\n".join(md_lines)

            # If no direction column, skip direction analysis
            if not direction_column or direction_column not in df.columns:
                md_lines.append(
                    "*Direction data not available in measurements - skipping direction analysis*\n"
                )
                return "\n".join(md_lines)

            # Filter valid data
            valid_df = df[[target_column, direction_column]].dropna()
            total_samples = len(valid_df)

            if total_samples == 0:
                md_lines.append("*No valid wind speed/direction data found*\n")
                return "\n".join(md_lines)

            # Separate calm vs windy conditions
            calm_mask = valid_df[target_column] <= speed_threshold
            calm_count = calm_mask.sum()
            calm_pct = (calm_count / total_samples) * 100

            windy_df = valid_df[~calm_mask]
            windy_count = len(windy_df)

            md_lines.append(
                f"**Measurement station**: {target_column.split()[0].title()}"
            )
            md_lines.append(f"**Total samples**: {total_samples:,}")
            md_lines.append(
                f"**Speed threshold**: >{speed_threshold} m/s for direction analysis\n"
            )

            # Overall distribution
            md_lines.append("### Wind Speed Distribution\n")
            md_lines.append(
                f"- **Calm** (≤{speed_threshold} m/s): {calm_pct:.1f}% ({calm_count:,} samples)"
            )
            md_lines.append(
                f"- **Windy** (>{speed_threshold} m/s): {100-calm_pct:.1f}% ({windy_count:,} samples)\n"
            )

            if windy_count == 0:
                md_lines.append(
                    "*No samples exceed speed threshold - no dominant direction analysis*\n"
                )
                return "\n".join(md_lines)

            # 8-compass direction analysis (only for windy conditions)
            md_lines.append(
                f"### Dominant Wind Directions (Speed >{speed_threshold} m/s)\n"
            )

            # Define 8 compass sectors (45° each, centered on cardinal/ordinal)
            compass_sectors = {
                "N": (337.5, 22.5),  # North: 337.5-22.5
                "NE": (22.5, 67.5),  # Northeast: 22.5-67.5
                "E": (67.5, 112.5),  # East: 67.5-112.5
                "SE": (112.5, 157.5),  # Southeast: 112.5-157.5
                "S": (157.5, 202.5),  # South: 157.5-202.5
                "SW": (202.5, 247.5),  # Southwest: 202.5-247.5
                "W": (247.5, 292.5),  # West: 247.5-292.5
                "NW": (292.5, 337.5),  # Northwest: 292.5-337.5
            }

            # Count samples in each sector
            sector_counts = {}
            for sector, (start, end) in compass_sectors.items():
                if start > end:  # Handle North wrap-around
                    mask = (windy_df[direction_column] >= start) | (
                        windy_df[direction_column] < end
                    )
                else:
                    mask = (windy_df[direction_column] >= start) & (
                        windy_df[direction_column] < end
                    )
                sector_counts[sector] = mask.sum()

            # Sort by count (descending)
            sorted_sectors = sorted(sector_counts.items(), key=lambda x: -x[1])

            # Display table
            md_lines.append("| Direction | Degrees | Sample Count | Percentage |")
            md_lines.append("|-----------|---------|--------------|------------|")

            for sector, count in sorted_sectors:
                pct = (count / windy_count) * 100
                start, end = compass_sectors[sector]
                if start > end:
                    degree_range = f"{start:.1f}°-360°, 0°-{end:.1f}°"
                else:
                    degree_range = f"{start:.1f}°-{end:.1f}°"
                md_lines.append(
                    f"| {sector:^9} | {degree_range:^17} | {count:^12,} | {pct:^10.1f}% |"
                )

            # Dominant pattern summary
            md_lines.append("")
            primary = sorted_sectors[0]
            secondary = sorted_sectors[1] if len(sorted_sectors) > 1 else None

            primary_pct = (primary[1] / windy_count) * 100
            md_lines.append(
                f"**Primary direction**: {primary[0]} ({primary_pct:.1f}% of windy samples)"
            )

            if secondary and secondary[1] > 0:
                secondary_pct = (secondary[1] / windy_count) * 100
                md_lines.append(
                    f"**Secondary direction**: {secondary[0]} ({secondary_pct:.1f}% of windy samples)"
                )

            # Direction variability
            direction_std = windy_df[direction_column].std()
            md_lines.append(
                f"\n**Direction variability**: {direction_std:.1f}° standard deviation"
            )

            # Interpretation hints
            md_lines.append("\n**Interpretation**:")
            if primary_pct > 50:
                md_lines.append(
                    f"- **Strongly directional**: {primary[0]} dominates with >{primary_pct:.0f}% of windy conditions"
                )
            elif primary_pct > 30:
                md_lines.append(
                    f"- **Moderately directional**: {primary[0]} is primary but not dominant"
                )
            else:
                md_lines.append(
                    f"- **Variable directions**: No single dominant direction (highest is {primary_pct:.0f}%)"
                )

            # Thermal wind hint (for lake locations)
            if "N" in [primary[0], secondary[0] if secondary else None] and "S" in [
                primary[0],
                secondary[0] if secondary else None,
            ]:
                md_lines.append(
                    "- **Thermal pattern detected**: N-S bidirectional pattern suggests valley/lake thermal winds"
                )

        except Exception as e:
            logger.warning(f"Error analyzing measurement directions: {e}")
            md_lines.append(f"\n⚠️ **Error**: Could not analyze directions - {str(e)}\n")

        md_lines.append("")
        return "\n".join(md_lines)

    def save_markdown_report(
        self,
        output_path: Path,
        measurements_path: Path = None,
        target_column: str = None,
        direction_column: str = None,
    ):
        """
        Save comprehensive markdown report.

        Args:
            output_path: Path to save report (Markdown format)
        """
        from datetime import datetime

        report = self.generate_report()
        md_lines = []

        # Header
        md_lines.append("# Feature Engineering Report")
        md_lines.append(
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        md_lines.append(
            f"\n**Purpose:** Documentation of engineered features from raw GFS variables\n"
        )
        md_lines.append("---\n")

        # Executive Summary
        summary = report["summary"]
        md_lines.append("## Executive Summary\n")
        md_lines.append(f"- **Total features engineered**: {summary['total_features']}")
        md_lines.append(f"- **Total data rows**: {summary['total_rows']:,}")
        md_lines.append(
            f"- **Date range**: {summary['date_range']['start']} to {summary['date_range']['end']}"
        )
        md_lines.append(f"- **Coverage**: {summary['date_range']['days']} days")
        md_lines.append(
            f"- **Missing values**: {summary['missing_values']['missing_pct']:.2f}%"
        )
        md_lines.append(
            f"  - Total cells: {summary['missing_values']['total_cells']:,}"
        )
        md_lines.append(
            f"  - Missing cells: {summary['missing_values']['missing_cells']:,}\n"
        )

        # Input Data: Wind Direction Analysis (if measurements provided)
        if measurements_path and target_column:
            direction_analysis = self._analyze_measurement_directions(
                measurements_path=measurements_path,
                target_column=target_column,
                direction_column=direction_column,
                speed_threshold=10.0,
            )
            md_lines.append(direction_analysis)
            md_lines.append("---\n")

        # Feature Categories
        categories = report["categories"]
        md_lines.append("## Feature Categories\n")
        md_lines.append("Breakdown of features by engineering stage:\n")
        md_lines.append("| Category | Count | % of Total |")
        md_lines.append("|----------|-------|------------|")

        total_features = summary["total_features"]
        for category, count in sorted(
            categories["COUNTS"].items(), key=lambda x: -x[1]
        ):
            pct = count / total_features * 100
            md_lines.append(f"| {category} | {count} | {pct:.1f}% |")

        md_lines.append("")

        # Category Details
        md_lines.append("### Category Descriptions\n")
        md_lines.append(
            "- **RAW**: Direct GFS variables (wind, pressure, temperature, etc.)"
        )
        md_lines.append(
            "- **SPATIAL_GRADIENT**: Gradients across spatial grid (N/S/E/W/NE/NW/SE/SW)"
        )
        md_lines.append(
            "- **TEMPORAL_LAG**: Historical values (1h, 3h, 6h, 1d, 2d, 3d back)"
        )
        md_lines.append(
            "- **ROLLING_STATS**: Moving window statistics (mean, std, max, min)"
        )
        md_lines.append(
            "- **TIME_DERIVATIVE**: Change rates (dT/dt, dP/dt, acceleration)"
        )
        md_lines.append(
            "- **PHYSICS**: Physics-derived indicators (stability, buoyancy, thermal)"
        )
        md_lines.append(
            "- **CYCLICAL**: Sine/cosine encoded temporal features (hour, month)"
        )
        md_lines.append(
            "- **BINARY**: Binary indicators (is_afternoon, is_summer, etc.)\n"
        )

        # Data Quality
        quality = report["quality"]
        md_lines.append("## Data Quality Analysis\n")

        # Find features with issues
        missing_issues = []
        zero_heavy = []
        inf_issues = []
        low_variance = []

        for feature, metrics in quality.items():
            if metrics["missing_pct"] > 5:
                missing_issues.append((feature, metrics["missing_pct"]))
            if metrics["zero_pct"] > 80:
                zero_heavy.append((feature, metrics["zero_pct"]))
            if metrics["inf_count"] > 0:
                inf_issues.append((feature, metrics["inf_count"]))

        # Missing values
        if missing_issues:
            md_lines.append("### Features with >5% Missing Values\n")
            md_lines.append("| Feature | Missing % |")
            md_lines.append("|---------|-----------|")
            for feat, pct in sorted(missing_issues, key=lambda x: -x[1])[:20]:
                md_lines.append(f"| `{feat}` | {pct:.1f}% |")
            if len(missing_issues) > 20:
                md_lines.append(f"\n*...and {len(missing_issues) - 20} more features*")
            md_lines.append("")
        else:
            md_lines.append("✅ **No features with >5% missing values**\n")

        # Infinite values
        if inf_issues:
            md_lines.append("### Features with Infinite Values\n")
            md_lines.append("| Feature | Inf Count |")
            md_lines.append("|---------|-----------|")
            for feat, count in sorted(inf_issues, key=lambda x: -x[1])[:10]:
                md_lines.append(f"| `{feat}` | {count} |")
            md_lines.append("")
        else:
            md_lines.append("✅ **No features with infinite values**\n")

        # Zero-heavy features
        if zero_heavy:
            md_lines.append("### Features with >80% Zero Values\n")
            md_lines.append("| Feature | Zero % |")
            md_lines.append("|---------|--------|")
            for feat, pct in sorted(zero_heavy, key=lambda x: -x[1])[:20]:
                md_lines.append(f"| `{feat}` | {pct:.1f}% |")
            if len(zero_heavy) > 20:
                md_lines.append(f"\n*...and {len(zero_heavy) - 20} more features*")
            md_lines.append("")

        # High Correlations
        correlations = report["correlations"]
        if correlations:
            md_lines.append("## High Correlations (|r| ≥ 0.95)\n")
            md_lines.append(
                "Feature pairs with very high correlation (potential redundancy):\n"
            )
            md_lines.append("| Feature 1 | Feature 2 | Correlation |")
            md_lines.append("|-----------|-----------|-------------|")

            for corr_item in correlations[:30]:  # Show top 30
                if isinstance(corr_item, dict):
                    feat1 = corr_item["feature1"]
                    feat2 = corr_item["feature2"]
                    corr = corr_item["correlation"]
                else:
                    feat1, feat2, corr = corr_item

                md_lines.append(f"| `{feat1}` | `{feat2}` | {corr:+.3f} |")

            if len(correlations) > 30:
                md_lines.append(f"\n*...and {len(correlations) - 30} more pairs*")
            md_lines.append("")
        else:
            md_lines.append("## High Correlations\n")
            md_lines.append("✅ **No highly correlated feature pairs (|r| ≥ 0.95)**\n")

        # Distribution Statistics (sample)
        distributions = report["distributions"]
        md_lines.append("## Distribution Statistics (Sample)\n")
        md_lines.append("Statistical properties of key features:\n")

        # Show stats for first 10 features of each category
        for category in [
            "RAW",
            "SPATIAL_GRADIENT",
            "TEMPORAL_LAG",
            "ROLLING_STATS",
            "PHYSICS",
        ]:
            if category in categories and categories[category]:
                md_lines.append(f"\n### {category} Features (Sample)\n")
                md_lines.append("| Feature | Mean | Std | Min | Max | Skew |")
                md_lines.append("|---------|------|-----|-----|-----|------|")

                sample_features = categories[category][:5]  # First 5 from each category
                for feat in sample_features:
                    if feat in distributions:
                        stats = distributions[feat]
                        md_lines.append(
                            f"| `{feat}` | {stats['mean']:.2f} | {stats['std']:.2f} | "
                            f"{stats['min']:.2f} | {stats['max']:.2f} | {stats['skew']:.2f} |"
                        )
                md_lines.append("")

        # Recommendations
        md_lines.append("## Recommendations\n")
        md_lines.append("### For Feature Selection (Phase 2):\n")

        if correlations and len(correlations) > 20:
            md_lines.append(
                f"1. **Remove redundant features**: {len(correlations)} pairs with |r| ≥ 0.95"
            )
            md_lines.append(
                f"   - Consider correlation pruning to reduce multicollinearity"
            )

        if missing_issues:
            md_lines.append(
                f"2. **Handle missing values**: {len(missing_issues)} features with >5% missing"
            )
            md_lines.append(
                f"   - XGBoost handles missing natively, but consider imputation for critical features"
            )

        if zero_heavy:
            md_lines.append(
                f"3. **Review zero-heavy features**: {len(zero_heavy)} features with >80% zeros"
            )
            md_lines.append(f"   - May indicate rare events or data quality issues")

        md_lines.append("\n### For Model Training:\n")
        md_lines.append(
            f"1. **Feature count**: {total_features} features is large - use feature selection"
        )
        md_lines.append(f"2. **Recommended approaches**:")
        md_lines.append(
            f"   - Train Phase 1 Model 1 with all features to rank importance"
        )
        md_lines.append(f"   - Select top 50-100 features for Phase 2 Model 2")
        md_lines.append(f"   - Consider balanced category selection (not just top-N)")

        # Appendix
        md_lines.append("\n---\n")
        md_lines.append("## Appendix: Feature Engineering Pipeline\n")
        md_lines.append("### 13-Stage Transformation Process:\n")
        md_lines.append("1. **Pivot to wide format** - Convert raw GFS data")
        md_lines.append(
            "2. **Comprehensive pressure gradients** - 8 directions + scales"
        )
        md_lines.append("3. **Surface gradients** - Temperature, solar, cloud, soil")
        md_lines.append("4. **Time derivatives** - Change rates and acceleration")
        md_lines.append("5. **Temporal lags** - 1h, 3h, 6h, 1d, 2d, 3d")
        md_lines.append("6. **Rolling statistics** - 3h, 6h, 12h, 24h, 3d, 7d windows")
        md_lines.append("7. **Wind shear** - Vertical gradients (100m - 10m)")
        md_lines.append("8. **Physics features** - Thermal indicators, stability")
        md_lines.append("9. **Cyclical encoding** - Hour/month sin/cos")
        md_lines.append("10. **GFS bias correction** - Historical error patterns")
        md_lines.append("11. **Categorical features** - Hour groups, season")
        md_lines.append("12. **Composite indices** - Multi-variable combinations")
        md_lines.append("13. **Data quality flags** - Missing data indicators\n")

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Markdown feature report saved to {output_path}")

    def _convert_to_json_serializable(self, obj):
        """
        Recursively convert numpy/pandas types to native Python types.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {
                key: self._convert_to_json_serializable(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj


def save_feature_list(df: pd.DataFrame, output_path: Path):
    """
    Save comprehensive feature list with descriptions.

    Args:
        df: DataFrame with engineered features
        output_path: Path to save feature list (text format)
    """
    feature_cols = [c for c in df.columns if c not in ["time", "forecast_made"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("WINDMOJO FEATURE LIST\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Features: {len(feature_cols)}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")

        # Group by category
        reporter = FeatureReport(df)
        categories = reporter._categorize_features()

        for category, features in sorted(categories.items()):
            if category == "COUNTS":
                continue

            f.write(f"\n{category} ({len(features)} features)\n")
            f.write("-" * 80 + "\n")

            for feature in sorted(features):
                # Get basic stats
                if feature in df.columns and df[feature].dtype in [
                    np.float64,
                    np.int64,
                ]:
                    stats = df[feature].describe()
                    f.write(f"  {feature}\n")
                    f.write(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n")
                    f.write(
                        f"    Mean ± Std: {stats['mean']:.2f} ± {stats['std']:.2f}\n"
                    )
                else:
                    f.write(f"  {feature}\n")

                f.write("\n")

    logger.info(f"Feature list saved to {output_path}")

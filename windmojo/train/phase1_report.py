"""
Phase 1 Model 1 Report Generation

Creates comprehensive analysis report from pilot training results.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_phase1_report(results_path: Path, importance_dir: Path, output_path: Path):
    """
    Generate comprehensive Phase 1 Model 1 report.

    Args:
        results_path: Path to phase1_model1_results.json
        importance_dir: Directory containing importance JSON files
        output_path: Path to save markdown report
    """
    logger.info("Generating Phase 1 Model 1 report...")

    # Load results
    with open(results_path, "r") as f:
        results = json.load(f)

    # Parse results
    hour_summaries = _extract_hour_summaries(results)
    top_features = _load_top_features(importance_dir)

    # Generate markdown report
    report = _create_markdown_report(hour_summaries, top_features)

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"  ✓ Report saved to {output_path}")

    # Also save JSON summary
    json_path = output_path.with_suffix(".json")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "hour_summaries": hour_summaries,
        "top_features_by_hour": top_features,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  ✓ JSON summary saved to {json_path}")


def _extract_hour_summaries(results: Dict) -> Dict:
    """Extract summary metrics for each hour."""
    summaries = {}

    for key, value in results.items():
        if key.endswith("_summary"):
            hour = key.replace("_summary", "")
            summaries[hour] = value

    return summaries


def _load_top_features(importance_dir: Path, top_n: int = 20) -> Dict[str, List]:
    """Load top N features for each hour."""
    top_features = {}

    importance_files = list(importance_dir.glob("**/importance_*.json"))

    for importance_file in importance_files:
        # Extract hour from filename: "importance_0800.json" → "08:00"
        hour = importance_file.stem.replace("importance_", "")
        hour_formatted = f"{hour[:2]}:{hour[2:]}"

        with open(importance_file, "r") as f:
            importance_data = json.load(f)

        features = importance_data["feature_importance"][:top_n]
        top_features[hour_formatted] = features

    return top_features


def _create_markdown_report(hour_summaries: Dict, top_features: Dict[str, List]) -> str:
    """Create markdown formatted report."""

    report = []

    report.append("# Phase 1: Model 1 Pilot Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n**Purpose:** Feature discovery with ALL 433 features\n")

    report.append("---\n")

    # Executive Summary
    report.append("## Executive Summary\n")
    report.append(
        "Phase 1 Model 1 trains on ALL 433 features with intentional overfitting "
    )
    report.append(
        "to discover which features matter most. This pilot uses Jan-Feb 2025 data "
    )
    report.append("(~60 days) with weekly walk-forward validation.\n")

    # Performance by Hour
    report.append("## Performance by Hour\n")
    report.append("| Hour | Test MAE | Test RMSE | Test R² | Folds |")
    report.append("|------|----------|-----------|---------|-------|")

    for hour in sorted(hour_summaries.keys()):
        summary = hour_summaries[hour]
        mae = summary["avg_test_mae"]
        rmse = summary["avg_test_rmse"]
        r2 = summary["avg_test_r2"]
        n_folds = len([k for k in summary.keys() if k.startswith("avg_")])

        report.append(
            f"| {hour} | {mae:.3f} ± {summary['std_test_mae']:.3f} | "
            f"{rmse:.3f} ± {summary['std_test_rmse']:.3f} | "
            f"{r2:.3f} ± {summary['std_test_r2']:.3f} | 4 |"
        )

    report.append("")

    # Analysis
    report.append("### Analysis\n")

    best_hour = min(hour_summaries.items(), key=lambda x: x[1]["avg_test_mae"])[0]
    worst_hour = max(hour_summaries.items(), key=lambda x: x[1]["avg_test_mae"])[0]

    best_mae = hour_summaries[best_hour]["avg_test_mae"]
    worst_mae = hour_summaries[worst_hour]["avg_test_mae"]

    report.append(f"- **Best performing hour:** {best_hour} (MAE: {best_mae:.3f} m/s)")
    report.append(
        f"- **Worst performing hour:** {worst_hour} (MAE: {worst_mae:.3f} m/s)"
    )
    report.append(
        f"- **Performance variation:** {worst_mae - best_mae:.3f} m/s difference\n"
    )

    # Top 20 Features Overall
    report.append("## Top 20 Features Overall\n")
    report.append("Aggregated feature importance across all hours:\n")

    overall_top_features = _aggregate_top_features(top_features, top_n=20)
    report.append("| Rank | Feature | Avg Gain | Category | Appears in Hours |")
    report.append("|------|---------|----------|----------|------------------|")

    for i, feat_info in enumerate(overall_top_features, 1):
        name = feat_info["feature"]
        avg_gain = feat_info["avg_gain"]
        category = feat_info["category"]
        hour_count = feat_info["hour_count"]

        report.append(
            f"| {i} | `{name}` | {avg_gain:.1f} | {category} | {hour_count} |"
        )

    report.append("")

    # Top Features by Category
    report.append("## Top Features by Category\n")
    report.append(
        "Top 10 features in each major category (aggregated across all hours):\n"
    )

    category_top_features = _get_top_features_by_category(top_features, top_n=10)

    for category in sorted(category_top_features.keys()):
        report.append(f"\n### {category}\n")
        report.append("| Rank | Feature | Avg Gain | Appears in Hours |")
        report.append("|------|---------|----------|------------------|")

        for i, feat_info in enumerate(category_top_features[category], 1):
            name = feat_info["feature"]
            avg_gain = feat_info["avg_gain"]
            hour_count = feat_info["hour_count"]

            report.append(f"| {i} | `{name}` | {avg_gain:.1f} | {hour_count} |")

        report.append("")

    # Note: Per-hour detailed rankings removed for brevity
    # Use feature_importance JSON files for hour-specific analysis if needed

    # Wind Direction Analysis
    wind_direction_analysis = _analyze_wind_directions(top_features)
    report.append(wind_direction_analysis)

    # Feature Category Analysis
    report.append("## Feature Category Analysis\n")

    category_stats = _analyze_feature_categories(top_features)

    report.append("Top 50 features by category:\n")
    report.append("| Category | Count | % of Top 50 |")
    report.append("|----------|-------|-------------|")

    for hour in sorted(category_stats.keys()):
        report.append(f"\n### {hour}\n")
        for category, count in sorted(
            category_stats[hour].items(), key=lambda x: -x[1]
        ):
            pct = count / 50 * 100
            report.append(f"| {category} | {count} | {pct:.1f}% |")

    report.append("")

    # Recommendations
    report.append("## Recommendations\n")

    report.append("### For Full Training:\n")
    report.append("1. Run full Model 1 training on all 268 days")
    report.append("2. Extract final feature importance rankings")
    report.append(
        "3. Identify features with zero importance (candidates for removal)\n"
    )

    report.append("### For Phase 2 Model 2:\n")

    # Calculate typical top-N thresholds
    avg_top50_pct = np.mean(
        [
            (
                top_features[hour][49]["cumulative_pct"]
                if len(top_features[hour]) > 49
                else 100
            )
            for hour in top_features.keys()
        ]
    )

    report.append(
        f"1. **Top 50 features** explain ~{avg_top50_pct:.1f}% of importance on average"
    )
    report.append(f"2. Consider keeping **top 75-100 features** for Model 2")
    report.append(f"3. Remove features with zero importance across all hours")
    report.append(f"4. Engineer interaction features based on top 10 features\n")

    report.append("### Data Quality Notes:\n")
    report.append("- Missing target values were skipped (rows with no measurement)")
    report.append("- Feature missing values handled by XGBoost natively")
    report.append("- Weekly walk-forward validation ensures no data leakage\n")

    # Appendix
    report.append("---\n")
    report.append("## Appendix: Model Configuration\n")
    report.append("```python")
    report.append("# Phase 1 Model 1 Parameters (Intentional Overfitting)")
    report.append("params = {")
    report.append("    'max_depth': 8,")
    report.append("    'learning_rate': 0.05,")
    report.append("    'n_estimators': 1000,")
    report.append("    'subsample': 1.0,")
    report.append("    'colsample_bytree': 1.0,  # Use ALL features")
    report.append("    'objective': 'reg:squarederror',")
    report.append("    'eval_metric': 'mae',")
    report.append("}")
    report.append("```\n")

    return "\n".join(report)


def _aggregate_top_features(
    top_features: Dict[str, List], top_n: int = 20
) -> List[Dict]:
    """
    Aggregate feature importance across all hours.

    Args:
        top_features: Dictionary of features by hour
        top_n: Number of top features to return

    Returns:
        List of feature dicts with avg_gain, category, hour_count
    """
    from collections import defaultdict

    # Aggregate gains across hours
    feature_gains = defaultdict(list)
    feature_categories = {}

    for hour, features in top_features.items():
        for feat in features:
            feature_name = feat["feature"]
            feature_gains[feature_name].append(feat["gain_importance"])
            feature_categories[feature_name] = feat.get("category", "UNKNOWN")

    # Calculate average gain and hour count
    aggregated = []
    for feature_name, gains in feature_gains.items():
        aggregated.append(
            {
                "feature": feature_name,
                "avg_gain": np.mean(gains),
                "category": feature_categories[feature_name],
                "hour_count": len(gains),
            }
        )

    # Sort by average gain
    aggregated.sort(key=lambda x: x["avg_gain"], reverse=True)

    return aggregated[:top_n]


def _get_top_features_by_category(
    top_features: Dict[str, List], top_n: int = 10
) -> Dict[str, List]:
    """
    Get top N features for each category.

    Args:
        top_features: Dictionary of features by hour
        top_n: Number of top features per category

    Returns:
        Dictionary mapping category -> list of top features
    """
    from collections import defaultdict

    # Aggregate by category
    category_features = defaultdict(lambda: defaultdict(list))

    for hour, features in top_features.items():
        for feat in features:
            category = feat.get("category", "UNKNOWN")
            feature_name = feat["feature"]
            category_features[category][feature_name].append(feat["gain_importance"])

    # Calculate average gain per category
    result = {}
    for category, features_dict in category_features.items():
        features_list = []
        for feature_name, gains in features_dict.items():
            features_list.append(
                {
                    "feature": feature_name,
                    "avg_gain": np.mean(gains),
                    "hour_count": len(gains),
                }
            )

        # Sort by average gain and take top N
        features_list.sort(key=lambda x: x["avg_gain"], reverse=True)
        result[category] = features_list[:top_n]

    return result


def _analyze_wind_directions(top_features: Dict[str, List]) -> str:
    """
    Analyze wind direction patterns in top features.

    Args:
        top_features: Dictionary of features by hour

    Returns:
        Markdown string with wind direction analysis
    """
    analysis = []

    analysis.append("## Wind Direction Analysis\n")
    analysis.append("Analysis of wind direction features in top 20:\n")

    # Find all wind direction features in top 20
    direction_features = defaultdict(list)

    for hour, features in sorted(top_features.items()):
        for i, feat in enumerate(features[:20], 1):
            feature_name = feat["feature"]

            # Check if it's a wind direction feature
            if (
                "direction" in feature_name.lower()
                or "wind_dir" in feature_name.lower()
            ):
                direction_features[hour].append(
                    {
                        "rank": i,
                        "feature": feature_name,
                        "gain": feat["gain_importance"],
                        "category": feat.get("category", "UNKNOWN"),
                    }
                )

    if direction_features:
        analysis.append("### Wind Direction Features by Hour\n")
        analysis.append(
            "| Hour | Count | Top Direction Feature | Rank | Gain | Category |"
        )
        analysis.append(
            "|------|-------|----------------------|------|------|----------|"
        )

        for hour in sorted(direction_features.keys()):
            feats = direction_features[hour]
            count = len(feats)

            if feats:
                top_feat = feats[0]  # Highest ranked
                analysis.append(
                    f"| {hour} | {count} | `{top_feat['feature']}` | "
                    f"{top_feat['rank']} | {top_feat['gain']:.1f} | {top_feat['category']} |"
                )

        analysis.append("")

        # Summary
        total_hours = len(top_features)
        hours_with_direction = len(direction_features)
        pct = hours_with_direction / total_hours * 100 if total_hours > 0 else 0

        analysis.append("### Summary\n")
        analysis.append(
            f"- **Hours with direction features in top 20**: {hours_with_direction}/{total_hours} ({pct:.0f}%)"
        )

        # Find most common direction features
        all_direction_features = []
        for hour_feats in direction_features.values():
            all_direction_features.extend([f["feature"] for f in hour_feats])

        from collections import Counter

        feature_counts = Counter(all_direction_features)

        if feature_counts:
            analysis.append(f"- **Most common direction features**:")
            for feat_name, count in feature_counts.most_common(5):
                analysis.append(f"  - `{feat_name}`: appears in {count} hours")
    else:
        analysis.append("*No wind direction features found in top 20 for any hour.*\n")
        analysis.append(
            "This suggests wind speed/magnitude features dominate importance."
        )

    analysis.append("")

    # Add compass direction gradient analysis
    analysis.append(_analyze_compass_gradients(top_features))

    return "\n".join(analysis)


def _analyze_compass_gradients(top_features: Dict[str, List]) -> str:
    """
    Analyze which of the 8 compass directions dominate in spatial gradients.

    Args:
        top_features: Dictionary of features by hour

    Returns:
        Markdown string with compass direction analysis
    """
    from collections import Counter, defaultdict

    lines = []
    lines.append("## Directional Gradient Patterns (8 Compass Directions)\n")
    lines.append(
        "Analysis of which compass directions dominate in spatial gradients:\n"
    )

    # 8 compass directions to look for
    compass_directions = ["_N_", "_S_", "_E_", "_W_", "_NE_", "_NW_", "_SE_", "_SW_"]
    direction_labels = {
        "_N_": "N",
        "_S_": "S",
        "_E_": "E",
        "_W_": "W",
        "_NE_": "NE",
        "_NW_": "NW",
        "_SE_": "SE",
        "_SW_": "SW",
    }

    # Aggregate all features across all hours (use top 50)
    all_features = []
    for hour, features in top_features.items():
        for feat in features[:50]:  # Top 50 features
            all_features.append(feat)

    # Count direction occurrences and sum importance
    direction_counts = Counter()
    direction_importance = defaultdict(float)

    for feat in all_features:
        feature_name = feat["feature"]
        gain = feat["gain_importance"]

        # Check for gradient patterns (exclude non-gradient features)
        if any(
            keyword in feature_name.lower()
            for keyword in ["grad", "diff", "curvature", "shear"]
        ):
            for pattern, label in direction_labels.items():
                if pattern in feature_name:
                    direction_counts[label] += 1
                    direction_importance[label] += gain
                    break  # Only count once per feature

    if direction_counts:
        lines.append("### Gradient Direction Summary\n")
        lines.append(
            "| Direction | Feature Count | Total Importance | Avg Importance | Rank |"
        )
        lines.append(
            "|-----------|---------------|------------------|----------------|------|"
        )

        # Sort by total importance
        sorted_directions = sorted(direction_importance.items(), key=lambda x: -x[1])

        for rank, (direction, total_imp) in enumerate(sorted_directions, 1):
            count = direction_counts[direction]
            avg_imp = total_imp / count if count > 0 else 0
            lines.append(
                f"| {direction:^9} | {count:^13} | {total_imp:^16.1f} | {avg_imp:^14.1f} | {rank:^4} |"
            )

        lines.append("")

        # Dominant direction analysis
        if sorted_directions:
            dominant = sorted_directions[0][0]
            dominant_count = direction_counts[dominant]
            lines.append(
                f"**Dominant gradient direction**: {dominant} ({dominant_count} features)"
            )
            lines.append("")

            # Interpretation hints
            lines.append("**Interpretation**:")
            if dominant in ["N", "S"]:
                lines.append(
                    f"- {dominant} gradients dominate → North-South pressure/temperature patterns important"
                )
            elif dominant in ["E", "W"]:
                lines.append(
                    f"- {dominant} gradients dominate → East-West pressure/temperature patterns important"
                )
            else:
                lines.append(
                    f"- {dominant} gradients dominate → Diagonal pressure/temperature patterns important"
                )
            lines.append("")

    else:
        lines.append("*No compass direction gradients found in top 50 features.*")
        lines.append(
            "This suggests non-gradient features dominate (raw values, lags, rolling stats)."
        )
        lines.append("")

    return "\n".join(lines)


def _analyze_feature_categories(
    top_features: Dict[str, List],
) -> Dict[str, Dict[str, int]]:
    """Count feature categories in top N features for each hour."""
    category_stats = {}

    for hour, features in top_features.items():
        categories = {}

        # Count top 50 features by category
        for feat in features[:50]:
            category = feat.get("category", "UNKNOWN")
            categories[category] = categories.get(category, 0) + 1

        category_stats[hour] = categories

    return category_stats

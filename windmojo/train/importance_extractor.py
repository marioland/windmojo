"""
Feature Importance Extraction

Extracts feature importance from Model 1 using:
- Gain: Total improvement in loss when splitting on feature
- SHAP: Individual prediction contributions (interpretability)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


def extract_importance(
    model: xgb.XGBRegressor, feature_columns: List[str], importance_type: str = "gain"
) -> pd.DataFrame:
    """
    Extract feature importance from trained Model 1.

    Args:
        model: Trained XGBoost model
        feature_columns: List of feature names
        importance_type: 'gain', 'cover', or 'weight'

    Returns:
        DataFrame with columns: feature, importance, rank
    """
    logger.info(f"Extracting {importance_type} importance...")

    # Get importance scores
    importance_dict = model.get_booster().get_score(importance_type=importance_type)

    # Create DataFrame
    importance_df = pd.DataFrame(
        [
            {"feature": feature, "importance": importance}
            for feature, importance in importance_dict.items()
        ]
    )

    # Some features may have zero importance (never used in splits)
    all_features = pd.DataFrame({"feature": feature_columns})
    importance_df = all_features.merge(importance_df, on="feature", how="left")
    importance_df["importance"] = importance_df["importance"].fillna(0)

    # Sort and rank
    importance_df = importance_df.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    # Calculate cumulative percentage
    total_importance = importance_df["importance"].sum()
    importance_df["cumulative_pct"] = (
        importance_df["importance"].cumsum() / total_importance * 100
    )

    logger.info(f"  ✓ Extracted importance for {len(importance_df)} features")
    logger.info(
        f"  Top feature: {importance_df.iloc[0]['feature']} (importance: {importance_df.iloc[0]['importance']:.1f})"
    )
    if len(importance_df) >= 50:
        logger.info(
            f"  Top 50 features explain: {importance_df.iloc[49]['cumulative_pct']:.1f}% of total importance"
        )
    if len(importance_df) >= 100:
        logger.info(
            f"  Top 100 features explain: {importance_df.iloc[99]['cumulative_pct']:.1f}% of total importance"
        )

    return importance_df


def calculate_shap_values(
    model: xgb.XGBRegressor,
    data: pd.DataFrame,
    feature_columns: List[str],
    max_samples: int = 500,
) -> Dict:
    """
    Calculate SHAP values for interpretability.

    Args:
        model: Trained XGBoost model
        data: Dataset to calculate SHAP on (typically test set)
        feature_columns: List of feature names
        max_samples: Limit samples for speed (SHAP is slow)

    Returns:
        Dict with SHAP values and summary
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Skipping SHAP calculation.")
        logger.warning("Install with: pip install shap")
        return None

    logger.info(f"Calculating SHAP values (this may take a while)...")

    # Limit samples for speed
    if len(data) > max_samples:
        logger.info(
            f"  Sampling {max_samples} of {len(data)} rows for SHAP calculation"
        )
        data_sample = data.sample(n=max_samples, random_state=42)
    else:
        data_sample = data

    X = data_sample[feature_columns]

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Calculate mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame
    shap_df = pd.DataFrame({"feature": feature_columns, "mean_abs_shap": mean_abs_shap})

    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(
        drop=True
    )
    shap_df["rank"] = range(1, len(shap_df) + 1)

    logger.info(f"  ✓ SHAP values calculated")
    logger.info(
        f"  Top feature: {shap_df.iloc[0]['feature']} (SHAP: {shap_df.iloc[0]['mean_abs_shap']:.3f})"
    )

    return {
        "shap_values": shap_values.tolist(),  # For JSON serialization
        "shap_ranking": shap_df.to_dict(orient="records"),
        "feature_names": feature_columns,
        "samples_used": len(data_sample),
    }


def combine_importance_rankings(
    gain_df: pd.DataFrame, shap_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Combine Gain and SHAP rankings into single DataFrame.

    Args:
        gain_df: DataFrame from extract_importance()
        shap_df: DataFrame from calculate_shap_values()['shap_ranking']

    Returns:
        Combined DataFrame with both rankings
    """
    # Include category column if it exists
    cols = ["feature", "importance", "rank", "cumulative_pct"]
    if "category" in gain_df.columns:
        cols.append("category")

    combined = gain_df[cols].copy()
    combined = combined.rename(
        columns={"importance": "gain_importance", "rank": "gain_rank"}
    )

    if shap_df is not None:
        shap_df = pd.DataFrame(shap_df) if isinstance(shap_df, list) else shap_df
        shap_subset = shap_df[["feature", "mean_abs_shap", "rank"]]
        shap_subset = shap_subset.rename(columns={"rank": "shap_rank"})

        combined = combined.merge(shap_subset, on="feature", how="left")

        # Calculate average rank
        combined["avg_rank"] = (
            combined["gain_rank"] + combined["shap_rank"].fillna(999)
        ) / 2
        combined = combined.sort_values("avg_rank").reset_index(drop=True)
        combined["combined_rank"] = range(1, len(combined) + 1)
    else:
        combined["combined_rank"] = combined["gain_rank"]

    return combined


def save_importance(
    importance_df: pd.DataFrame, output_path: Path, shap_data: Optional[Dict] = None
):
    """
    Save feature importance to JSON file.

    Args:
        importance_df: DataFrame with importance rankings
        output_path: Path to save JSON
        shap_data: Optional SHAP data dict
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use 'importance' or 'gain_importance' column depending on what's available
    imp_col = (
        "gain_importance"
        if "gain_importance" in importance_df.columns
        else "importance"
    )

    output_data = {
        "feature_importance": importance_df.to_dict(orient="records"),
        "summary": {
            "total_features": len(importance_df),
            "features_with_nonzero_importance": int((importance_df[imp_col] > 0).sum()),
            "top_10_explain_pct": (
                float(
                    importance_df.iloc[min(9, len(importance_df) - 1)]["cumulative_pct"]
                )
                if len(importance_df) >= 10
                else 100.0
            ),
            "top_50_explain_pct": (
                float(importance_df.iloc[49]["cumulative_pct"])
                if len(importance_df) > 49
                else 100.0
            ),
            "top_100_explain_pct": (
                float(importance_df.iloc[99]["cumulative_pct"])
                if len(importance_df) > 99
                else 100.0
            ),
        },
    }

    if shap_data:
        output_data["shap"] = {
            "samples_used": shap_data["samples_used"],
            "top_10_features": shap_data["shap_ranking"][:10],
        }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Feature importance saved to {output_path}")


def categorize_features(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature category labels based on naming patterns.

    Args:
        importance_df: DataFrame with feature importance

    Returns:
        DataFrame with added 'category' column
    """

    def get_category(feature_name: str) -> str:
        """Determine feature category from name."""
        name_lower = feature_name.lower()

        if any(x in name_lower for x in ["grad", "shear", "curvature"]):
            return "SPATIAL_GRADIENT"
        elif any(
            x in name_lower
            for x in ["lag_1h", "lag_3h", "lag_6h", "lag_1d", "lag_2d", "lag_3d"]
        ):
            return "TEMPORAL_LAG"
        elif any(x in name_lower for x in ["change_1h", "change_3h", "accel"]):
            return "TIME_DERIVATIVE"
        elif any(
            x in name_lower
            for x in ["rolling", "_std", "_max", "_min", "_sum", "_mean"]
        ):
            return "ROLLING_STATS"
        elif any(
            x in name_lower
            for x in [
                "stability",
                "richardson",
                "buoyancy",
                "heating",
                "cooling",
                "thermal",
                "soil_moisture",
                "gust_factor",
                "gust_excess",
            ]
        ):
            return "PHYSICS"
        elif any(x in name_lower for x in ["_sin", "_cos"]):
            return "CYCLICAL"
        elif any(x in name_lower for x in ["is_", "clear_sky_factor"]):
            return "BINARY"
        else:
            return "RAW"

    importance_df["category"] = importance_df["feature"].apply(get_category)

    return importance_df

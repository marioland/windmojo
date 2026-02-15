"""
Feature Categorization System

Categorizes synoptic features as:
- location_agnostic: Generic atmospheric patterns that work anywhere
- location_aware: Features that use specific geographic locations (Alps, Lake Como, etc.)
- borderline: Features that use regional names but represent generic patterns
- temporal: Time-based features (hour, day of year)

This enables feature selection strategies like:
- "Include at least N location-agnostic features"
- "Balance between location-aware and location-agnostic"
- Analysis of which category contributes most to model performance
"""

from typing import Dict, List

# Comprehensive feature categorization
FEATURE_CATEGORIES = {
    # ===== PRESSURE CENTER FEATURES (42 total) =====
    # All pressure center features are location-agnostic (they identify H/L anywhere)
    # High pressure systems (21 features)
    "high1_lat": "location_agnostic",
    "high1_lon": "location_agnostic",
    "high1_z500": "location_agnostic",
    "high1_dist_km": "location_aware",  # Distance to Lake Como
    "high1_bearing": "location_aware",  # Bearing from Lake Como
    "high1_gradient": "location_agnostic",
    "high1_exists": "location_agnostic",
    "high2_lat": "location_agnostic",
    "high2_lon": "location_agnostic",
    "high2_z500": "location_agnostic",
    "high2_dist_km": "location_aware",
    "high2_bearing": "location_aware",
    "high2_gradient": "location_agnostic",
    "high2_exists": "location_agnostic",
    "high3_lat": "location_agnostic",
    "high3_lon": "location_agnostic",
    "high3_z500": "location_agnostic",
    "high3_dist_km": "location_aware",
    "high3_bearing": "location_aware",
    "high3_gradient": "location_agnostic",
    "high3_exists": "location_agnostic",
    # Low pressure systems (21 features)
    "low1_lat": "location_agnostic",
    "low1_lon": "location_agnostic",
    "low1_z500": "location_agnostic",
    "low1_dist_km": "location_aware",
    "low1_bearing": "location_aware",
    "low1_gradient": "location_agnostic",
    "low1_exists": "location_agnostic",
    "low2_lat": "location_agnostic",
    "low2_lon": "location_agnostic",
    "low2_z500": "location_agnostic",
    "low2_dist_km": "location_aware",
    "low2_bearing": "location_aware",
    "low2_gradient": "location_agnostic",
    "low2_exists": "location_agnostic",
    "low3_lat": "location_agnostic",
    "low3_lon": "location_agnostic",
    "low3_z500": "location_agnostic",
    "low3_dist_km": "location_aware",
    "low3_bearing": "location_aware",
    "low3_gradient": "location_agnostic",
    "low3_exists": "location_agnostic",
    # ===== REGIONAL AGGREGATION FEATURES (64 original + 10 new = 74 total) =====
    # Iceland region (8 features) - location-agnostic
    "iceland_pressure_msl_mean": "location_agnostic",
    "iceland_pressure_msl_std": "location_agnostic",
    "iceland_z500_mean": "location_agnostic",
    "iceland_z500_std": "location_agnostic",
    "iceland_z850_mean": "location_agnostic",
    "iceland_z850_std": "location_agnostic",
    "iceland_temp_850_mean": "location_agnostic",
    "iceland_temp_850_std": "location_agnostic",
    # Azores region (8 features) - location-agnostic
    "azores_pressure_msl_mean": "location_agnostic",
    "azores_pressure_msl_std": "location_agnostic",
    "azores_z500_mean": "location_agnostic",
    "azores_z500_std": "location_agnostic",
    "azores_z850_mean": "location_agnostic",
    "azores_z850_std": "location_agnostic",
    "azores_temp_850_mean": "location_agnostic",
    "azores_temp_850_std": "location_agnostic",
    # Atlantic region (8 features) - location-agnostic
    "atlantic_pressure_msl_mean": "location_agnostic",
    "atlantic_pressure_msl_std": "location_agnostic",
    "atlantic_z500_mean": "location_agnostic",
    "atlantic_z500_std": "location_agnostic",
    "atlantic_z850_mean": "location_agnostic",
    "atlantic_z850_std": "location_agnostic",
    "atlantic_temp_850_mean": "location_agnostic",
    "atlantic_temp_850_std": "location_agnostic",
    # Scandinavia region (8 features) - location-agnostic
    "scandinavia_pressure_msl_mean": "location_agnostic",
    "scandinavia_pressure_msl_std": "location_agnostic",
    "scandinavia_z500_mean": "location_agnostic",
    "scandinavia_z500_std": "location_agnostic",
    "scandinavia_z850_mean": "location_agnostic",
    "scandinavia_z850_std": "location_agnostic",
    "scandinavia_temp_850_mean": "location_agnostic",
    "scandinavia_temp_850_std": "location_agnostic",
    # Central Europe region (8 features) - borderline
    "central_europe_pressure_msl_mean": "borderline",
    "central_europe_pressure_msl_std": "borderline",
    "central_europe_z500_mean": "borderline",
    "central_europe_z500_std": "borderline",
    "central_europe_z850_mean": "borderline",
    "central_europe_z850_std": "borderline",
    "central_europe_temp_850_mean": "borderline",
    "central_europe_temp_850_std": "borderline",
    # Mediterranean region (8 features) - location-aware
    "mediterranean_pressure_msl_mean": "location_aware",
    "mediterranean_pressure_msl_std": "location_aware",
    "mediterranean_z500_mean": "location_aware",
    "mediterranean_z500_std": "location_aware",
    "mediterranean_z850_mean": "location_aware",
    "mediterranean_z850_std": "location_aware",
    "mediterranean_temp_850_mean": "location_aware",
    "mediterranean_temp_850_std": "location_aware",
    # Alps North region (8 features) - location-aware
    "alps_north_pressure_msl_mean": "location_aware",
    "alps_north_pressure_msl_std": "location_aware",
    "alps_north_z500_mean": "location_aware",
    "alps_north_z500_std": "location_aware",
    "alps_north_z850_mean": "location_aware",
    "alps_north_z850_std": "location_aware",
    "alps_north_temp_850_mean": "location_aware",
    "alps_north_temp_850_std": "location_aware",
    # Alps South region (8 features) - location-aware
    "alps_south_pressure_msl_mean": "location_aware",
    "alps_south_pressure_msl_std": "location_aware",
    "alps_south_z500_mean": "location_aware",
    "alps_south_z500_std": "location_aware",
    "alps_south_z850_mean": "location_aware",
    "alps_south_z850_std": "location_aware",
    "alps_south_temp_850_mean": "location_aware",
    "alps_south_temp_850_std": "location_aware",
    # Cross-regional indices (8 original features)
    "nao_proxy": "location_agnostic",  # Azores-Iceland
    "foehn_index": "location_aware",  # Alps-specific
    "med_ce_gradient": "borderline",  # Uses named regions
    "scandi_blocking_strength": "borderline",
    "zonal_flow_strength": "borderline",
    "thermal_advection": "borderline",
    "alpine_thickness": "location_aware",  # Alps-specific
    "atlantic_ridge_strength": "borderline",
    # Location-agnostic gradient features (10 new features)
    "pressure_gradient_ns": "location_agnostic",
    "pressure_gradient_ew": "location_agnostic",
    "pressure_gradient_magnitude": "location_agnostic",
    "pressure_gradient_direction": "location_agnostic",
    "pressure_laplacian": "location_agnostic",
    "temperature_gradient_ns": "location_agnostic",
    "temperature_gradient_ew": "location_agnostic",
    "temperature_gradient_magnitude": "location_agnostic",
    "thermal_wind_strength": "location_agnostic",
    "baroclinicity_index": "location_agnostic",
    # ===== TELECONNECTION INDICES (9 features) =====
    # All teleconnection indices are location-agnostic
    "nao_index": "location_agnostic",
    "nao_positive": "location_agnostic",
    "ea_index": "location_agnostic",
    "scand_index": "location_agnostic",
    "scandi_blocking": "location_agnostic",
    "genoa_low_index": "location_aware",  # Genoa Low is location-specific
    "foehn_gradient": "location_aware",  # Alps-specific
    "alpine_thermal_stability": "location_aware",  # Alps-specific
    "breva_favorable": "location_aware",  # Lake Como-specific
    # ===== SYNOPTIC INTERACTION FEATURES (30 new features) =====
    # All interaction features are location-agnostic
    # High-low distances (9 features)
    "h1_l1_distance": "location_agnostic",
    "h1_l2_distance": "location_agnostic",
    "h1_l3_distance": "location_agnostic",
    "h2_l1_distance": "location_agnostic",
    "h2_l2_distance": "location_agnostic",
    "h2_l3_distance": "location_agnostic",
    "h3_l1_distance": "location_agnostic",
    "h3_l2_distance": "location_agnostic",
    "h3_l3_distance": "location_agnostic",
    # High-low pressure differences (9 features)
    "h1_l1_pressure_diff": "location_agnostic",
    "h1_l2_pressure_diff": "location_agnostic",
    "h1_l3_pressure_diff": "location_agnostic",
    "h2_l1_pressure_diff": "location_agnostic",
    "h2_l2_pressure_diff": "location_agnostic",
    "h2_l3_pressure_diff": "location_agnostic",
    "h3_l1_pressure_diff": "location_agnostic",
    "h3_l2_pressure_diff": "location_agnostic",
    "h3_l3_pressure_diff": "location_agnostic",
    # Strength ratios (5 features)
    "high1_high2_ratio": "location_agnostic",
    "low1_low2_ratio": "location_agnostic",
    "high_mean_strength": "location_agnostic",
    "low_mean_strength": "location_agnostic",
    "high_low_balance": "location_agnostic",
    # Pattern geometry (4 features)
    "high_dispersion": "location_agnostic",
    "low_dispersion": "location_agnostic",
    "pattern_lat_range": "location_agnostic",
    "pattern_lon_range": "location_agnostic",
    # Configuration counts (3 features)
    "northern_highs": "location_agnostic",
    "southern_lows": "location_agnostic",
    "north_south_ratio": "location_agnostic",
    # ===== TEMPORAL FEATURES (4 features) =====
    "hour_sin": "temporal",
    "hour_cos": "temporal",
    "day_of_year_sin": "temporal",
    "day_of_year_cos": "temporal",
}


def get_feature_category(feature_name: str) -> str:
    """
    Get the category of a feature.

    Args:
        feature_name: Name of the feature

    Returns:
        Category: 'location_agnostic', 'location_aware', 'borderline', or 'temporal'
        Returns 'unknown' if feature not in mapping
    """
    return FEATURE_CATEGORIES.get(feature_name, "unknown")


def get_features_by_category(category: str) -> List[str]:
    """
    Get all features in a specific category.

    Args:
        category: Category to filter by

    Returns:
        List of feature names in that category
    """
    return [feature for feature, cat in FEATURE_CATEGORIES.items() if cat == category]


def get_category_summary() -> Dict[str, int]:
    """
    Get count of features in each category.

    Returns:
        Dict of {category: count}
    """
    summary = {}
    for cat in ["location_agnostic", "location_aware", "borderline", "temporal"]:
        summary[cat] = len(get_features_by_category(cat))
    return summary


def analyze_feature_importance(
    feature_importances: Dict[str, float], top_n: int = 20
) -> Dict:
    """
    Analyze feature importance by category.

    Args:
        feature_importances: Dict of {feature_name: importance_value}
        top_n: Number of top features to analyze

    Returns:
        Dict with analysis results
    """
    # Sort by importance
    sorted_features = sorted(
        feature_importances.items(), key=lambda x: x[1], reverse=True
    )[:top_n]

    # Categorize top features
    category_counts = {
        "location_agnostic": 0,
        "location_aware": 0,
        "borderline": 0,
        "temporal": 0,
    }
    category_importance = {
        "location_agnostic": 0.0,
        "location_aware": 0.0,
        "borderline": 0.0,
        "temporal": 0.0,
    }

    for feature, importance in sorted_features:
        cat = get_feature_category(feature)
        if cat in category_counts:
            category_counts[cat] += 1
            category_importance[cat] += importance

    return {
        "top_features": sorted_features,
        "category_counts": category_counts,
        "category_importance": category_importance,
        "total_features": len(sorted_features),
    }

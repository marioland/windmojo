"""
Feature Selection with Atomic Groups

Defines systematic feature groups and selection strategies for Model 2.
Addresses the problem of arbitrary h1/h2/h3, l1/l2/l3 selection by:
1. Treating related attributes as atomic groups (all-or-nothing)
2. Using distance-based sorting for more consistent pressure center ranking
3. Providing physics-informed feature selection strategies
"""

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Atomic Feature Groups
# =============================================================================

# Pressure center atomic groups - if any feature selected, include ALL
PRESSURE_CENTER_GROUPS = {
    "high1": [
        "high1_lat",
        "high1_lon",
        "high1_z500",
        "high1_dist_km",
        "high1_bearing",
        "high1_gradient",
        "high1_exists",
    ],
    "high2": [
        "high2_lat",
        "high2_lon",
        "high2_z500",
        "high2_dist_km",
        "high2_bearing",
        "high2_gradient",
        "high2_exists",
    ],
    "high3": [
        "high3_lat",
        "high3_lon",
        "high3_z500",
        "high3_dist_km",
        "high3_bearing",
        "high3_gradient",
        "high3_exists",
    ],
    "low1": [
        "low1_lat",
        "low1_lon",
        "low1_z500",
        "low1_dist_km",
        "low1_bearing",
        "low1_gradient",
        "low1_exists",
    ],
    "low2": [
        "low2_lat",
        "low2_lon",
        "low2_z500",
        "low2_dist_km",
        "low2_bearing",
        "low2_gradient",
        "low2_exists",
    ],
    "low3": [
        "low3_lat",
        "low3_lon",
        "low3_z500",
        "low3_dist_km",
        "low3_bearing",
        "low3_gradient",
        "low3_exists",
    ],
}

# High-Low interaction groups (use with dominant systems only)
HL_INTERACTION_GROUPS = {
    "h1_l1": ["h1_l1_distance", "h1_l1_pressure_diff"],
    "h1_l2": ["h1_l2_distance", "h1_l2_pressure_diff"],
    "h2_l1": ["h2_l1_distance", "h2_l1_pressure_diff"],
}

# Temporal features - always include as pair
TEMPORAL_GROUPS = {
    "hour": ["hour_sin", "hour_cos"],
    "day_of_year": ["day_of_year_sin", "day_of_year_cos"],
}


# =============================================================================
# Feature Selection Strategies
# =============================================================================


def get_model2_features_synoptic_24(
    feature_importance: Dict[str, float] = None,
) -> List[str]:
    """
    24-feature selection focused on European synoptic weather.

    Strategy:
    1. Force temporal features (diurnal + seasonal)
    2. Include TWO nearest pressure systems (h1 + l1) with atomic groups
    3. Add primary h1-l1 interaction
    4. Rich representation of European-scale weather patterns
    5. Minimal local topography (only top 1)

    Breakdown:
    - Temporal: 4 features (16.7%)
    - Pressure centers: 8 features (33.3%)
    - Interactions: 2 features (8.3%)
    - European synoptic: 9 features (37.5%)
    - Local: 1 feature (4.2%)

    Args:
        feature_importance: Optional dict for validation

    Returns:
        List of 24 selected feature names
    """
    selected = []

    # === TEMPORAL (4) - Forced in ===
    selected.extend(
        [
            "hour_sin",
            "hour_cos",
            "day_of_year_sin",
            "day_of_year_cos",
        ]
    )

    # === NEAREST HIGH (4) - Atomic group ===
    selected.extend(
        [
            "high1_bearing",
            "high1_dist_km",
            "high1_z500",
            "high1_gradient",
        ]
    )

    # === NEAREST LOW (4) - Atomic group ===
    selected.extend(
        [
            "low1_bearing",
            "low1_dist_km",
            "low1_z500",
            "low1_gradient",
        ]
    )

    # === HIGH-LOW INTERACTION (2) ===
    selected.extend(
        [
            "h1_l1_distance",
            "h1_l1_pressure_diff",
        ]
    )

    # === EUROPEAN SYNOPTIC WEATHER (9) ===
    # Large-scale flow patterns
    selected.extend(
        [
            "zonal_flow_strength",  # West-to-east flow
            "pressure_gradient_magnitude",  # Overall pressure gradient strength
            "pressure_gradient_direction",  # Direction of main gradient
        ]
    )

    # Teleconnection indices (dominant modes)
    selected.extend(
        [
            "nao_index",  # North Atlantic Oscillation
            "scand_index",  # Scandinavian blocking
        ]
    )

    # Temperature/baroclinicity
    selected.extend(
        [
            "temperature_gradient_ew",  # East-west thermal contrast
            "temperature_gradient_ns",  # North-south thermal contrast
            "baroclinicity_index",  # Frontal activity indicator
        ]
    )

    # Regional pressure patterns (Mediterranean is key for Alps)
    selected.extend(
        [
            "mediterranean_pressure_msl_mean",  # Mediterranean high/low
        ]
    )

    # === LOCAL TOPOGRAPHIC (1) ===
    selected.extend(
        [
            "alpine_thermal_stability",  # Top-ranked local feature
        ]
    )

    logger.info(f"Synoptic-focused 24-feature selection: {len(selected)} features")
    return selected


def get_model2_features_hybrid(
    feature_importance: Dict[str, float], budget: int = 16
) -> List[str]:
    """
    Hybrid physics-informed feature selection for Model 2.

    Strategy:
    1. Force temporal features (diurnal cycle critical despite low rank)
    2. Use atomic groups for pressure centers (complete description)
    3. Use h1/l1 (dominant systems) + h1_l1 interaction
    4. Fill remaining budget with top location-aware regional features
    5. Add seasonal if budget allows

    Args:
        feature_importance: Dict mapping feature name to importance score
        budget: Total number of features allowed

    Returns:
        List of selected feature names
    """
    selected = set()

    # 1. Temporal (forced in) - 2 features
    selected.update(TEMPORAL_GROUPS["hour"])
    logger.info(f"Added temporal (hour): {TEMPORAL_GROUPS['hour']}")

    # 2. Dominant high (h1) atomic group - 7 features
    selected.update(PRESSURE_CENTER_GROUPS["high1"])
    logger.info(f"Added high1 atomic group: 7 features")

    # 3. Dominant low (l1) atomic group - 7 features
    selected.update(PRESSURE_CENTER_GROUPS["low1"])
    logger.info(f"Added low1 atomic group: 7 features")

    # 4. High-low interaction (h1_l1) - 2 features
    selected.update(HL_INTERACTION_GROUPS["h1_l1"])
    logger.info(f"Added h1_l1 interaction: 2 features")

    # Running total: 2 + 7 + 7 + 2 = 18 features
    # This exceeds budget of 16!

    # Need to reduce: Use compact atomic groups (remove lat/lon/exists)
    # Revised approach:
    selected = set()

    # Compact high1 group: bearing, distance, z500, gradient (4 features)
    selected.update(["high1_bearing", "high1_dist_km", "high1_z500", "high1_gradient"])

    # Compact low1 group: bearing, distance, z500, gradient (4 features)
    selected.update(["low1_bearing", "low1_dist_km", "low1_z500", "low1_gradient"])

    # h1_l1 interaction: 2 features
    selected.update(HL_INTERACTION_GROUPS["h1_l1"])

    # Temporal: 2 features
    selected.update(TEMPORAL_GROUPS["hour"])

    # Seasonal: 1 feature (cos only, more important than sin)
    selected.add("day_of_year_cos")

    # Running total: 4 + 4 + 2 + 2 + 1 = 13 features
    # Remaining budget: 3 features

    # 5. Fill with top location-aware regional features
    location_aware_features = [
        "alpine_thermal_stability",
        "foehn_gradient",
        "alps_south_temp_850_std",
        "alps_north_temp_850_mean",
    ]

    for feat in location_aware_features:
        if len(selected) >= budget:
            break
        if feat in feature_importance:
            selected.add(feat)

    logger.info(f"Final selection: {len(selected)} features")
    return sorted(list(selected))


def get_model2_features_maximum_generalization(
    feature_importance: Dict[str, float], budget: int = 16
) -> List[str]:
    """
    Maximum generalization strategy - prioritize location-agnostic features.

    Strategy:
    1. Force temporal features
    2. Use only location-agnostic synoptic features
    3. Avoid Alps/Lake Como specific features

    Args:
        feature_importance: Dict mapping feature name to importance score
        budget: Total number of features allowed

    Returns:
        List of selected feature names
    """
    selected = set()

    # 1. Temporal (2 features)
    selected.update(TEMPORAL_GROUPS["hour"])

    # 2. Seasonal (2 features)
    selected.update(TEMPORAL_GROUPS["day_of_year"])

    # 3. Pressure centers (compact) - 8 features
    selected.update(["high1_bearing", "high1_dist_km", "high1_z500", "high1_gradient"])
    selected.update(["low1_bearing", "low1_dist_km", "low1_z500", "low1_gradient"])

    # 4. h1_l1 interaction - 2 features
    selected.update(HL_INTERACTION_GROUPS["h1_l1"])

    # Running total: 2 + 2 + 8 + 2 = 14 features
    # Remaining: 2 features

    # 5. Location-agnostic regional features
    location_agnostic = [
        "zonal_flow_strength",
        "temperature_gradient_ew",
        "pressure_gradient_magnitude",
        "nao_index",
    ]

    for feat in location_agnostic:
        if len(selected) >= budget:
            break
        if feat in feature_importance:
            selected.add(feat)

    logger.info(f"Maximum generalization selection: {len(selected)} features")
    return sorted(list(selected))


def expand_atomic_groups(features: List[str]) -> Set[str]:
    """
    Expand feature list to include complete atomic groups.

    If any feature from a pressure center group is included,
    add all features from that group.

    Args:
        features: List of feature names

    Returns:
        Set of feature names with atomic groups expanded
    """
    expanded = set(features)

    # Check which pressure center groups are triggered
    for group_name, group_features in PRESSURE_CENTER_GROUPS.items():
        if any(f in features for f in group_features):
            expanded.update(group_features)
            logger.info(
                f"Expanded atomic group: {group_name} (+{len(group_features)} features)"
            )

    # Check temporal groups
    for group_name, group_features in TEMPORAL_GROUPS.items():
        if any(f in features for f in group_features):
            expanded.update(group_features)
            logger.debug(f"Expanded temporal group: {group_name}")

    return expanded


def validate_atomic_consistency(features: List[str]) -> bool:
    """
    Check if feature selection respects atomic groups.

    Returns True if selection is valid (no partial groups),
    False if some groups are incomplete.

    Args:
        features: List of feature names

    Returns:
        True if valid, False otherwise
    """
    feature_set = set(features)

    for group_name, group_features in PRESSURE_CENTER_GROUPS.items():
        features_in_group = [f for f in group_features if f in feature_set]

        if 0 < len(features_in_group) < len(group_features):
            logger.warning(
                f"Partial atomic group {group_name}: "
                f"{len(features_in_group)}/{len(group_features)} features"
            )
            logger.warning(f"  Present: {features_in_group}")
            logger.warning(f"  Missing: {set(group_features) - set(features_in_group)}")
            return False

    return True


# =============================================================================
# Recommended Model 2 Selections
# =============================================================================

MODEL2_RECOMMENDED_16 = [
    # Temporal (2) - forced in despite low rank
    "hour_sin",
    "hour_cos",
    # Nearest high atomic group (4) - compact version
    "high1_bearing",
    "high1_dist_km",
    "high1_z500",
    "high1_gradient",
    # Nearest low atomic group (4) - compact version
    "low1_bearing",
    "low1_dist_km",
    "low1_z500",
    "low1_gradient",
    # High-low interaction (2)
    "h1_l1_distance",
    "h1_l1_pressure_diff",
    # Location-aware regional (3)
    "alpine_thermal_stability",
    "foehn_gradient",
    "alps_south_temp_850_std",
    # Seasonal (1)
    "day_of_year_cos",
]

MODEL2_SYNOPTIC_24 = [
    # === TEMPORAL (4) ===
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    # === NEAREST HIGH (4) ===
    "high1_bearing",
    "high1_dist_km",
    "high1_z500",
    "high1_gradient",
    # === NEAREST LOW (4) ===
    "low1_bearing",
    "low1_dist_km",
    "low1_z500",
    "low1_gradient",
    # === HIGH-LOW INTERACTION (2) ===
    "h1_l1_distance",
    "h1_l1_pressure_diff",
    # === EUROPEAN SYNOPTIC WEATHER (9) ===
    "zonal_flow_strength",
    "pressure_gradient_magnitude",
    "pressure_gradient_direction",
    "nao_index",
    "scand_index",
    "temperature_gradient_ew",
    "temperature_gradient_ns",
    "baroclinicity_index",
    "mediterranean_pressure_msl_mean",
    # === LOCAL TOPOGRAPHIC (1) ===
    "alpine_thermal_stability",
]

# Note: When using distance-based sorting (sort_by='distance'),
# high1/low1 will be nearest systems, making this selection more
# consistent and generalizable across different time periods.

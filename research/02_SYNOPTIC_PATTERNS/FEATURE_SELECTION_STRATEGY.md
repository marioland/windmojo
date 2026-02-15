# Atomic Feature Selection Strategy

**Date:** 2025-01-21
**Author:** Windmojo Contributors
**Status:** Implemented

## Problem Statement

The current feature selection approach for Model 2 has three critical issues:

### 1. Arbitrary h1/h2/h3 Selection
Current Model 1b uses strength-based sorting where:
- `h1` = strongest high (highest z500)
- `l1` = strongest low (lowest z500)

However, feature importance rankings show:
- `low3_bearing` ranks #2 (importance: 1,226)
- `low1_bearing` ranks #173 (importance: 102)

This reveals that **the 3rd-strongest low is 12x more important than the strongest low**. Why?
- `l1` might be far away (e.g., Iceland low)
- `l3` might be weaker but **nearer** (e.g., Genoa low)
- Selecting "h2_l1_pressure_diff" vs "h1_l3_pressure_diff" becomes arbitrary and dataset-specific

### 2. Incomplete Feature Groups
Selecting individual features from pressure centers is physically incomplete:
- `high1_bearing` tells direction, but without `high1_dist_km` and `high1_z500`, we can't calculate geostrophic wind
- Taking bearing without distance is like having a compass but no map

### 3. Forced Temporal Features Despite Low Rank
- `hour_sin` ranks #172, `hour_cos` ranks #179 (very low importance)
- BUT: Diurnal wind cycles are **physically essential**
- Lake breezes occur 10-14:00, mountain valleys have night/day patterns
- Pure importance-based selection would miss this critical physics

## Solution: 3-Part Strategy

### Part 1: Distance-Based Sorting

**Change in `pressure_centers.py`:**

```python
def __init__(self, ..., sort_by: str = 'strength'):
    ...
    self.sort_by = sort_by  # 'strength' or 'distance'

def detect_and_parameterize(self, ...):
    # Calculate distance to target for ALL detected centers
    for center in highs + lows:
        center['distance_to_target'] = haversine(...)

    if self.sort_by == 'distance':
        # h1 = NEAREST high, l1 = NEAREST low
        highs = sorted(highs, key=lambda x: x['distance_to_target'])[:self.n_highs]
        lows = sorted(lows, key=lambda x: x['distance_to_target'])[:self.n_lows]
    else:  # 'strength'
        # h1 = STRONGEST high, l1 = STRONGEST low
        highs = sorted(highs, key=lambda x: x['z500'], reverse=True)[:self.n_highs]
        lows = sorted(lows, key=lambda x: x['z500'])[:self.n_lows]
```

**Rationale:**
- Nearest systems dominate local winds more than distant strong systems
- Distance-based sorting ensures `h1`/`l1` are **consistently nearest**, not arbitrarily strongest
- More generalizable across time periods and locations

**Configuration:**
```yaml
synoptic_features:
  pressure_centers:
    sort_by: 'distance'  # NEW parameter
```

### Part 2: Atomic Feature Groups

**Definition:** Related features must be selected **together** or **not at all**.

#### Pressure Center Atomic Groups

**Complete group** (7 features per center):
```python
HIGH1_FULL = [
    'high1_lat', 'high1_lon',          # Position
    'high1_z500',                       # Intensity
    'high1_dist_km', 'high1_bearing',  # Location-aware
    'high1_gradient',                   # Strength
    'high1_exists'                      # Flag
]
```

**Compact group** (4 features per center - recommended):
```python
HIGH1_COMPACT = [
    'high1_bearing',    # Direction from target to high
    'high1_dist_km',    # Distance to high
    'high1_z500',       # Intensity of high
    'high1_gradient'    # Strength of high
]
```

**Rationale:**
- Geostrophic wind ∝ pressure gradient / distance
- All 4 features are needed to estimate wind forcing from a pressure center
- Taking only `bearing` or only `distance` is physically incomplete

#### High-Low Interaction Groups

```python
H1_L1_INTERACTION = [
    'h1_l1_distance',      # Separation of primary systems
    'h1_l1_pressure_diff'  # Main synoptic gradient
]
```

**Rationale:**
- These two features together describe the primary synoptic-scale pressure gradient
- This gradient drives large-scale flow patterns

#### Temporal Groups

```python
HOUR_CYCLE = ['hour_sin', 'hour_cos']           # Diurnal
SEASONAL_CYCLE = ['day_of_year_sin', 'day_of_year_cos']  # Annual
```

**Rationale:**
- Sin/cos pairs encode periodic cycles without discontinuity at 0°/360°
- Both components needed to represent full cycle (sin for phase, cos for magnitude)

### Part 3: Model 2 Feature Selection (16 features)

Based on atomic grouping and distance-based sorting:

```python
MODEL2_RECOMMENDED_16 = [
    # === Temporal (2) - FORCED despite low rank ===
    'hour_sin',
    'hour_cos',

    # === Nearest High (4) - Atomic group ===
    'high1_bearing',     # Direction to nearest high
    'high1_dist_km',     # Distance to nearest high
    'high1_z500',        # Intensity of nearest high
    'high1_gradient',    # Strength of nearest high

    # === Nearest Low (4) - Atomic group ===
    'low1_bearing',      # Direction to nearest low
    'low1_dist_km',      # Distance to nearest low
    'low1_z500',         # Intensity of nearest low
    'low1_gradient',     # Strength of nearest low

    # === High-Low Interaction (2) ===
    'h1_l1_distance',       # Separation of primary systems
    'h1_l1_pressure_diff',  # Main synoptic gradient

    # === Location-Aware Regional (3) ===
    'alpine_thermal_stability',  # Rank #1, importance 1,337
    'foehn_gradient',            # Rank #4, importance 742
    'alps_south_temp_850_std',   # Rank #3, importance 944

    # === Seasonal (1) ===
    'day_of_year_cos',  # Rank #5, importance 717
]
```

**Category Breakdown:**
- **Location-agnostic:** 10 features (62.5%) - h1/l1 groups + h1_l1 interaction
- **Location-aware:** 3 features (18.75%) - Alps-specific
- **Temporal:** 3 features (18.75%) - hour + seasonal

**Physics Completeness:**
- Nearest high: Complete gradient description ✓
- Nearest low: Complete gradient description ✓
- Main synoptic gradient (h1-l1): Complete ✓
- Diurnal cycle: Complete (sin+cos) ✓
- Seasonal cycle: Partial (cos only, sin omitted due to budget)
- Local topography: Top 3 Alps features ✓

## Implementation

### Files Created/Modified

1. **`windmojo/features/pressure_centers.py`**
   - Added `sort_by` parameter to `__init__` (lines 34, 52)
   - Added distance calculation before sorting (lines 92-97)
   - Added conditional sorting logic (lines 100-109)

2. **`windmojo/features/synoptic_pipeline.py`**
   - Added `sort_by` parameter passthrough (line 57)

3. **`windmojo/features/feature_selection.py`** (NEW)
   - Atomic group definitions
   - Helper functions: `expand_atomic_groups()`, `validate_atomic_consistency()`
   - Selection strategies: `get_model2_features_hybrid()`, `get_model2_features_maximum_generalization()`
   - Recommended 16-feature set: `MODEL2_RECOMMENDED_16`

4. **`scripts/compare_sorting_methods.py`** (NEW)
   - Trains Model 1b with both `sort_by='strength'` and `sort_by='distance'`
   - Compares feature importance rankings
   - Generates comparison report

### Usage Example

```python
from windmojo.features.synoptic_pipeline import SynopticFeaturePipeline
from windmojo.features.feature_selection import MODEL2_RECOMMENDED_16, expand_atomic_groups

# Configure pipeline with distance-based sorting
config = {
    'center': {'latitude': 46.083333, 'longitude': 9.3},
    'synoptic_features': {
        'pressure_centers': {
            'sort_by': 'distance'  # Use nearest systems
        }
    }
}

pipeline = SynopticFeaturePipeline(config=config)

# Extract features
features_df = pipeline.transform(grid_df)

# Select Model 2 features (with atomic groups)
model2_features = MODEL2_RECOMMENDED_16

# Validate atomic consistency
from windmojo.features.feature_selection import validate_atomic_consistency
assert validate_atomic_consistency(model2_features), "Incomplete atomic groups!"

# Train Model 2
X = features_df[model2_features]
y = target_df['avgspeed']
model.fit(X, y)
```

## Benefits

### 1. Consistency
**Before:** `h1` might be Iceland high (1500km away) in one sample, Azores high (2500km away) in another
**After:** `h1` is always the nearest high, providing consistent meaning across all samples

### 2. Generalization
**Before:** Selecting "h2_l1_pressure_diff" is arbitrary (why not h1_l3?)
**After:** Always use h1_l1 (nearest to nearest) - principled, not arbitrary

### 3. Physical Completeness
**Before:** Might select `high1_bearing` without `high1_dist_km` - incomplete gradient
**After:** Atomic groups ensure we get bearing + distance + intensity together

### 4. Physics-Informed
**Before:** Pure importance would exclude hour_sin/cos (ranked #172-179)
**After:** Force temporal features despite low rank - they encode essential diurnal physics

## Evidence from Current Model 1b

Looking at the feature importance report (`model1b_synoptic_report.md`):

**Top 10 features (strength-based sorting):**
1. alpine_thermal_stability (1,337)
2. **low3_bearing (1,226)** ← 3rd strongest low!
3. alps_south_temp_850_std (944)
4. foehn_gradient (742)
5. day_of_year_cos (717)
6. foehn_index (648)
7. **h2_l1_pressure_diff (615)** ← 2nd strongest high, 1st strongest low
8. day_of_year_sin (516)
9. **h2_l3_pressure_diff (496)** ← 2nd strongest high, 3rd strongest low
10. low2_gradient (476)

**Key observations:**
- `low3` appears in ranks #2, #9 (very important)
- `low1` appears nowhere in top 20 (less important)
- `h2` appears in ranks #7, #9 (important)
- `h1` appears rarely

**Interpretation:**
- Strength ≠ importance for local winds
- Distance matters more than intensity
- Distance-based sorting would make `h1`/`l1` consistently important

## Alternative Strategies

### Maximum Generalization (16 features)
For training on Lake Como but deploying elsewhere:

```python
MODEL2_MAXIMUM_GENERALIZATION = [
    # Temporal (4)
    'hour_sin', 'hour_cos',
    'day_of_year_sin', 'day_of_year_cos',

    # Pressure centers (8)
    'high1_bearing', 'high1_dist_km', 'high1_z500', 'high1_gradient',
    'low1_bearing', 'low1_dist_km', 'low1_z500', 'low1_gradient',

    # Interactions (2)
    'h1_l1_distance', 'h1_l1_pressure_diff',

    # Location-agnostic regional (2)
    'zonal_flow_strength',
    'temperature_gradient_ew',
]
```

**Trade-off:** Better generalization, but loses Alps-specific knowledge (lower accuracy at Lake Como)

### Pure Location-Agnostic (14 features)
Completely remove location-specific features:

```python
MODEL2_PURE_AGNOSTIC = [
    # Temporal (4)
    'hour_sin', 'hour_cos',
    'day_of_year_sin', 'day_of_year_cos',

    # Pressure centers (8)
    'high1_bearing', 'high1_dist_km', 'high1_z500', 'high1_gradient',
    'low1_bearing', 'low1_dist_km', 'low1_z500', 'low1_gradient',

    # Interactions (2)
    'h1_l1_distance', 'h1_l1_pressure_diff',
]
```

**Trade-off:** Maximum generalization, significant accuracy loss at Lake Como

## Testing Strategy

To validate this approach:

1. **Train Model 1b with distance-based sorting**
   ```bash
   # Modify config/lake_como/dervio.yaml
   synoptic_features:
     pressure_centers:
       sort_by: 'distance'

   # Retrain
   uv run python scripts/train_model1b_synoptic.py
   ```

2. **Compare feature importance rankings**
   - Check if `high1`/`low1` move up in importance
   - Check if `high2`/`high3`/`low2`/`low3` become less arbitrary

3. **Train Model 2 with recommended 16 features**
   - Use distance-based sorting
   - Select `MODEL2_RECOMMENDED_16`
   - Compare performance vs full Model 1b

4. **Test generalization**
   - Train at Lake Como
   - Test at different location (if data available)
   - Measure performance degradation

## Recommendations

1. **For Model 2 at Lake Como:** Use `MODEL2_RECOMMENDED_16`
   - Balances accuracy and feature count
   - Includes critical Alps features
   - Uses atomic groups for consistency

2. **For generalization:** Use `MODEL2_MAXIMUM_GENERALIZATION`
   - Removes Alps-specific features
   - All features should work anywhere
   - Accept moderate accuracy loss

3. **Always use distance-based sorting** for Model 2
   - More consistent than strength-based
   - Better matches feature importance patterns
   - More generalizable

4. **Validate atomic consistency** before training
   ```python
   from windmojo.features.feature_selection import validate_atomic_consistency
   assert validate_atomic_consistency(selected_features)
   ```

## Future Work

1. **Test distance-based sorting empirically**
   - Retrain Model 1b with `sort_by='distance'`
   - Compare test MAE/RMSE
   - Analyze feature importance shifts

2. **Create ensemble of sorting methods**
   - Include features from both `sort_by='strength'` and `sort_by='distance'`
   - Let model learn which to weight higher

3. **Optimize atomic groups**
   - Test if 4-feature compact groups are sufficient
   - Or if full 7-feature groups improve accuracy

4. **Cross-location validation**
   - Train at Lake Como, test at Lake Garda
   - Measure generalization with location-agnostic features

5. **Dynamic pressure center weighting**
   - Instead of fixed h1/l1/h2/l2, use distance-weighted ensemble
   - `wind ~ Σ (pressure_gradient[i] / distance[i]²)` for all detected centers

##Files Reference

- **Implementation:** `windmojo/features/pressure_centers.py:28-52, 92-109`
- **Pipeline:** `windmojo/features/synoptic_pipeline.py:52-58`
- **Selection Module:** `windmojo/features/feature_selection.py`
- **Comparison Script:** `scripts/compare_sorting_methods.py`
- **Current Report:** `reports/model1b_synoptic_report.md`

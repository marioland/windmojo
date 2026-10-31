
# Synoptic weather feature engineering

> **Historical (2026-10-31):** design of the synoptic grid experiment (Model 1b). The
> approach was discontinued (see [SYNOPTIC_ONLY_LESSONS_LEARNED.md](SYNOPTIC_ONLY_LESSONS_LEARNED.md))
> and its code (synoptic downloaders, pressure centers, teleconnections, regional
> aggregation) was removed; it remains in the git history.

This summary addresses the suitability of the chosen data parameters (coverage, spatial resolution, and temporal resolution) for effectively analyzing the five key Central European wind indicators previously discussed.

---

## 🌍 Data Parameters & Efficiency

The selected parameters create a large, manageable grid suitable for synoptic-scale analysis.

| Parameter | Value | Assessment |
| :--- | :--- | :--- |
| **Coverage (Spatial Extent)** | $30^\circ\text{N}-75^\circ\text{N}$, $25^\circ\text{W}-40^\circ\text{E}$ | **Excellent.** Captures all critical centers of action (Azores, Iceland/NAO points, Mediterranean, Scandinavia). |
| **Spatial Resolution** | $0.4^\circ$ | **Ideal.** Provides approximately **18,532 points** per time step, offering enough detail to map system centers and gradients without excessive data load. |
| **Temporal Resolution** | **Single Daily Snapshot (12:00 UTC)** | **Major Constraint.** This is the primary limiting factor for analyzing dynamic wind events. |

---

## ✅ Spatial Performance: Indicators Fully Covered

The combined wide coverage and $0.4^\circ$ resolution provide an excellent spatial basis for *all* five indicators. You will be able to accurately locate and map the shape of these features at 12:00 UTC.

1.  **NAO Index (1):** The necessary pressure points (Iceland/Azores) are well within the grid.
2.  **Central European Pressure Anomaly (2):** The fine resolution is excellent for precise pressure gradient analysis across Central Europe.
3.  **Azores High Extension & Position (3):** The $25^\circ\text{W}$ boundary allows tracking the high's influence and ridging towards the continent.
4.  **Scandinavian/Baltic High Presence (4):** The $75^\circ\text{N}$ coverage ensures the high and its associated easterly flow component are fully captured.

---

## 🔑 Conclusion: The Trade-Off

The current parameters prioritize **spatial resolution** and **broad coverage** over **temporal accuracy**.

* **Result:** The system is highly effective for determining the **general, slow-changing synoptic pattern** (the position of large, stable Highs).
* **Limitation:** It is **insufficient** for reliably analyzing the **dynamics, maximum intensity, or precise movement (track)** of fast-moving Low-Pressure Systems that generate the most significant wind events.

To make the overall analysis reliable for all five indicators, especially the wind-generating Lows, the **temporal resolution should be increased** to at least $12$ hours (00:00 UTC and 12:00 UTC) or ideally **6 hours** (00, 06, 12, 18 UTC).

---

## 🎯 Model 1b: From Grid to XGBoost Features

### Strategy: Lunch-Time Synoptic Features for Afternoon Wind Prediction

**Key Insight:** For Lake Como thermal winds (Breva/Tivano), the **12:00 UTC synoptic situation** determines afternoon wind strength. We use the 18,532-point pressure grid to extract ~30-50 features that capture large-scale weather patterns.

---

## 📊 Grid-to-Features Transformation Pipeline

### Input: 18,532 Grid Points @ 12:00 UTC
- **Coverage:** 30°N-75°N, 25°W-40°E
- **Resolution:** 0.4° (~44 km spacing)
- **Variables:** `geopotential_height_500hPa`, `geopotential_height_850hPa`, `temperature_850hPa`, `pressure_msl`

### Output: ~40 Synoptic Features
Three complementary approaches extract different aspects of the weather pattern.

---

## 🔧 Approach #1: Pressure Center Parameterization

**Goal:** Detect and characterize individual highs/lows in the pressure field.

### Algorithm

```python
def detect_pressure_centers(grid_data):
    """
    Detect local maxima/minima in geopotential height field.
    Uses 500 hPa geopotential as it's less affected by topography.
    """
    # 1. Smooth field to remove noise
    z500_smooth = gaussian_filter(grid_data['z500'], sigma=2)

    # 2. Find local maxima (highs) and minima (lows)
    highs = find_local_maxima(z500_smooth, min_distance=5, threshold=5500)
    lows = find_local_minima(z500_smooth, min_distance=5, threshold=5600)

    # 3. Rank by intensity
    highs = sorted(highs, key=lambda h: h['z500'], reverse=True)[:3]  # Top 3 highs
    lows = sorted(lows, key=lambda l: l['z500'])[:3]  # Top 3 lows

    return highs, lows

def parameterize_center(center, center_type='high'):
    """
    Extract features from a pressure center.
    """
    features = {}

    # Position
    features[f'{center_type}1_lat'] = center['lat']
    features[f'{center_type}1_lon'] = center['lon']

    # Intensity
    features[f'{center_type}1_z500'] = center['z500']
    features[f'{center_type}1_anomaly'] = center['z500'] - climatology_mean(center['lat'], center['lon'])

    # Geometry (fit ellipse to closed contour)
    contour = find_closed_contour(center, threshold=20)  # 20 gpm contour
    features[f'{center_type}1_area_km2'] = calculate_area(contour)
    features[f'{center_type}1_eccentricity'] = fit_ellipse(contour)['eccentricity']
    features[f'{center_type}1_orientation'] = fit_ellipse(contour)['angle']

    # Gradient (pressure tightness)
    features[f'{center_type}1_max_gradient'] = max_gradient_around_center(center, radius=500km)

    # Distance to Lake Como (45.96°N, 9.25°E)
    features[f'{center_type}1_dist_to_como'] = haversine_distance(center['lat'], center['lon'], 45.96, 9.25)
    features[f'{center_type}1_bearing_to_como'] = bearing(center['lat'], center['lon'], 45.96, 9.25)

    return features
```

### Features Generated (12 per high, 12 per low, top 3 each)

**For strongest High:**
- `high1_lat`, `high1_lon` - Position
- `high1_z500`, `high1_anomaly` - Intensity
- `high1_area_km2`, `high1_eccentricity`, `high1_orientation` - Shape
- `high1_max_gradient` - Tightness
- `high1_dist_to_como`, `high1_bearing_to_como` - Relation to prediction site
- `high1_exists` - Binary flag (1 if detected, 0 otherwise)
- `high1_moving_speed`, `high1_moving_direction` - Dynamics (if temporal data available)

**Repeat for `high2`, `high3`, `low1`, `low2`, `low3`**

**Total from Approach #1:** ~36 features (3 highs × 12 + 3 lows × 12)

---

## 🌍 Approach #2: Regional Aggregation

**Goal:** Extract regional pressure statistics and cross-regional gradients.

### Regions Defined

```python
REGIONS = {
    'iceland': {'lat': (63, 67), 'lon': (-25, -15)},      # Iceland Low region
    'azores': {'lat': (35, 40), 'lon': (-30, -20)},       # Azores High region
    'atlantic': {'lat': (45, 55), 'lon': (-20, -5)},      # Atlantic approach
    'scandinavia': {'lat': (60, 70), 'lon': (5, 25)},     # Scandinavian High
    'central_europe': {'lat': (47, 53), 'lon': (5, 20)},  # Central Europe
    'mediterranean': {'lat': (38, 45), 'lon': (5, 20)},   # Mediterranean
    'alps_north': {'lat': (47, 49), 'lon': (6, 14)},      # North of Alps
    'alps_south': {'lat': (44, 46), 'lon': (7, 13)},      # South of Alps (includes Como)
}
```

### Feature Calculations

```python
def regional_features(grid_data, region_name, region_bounds):
    """
    Calculate aggregate statistics for a region.
    """
    # Extract region subset
    mask = (grid_data['lat'] >= region_bounds['lat'][0]) & \
           (grid_data['lat'] <= region_bounds['lat'][1]) & \
           (grid_data['lon'] >= region_bounds['lon'][0]) & \
           (grid_data['lon'] <= region_bounds['lon'][1])

    region_data = grid_data[mask]

    features = {
        f'{region_name}_pressure_mean': region_data['pressure_msl'].mean(),
        f'{region_name}_pressure_std': region_data['pressure_msl'].std(),
        f'{region_name}_z500_mean': region_data['z500'].mean(),
        f'{region_name}_temp_850_mean': region_data['temp_850'].mean(),
    }

    return features

def cross_regional_features(regional_stats):
    """
    Calculate meteorologically meaningful gradients.
    """
    features = {}

    # NAO Proxy: Azores - Iceland pressure difference
    features['nao_proxy'] = (regional_stats['azores_pressure_mean'] -
                             regional_stats['iceland_pressure_mean']) / 100

    # Foehn Index: South - North Alps gradient (CRITICAL for Lake Como!)
    features['foehn_index'] = (regional_stats['alps_south_pressure_mean'] -
                               regional_stats['alps_north_pressure_mean'])

    # Mediterranean-Central Europe gradient
    features['med_ce_gradient'] = (regional_stats['mediterranean_pressure_mean'] -
                                   regional_stats['central_europe_pressure_mean'])

    # Scandinavian blocking indicator
    features['scandi_blocking_strength'] = (regional_stats['scandinavia_pressure_mean'] -
                                            regional_stats['central_europe_pressure_mean'])

    # Atlantic-Europe flow
    features['zonal_flow_strength'] = (regional_stats['atlantic_z500_mean'] -
                                       regional_stats['central_europe_z500_mean'])

    # Thermal gradient (850 hPa temperature difference)
    features['thermal_advection'] = (regional_stats['mediterranean_temp_850_mean'] -
                                     regional_stats['central_europe_temp_850_mean'])

    return features
```

### Features Generated

**Per-region stats (8 regions × 4 variables):** 32 features
**Cross-regional gradients:** 6 features

**Total from Approach #2:** ~38 features

---

## 📈 Approach #4: Teleconnection Indices

**Goal:** Calculate standard meteorological indices from grid data.

### NAO (North Atlantic Oscillation)

```python
def calculate_nao_index(grid_data):
    """
    Standard NAO: Pressure difference between Azores and Iceland.
    Uses official definition points.
    """
    # Standard points
    p_azores = get_point_value(grid_data, lat=38, lon=-26, var='pressure_msl')
    p_iceland = get_point_value(grid_data, lat=65, lon=-20, var='pressure_msl')

    # Raw difference
    nao_raw = (p_azores - p_iceland) / 100  # Convert to hPa, scale to ~10

    # Normalized (requires climatology - use historical mean/std)
    # nao_normalized = (nao_raw - historical_mean) / historical_std

    return {
        'nao_index': nao_raw,
        'nao_positive': 1 if nao_raw > 0 else 0,  # Binary NAO phase
    }
```

### EA (East Atlantic Pattern)

```python
def calculate_ea_index(grid_data):
    """
    East Atlantic pattern: Captures north-south dipole shifted east of NAO.
    """
    # Approximate EA centers
    p_north = get_point_value(grid_data, lat=55, lon=-10, var='pressure_msl')  # UK/Ireland
    p_south = get_point_value(grid_data, lat=40, lon=-10, var='pressure_msl')  # Portugal

    ea_index = (p_north - p_south) / 100

    return {
        'ea_index': ea_index,
    }
```

### SCAND (Scandinavian Pattern)

```python
def calculate_scand_index(grid_data):
    """
    Scandinavian blocking pattern: High over Scandinavia.
    """
    # Center over Scandinavia
    p_scandi = get_point_value(grid_data, lat=65, lon=20, var='pressure_msl')

    # Reference: Central Europe pressure
    p_central = get_point_value(grid_data, lat=50, lon=10, var='pressure_msl')

    scand_index = (p_scandi - p_central) / 100

    return {
        'scand_index': scand_index,
        'scandi_blocking': 1 if scand_index > 10 else 0,  # >10 hPa difference
    }
```

### Alpine-Specific Indices

```python
def calculate_alpine_indices(grid_data):
    """
    Custom indices for Alpine wind prediction.
    """
    # Genoa Low Index (Mediterranean cyclogenesis)
    p_genoa = get_point_value(grid_data, lat=44.4, lon=8.9, var='pressure_msl')
    p_milan = get_point_value(grid_data, lat=45.5, lon=9.2, var='pressure_msl')
    genoa_low_strength = (p_milan - p_genoa)

    # Foehn Precondition: South-north pressure gradient
    p_south_alps = get_point_value(grid_data, lat=44, lon=9, var='pressure_msl')
    p_north_alps = get_point_value(grid_data, lat=48, lon=9, var='pressure_msl')
    foehn_gradient = (p_south_alps - p_north_alps)

    # Thermal stability (850 hPa - surface temperature difference)
    t_850 = get_point_value(grid_data, lat=46, lon=9, var='temp_850')
    t_surface = get_point_value(grid_data, lat=46, lon=9, var='temp_2m')
    thermal_stability = (t_850 - t_surface) / 1.5  # Lapse rate per km

    return {
        'genoa_low_index': genoa_low_strength,
        'foehn_gradient': foehn_gradient,
        'alpine_thermal_stability': thermal_stability,
        'breva_favorable': 1 if (foehn_gradient < -2 and thermal_stability < -6) else 0,
    }
```

### Features Generated

- `nao_index`, `nao_positive` (2)
- `ea_index` (1)
- `scand_index`, `scandi_blocking` (2)
- `genoa_low_index`, `foehn_gradient`, `alpine_thermal_stability`, `breva_favorable` (4)

**Total from Approach #4:** ~9 features

---

## 🎁 Complete Feature Set for Model 1b

### Summary Table

| Category | Approach | # Features | Description |
|----------|----------|------------|-------------|
| **Pressure Centers** | #1 Parameterization | 36 | Position, intensity, shape of top 3 highs/lows |
| **Regional Stats** | #2 Aggregation | 38 | Regional means, std, cross-regional gradients |
| **Teleconnection** | #4 Indices | 9 | NAO, EA, SCAND, Alpine-specific indices |
| **TOTAL** | | **~83** | Complete synoptic feature set |

### Recommended Subset for Initial Model 1b

**Start with most interpretable features (40 total):**

1. **Pressure Centers (18):**
   - Top 2 highs: position, intensity, distance to Como (2 × 6 = 12)
   - Top 1 low: position, intensity, distance to Como (1 × 6 = 6)

2. **Regional Gradients (13):**
   - NAO proxy, Foehn index, Med-CE gradient, Scandi blocking, Zonal flow, Thermal advection
   - Regional means: alps_south, alps_north, mediterranean, central_europe, scandinavia, iceland, azores (7)

3. **Teleconnection Indices (9):**
   - NAO, EA, SCAND, Genoa Low, Foehn gradient, Thermal stability, Breva favorable, NAO positive, Scandi blocking

---

## 🚀 Implementation Roadmap

### Phase 1: Data Collection (1 day)
1. Download 18,532-point grid from Open-Meteo for Lake Como region
2. Variables: `geopotential_height_500hPa`, `geopotential_height_850hPa`, `temperature_850hPa`, `pressure_msl`
3. Time: 12:00 UTC daily for training period (Jan-Sep 2024, 269 days)
4. Storage: `data/lake_como/synoptic_grid/grid_12utc_YYYYMMDD.nc` (NetCDF format)

### Phase 2: Feature Engineering (1-2 days)
1. Implement three feature extraction functions:
   - `extract_pressure_centers()` → 18 features
   - `extract_regional_stats()` → 13 features
   - `extract_teleconnection_indices()` → 9 features
2. Create pipeline: Grid → 40 features
3. Output: `data/lake_como/dervio/features/dervio_synoptic_features.parquet`

### Phase 3: Model 1b Training (2-3 hours)
1. Train XGBoost with ONLY these 40 synoptic features
2. Target: `dervio speed`
3. Scope: Full (269 days)
4. Hours: 08:00-18:00 UTC (13 hourly models)
5. Extract feature importance to determine which synoptic features matter most

### Phase 4: Evaluation
1. Compare Model 1b (40 synoptic) vs Model 1a (433 mixed features)
2. Identify high-ranking synoptic features
3. Combine top synoptic features with top local features for final model

---

## 💡 Expected Insights

**Questions Model 1b will answer:**

1. **Do large-scale patterns matter for local wind?**
   - If MAE is competitive with local features → Yes, synoptic forcing is important
   - If MAE is poor → Local effects dominate

2. **Which patterns drive Lake Como winds?**
   - High feature importance for Foehn gradient → Breva driven by Alpine pressure gradient
   - High importance for NAO → Large-scale Atlantic flow modulates local winds
   - High importance for Genoa Low → Mediterranean cyclones enhance thermal circulation

3. **Can we simplify the model?**
   - If only 10-15 of 40 features rank high → Use sparse synoptic representation
   - If all features matter → Complex synoptic situation, need full characterization

4. **Is 12:00 UTC sufficient?**
   - If temporal derivatives are important (not available with single snapshot) → Need multi-time analysis
   - If static pattern features work well → Single daily snapshot is adequate
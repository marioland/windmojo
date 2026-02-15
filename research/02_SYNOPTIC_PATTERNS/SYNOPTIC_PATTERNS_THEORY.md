# Synoptic Patterns Theory for Central European Wind Prediction

**Understanding and encoding large-scale weather patterns for machine learning applications**

This guide explains the synoptic-scale weather patterns that control wind in Central Europe and provides theoretical frameworks for encoding these spatial patterns into features suitable for tree-based machine learning models like XGBoost.

---

## Table of Contents

1. [Top European Weather Indicators](#top-european-weather-indicators)
2. [Top Central European Wind Indicators](#top-central-european-wind-indicators)
3. [Classic Central European Wind Patterns](#classic-central-european-wind-patterns)
4. [Encoding Synoptic Patterns for Machine Learning](#encoding-synoptic-patterns-for-machine-learning)
5. [Feature Engineering Approaches](#feature-engineering-approaches)

---

## Top European Weather Indicators

These are the **top 10 weather indicators** that best capture the general synoptic weather situation over Europe:

### 1. Azores High Position & Strength
- Controls Atlantic storm track into Europe
- Position (extended toward Europe vs. retracted) determines if weather systems hit Iberia, UK, or are deflected north
- Summer extension brings Mediterranean heat; winter retreat allows storms

### 2. Icelandic Low Position & Depth
- Partner to Azores High in determining NAO state
- Deep + northeast position = strong westerlies, mild/wet in Northern Europe
- Weak/displaced = blocked patterns, cold potential

### 3. NAO Index (North Atlantic Oscillation)
- Single most important number for European weather
- Positive = strong westerlies, storms hit northern Europe
- Negative = blocked patterns, storms hit Mediterranean, cold outbreaks possible

### 4. Scandinavian/Baltic High Presence
- "Scandi blocking" - crucial for cold outbreaks
- Acts as wall deflecting Atlantic systems south
- Often brings "Beast from the East" scenarios

### 5. Genoa/Mediterranean Low
- Key cyclogenesis region (Gulf of Genoa)
- Brings heavy precipitation to Alps, Balkans, Central Europe
- Position determines if moisture hits Italy, Balkans, or moves north

### 6. Russian/Siberian High Extension
- Western extent of the massive winter Asian anticyclone
- Extension to ~30°E brings severe cold to Europe
- Position of western edge determines cold air penetration

### 7. Bay of Biscay Pressure
- Strategic location for steering storms
- High here = Iberian blocking, storms deflect to UK/Norway
- Low here = direct storm impact on Western Europe

### 8. Central European Pressure Anomaly
- Pressure over Germany/Poland region
- High = stable, dry conditions, heat in summer, inversions in winter
- Low = active weather, precipitation, frontal passages

### 9. Greenland High (Blocking)
- Creates negative NAO when present
- Reverses normal westerly flow
- Brings Arctic air to Europe via northerly flow
- Often paired with Mediterranean storms

### 10. Jet Stream Position
- Latitude over Atlantic (35°N to 65°N range)
- Strength (speed) and waviness
- North = storms hit Scotland/Norway
- South = storms hit Iberia/Mediterranean
- Wavy = blocking patterns, extreme weather

### Bonus Indicators

Often included in operational forecasting:
- **UK-Biscay Gradient**: Pressure difference indicating storm track
- **Baltic-Mediterranean Gradient**: North-south pressure difference
- **500 hPa Geopotential Height** over Central Europe: Upper-level pattern

---

## Top Central European Wind Indicators

For **wind specifically in Central Europe**, here are the **top 5 indicators** ranked by their direct influence:

### 1. NAO Index (North Atlantic Oscillation)

**Direct wind impact**: Controls westerly wind strength

- **NAO+**: Strong westerlies, persistent W/SW winds in Central Europe (10-20 m/s common)
- **NAO-**: Weak/variable winds, more easterly component possible
- Single best predictor for monthly mean wind speeds

### 2. Central European Pressure Anomaly

**Most direct local control** on wind

- Low pressure = cyclonic circulation, strong winds, typically SW before front, NW after
- High pressure = weak winds, subsidence, wind direction depends on position within high
- Pressure gradient across Germany-Poland axis determines wind speed

### 3. Azores High Extension & Position

Controls the **pressure gradient** between Atlantic and continent

- Extended northeast (toward UK/France) = strong SW flow over Central Europe
- Retracted southwest = weaker pressure gradient, reduced winds
- Ridge extending to Central Europe = calm conditions

### 4. Scandinavian/Baltic High Presence

Creates **easterly winds** when present (reverses normal westerly flow)

- Sets up strong pressure gradient with lower pressure to the west
- Winter: brings cold, dry easterly winds ("Russian wind")
- Position determines if winds are NE (high over Norway) or E/SE (high over Baltic)

### 5. Genoa/Mediterranean Low Track

Creates strong **southerly flow** ahead of system (Föhn conditions in Alps)

- **Vb track** (moves northeast): brings strong S→SW→NW wind shift
- Tight pressure gradient between Mediterranean low and Central European high
- Can generate strongest winds in Central Europe (storms like Kyrill, Lothar)

---

## Classic Central European Wind Patterns

### 1. Strong Westerly (Most Common)

**Conditions:**
- NAO+, Azores High extended, no blocking

**Wind:**
- Speed: 15-25 m/s
- Direction: 240-270°

### 2. Föhn/Southerly

**Conditions:**
- Mediterranean Low + higher pressure north

**Wind:**
- Speed: 15-30 m/s
- Direction: 160-200°

### 3. Easterly/Continental

**Conditions:**
- Scandinavian High + lower pressure west

**Wind:**
- Speed: 10-20 m/s
- Direction: 060-120°

### 4. Calm

**Conditions:**
- High pressure centered over Central Europe

**Wind:**
- Speed: <5 m/s
- Variable direction

---

## Encoding Synoptic Patterns for Machine Learning

This is a fascinating challenge in meteorological ML! Synoptic-scale weather patterns (like those on pressure charts) are difficult to encode for tree-based models like XGBoost. Here are several theoretical approaches to transform spatial weather information into ML-friendly features:

### The Challenge

- **Spatial data**: Pressure fields are 2D grids with thousands of points
- **Tree-based models**: XGBoost requires tabular features (single numbers per sample)
- **Goal**: Extract meaningful numerical features that capture the essence of weather patterns

### Core Principles

Each meteorological indicator needs multiple representations to capture its different aspects:
- **Position** (where)
- **Intensity** (how strong)
- **Geometry** (what shape)
- **Motion** (where going)
- **Gradients** (impact on flow)

---

## Feature Engineering Approaches

### 1. Pressure Center Parameterization

Instead of raw pressure fields, characterize each pressure system:

**Features to extract:**
- **Position**: Latitude/longitude of high/low centers
- **Intensity**: Central pressure value, pressure anomaly from climatology
- **Geometry**: Area, eccentricity, orientation angle
- **Gradient**: Maximum pressure gradient, steepness index

**Example features:**
```python
[H1_lat, H1_lon, H1_pressure, H1_area, H1_eccentricity,
 L1_lat, L1_lon, L1_depth, L1_gradient_max, ...]
```

**Advantages:**
- Compact representation
- Interpretable features
- Captures system characteristics

**Disadvantages:**
- Requires center detection algorithm
- May miss multi-center patterns
- Sensitive to detection thresholds

### 2. Regional Aggregation Approach

Divide Europe into meaningful meteorological regions:

**Process:**
1. Define regions (e.g., Iceland, Azores, Scandinavia, Central Europe, Mediterranean)
2. Calculate aggregate statistics for each region:
   - Mean pressure
   - Pressure anomaly
   - Gradient magnitude
3. Create cross-regional features:
   - Pressure differences between regions (e.g., `P_Iceland - P_Azores` for NAO-like index)
   - Gradients between regions

**Example features:**
```python
{
  'iceland_pressure_mean': 1002,
  'azores_pressure_mean': 1024,
  'nao_proxy': 22,  # Azores - Iceland
  'ce_pressure_mean': 1015,
  'med_ce_gradient': -3,  # Mediterranean - Central Europe
}
```

**Advantages:**
- Meteorologically meaningful
- Reduces dimensionality naturally
- Easy to implement

**Disadvantages:**
- Region boundaries are arbitrary
- May miss fine-scale features
- Averages can obscure details

### 3. Weather Regime Classification

Use established European weather regimes as categorical/ordinal features:

**Common European regimes:**
- Atlantic Ridge (AR)
- Scandinavian Blocking (SB)
- NAO+ / NAO-
- East Atlantic pattern
- Greenland Blocking

**Encoding options:**
- One-hot vectors
- Ordinal values
- "Regime strength" or "similarity score" to each regime

**Example features:**
```python
{
  'regime_primary': 'NAO+',  # Categorical
  'regime_nao_pos_strength': 0.85,  # Similarity score
  'regime_scandi_blocking': 0.12,
  'regime_atlantic_ridge': 0.03,
}
```

**Advantages:**
- Captures complex patterns with simple labels
- Based on established meteorology
- Handles non-linear relationships

**Disadvantages:**
- Requires regime classification (subjective or automated)
- May oversimplify
- Transition periods between regimes

### 4. Teleconnection Indices

Use established meteorological indices that capture large-scale patterns:

**Standard indices:**
- **NAO** (North Atlantic Oscillation): `P_Azores - P_Iceland`
- **EA** (East Atlantic) pattern
- **SCAND** (Scandinavian) pattern
- **EA/WR** (East Atlantic/West Russia) pattern

**Custom indices for specific regions:**
- Mediterranean-Central Europe gradient
- Alpine Föhn index: `P_south_alps - P_north_alps`

**Example features:**
```python
{
  'nao_index': 1.2,
  'ea_index': -0.3,
  'scand_index': 0.7,
  'foehn_index': -4.5,  # Negative = favorable for Föhn
}
```

**Advantages:**
- Standard, reproducible
- Well-studied relationships
- Single numbers capture complex patterns

**Disadvantages:**
- Limited to predefined patterns
- May miss novel situations
- Specific to calculation points

### 5. Graph-Based Features

Represent the weather map as a graph:

**Structure:**
- **Nodes**: Pressure centers or grid points
- **Edges**: Pressure gradients or geostrophic wind relationships

**Features to extract:**
- Graph metrics: centrality, clustering coefficients, shortest paths between pressure centers
- Convert to numerical features: number of nodes, average degree, graph diameter

**Example features:**
```python
{
  'num_pressure_centers': 5,
  'avg_node_degree': 2.4,
  'graph_diameter': 3,
  'clustering_coefficient': 0.65,
}
```

**Advantages:**
- Captures topological relationships
- Handles variable number of centers
- Novel approach

**Disadvantages:**
- Complex to implement
- Not interpretable
- Requires graph theory expertise

### 6. Fourier/Spectral Decomposition

Transform spatial pressure fields into frequency domain:

**Process:**
- 2D Fourier transform of pressure field → select dominant wavenumbers
- Spherical harmonics for better representation on Earth's surface
- Keep only the first N coefficients as features
- Captures large-scale wave patterns (Rossby waves)

**Example features:**
```python
{
  'fourier_coef_0_0': 1013.2,  # Mean pressure
  'fourier_coef_1_0': 12.5,    # Zonal wave 1
  'fourier_coef_0_1': -8.3,    # Meridional wave 1
  'fourier_coef_2_0': 3.2,     # Zonal wave 2
  ...
}
```

**Advantages:**
- Mathematically rigorous
- Captures wave patterns
- Compact representation

**Disadvantages:**
- Not interpretable
- Assumes periodic patterns
- Difficult to relate to physical processes

### 7. Distance and Directional Features

Create relational features between your prediction location and pressure systems:

**Features:**
- Distance to nearest high/low
- Direction to nearest high/low (encoded as sin/cos for cyclical nature)
- Number of highs/lows within certain radius
- "Pressure tendency" based on approaching systems

**Example features for Lake Como (45.96°N, 9.25°E):**
```python
{
  'dist_to_nearest_high': 850,  # km
  'direction_to_high_sin': 0.87,  # ~60° (NE)
  'direction_to_high_cos': 0.50,
  'dist_to_nearest_low': 420,  # km
  'num_lows_within_1000km': 2,
  'approaching_system_speed': -15,  # km/h, negative = approaching
}
```

**Advantages:**
- Very relevant for local prediction
- Simple to compute
- Directly interpretable

**Disadvantages:**
- Requires center detection
- May miss patterns without clear centers
- Limited to nearest systems

### 8. Empirical Orthogonal Functions (EOFs)

Use PCA on historical pressure fields:

**Process:**
- Compute PCA on historical pressure grids
- Project current pressure field onto principal components
- Use EOF coefficients as features

**First few EOFs often correspond to known weather patterns:**
- EOF 1: Often captures NAO pattern
- EOF 2: Often captures East Atlantic pattern
- EOF 3+: Secondary patterns

**Example features:**
```python
{
  'eof_1_loading': 2.5,   # Strong NAO+ pattern
  'eof_2_loading': -0.8,
  'eof_3_loading': 1.2,
  'pca_explained_variance': 0.65,  # How well EOFs capture current pattern
}
```

**Advantages:**
- Data-driven (learns from your domain)
- Reduces dimensionality optimally
- Captures dominant patterns

**Disadvantages:**
- Requires large historical dataset
- EOFs can be unstable
- Not always interpretable

### 9. Frontal Zone Encoding

Characterize frontal systems:

**Features:**
- Front type (cold/warm/occluded/stationary)
- Distance to nearest front
- Front orientation and speed
- Temperature gradient across front
- Binary indicators for front presence in regions

**Example features:**
```python
{
  'front_present': 1,  # Binary
  'front_type': 'cold',  # Categorical
  'dist_to_front': 250,  # km
  'front_orientation': 045,  # degrees (SW-NE)
  'front_speed': 35,  # km/h
  'temp_gradient_across_front': 8,  # °C per 100 km
}
```

**Advantages:**
- Directly relevant for wind prediction
- Captures dynamic features
- Meteorologically meaningful

**Disadvantages:**
- Requires front detection algorithm
- Fronts are often diffuse
- Subjectivity in classification

### 10. Hybrid Embedding Approach

Combine deep learning with XGBoost:

**Process:**
1. Use a CNN or Vision Transformer to create embeddings from pressure maps
2. Extract the embedding vector (e.g., 128 dimensions)
3. Feed this embedding along with other features to XGBoost

**Example:**
```python
# CNN processes pressure map → 128-dim embedding
cnn_features = cnn_model.predict(pressure_map)  # [0.23, -0.45, 0.88, ...]

# Combine with traditional features
all_features = concat(cnn_features, traditional_features)

# Train XGBoost
xgb_model.fit(all_features, target)
```

**Advantages:**
- Lets neural network learn optimal representation
- Can capture complex spatial patterns
- Combines strengths of deep learning and tree models

**Disadvantages:**
- Requires training CNN first
- More complex pipeline
- Less interpretable
- Requires more data

---

## Detailed Feature Engineering: Top 5 Wind Indicators

### 1. NAO Index - Multi-representation Approach

```python
# Basic NAO features
'nao_index': 1.2,                          # Raw index value
'nao_index_squared': 1.44,                 # Non-linear response to extremes
'nao_index_abs': 1.2,                      # Magnitude (strong pattern either way)
'nao_category': 2,                         # Categorical: -2,-1,0,1,2 (strong neg to strong pos)

# Temporal NAO features
'nao_3day_mean': 1.1,                      # Smoothed/persistent signal
'nao_7day_trend': 0.3,                     # Increasing/decreasing (pattern change)
'nao_persistence_days': 12,                # How many days in same phase

# NAO-based gradient estimation
'nao_implied_westerly_speed': 12.5,        # Empirical formula: 10 + 5*NAO
'nao_storm_track_latitude': 55.0,          # 50 + 5*NAO (shifts north when positive)
```

### 2. Central European Pressure Anomaly - Local Field Characterization

```python
# Absolute and anomaly values
'pressure_ce': 1015,                       # Absolute pressure (hPa)
'pressure_ce_anomaly': -3,                 # Departure from climatology
'pressure_ce_standardized': -0.8,          # (P - P_clim) / σ_clim

# Spatial pressure features (sample points around CE)
'pressure_gradient_magnitude': 8.5,        # |∇P| in hPa/1000km
'pressure_gradient_direction': 265,        # Direction of strongest gradient
'pressure_laplacian': -2.1,               # ∇²P (curvature - indicates low/high center)

# Multi-point sampling (create pressure field representation)
'pressure_munich': 1013,
'pressure_berlin': 1015,
'pressure_warsaw': 1017,
'pressure_vienna': 1014,
'zonal_gradient_ce': -4,                  # P_west - P_east across CE
'meridional_gradient_ce': 2,              # P_north - P_south across CE

# Temporal pressure features
'pressure_ce_3h_tendency': -2.5,          # Falling/rising pressure
'pressure_ce_24h_change': -8,             # Longer-term trend
```

### 3. Azores High - Position & Intensity Encoding

```python
# Position features (where is the high?)
'azores_high_lat': 38.5,                  # Latitude of center
'azores_high_lon': -28.0,                 # Longitude of center
'azores_high_distance_to_ce': 2850,       # km from Central Europe
'azores_high_bearing_from_ce': 248,       # Direction from CE (degrees)

# Extension features (how far does it reach?)
'azores_ridge_longitude': -8,             # Easternmost extent of 1020 hPa contour
'azores_ridge_latitude': 45,              # Northernmost extent
'azores_extension_index': 0.7,            # 0-1 scale: (ridge_lon - (-30))/30
'ridge_to_ce': 0,                         # Binary: does ridge reach CE?

# Intensity features
'azores_max_pressure': 1024,              # Central pressure
'azores_strength_anomaly': +6,            # Departure from normal
'azores_area_above_1020': 2.5e6,         # km² above threshold

# Gradient features (impact on flow)
'azores_ce_pressure_diff': 9,             # P_azores - P_ce
'azores_ce_gradient_component': 3.2,      # Gradient along Azores-CE axis
```

### 4. Scandinavian/Baltic High - Blocking Characterization

```python
# Presence and intensity
'scandi_high_exists': 1,                  # Binary indicator
'scandi_high_pressure': 1028,             # Maximum pressure if exists, else 0
'scandi_blocking_index': 0.75,            # 0-1 continuous blocking strength

# Position encoding (critical for wind direction)
'scandi_high_lat': 62,                    # Latitude of center
'scandi_high_lon': 18,                    # Longitude of center
'scandi_position_type': 2,                # 1=Norwegian, 2=Swedish, 3=Baltic, 4=Russian

# Blocking geometry
'scandi_meridional_extent': 15,           # Degrees latitude (how tall)
'scandi_zonal_extent': 20,                # Degrees longitude (how wide)
'scandi_eccentricity': 0.3,              # Shape: 0=circular, 1=elongated
'scandi_orientation': 35,                 # Major axis angle

# Impact features
'scandi_ce_gradient': 13,                 # Pressure difference to CE
'easterly_forcing': 8.5,                  # Proxy for easterly wind strength
'blocking_age_days': 5,                   # Persistence (older = more stable)

# Temperature gradient (often couples with blocking)
'temp_gradient_east_west': -12,           # °C difference (cold east)
```

### 5. Genoa/Mediterranean Low - Cyclone Tracking

```python
# Existence and intensity
'med_low_exists': 1,                      # Binary
'med_low_pressure': 996,                  # Central pressure if exists
'med_low_depth_anomaly': -12,             # How deep relative to normal

# Position and motion
'med_low_lat': 44.5,                      # Current latitude
'med_low_lon': 8.2,                       # Current longitude
'med_low_track_type': 5,                  # 1-6: Vb track=5, Va=4, etc.
'med_low_speed': 35,                      # km/h movement speed
'med_low_direction': 45,                  # Movement direction (degrees)

# Distance and approach features
'med_low_distance_to_ce': 650,            # Current distance to CE
'med_low_approach_rate': -25,             # km/h (negative = approaching)
'time_to_ce_arrival': 26,                 # Hours until closest approach

# Cyclone characteristics
'med_low_warm_sector_width': 300,         # km (determines wind shift timing)
'med_low_pressure_gradient': 18,          # Max gradient around low (hPa/500km)
'med_low_frontal_speed': 45,              # Cold front propagation speed

# Interaction with topography
'alpine_barrier_effect': 1,               # Binary: will cause Föhn?
'adriatic_enhancement': 0,                # Bora wind potential
```

---

## Advanced Composite Features

Beyond individual indicators, create interaction features that capture non-linear relationships:

```python
# Interaction features (multiplicative/ratios capture non-linear effects)
'nao_azores_interaction': 1.2 * 0.7,      # NAO * extension_index
'pressure_squeeze': gradient_NS * gradient_EW,  # 2D gradient intensity
'blocking_vs_nao': scandi_index / (nao + 2),   # Competing patterns

# Circulation indices
'zonal_index': P_40N - P_60N,             # Strength of westerly flow
'meridional_index': P_west - P_east,       # Cross-Europe flow
'cyclonicity': low_count - high_count,     # Net cyclonic tendency

# Pattern stability
'pattern_persistence': min(nao_days, azores_days, scandi_days),
'transition_probability': pattern_change_historical_likelihood,

# Wind-specific derivatives
'theoretical_geostrophic_wind': k * gradient_magnitude / sin(lat),
'wind_direction_mode': atan2(meridional_gradient, zonal_gradient),
'shear_indicator': gradient_direction_change_per_100km,
```

---

## Feature Selection Strategy

For XGBoost, include these feature groups:

### 1. Core Features (15-20)
Direct measurements and positions
- Pressure center locations
- Central pressure values
- Index values (NAO, SCAND, etc.)

### 2. Gradient Features (5-10)
Pressure differences, directional gradients
- Cross-regional pressure differences
- Maximum gradients
- Directional components

### 3. Temporal Features (5-10)
Tendencies, persistence, trends
- Pressure change rates
- Pattern persistence
- Movement speeds

### 4. Interaction Features (5-10)
Products and ratios of main indicators
- NAO × Azores extension
- Competing pattern ratios
- Gradient products

### 5. Categorical Encodings (3-5)
Pattern types, track classifications
- Weather regime
- Cyclone track type
- Blocking type

**Total: ~40-50 features** comprehensively capture synoptic control on Central European winds.

---

## Key Principles

### Multiple Representations

Each meteorological indicator needs multiple representations to capture its different aspects:

1. **Position** (where): Latitude, longitude, distance, bearing
2. **Intensity** (how strong): Central value, anomaly from climatology, standardized value
3. **Geometry** (what shape): Area, eccentricity, orientation
4. **Motion** (where going): Speed, direction, approach rate
5. **Gradients** (impact on flow): Pressure differences, maximum gradients

### Spatial to Tabular Transformation

The fundamental challenge is transforming 2D spatial fields into 1D feature vectors:

```
Pressure Field (2D Grid)  →  Feature Engineering  →  Feature Vector (1D)
[18,532 grid points]                                [40-50 features]
```

### Meteorological Meaningfulness

Prefer features with clear meteorological interpretation:
- NAO index → westerly strength
- Föhn gradient → Alpine wind
- Blocking index → easterly flow

This makes models interpretable and ensures features capture actual physical processes.

### Scale Hierarchy

Capture patterns at multiple scales:
- **Synoptic scale** (1000+ km): NAO, Azores High, Scandinavian blocking
- **Regional scale** (100-1000 km): Central European pressure, Mediterranean lows
- **Local scale** (<100 km): Cross-Alpine gradients, local pressure tendencies

---

## Practical Considerations

### Data Requirements

Different approaches have different data needs:

- **Simple indices**: Just need pressure at a few points
- **Regional aggregation**: Need complete grid coverage
- **EOF/PCA**: Require long historical dataset
- **CNN embeddings**: Need large training dataset

### Computational Cost

- **Cheapest**: Direct index calculation, distance features
- **Moderate**: Regional aggregation, pressure center detection
- **Expensive**: Spectral decomposition, EOF calculation, CNN training

### Interpretability vs. Performance

- **Most interpretable**: Teleconnection indices, regional aggregations
- **Balanced**: Pressure center parameterization, distance features
- **Least interpretable**: Fourier coefficients, CNN embeddings, graph metrics

Choose approach based on your priorities: if model explanation is critical, use interpretable features; if maximum performance is needed, consider hybrid approaches.

---

## Expected Insights from Feature Importance

After training XGBoost with these features, feature importance will reveal:

### 1. Which Patterns Matter Most

- High importance for NAO → Large-scale Atlantic flow dominates
- High importance for Mediterranean Low → Local cyclogenesis critical
- High importance for Föhn gradient → Alpine effects dominate

### 2. Position vs. Intensity

- If position features (lat/lon) rank high → Pattern location is key
- If intensity features (pressure values) rank high → Pattern strength matters more

### 3. Temporal vs. Spatial

- If tendency features rank high → Rate of change drives wind
- If static pattern features rank high → Current configuration sufficient

### 4. Local vs. Remote

- If distance features rank high → Proximity to systems matters
- If regional features rank high → Large-scale pattern dominates

---

## Conclusion

Encoding synoptic weather patterns for machine learning requires careful consideration of:

1. **Physical processes**: Features should represent actual meteorological controls on wind
2. **Scale**: Capture patterns from synoptic to local
3. **Dimensionality**: Balance between information content and model complexity
4. **Interpretability**: Trade-off between performance and understanding

The recommended approach combines multiple methods:
- **Teleconnection indices** for standardized large-scale patterns
- **Regional aggregation** for spatial coverage
- **Pressure center parameterization** for system characteristics
- **Distance features** for local relevance

This multi-faceted approach provides ~40-50 features that comprehensively encode the synoptic control on Central European winds while maintaining interpretability and computational efficiency.

---

**Document Status**: Consolidated from EUROPEAN_GENERAL_WEATHER.md and GENERAL_WEATHER_SITUATIONS_FOR_ML.md
**Last Updated**: 2025-11-27
**Version**: 1.0

# Open-Meteo API Comparison: Focus on Ensemble Data

## Quick Summary

| Feature | Historical Forecast API | Ensemble API |
|---------|------------------------|--------------|
| **Data Type** | ❌ **Deterministic only** (single forecast) | ✅ **Ensemble data** (20-51 members) |
| **Time Coverage** | Past (2-5 years historical) | Future (7-35 days ahead) |
| **Spatial Resolution** | High (1-9 km) | Medium (25-50 km) |
| **Use Case** | ML training, historical analysis | Uncertainty estimation, risk assessment |
| **Best for XGBoost** | ✅ Primary choice | Special cases only |

---

## Historical Forecast API

**Endpoint**: `https://historical-forecast-api.open-meteo.com/v1/forecast`

### What it provides:
- **Single deterministic forecasts** archived continuously
- Takes the first hours of each model run and combines them into a seamless time series
- 2-5 years of historical data (depending on model)
- Same models and variables as the current Forecast API

### Key characteristics:
- ❌ **NO ensemble data** - only one forecast per timestamp
- ✅ High spatial resolution (1-9 km)
- ✅ Initialized with real measurements (stations, satellites, radar)
- ✅ Nearly as accurate as direct measurements
- ✅ Global coverage

### Why it's called "Historical Forecast":
It archives weather *forecasts* from the past (not actual measurements). Each archived value was a forecast at the time it was made, initialized with the best available observational data.

---

## Ensemble API

**Endpoint**: `https://api.open-meteo.com/v1/ensemble`

### What it provides:
- **Multiple forecast members** (ensemble predictions)
- Each member runs with slightly different initial conditions/parameters
- Provides probability distributions and uncertainty estimates
- 7-35 days into the future (model dependent)

### Key characteristics:
- ✅ **Full ensemble data** with 20-51 members per model
- ✅ Uncertainty quantification (spread, percentiles, probabilities)
- ✅ Longer forecast horizons (up to 35 days for GFS)
- ❌ NO extensive historical archive (only recent weeks via "Past Days" feature)
- ❌ Lower spatial resolution (25-50 km)

### Available ensemble models:
- **ECMWF IFS Ensemble**: 51 members, up to 15 days
- **GFS Ensemble**: 31 members, up to 35 days  
- **DWD ICON-EU EPS**: 40 members, up to 7 days (high resolution for Europe)
- Other national ensemble systems

---

## The Critical Difference: Ensemble Data

### What is Ensemble Data?

Instead of one single forecast, ensemble models generate multiple forecasts (members):

```
Single Forecast (Historical Forecast API):
Time: 2024-01-15 12:00
Temperature: 5.2°C
Pressure: 1013.5 hPa

Ensemble Forecast (Ensemble API):
Time: 2024-01-15 12:00
Temperature: 
  - Member 1: 4.8°C
  - Member 2: 5.5°C
  - Member 3: 4.9°C
  - ... (48 more members)
  - Mean: 5.2°C
  - Std Dev: 0.8°C
  - 10th percentile: 4.1°C
  - 90th percentile: 6.3°C
```

### Why Ensemble Data Matters

**Advantages of ensemble forecasts:**

1. **Uncertainty quantification**: Know how confident the forecast is
2. **Probability estimates**: P(Temperature > 30°C) = 35%
3. **Risk assessment**: Evaluate extreme event likelihood
4. **Improved accuracy**: Ensemble mean often more accurate than single forecast
5. **Decision support**: Better for risk-based decisions

**Example applications:**
- Energy trading: "What's the risk of extreme heat affecting demand?"
- Insurance: "What's the probability of damaging wind speeds?"
- Agriculture: "What's the chance of frost in the next 10 days?"
- Flood forecasting: "How likely is >50mm precipitation?"

---

## Why Historical Forecast API Has NO Ensemble Data

### Technical reasons:

1. **Storage constraints**: Ensemble data is massive
   - Single forecast: ~100 GB for 5 years
   - Ensemble (51 members): ~5 TB for 5 years
   - Multiplied by all available models: 50-100+ TB

2. **Operational focus**: Historical Forecast API designed for:
   - Machine learning training
   - Validation against observations
   - High-resolution deterministic forecasts

3. **Archive availability**: National weather services don't always provide:
   - Long-term ensemble archives
   - Public access to historical ensemble data
   - Consistent ensemble configurations over years

### What you get instead:

The Historical Forecast API provides the **control run** or **deterministic forecast** - essentially what would be "member 0" if it were an ensemble, but with higher resolution.

---

## For XGBoost Training: Which API?

### Use Historical Forecast API (Deterministic) when:

✅ **Recommended for 90% of use cases**

- You need training data (historical forecasts)
- Prediction horizon is 1-7 days
- High spatial resolution is important
- You want consistency between training and production
- Computational efficiency matters
- Standard regression/classification tasks

**Example**: Energy demand forecasting, solar power prediction, retail demand

### Use Ensemble API when:

🎯 **Specialized use cases only**

- Uncertainty quantification is critical to your business
- You need probability estimates (not just point forecasts)
- Predicting extreme/rare events
- Risk management applications
- Forecast horizon is 7-35 days
- You're building probabilistic models

**Example**: Insurance risk models, extreme weather warnings, long-term planning

---

## Feature Engineering Approaches

### With Historical Forecast API (Single Forecast)

```python
# Direct weather variables
features = [
    'temperature_2m',
    'pressure_msl',
    'wind_speed_10m',
    'precipitation',
    'cloud_cover',
]

# Temporal features
features += [
    'temp_lag_1h',
    'temp_lag_24h',
    'temp_rolling_mean_6h',
    'pressure_change_3h',
]

# Spatial features (if using multiple locations)
features += [
    'temp_gradient_north_south',
    'pressure_difference_location_A_B',
]
```

### With Ensemble API (Multiple Members)

```python
# Option 1: Aggregate statistics
features = [
    'temperature_ensemble_mean',
    'temperature_ensemble_std',        # Uncertainty
    'temperature_ensemble_min',
    'temperature_ensemble_max',
    'temperature_p10',                 # 10th percentile
    'temperature_p90',                 # 90th percentile
    'temperature_iqr',                 # Inter-quartile range
]

# Option 2: Probability features
features += [
    'prob_temp_above_30',              # P(T > 30°C)
    'prob_precip_above_10mm',          # P(Precip > 10mm)
    'prob_wind_above_50kmh',           # P(Wind > 50 km/h)
]

# Option 3: Ensemble spread (uncertainty measure)
features += [
    'ensemble_spread',                 # Std dev / mean
    'ensemble_range',                  # Max - min
    'coefficient_of_variation',        # Std dev / mean
]

# Option 4: Use individual members
features += [
    'temp_member_01',
    'temp_member_02',
    # ... up to member_51
    # (Usually not recommended - too many features)
]
```

---

## Practical Examples

### Example 1: Solar Power Forecasting

**Approach**: Historical Forecast API ✅

```python
# Fetch historical forecast data
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
    'latitude': 47.37,
    'longitude': 8.54,
    'start_date': '2022-01-01',
    'end_date': '2024-12-31',
    'hourly': [
        'temperature_2m',
        'cloud_cover',
        'direct_radiation',
        'diffuse_radiation',
        'wind_speed_10m'
    ]
}

# Train XGBoost on historical data
# Use same API (Forecast API) for production predictions
```

**Why deterministic is better here**:
- Need high spatial resolution for local cloud patterns
- Short prediction horizon (1-2 days)
- Consistency between training and production is critical
- Don't need uncertainty - just accurate point predictions

### Example 2: Flood Risk Assessment

**Approach**: Ensemble API 🎯

```python
# Fetch ensemble forecast
url = "https://api.open-meteo.com/v1/ensemble"
params = {
    'latitude': 47.37,
    'longitude': 8.54,
    'hourly': 'precipitation',
    'models': 'ecmwf_ifs025'  # 51 ensemble members
}

# Calculate risk metrics
precipitation_members = response['hourly']['precipitation']
prob_heavy_rain = (precipitation_members > 50).mean()  # Probability > 50mm

# Use in risk model
features = {
    'precip_prob_gt_50mm': prob_heavy_rain,
    'precip_ensemble_max': max(precipitation_members),
    'precip_ensemble_spread': std(precipitation_members),
}
```

**Why ensemble is better here**:
- Extreme events are low-probability, high-impact
- Need to know: "How likely?" not just "How much?"
- Ensemble spread indicates forecast confidence
- Can trigger warnings based on probability thresholds

---

## Hybrid Approach: Combining Both APIs

For sophisticated applications, use both:

### Strategy 1: Time-based split

```python
# Short-term (0-3 days): Historical Forecast API
# - High resolution, high accuracy
# - Deterministic predictions

# Medium-term (4-10 days): Ensemble API  
# - Lower resolution, uncertainty estimates
# - Probabilistic predictions
```

### Strategy 2: Ensemble uncertainty as feature

```python
# Primary model: Train on Historical Forecast API
X_train = historical_forecast_features

# Add ensemble spread as "confidence" feature
X_train['forecast_uncertainty'] = ensemble_spread_from_current_forecast

# XGBoost learns: "When uncertainty is high, adjust predictions"
```

### Strategy 3: Ensemble for extreme events only

```python
# Normal conditions: Use Historical Forecast deterministic
if ensemble_spread < threshold:
    use_deterministic_model()
    
# Uncertain conditions: Use ensemble statistics
else:
    use_probabilistic_model()
```

---

## Summary: Key Takeaways

### Historical Forecast API (No Ensemble)
- ✅ **Best for XGBoost training** in most cases
- Single forecast per timestamp (deterministic)
- 2-5 years of historical data
- High resolution (1-9 km)
- Consistent with production Forecast API
- Simple, efficient, practical

### Ensemble API (Full Ensemble Data)
- ✅ **Best for uncertainty and risk** applications
- 20-51 forecasts per timestamp (probabilistic)
- Only future forecasts (7-35 days)
- Lower resolution (25-50 km)
- Rich uncertainty information
- Complex but powerful for specific use cases

### Decision Framework

```
Do you need to quantify uncertainty or risk?
├─ NO → Historical Forecast API (deterministic)
│        → Train XGBoost on 2-5 years of data
│        → Use Forecast API for production
│
└─ YES → Consider Ensemble API
         ├─ Have historical ensemble data available? 
         │  ├─ NO → Problem: Can't train with ensemble features
         │  │        Solution: Train on Historical Forecast,
         │  │        add ensemble features in production only
         │  │
         │  └─ YES → Use ensemble statistics as features
         │           (mean, std, percentiles, probabilities)
         │
         └─ Is prediction horizon > 7 days?
            ├─ YES → Ensemble API is necessary
            └─ NO → Historical Forecast likely sufficient
```

### The Fundamental Trade-off

**Historical Forecast API**: 
- Data availability ✅ (years of history)
- Ensemble information ❌ (single forecast only)

**Ensemble API**: 
- Data availability ❌ (only recent/future)
- Ensemble information ✅ (full uncertainty)

For training XGBoost, **data availability wins** in most cases. You can't train a model without historical data, even if ensemble information would be theoretically valuable.

---

## Conclusion

**Default recommendation**: Start with **Historical Forecast API** for XGBoost training.

It provides:
- Sufficient historical data (2-5 years)
- High-quality deterministic forecasts
- Consistency between training and production
- Simple, interpretable features
- Computational efficiency

Only move to Ensemble API when:
- Your application explicitly requires uncertainty quantification
- Risk assessment is core to your business logic  
- You're predicting extreme/rare events
- You have access to historical ensemble data or can live without it

I'd be glad to provide a shorter summary, focusing on the **GFS (Global Forecast System) model** for all your data retrieval needs. This approach ensures both **model consistency** and **data efficiency**.





# Grid download for synaptic situation

To analyze the five Central European wind indicators effectively, you need a single, wide-ranging grid fetched from the GFS API.

-----

### 1\. Data Parameters (The GFS Grid)

| Parameter | Recommended Value | Reason |
| :--- | :--- | :--- |
| **Abdeckung (Coverage)** | **$30^\circ\text{N}-75^\circ\text{N}$, $25^\circ\text{W}-40^\circ\text{E}$** | Captures all relevant action centers (NAO, Azore High, Genoa Low) in the North Atlantic/European region. |
| **Räumliche Auflösung** | **$0.4^\circ$** (approx. 18,500 points) | Provides the best balance: high enough detail to detect tight pressure systems (like the Genoa Low) without overwhelming processing resources. |
| **Zeitliche Auflösung** | **Ideal: 4 Times Daily (00, 06, 12, 18 UTC)** | **Critical for accuracy.** A single 12:00 UTC snapshot is insufficient for tracking the **maximum intensity** and **true track** of fast-moving Lows. Use 4 samples daily to capture the systems' full dynamics. |

-----

### 2\. The GFS Approach

The **GFS API** is the solution for your model consistency requirement because it supports the efficient grid syntax for current forecasts, and its historical data is available via the Open-Meteo Historical Forecast API.

  * **For Training Data:** Use the **Historical Forecast API** (which relies on GFS/ECMWF runs) for model consistency.
  * **For Live Prediction:** Use the **GFS API** (`/v1/gfs`) for live data retrieval.

The efficient grid download is achieved using the **`START,END,STEP`** notation in your API call:

```url
# Example GFS API call for a 7-day forecast
https://api.open-meteo.com/v1/gfs?
    latitude=30,75,0.4& 
    longitude=-25,40,0.4&
    hourly=mean_sea_level_pressure&
    forecast_days=7
```
# Failed Synoptic-Only Approach

**Date:** 2025-01-21
**Status:** Discontinued
**Models:** Model 1b (167 features), Model 2 (24 features)

---

## What We Tried to Achieve

### Goal: Synoptic-Scale Wind Prediction

The objective was to predict local wind speeds **solely from large-scale atmospheric patterns** without using any local measurements. This approach aimed to:

1. **Capture European synoptic weather regimes**
   - North Atlantic Oscillation (NAO)
   - Scandinavian blocking patterns
   - Mediterranean cyclogenesis
   - Alpine Foehn situations
   - Azores High / Icelandic Low configurations

2. **Represent the pressure field as feature vectors**
   - Extract positions of the 3 strongest highs (h1, h2, h3)
   - Extract positions of the 3 strongest lows (l1, l2, l3)
   - For each pressure center, encode:
     - **Bearing**: Direction from target location (0-360°)
     - **Distance**: How far away (km)
     - **Strength**: z500 geopotential height (intensity)
     - **Gradient**: Pressure gradient strength

3. **Encode the "isobar map" as compact features**

   **The core idea:** By knowing where the 3 nearest highs and 3 nearest lows are positioned (bearing + distance), combined with their strength and gradient, the model could reconstruct the approximate pressure field over Europe.

   **Like telling the model:**
   - "There's a strong high 800km to the west (Azores High)"
   - "A weaker high 1200km northeast (Russian High)"
   - "A deep low 600km northwest (Iceland Low)"

   **This compact representation should theoretically:**
   - Capture the general weather situation in Europe
   - Allow the model to infer wind patterns from pressure gradients
   - Work anywhere (not location-specific)

   **Why 3 highs and 3 lows?**
   - 1-2 systems: Too simple, misses complex multi-center patterns
   - 3 systems: Captures primary + secondary influences
   - 4+ systems: Diminishing returns, added complexity

4. **Generalize across locations**
   - Train on Lake Como, deploy anywhere in the Alps (or beyond)
   - Avoid location-specific features
   - Pure physics: geostrophic wind → surface wind

5. **Physics-informed feature engineering**
   - Geostrophic wind ∝ pressure gradient / Coriolis force
   - Thermal wind from temperature gradients at different pressure levels
   - Baroclinicity (frontal activity) from temperature/pressure coupling
   - Teleconnection patterns (large-scale oscillations)

### Implementation

**Model 1b (Experiment 8) - Full feature set:**
- **167 synoptic features** (all possible combinations)
- Pressure centers: 3 highs × 7 features + 3 lows × 7 features = 42 features
  - For each center: lat, lon, z500, distance, bearing, gradient, exists_flag
- Interactions: h1-l1, h1-l2, h2-l1, h2-l3 distances and pressure differences
- Regional: Zonal flow, NAO index, Scandinavian index, baroclinicity, Mediterranean pressure
- Temporal: hour_sin/cos, day_of_year_sin/cos
- **Test MAE: 5.5 kt** (poor performance)

**Model 2 (24-feature selection) - Reduced set:**
- **24 features** with atomic grouping strategy
- Focus: European synoptic weather (37.5% of features)
- Distance-based sorting (h1 = nearest high, not strongest)
- Breakdown:
  - Temporal: 4 features (hour/day cycles)
  - High1 group: 4 features (bearing, distance, z500, gradient)
  - Low1 group: 4 features (bearing, distance, z500, gradient)
  - H1-L1 interaction: 2 features
  - European synoptic: 9 features (NAO, zonal flow, baroclinicity, etc.)
  - Local: 1 feature (alpine thermal stability)
- **Test MAE: 7.0 kt** (worse than Model 1b)

---

## Problems Encountered

### 1. Approach Too Simplistic for Terrain Interference

**The Alps are not flat.**

Local terrain effects completely dominate synoptic forcing in complex mountainous regions:

**Valley channeling:**
- Breva (southerly wind up Lake Como) flows along valley axis regardless of synoptic wind direction
- Valley geometry constrains wind direction more than pressure gradients
- 3D terrain creates local pressure perturbations not captured in synoptic z500

**Mountain/valley breezes (thermal circulation):**
- Daytime: Slopes heat → air rises → valley wind flows upslope
- Nighttime: Slopes cool → air sinks → mountain wind flows downslope
- These thermal circulations can be 10-15 kt, overwhelming synoptic forcing

**Foehn effects:**
- Air forced over Alps warms adiabatically on descent
- Creates strong, warm, dry downslope winds
- Highly dependent on specific topography (ridge height, valley orientation)
- Synoptic flow direction matters, but local geometry determines strength

**Lake-land temperature contrasts:**
- Water heats/cools slower than land → local pressure differences
- Lake breeze circulation independent of synoptic patterns
- Peak strength 10-14:00 when land is warmest

**Same synoptic situation, different local outcomes:**
- High over Western Europe can produce:
  - Strong Breva (if sunny and warm → strong thermal forcing)
  - Weak Breva (if cloudy → reduced heating)
  - No wind at all (if inversion layer present → blocked circulation)

**Missing critical local information:**
- Solar radiation intensity (drives thermal circulation)
- Local temperature gradients (valley-scale pressure differences)
- Atmospheric stability (inversion layers block wind)
- Humidity/cloud cover (affects heating rate)
- Surface roughness (terrain drag)

**Conclusion:** Synoptic features are **necessary but not sufficient** for complex terrain. Local physics dominates.

---

### 2. Modeling and Grouping Challenges

**XGBoost cannot enforce atomic feature groups.**

**The problem:** Features representing a single pressure center should be used together:
- `high1_bearing` + `high1_dist_km` + `high1_z500` + `high1_gradient` = complete description
- Bearing alone = direction but no position (meaningless)
- Distance alone = proximity but no direction (incomplete)
- Gradient alone = strength but no location (useless)

**Why it matters:**
- To reconstruct the pressure field, you need ALL components
- A high at 45° bearing and 800 km distance has very different impact than same bearing at 1500 km
- The model needs bearing AND distance together to understand position

**XGBoost treats features independently:**
- Decision trees split on one feature at a time
- A tree might split on `high1_bearing > 180°` alone
- Completely ignores `high1_dist_km`, making the bearing split meaningless
- No way to enforce "all or nothing" usage of feature groups

**Attempted solutions and why they failed:**

1. **`interaction_constraints` parameter:**
   - What it does: Specifies which features CAN interact in trees
   - Example: `[[high1_bearing, high1_dist_km], [low1_bearing, low1_dist_km]]`
   - Limitation: Only **permits** interactions, doesn't **require** them
   - A tree can still use `high1_bearing` alone without `high1_dist_km`
   - Result: Doesn't solve the atomic group problem

2. **`monotone_constraints` parameter:**
   - What it does: Enforces directional relationships (feature ↑ → target ↑)
   - Example: `{'temperature': 1}` means higher temp must increase prediction
   - Limitation: Controls individual feature-target relationships, not feature grouping
   - Result: Completely irrelevant for our use case

3. **Manual feature validation:**
   - What we did: Created `validate_atomic_consistency()` function
   - Checks that selected features include complete groups
   - Limitation: Only validates our manual selection, doesn't constrain XGBoost's internal tree building
   - Result: We ensure we SELECT complete groups, but XGBoost can still USE them partially

**Better approach (not pursued):**
- Create composite features: `high1_wind_forcing = gradient / (distance + 1)`
- Encodes all atomic group information in a single feature
- XGBoost sees it as one variable (atomic by construction)
- But: Loses the "isobar map" interpretability we wanted
- Still doesn't guarantee the model uses composites correctly

**Conclusion:** Tree-based models are **fundamentally unsuited** for enforcing structured feature dependencies. This is a conceptual mismatch, not a parameter tuning problem.

---

### 3. Location-Specific vs General Trade-off

**Cannot have both accuracy and generalization in complex terrain.**

**Alps-specific features (highly important in Model 1b):**
- `alpine_thermal_stability` - Rank #1, importance 1,337
  - Temperature difference between Alps and Po Valley
  - Predicts thermal wind strength in Lake Como region
  - **Completely useless** outside the Alps

- `foehn_gradient` - Rank #4, importance 742
  - North-south pressure difference across Alps
  - Indicates Foehn potential
  - **Only meaningful** for Alpine locations

- `alps_south_temp_850_std` - Rank #3, importance 944
  - Temperature variability over northern Italy
  - Related to Adriatic cyclogenesis and Po Valley heating
  - **Geographic-specific**

- `mediterranean_pressure_msl_mean` - Important for Alpine winds
  - Genoa low and Mediterranean cyclones drive Foehn events
  - **Regional pattern**, not universal

**Location-agnostic features (lower importance):**
- `zonal_flow_strength` - West-to-east flow
- `temperature_gradient_ew` - East-west thermal contrast
- `pressure_gradient_magnitude` - Overall gradient strength
- These work anywhere but have **weak predictive power** for Lake Como

**The fundamental dilemma:**
- **Include Alps features** → Better accuracy at Lake Como (MAE 5.5), won't generalize to Pyrenees/Rockies/New Zealand
- **Exclude Alps features** → Worse accuracy (MAE 7.0), but transferable to other mountain regions
- **No middle ground** - you must choose

**Model 2's generalization attempt:**
- Used distance-based sorting (h1 = nearest, not strongest) for consistency
- Emphasized European synoptic patterns (9 features, 37.5%)
- Minimized local features (1 feature, 4.2%)
- **Result:** Performance degraded from MAE 5.5 to 7.0 kt
- **Conclusion:** Lost accuracy without gaining enough generalization benefit

**Why this is unavoidable:**
- Complex terrain creates unique local physics
- "Universal" features smooth over these local details
- A model that works everywhere works poorly everywhere
- Location-specific models will always outperform for their specific location

**Conclusion:** **Generalization requires sacrificing accuracy.** For operational forecasting, location-specific models are the right choice.

---

### 4. Insufficient Data (9 Months)

**Not enough samples to learn seasonal patterns and interannual variability.**

**Available data:**
- January 2025 - September 2025 (9 months planned)
- Currently: ~6,000 hourly samples
- Only one summer season (May-Sep 2025)
- No winter data from previous years (Oct 2024 - Apr 2025 missing)

**Why this is inadequate:**

**Seasonal patterns need multi-year data:**
- NAO (North Atlantic Oscillation) phase shifts year-to-year
  - 2024 winter: NAO+ (mild, wet)
  - 2025 winter: NAO- (cold, dry)
  - Model trained on 2025 alone won't know typical NAO+ behavior

- Blocking patterns vary annually
  - Some years: frequent Scandinavian blocking (cold easterly flow)
  - Other years: persistent Azores High (warm westerlies)
  - Need multiple years to learn "normal" vs "anomalous"

- Mediterranean cyclone frequency changes with climate oscillations
  - El Niño years: Different cyclone tracks
  - La Niña years: Different frequency/intensity
  - One season ≠ representative sample

**Consequences of limited data:**
- Model cannot learn "typical July wind" vs "typical February wind"
- Overfits to 2025 weather patterns (e.g., if 2025 was anomalously blocked)
- Cannot distinguish rare events from normal variability
- Seasonal decomposition fails (only 1 cycle of each season)

**What we're missing:**
- Extreme events: Deep lows (storms), strong blocking (heatwaves)
- Transition periods: Spring/fall shoulder seasons with mixed regimes
- Long-term trends: Climate change effects on synoptic patterns
- Interannual comparisons: Year-to-year consistency checks

**Recommended minimum:**
- **3+ years** of continuous data
- Captures multiple cycles of all seasons
- Includes both NAO+ and NAO- phases
- Provides enough rare events (e.g., 1-in-2-year blocking patterns)
- Allows proper train/validation/test splits by year

**Comparison to operational models:**
- GFS: Trained on 40+ years of reanalysis data
- ECMWF: 50+ years of observations assimilated
- Our 9 months is **0.75% of ECMWF's data**

**Conclusion:** **Multi-year data is essential** for synoptic-scale learning. 9 months is inadequate for reliable patterns. This alone would justify stopping this approach.

---

### 5. Ensemble Data Unavailable

**Only deterministic forecasts accessible; consistent historical and forecast data needed.**

**Why ensemble forecasts would be better:**

**Ensemble forecasts capture uncertainty:**
- Run model 50+ times with perturbed initial conditions
- Spread shows uncertainty in pressure center positions
- Example: 30% of members have high centered 100km east of mean position
- Provides probabilistic information: "70% chance high stays west"

**Better for machine learning:**
- Neural networks can learn from distribution (mean + spread)
- Can predict: "When ensemble spread is large, local wind is less predictable"
- Captures non-linear relationships between uncertainty and forecast skill

**More robust to initial condition errors:**
- Single deterministic forecast can be wrong about pressure center position
- Ensemble mean often more accurate than any single member
- Ensemble spread indicates forecast confidence

**Better match for neural networks:**
- LSTMs/Transformers can process ensemble members as sequences
- Can learn: "Model disagreement in high position → larger wind forecast error"
- XGBoost cannot easily leverage ensemble spread (need to engineer features like "std_dev_high1_position")

**Why we used deterministic only:**

**Consistency requirement:**
- Training data (historical) must match forecast data (operational)
- If we train on ensemble archive but forecast with deterministic → distribution mismatch
- Model calibration breaks down (trained on spread, gets single value at inference)

**Historical ensemble archives not easily accessible:**
- Open-Meteo API provides deterministic forecasts (GFS, ECMWF)
- Historical ensemble data requires:
  - Direct ECMWF API access (expensive)
  - Or downloading from research archives (TIGGE, S2S database)
  - Complex preprocessing to align members

**Computational cost:**
- Ensemble data is 10-50x larger than deterministic (51 members vs 1)
- Downloads: 11,200 grid points × 51 members = 571,200 data points per hour
- Storage: ~30 GB vs ~1 GB for 9 months of data
- Processing time: Proportionally longer

**Simplicity:**
- Deterministic forecasts are straightforward: one value per variable per grid point
- Ensemble requires: member management, spread calculation, percentile extraction
- Adds complexity to already complex pipeline

**Alternative approach (not pursued):**
- Could train on ERA5 reanalysis (deterministic, high quality historical data)
- Forecast with GFS deterministic (operational, free)
- Risk: ERA5 and GFS have different biases/characteristics
- Calibration mismatch likely

**Conclusion:** **Deterministic-only data limits model sophistication.** Ensemble forecasts would improve synoptic-scale modeling but require infrastructure and cost we cannot justify for a single-location model.

---

### 6. Heavy Computational Cost for Synoptic Grids

**Large-scale downloads are expensive, slow, and disproportionate to value.**

**Grid requirements for synoptic features:**

**Spatial coverage:**
- European domain: 35°N to 75°N, 30°W to 40°E
- Why so large? To capture:
  - Azores High (often at 40°N, 20°W)
  - Icelandic Low (often at 65°N, 20°W)
  - Russian High (sometimes at 60°N, 40°E)
  - Mediterranean lows (35-45°N)

**Resolution:**
- 0.5° × 0.5° latitude/longitude (approximately 50km × 50km)
- Finer resolution would be better but prohibitively expensive
- Coarser resolution might miss smaller pressure centers

**Grid size:**
- Latitude: (75-35) / 0.5 = 80 points
- Longitude: (40-(-30)) / 0.5 = 140 points
- Total: 80 × 140 = **11,200 grid points**

**Variables needed per timestep:**
- Geopotential height at 500 hPa (z500) - pressure center detection
- Mean sea level pressure (msl) - surface pressure gradients
- Temperature at 850 hPa - lower atmosphere thermal structure
- Temperature at 700 hPa - mid-level baroclinicity
- Temperature at surface (2m) - surface thermal gradients
- U wind component at 500 hPa - zonal flow
- V wind component at 500 hPa - meridional flow
- Relative humidity (for cloud/precipitation proxies)
- **Total: ~8 variables × 11,200 points = 89,600 data points per hour**

**Download costs:**

**API rate limits:**
- Open-Meteo: ~10,000 requests per day (generous but finite)
- Each grid point × variable = one data request (approximately)
- For one hour of data: 89,600 requests (would take 9 days!)
- Actual: Can batch requests, but still slow

**Actual download approach:**
- Batch by regions (multiple grid points per request)
- Cache aggressively (don't re-download)
- Retry logic for failed requests
- **Result: Several hours per month of synoptic data**

**Comparison:**
- Synoptic grid: 89,600 data points/hour
- Local Lake Como grid (9-point circular): 9 points × 8 variables = **72 data points/hour**
- **Synoptic is 1,245x more data than local**

**Storage requirements:**
- Parquet files: ~50-100 MB per day of synoptic grids (compressed)
- 9 months: ~15-30 GB of raw synoptic data
- Local forecasts: ~100-200 MB for 9 months
- **Synoptic is 100x larger storage**

**Processing time:**
- Pressure center detection: ~2-5 seconds per timestep (scipy peak finding)
- Feature engineering: ~1 second per timestep
- For 6,000 hours: ~3-10 hours of computation
- Local features: ~1 minute total

**Infrastructure requirements:**
- Need robust download pipeline with batching, retries, caching
- Need significant disk space for raw grids
- Need computational resources for pressure center detection
- **All this for marginal (if any) improvement over local forecasts**

**Conclusion:** **Synoptic grids are expensive** in every dimension (download time, storage, computation) compared to local forecasts. The cost-benefit ratio is poor for a single-location model.

---

## What Model 1a Will Handle

**Most pressure and regime influence already captured by local measurements.**

Our simpler **Model 1a** approach (local forecast-based) implicitly includes synoptic information without explicitly modeling it:

**1. Local forecast models already incorporate synoptic patterns:**
- GFS and ECMWF run global data assimilation
  - Ingest satellite observations, weather balloons, aircraft reports
  - Assimilate global pressure/temperature/wind observations
  - Solve full primitive equations (Navier-Stokes for atmosphere)

- Large-scale pressure patterns drive the local forecast
  - High over Western Europe → local high pressure forecast at Lake Como
  - Icelandic Low deepening → local pressure falling forecast
  - Mediterranean cyclone → local southerly wind increase forecast

- NAO and blocking patterns affect local forecasts
  - NAO+ → milder temperatures, westerly flow in forecast
  - Scandinavian blocking → colder temperatures, easterly flow in forecast
  - Model "knows" synoptic regime even if we don't extract it explicitly

**2. Local measurements capture downstream effects:**
- Surface pressure at Lake Como reflects synoptic-scale highs/lows
  - Don't need to extract "high1_z500" ourselves
  - It's already in the local pressure forecast (msl_surface_9point_mean)

- Temperature forecasts encode thermal wind and baroclinicity
  - Don't need to compute "temperature_gradient_ew" manually
  - It's already in the local temperature gradients between grid points

- Wind forecasts at 10m/100m include downscaling of geostrophic wind
  - GFS/ECMWF already convert synoptic pressure gradients to surface winds
  - Include terrain effects (though simplified)

**3. We avoid re-inventing the wheel:**
- Don't need to detect pressure centers ourselves
  - GFS/ECMWF already do this (and better - they use 4D-Var data assimilation)
- Don't need to compute geostrophic wind manually
  - Already done in their physics parameterizations
- Don't need to download huge synoptic grids
  - Just download the 9-point local forecast (cheap, fast)

**4. Focus on local refinement (our competitive advantage):**
- Model 1a learns: **"Given local forecast X, actual wind is usually Y"**

- Corrects for systematic biases:
  - Example: GFS consistently underestimates Breva by 3 kt
  - Model learns this offset and corrects it

- Learns terrain-specific adjustments:
  - Example: When GFS forecasts southerly 10 kt + sunny + afternoon → actual Breva is 18 kt
  - Model learns the local thermal amplification

- Captures what large models miss:
  - Valley channeling (not in 50km resolution GFS)
  - Lake-land contrasts (not fully resolved)
  - Local thermal circulations (parameterized poorly in large models)

**Key insight:**
We don't need to predict synoptic weather ourselves. We just need to **consume existing synoptic forecasts** (which are excellent) and refine them for complex local terrain (which is where we add value).

**Model 1a is simpler and better:**
- Fewer features (30-50 vs 167)
- Cheaper downloads (9 points vs 11,200 points)
- Faster training (minutes vs hours)
- Better performance (expected, based on local features importance)
- Easier to maintain (less complexity)

---

## What Was Achieved

Despite discontinuing this approach, we accomplished valuable work:

### 1. Validated Atomic Grouping Concept

**Developed systematic feature groups:**
- `PRESSURE_CENTER_GROUPS`: h1/h2/h3, l1/l2/l3 atomic sets
- `HL_INTERACTION_GROUPS`: h1-l1, h1-l2, h2-l1 interactions
- `TEMPORAL_GROUPS`: hour_sin/cos, day_of_year_sin/cos pairs

**Created validation framework:**
- `validate_atomic_consistency()` function checks if selection is complete
- Warns when partial groups are selected (e.g., high1_bearing without high1_dist_km)
- Documented in `feature_selection.py`

**Documented XGBoost limitations:**
- Proved that tree-based models cannot enforce atomic groups
- Showed `interaction_constraints` is insufficient
- This knowledge is valuable for future modeling decisions (know when NOT to use XGBoost)

### 2. Distance-Based Sorting Innovation

**Changed pressure center ranking:**
- Old approach (strength-based): h1 = strongest high (highest z500)
- New approach (distance-based): h1 = nearest high (closest to target)

**Why this matters:**
- Evidence from Model 1b feature importance:
  - `low3_bearing` ranked #2 (importance 1,226)
  - `low1_bearing` ranked #173 (importance 102)
  - **The 3rd-strongest low was 12x more important than the strongest low!**
  - Why? Because low3 was nearer to Lake Como than low1

**More consistent and generalizable:**
- Strength-based: h1 might be Azores High (2000km away) one day, Russian High (1500km away) next day
- Distance-based: h1 is always the nearest high, regardless of strength
- Feature meaning is consistent across samples

**Implementation:**
- Added `sort_by` parameter to `PressureCenterDetector`
- `sort_by='distance'` sorts by haversine distance to target
- `sort_by='strength'` sorts by z500 intensity (old behavior)
- Configurable via YAML: `synoptic_features.pressure_centers.sort_by`

### 3. Physics-Informed Feature Engineering

**Created comprehensive synoptic feature library:**

**Pressure center features:**
- Position: lat, lon
- Intensity: z500 geopotential height
- Location-awareness: distance, bearing to target
- Strength: gradient (pressure change per km)
- Existence flags: high1_exists, low1_exists

**Interaction features:**
- High-low separation: h1_l1_distance
- Main synoptic gradient: h1_l1_pressure_diff
- Secondary interactions: h1_l2, h2_l1, h2_l3

**Teleconnection indices:**
- NAO (North Atlantic Oscillation): Arctic vs Azores pressure difference
- Scandinavian pattern: Blocking over Scandinavia
- East Atlantic pattern: Secondary NAO-like mode

**Regional patterns:**
- Zonal flow strength: West-to-east flow intensity
- Meridional flow strength: North-to-south flow intensity
- Mediterranean pressure mean: Genoa low / cyclogenesis indicator

**Baroclinicity measures:**
- Temperature gradients: East-west, North-south at 850/700 hPa
- Thermal wind: Vertical wind shear from thermal gradients
- Baroclinicity index: Frontal activity indicator

**Well-documented code:**
- Type hints throughout
- Docstrings for all classes and methods
- Research notes explaining physics reasoning
- Clear separation of concerns (pressure_centers.py, synoptic_interactions.py, synoptic_pipeline.py)

### 4. Reusable Infrastructure

**Generic synoptic feature extraction pipeline:**
- Not Lake Como specific - can be used anywhere
- Configurable via YAML (grid extent, resolution, parameters)
- Modular design (easy to add new feature types)

**Could be valuable for:**
- Multi-location ensemble models (train on 10 lakes, deploy to 11th)
- Transfer learning experiments (pre-train on synoptic, fine-tune with local)
- Neural network weather prediction research (use as baseline features)
- Educational purposes (example of physics-informed ML)

**Code components:**
- `synoptic_pipeline.py`: Main orchestration
- `pressure_centers.py`: Detection and parameterization
- `synoptic_interactions.py`: High-low interactions
- `feature_selection.py`: Atomic grouping framework
- Training scripts with full pipeline examples

### 5. Quantified the Performance Gap

**Established baseline performance:**
- Synoptic-only (Model 1b): MAE = 5.5 kt
- Synoptic-only reduced (Model 2): MAE = 7.0 kt
- This is **poor** compared to persistence forecast (MAE ~4 kt) or climatology

**Provides comparison point:**
- When we build Model 1a (local features), we have a baseline
- Can quantify value of local features: "Improved from 5.5 kt to X kt"
- Justifies decision to abandon synoptic-only approach

**Evidence that local features are essential:**
- Not optional additions - they're the primary signal
- Synoptic features provide context but not precision
- Complex terrain requires local measurements

---

## Why We Stop Here

### Decision: Leverage Large Models, Stay Simple and Lightweight

**1. Large-scale weather models already exist and are excellent:**

**Global Forecast System (GFS):**
- 0.25° resolution (~25 km)
- 16-day forecasts, updated 4x daily
- Full global data assimilation (satellites, balloons, aircraft)
- Free and publicly accessible

**ECMWF (European Centre for Medium-Range Weather Forecasts):**
- 0.1° resolution (~10 km)
- 10-day forecasts, updated 2x daily
- World's most accurate global model (consistently wins forecast skill competitions)
- 50+ member ensemble available

**These models have:**
- **Decades of development:** 40+ years of continuous improvement
- **Supercomputer resources:** Run on some of world's fastest supercomputers
- **Massive data assimilation:** Ingest millions of observations daily
- **Full physics:** Solve primitive equations with sophisticated parameterizations
- **Expert teams:** Hundreds of PhD meteorologists and computational scientists

**We cannot compete at this level.** Nor should we try.

**2. We consume their output, not compete:**

**Our competitive advantage is local refinement, not synoptic-scale prediction:**
- Large models struggle with complex terrain (50km resolution cannot resolve valleys)
- Large models parameterize lake breezes (we can measure them directly)
- Large models have systematic biases (we can learn and correct them)

**Simple approach:**
- Download local forecasts (9-point grid around Lake Como)
- Cheap: 9 points × 8 variables = 72 data points per hour
- Fast: Download completes in seconds
- Accurate: GFS/ECMWF already include synoptic information

**Learn local adjustments:**
- Model 1a: "Given GFS forecast X, actual wind is Y"
- Captures terrain effects, thermal circulations, systematic biases
- Small model (30-50 features), fast training, easy deployment

**3. Stay simple and lightweight:**

**Small models have advantages:**
- Fast training: Minutes instead of hours
- Fast inference: Milliseconds per prediction
- Easy to understand: Can inspect feature importance, debug errors
- Low maintenance: Fewer moving parts, less to break

**Low computational cost:**
- Can run on modest hardware (laptop, Raspberry Pi)
- No need for GPU, cloud compute, distributed systems
- Cheap enough to run continuously in production

**Easy deployment:**
- No synoptic grid downloads in production (just 9-point local forecast)
- Small model file (~1 MB) easy to version control
- Simple pipeline: Download → Predict → Done

**Operational simplicity:**
- Fewer failure modes (no complex pressure center detection)
- Easier debugging (can manually inspect 9-point forecast)
- Faster iteration (re-train in minutes, not hours)

**4. Focus on what matters for local wind forecasting:**

**Signal breakdown (estimated):**
- Local measurements: **80%** of predictable signal
  - Local pressure, temperature, humidity
  - Solar radiation (heating drives thermal winds)
  - Wind persistence (today's wind predicts tomorrow's wind)

- Synoptic patterns (from GFS/ECMWF forecasts): **15%** of signal
  - Included implicitly in local pressure/temperature forecasts
  - Large-scale regime (blocking, NAO phase)

- Terrain/thermal effects: **5%** residual
  - What GFS/ECMWF miss (valley channeling, lake breeze details)
  - Where our Model 1a adds unique value

**Our strategy:**
- Get 80% from local forecasts (cheap, easy)
- Get 15% from synoptic information embedded in local forecasts (free)
- Get 5% from learning local adjustments (our value-add)
- Total: 100% of achievable skill

**What we don't do:**
- Re-extract synoptic features ourselves (expensive, marginal value)
- Try to compete with GFS/ECMWF (impossible)
- Build complex models we can't maintain (unnecessary)

**5. Let neural networks and large models handle synoptic patterns:**

**If we wanted to model synoptic weather directly:**
- Neural networks would be better suited than XGBoost
- LSTMs can model temporal evolution of pressure patterns
- Transformers can capture spatial relationships across Europe
- CNNs can process pressure/temperature fields as images

**But others are already doing this at scale:**
- **GraphCast (Google DeepMind):** 0.25° global forecasts in under 1 minute
- **Pangu-Weather (Huawei):** Competitive with ECMWF, 10,000x faster
- **FourCastNet (NVIDIA):** Extreme weather prediction with AI
- **MetNet (Google):** Nowcasting (0-12 hour) precipitation

**These models are:**
- Trained on decades of data (ERA5 reanalysis: 1940-present)
- Use massive compute (hundreds of GPUs, weeks of training)
- Openly published (papers, code, pre-trained weights available)

**We learn from their outputs:**
- Could ensemble GFS + ECMWF + GraphCast forecasts
- Use their synoptic predictions as input to our local model
- Add value in the "last mile" (global → local refinement)
- Stay in our competency area (local terrain physics)

---

## Lessons Learned

### For Future Projects

**1. Start with local, add synoptic if needed (not vice versa)**
- Local features provide most of the signal in complex terrain
- Synoptic-only approach was backwards
- Should have built Model 1a first, then evaluated if synoptic features add value

**2. Tree-based models struggle with structured feature dependencies**
- XGBoost cannot enforce "use these 4 features together"
- Solution: Create composite features (e.g., `wind_forcing = gradient / distance`)
- Or: Use neural networks (can learn feature interactions internally)

**3. Generalization requires sacrificing accuracy**
- Universal models work poorly everywhere
- Location-specific models work well for their location
- For operational forecasting, accept location-specific models

**4. Multi-year data is essential for seasonal patterns**
- Minimum: 3+ years of continuous data
- Needed to learn interannual variability (NAO phases, blocking frequency)
- 9 months is inadequate for robust synoptic-scale learning

**5. Ensemble data is important but hard to obtain**
- Deterministic forecasts are limiting
- Need consistent training (historical) and forecast (operational) data
- Ensemble archives require infrastructure we don't have

**6. Computational cost matters - don't over-engineer**
- Synoptic grids: 1,245x more data than local forecasts
- Downloads take hours, storage takes GBs
- Value doesn't justify cost for single-location models

**7. Don't reinvent the wheel - leverage existing tools**
- GFS/ECMWF are world-class synoptic models
- Use their output, don't try to replicate their work
- Focus on where we add unique value (local refinement)

**8. Simpler is better for operational systems**
- Small models: Easier to maintain, debug, deploy
- Fewer dependencies: Less to break in production
- Fast inference: Responsive user experience

### What We Preserve

**Code (kept for reference):**
- `windmojo/features/synoptic_pipeline.py`
- `windmojo/features/pressure_centers.py`
- `windmojo/features/synoptic_interactions.py`
- `windmojo/features/feature_selection.py`

**Scripts (archived):**
- `scripts/train_model1b_synoptic.py` - Model 1b (167 features)
- `scripts/train_model2_synoptic24.py` - Model 2 (24 features)
- `scripts/compare_sorting_methods.py` - Distance vs strength sorting

**Reports (documentation):**
- `reports/model1b_synoptic_report.md` - Full feature set results
- `reports/model2_synoptic24_report.md` - Reduced feature set results

**Research notes (this document):**
- Why approach failed
- What we learned
- Lessons for future work

**Knowledge value:**
- Knowing what doesn't work is valuable
- Failed experiments guide future decisions
- Documentation prevents repeating mistakes

---

## Conclusion

This synoptic-only approach was a **valuable learning experience** that clarified our path forward:

**What we learned:**
- Local features dominate in complex terrain
- Synoptic patterns are necessary but not sufficient
- Large-scale models (GFS/ECMWF) already do synoptic forecasting excellently
- Our value is in local refinement, not competing with global models

**Path forward:**
- **Model 1a:** Simple local forecast-based model
- Consume GFS/ECMWF output (cheap, fast, includes synoptic info)
- Learn local adjustments (terrain, thermal effects, biases)
- Stay lightweight (easy to maintain, deploy, iterate)

**Why this is the right decision:**
- Simpler (fewer features, faster downloads)
- Better (local features provide most signal)
- Cheaper (no synoptic grid infrastructure)
- Maintainable (small model, few dependencies)
- Pragmatic (leverage existing tools instead of rebuilding)

We preserve the code and documentation for future reference, but do not pursue synoptic-only modeling further. This topic we leave for the large models and research groups with appropriate resources. **We stay simple and light.**

---

*"The best models are not the most sophisticated, but the ones that actually work in production."*

[![CI](https://github.com/marioland/windmojo/actions/workflows/ci.yml/badge.svg)](https://github.com/marioland/windmojo/actions/workflows/ci.yml)

# Windmojo - ML-Powered Wind Forecasting for Any Location

**Find your wind mojo with location-specific ML forecasts**

## What Windmojo Does

Windmojo builds custom machine learning wind forecasting models for any location in the world. You provide:

1. **GPS coordinates** of your wind measurement station
2. **Historical wind data** (1 year minimum recommended)
3. **Location name** for organizing your models

Windmojo then:
- Downloads historical weather forecast data around your location
- Engineers ~160 features from atmospheric patterns using a 9-point circular grid
- Trains a unified XGBoost model (Spatial Heavy) that learns local wind characteristics
- Generates accurate wind forecasts with ~10% improvement over GFS baseline
- Creates validation diagrams comparing predictions with measurements

## Quick Start

```bash
# 1. Train a model (if you have migrated data)
python cli.py train --spot dervio

# 2. Evaluate and create validation diagrams
python cli.py evaluate --spot dervio

# 3. Migrate from old structure (if needed)
python cli.py migrate --from-dir data/lake_como/dervio --to-spot dervio
```

See [SIMPLIFIED_ARCHITECTURE.md](SIMPLIFIED_ARCHITECTURE.md) for complete documentation.

---

## Why Windmojo?

Global weather models (GFS, ECMWF) provide general wind forecasts, but they often miss crucial local effects:
- **Thermal winds** (sea/lake breezes, mountain/valley winds)
- **Topographic channeling** (canyon winds, gap flows)
- **Local convergence zones** (land-sea boundaries)
- **Micro-climate effects** (urban heat islands, local heating)

Windmojo learns these location-specific patterns from your historical wind data and creates a custom ML model.

### Primary Use Cases

- **Wind Sports**: Kitesurfing, windsurfing, sailing, paragliding
- **Coastal Locations**: Sea breeze prediction
- **Lake Environments**: Thermal wind forecasting (Ora, Breva, Pelér patterns)
- **Mountain Valleys**: Katabatic and anabatic wind systems
- **Complex Terrain**: Any location where local effects dominate

---

## Requirements

### Input Data

**Minimum:**
- **1 year of hourly wind measurements** (more is better)
- **GPS coordinates** (latitude, longitude)
- **Location name** for organizing your data

**Recommended:**
- 2-3 years of data for robust seasonal patterns
- Hourly or sub-hourly measurements
- Less than 20% missing data

**Data Format:**
CSV with a `timestamp` column in the format `YYYY-MM-DD_HHMM`, local time of the
spot's configured `timezone`. Wind speed in m/s; the column name is set in the config.

```csv
timestamp,dervio speed,dervio direction
2024-01-01_0000,5.2,180
2024-01-01_0100,4.8,175
```

See [data/README.md](data/README.md#measurement-file-format) for details.

---

## Key Features

### 8+1 Circular Grid Architecture

Windmojo uses a **9-point circular grid** (8 compass directions + center) to efficiently capture atmospheric patterns around your location:

**Grid Structure:**
- **9 total points**: 1 center + 8 surrounding points
- **Center point**: Your exact measurement location
- **8 perimeter points**: Positioned at the 8 compass directions (N, NE, E, SE, S, SW, W, NW)
- **Equal spacing**: All 8 points at the same configurable distance (radius) from center

**Configuration:**
- Configurable radius in your spot's config file
- **Default radius: 20 km**
- Tune based on location characteristics (narrow valleys vs. wide lakes)

**Variable Distribution:**

*Grid variables (9 points):*
- Surface: `temperature_2m`, `pressure_msl`, `cloud_cover`, `precipitation`, `solar_radiation`
- Wind: `wind_speed_10m`, `wind_direction_10m`
- Upper air: `temperature_850hPa`, `geopotential_height_500hPa`

*Center-only variables (1 point):*
- Surface: `relative_humidity_2m`, `wind_gusts_10m`
- Wind profile: `wind_speed_100m`, `wind_direction_100m`
- Upper atmosphere: `wind_speed_850hPa`, `wind_direction_850hPa`, `wind_speed_250hPa`, `wind_direction_250hPa`

**Benefits:**
- **Simpler data fetching**: 9 points instead of complex grids
- **Aligned with meteorology**: 8 compass directions match how wind patterns are described
- **Direct pressure gradient support**: Natural alignment with 8-direction pressure gradient calculations
- **Wind shear calculation**: Complete wind speed/direction at multiple altitudes
- **More efficient**: Less data to download and process
- **Configurable scale**: Adjust spatial coverage to match local wind phenomena

### Feature Engineering

Windmojo automatically engineers **100-300 features** from the downloaded data:

**Spatial Features:**
- Pressure gradients in 8 compass directions
- Temperature, cloud, and solar radiation gradients
- Cross-grid patterns and interactions

**Temporal Features:**
- Temporal lags (1h, 3h, 6h, 1d, 2d, 3d)
- Rolling statistics (3h, 6h, 12h, 24h, 3d, 7d)
- Time derivatives and change rates

**Physics-Based Features:**
- Wind shear (vertical gradients between atmospheric levels)
- Thermal indicators
- Cyclical encoding (hour, month, season)
- Stability indices

### Training Pipeline

**Phase 1 - Feature Discovery:**
- Trains XGBoost with all engineered features
- Identifies which atmospheric patterns matter most at your location
- Generates feature importance rankings

**Phase 2 - Production Model:**
- Selects top 30-80 features from Phase 1
- Trains optimized model with regularization
- Validates on held-out test data
- Ready for production forecasting

### Output

- **Trained ML Model**: Ready for daily forecast generation
- **Feature Importance Report**: Physical interpretation of what drives wind at your location
- **Validation Metrics**: MAE, RMSE, R² on test set
- **Interactive Visualizations**: Compare model vs. GFS vs. measurements

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Internet Connection**: For downloading weather data
- **UV Package Manager**: Modern Python package manager

### 1. Install UV Package Manager

**macOS / Linux / WSL:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows 11 (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart terminal after installation.

### 2. Clone Repository

```bash
git clone https://github.com/marioland/windmojo.git
cd windmojo
```

### 3. Install Dependencies

```bash
uv sync
```

---

## Quick Start

### 1. Prepare Your Data

Organize wind measurements:

```
data/[area]/[spot]/measurements/wind_data.csv
```

Example:
```
data/lake_como/dervio/measurements/wind_data.csv
```

**CSV Format** (`timestamp` must be `YYYY-MM-DD_HHMM`, local time):
```csv
timestamp,dervio speed,dervio direction
2024-01-01_0000,5.2,180
2024-01-01_0100,4.8,175
```

After download, feature engineering and feature selection, the spot is moved to
`data/spots/[spot]/` for `python cli.py train` / `evaluate` — see
[data/README.md](data/README.md).

### 2. Configure Your Location

Create `config/[area]/[spot].yaml`:

```yaml
location_name: "Dervio, Lake Como"
area: "lake_como"
spot: "dervio"

center:
  latitude: 46.083333
  longitude: 9.3

measurement_file: "data/lake_como/dervio/measurements/wind_data.csv"
timezone: "Europe/Rome"

grid:
  radius_km: 20  # Distance to perimeter points

training:
  test_split: 0.15

forecast:
  forecast_days: 7
```

### 3. Run the Pipeline

**Download Historical Data:**
```bash
uv run python -m windmojo.data_fetcher \
  --config config/lake_como/dervio.yaml \
  --mode historical
```
*Time: 5-15 minutes*

**Engineer Features:**
```bash
uv run python -m windmojo.feature_engineer \
  --config config/lake_como/dervio.yaml
```
*Time: 2-5 minutes*

**Train Phase 1 Model (Feature Discovery):**
```bash
uv run python -m windmojo.train_phase1_model1 \
  --config config/lake_como/dervio.yaml \
  --scope full \
  --hours 06:00,07:00,08:00,09:00,10:00,11:00,12:00,13:00,14:00,15:00,16:00,17:00,18:00 \
  --target-column "dervio speed" \
  --parallel-jobs 4
```
*Time: 10-20 minutes with 4 parallel jobs*

**Select Features:**
```bash
uv run python -m windmojo.feature_selection \
  --model1-dir data/lake_como/dervio/models/phase1_model1/full \
  --features-file data/lake_como/dervio/features/dervio_features_phase1.parquet \
  --output-dir data/lake_como/dervio/models/phase1_model1/full/feature_selections
```
*Time: 1 minute*

**Train Phase 2 Production Model:**
```bash
uv run python -m windmojo.train_phase2_model2_unified \
  --config config/lake_como/dervio.yaml \
  --approach gfs_balanced \
  --feature-selection data/lake_como/dervio/models/phase1_model1/full/feature_selections/features_gfs_balanced.json \
  --test-weeks data/lake_como/dervio/test_weeks.json
```
*Time: 2-10 minutes*

**Generate Forecasts:**
```bash
uv run python -m windmojo.forecast \
  --config config/lake_como/dervio.yaml \
  --model data/lake_como/dervio/models/phase2_model2/gfs_balanced/model2_gfs_balanced_unified.json
```

**Total pipeline time:** ~45-60 minutes (one-time setup)
**Daily forecast time:** ~1-2 minutes

---

## Example Results: Lake Como (Lecco)

Windmojo successfully trained production wind forecasting models for Lecco on Lake Como, Italy.

**Input Data:**
- Location: Lecco, Lake Como (45.857°N, 9.384°E)
- Measurements: January - October 2024 (269 days, ~9 months)
- Training hours: 06:00-18:00 (13 daylight hours)
- Total samples: 2,332

**Performance on Held-Out Test Weeks:**

| Model | MAE | vs. GFS | Improvement |
|-------|-----|---------|-------------|
| **gfs_balanced** (53 features) | **3.388 m/s** | 3.719 m/s | **+8.9%** ✅ |
| gfs_topn (75 features) | 3.399 m/s | 3.719 m/s | +8.6% ✅ |
| gfs_regime_indicators (40 features) | 3.557 m/s | 3.719 m/s | +4.4% ✅ |

**All 5 trained models beat GFS baseline** on unseen test weeks.

**Season Performance:**

| Season | GFS MAE | Best Model | Improvement |
|--------|---------|------------|-------------|
| Early (March) | 2.935 m/s | 3.248 m/s | Challenging |
| Mid (May-June) | 3.093 m/s | **2.731 m/s** | **12% better** ✅ |
| Late (August) | 5.130 m/s | **4.026 m/s** | **21% better** ✅ |

**Key Findings:**
- Summer thermal winds show biggest improvement (21%)
- GFS struggles with afternoon lake thermals
- Physics-based features alone (40 features) achieve 4.4% improvement
- Balanced feature selection (53 features) outperforms pure data-driven approaches

---

## Data & File Structure

Windmojo is migrating from an `area/spot` layout to a flat `spots/` layout. Download,
feature engineering and Phase 1/2 training use the legacy layout; `cli.py train` and
`cli.py evaluate` use the new one (details in [data/README.md](data/README.md)):

```
windmojo/
├── config/
│   ├── examples/dervio.yaml     # Example config (simplified format)
│   └── [area]/
│       └── [spot].yaml          # Legacy location configuration
├── data/
│   ├── [area]/                  # Legacy layout
│   │   └── [spot]/
│   │       ├── measurements/
│   │       │   └── wind_data.csv   # Your wind measurements
│   │       ├── raw/             # Downloaded forecast data (Parquet)
│   │       ├── features/        # Engineered features (Parquet)
│   │       └── models/          # Trained models
│   │           ├── phase1_model1/     # Feature discovery
│   │           ├── phase2_model2/     # Production models
│   │           └── phase2_selections/ # Feature selections (read by cli.py train)
│   └── spots/                   # New layout
│       └── [spot]/
│           ├── input/           # config.yaml + measurements.csv
│           ├── training/        # raw/, features/, validation/
│           └── models/          # model.json + metadata.json
└── windmojo/                    # Source code
    ├── data/                    # Data downloading
    ├── features/                # Feature engineering
    └── train/                   # Model training
```

---

## CLI Reference

### Core Commands

```bash
# Download historical weather data (auto-detects date range from measurements)
uv run python -m windmojo.data_fetcher \
  --config config/area/spot.yaml \
  --mode historical

# Engineer features from downloaded data
uv run python -m windmojo.feature_engineer \
  --config config/area/spot.yaml

# Train Phase 1 - Feature discovery (with parallelization)
uv run python -m windmojo.train_phase1_model1 \
  --config config/area/spot.yaml \
  --scope full \
  --hours 06:00,07:00,...,18:00 \
  --target-column "your_speed_column" \
  --parallel-jobs 4  # Use 8 for 8-core systems (6-7× speedup)

# Select top features (generates 5 approaches)
uv run python -m windmojo.feature_selection \
  --model1-dir data/area/spot/models/phase1_model1/full \
  --features-file data/area/spot/features/spot_features_phase1.parquet \
  --output-dir data/area/spot/models/phase1_model1/full/feature_selections

# Train Phase 2 - Production model
uv run python -m windmojo.train_phase2_model2_unified \
  --config config/area/spot.yaml \
  --approach gfs_balanced \
  --feature-selection data/area/spot/models/phase1_model1/full/feature_selections/features_gfs_balanced.json \
  --test-weeks data/area/spot/test_weeks.json

# Generate forecasts (production use)
uv run python -m windmojo.forecast \
  --config config/area/spot.yaml \
  --model data/area/spot/models/phase2_model2/gfs_balanced/model2_gfs_balanced_unified.json
```

---

## FAQ

### How much wind data do I need?

- **Minimum**: 3 months (basic training)
- **Recommended**: 1 year (full seasonal cycle)
- **Optimal**: 2-3 years (robust multi-year patterns)

### What wind measurement quality is required?

**Minimum Requirements:**
- Hourly measurements
- Fixed location (not moving)
- Less than 20% missing data

**Quality Indicators:**
- ✅ Calibrated anemometer (ultrasonic or cup)
- ✅ Unobstructed location
- ⚠️ Personal weather stations (may need filtering)
- ❌ Smartphone/handheld measurements (too inconsistent)

### What's a "good" MAE for wind forecasting?

| MAE (m/s) | Quality | Usability |
|-----------|---------|-----------|
| < 4.0 | Excellent | High confidence for planning |
| 4.0 - 6.0 | Good | Reliable for wind sports |
| 6.0 - 8.0 | Fair | General guidance |
| > 8.0 | Poor | Consider more training data |

**Note:** Afternoon MAE matters most for wind sports. If afternoon MAE < 5.0 m/s, model is highly useful.

### Can I use this for locations without thermal winds?

Yes! Windmojo works for any location:

**Works well for:**
- ✅ Coastal sea breezes
- ✅ Mountain valley winds
- ✅ Lake thermal systems
- ✅ Urban wind patterns
- ✅ Complex terrain

**Tip:** Locations with strong local effects see the most improvement over GFS.

### Can I use multiple measurement stations?

Yes! Train independent models for each station using the area/spot structure:

```
data/lake_como/dervio/
data/lake_como/colico/
data/lake_como/domaso/
```

Each location gets its own configuration file and trained model.

---

## Documentation

- **[research/](research/)**: Research documentation organized by topic
  - `01_FUNDAMENTALS/` - Wind physics reference
  - `02_SYNOPTIC_PATTERNS/` - Large-scale weather patterns
  - `03_FEATURE_ENGINEERING/` - Feature engineering approaches
  - `04_DATA_SOURCES/` - API documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Guidelines for contributors

---

## Limitations

1. **Data Requirement**: Needs minimum 1 year of wind measurements
2. **GFS Resolution**: Limited by 0.25° (~25 km) grid resolution
3. **Forecast Horizon**: Quality degrades beyond 7 days
4. **Computational Cost**: Training requires 30-60 minutes per location
5. **Local Effects Only**: Best for locations with strong local wind patterns

---

## Contributing

Contributions welcome! Areas of interest:
- New feature engineering approaches
- Alternative ML models (LightGBM, Neural Networks)
- Multi-location training
- GUI for non-technical users

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

See [LICENSE](LICENSE) file.

---

## Acknowledgments

- **Weather Data**: Open-Meteo API (https://open-meteo.com/)
- **Numerical Models**: GFS (NOAA)
- **ML Framework**: XGBoost, scikit-learn
- **Python Ecosystem**: pandas, numpy, matplotlib

---

## Roadmap

### Near-Term
- **Phase 3: Forecasting Pipeline** — Daily automated forecast generation
- **Additional Locations** — Expand beyond Lake Como (Lecco, other lakes, coastal spots)
- **API Endpoint** — REST API for serving predictions

### Mid-Term
- **Multi-Location Training** — Share learned patterns across similar locations
- **Alternative Models** — LightGBM, neural networks for comparison
- **GUI** — Web interface for non-technical users

### Long-Term: Autonomous Research Loop

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — an autonomous AI agent that modifies code, runs experiments, keeps improvements, and repeats indefinitely. The pattern maps naturally to Windmojo:

- **Agent explores** feature selection from 498 engineered features, XGBoost hyperparameters, new feature combinations, and training data weighting
- **XGBoost trains in seconds** (vs. 5 minutes for LLMs), enabling 1,000+ experiments overnight
- **Single metric** (MAE on held-out test weeks) provides clear optimization target
- **Phase 2 already proved** that feature selection matters more than model complexity (30 features beat 150)

See [research/05_TOOLS_AND_METHODS/AUTORESEARCH.md](research/05_TOOLS_AND_METHODS/AUTORESEARCH.md) for full analysis.

---

## Version History

- **v0.2.0** (2025-11-27)
  - Unified model architecture (one model for all hours)
  - 5 feature selection approaches
  - Physics-driven regime discovery
  - Enhanced feature engineering (498 features)
  - Test week validation with interactive visualizations
  - All models beat GFS baseline on held-out data
  - Example: Lake Como Lecco (8.9% improvement)

- **v0.1.0** (2025-11-02)
  - Initial release
  - Generic area/spot data structure
  - XGBoost training pipeline
  - Feature engineering and selection
  - CLI for data fetching and forecasting

---

**Find your wind mojo. Train your model. Catch the wind.**

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
- Engineers ~380 features from atmospheric patterns using a 9-point circular grid
- Trains one XGBoost model for all hours on all features, learning local wind characteristics
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

# 4. Live forecast as JSON (for websites)
python cli.py predict --spot dervio
```

See [SIMPLIFIED_ARCHITECTURE.md](SIMPLIFIED_ARCHITECTURE.md) for complete documentation.

**Forecasts on your website:** [DEPLOYMENT.md](DEPLOYMENT.md) explains how to run the
forecast on a schedule and show it on any website — including shared web hosting
without Python (cron on the host, GitHub Actions + SFTP, or own server), with a
ready-made forecast page in [`web/`](web/index.html).

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
spot's configured `timezone`. Wind speed in m/s, km/h, knots or mph (set `target.unit`
in the config, default m/s); the column name is set in the config.

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

Windmojo automatically engineers **~380 features** from the downloaded data:

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

One XGBoost model for all hours (`python cli.py train`):
- Uses every feature of the pipeline. XGBoost copes with redundant and weak features;
  a separate feature selection (the former Phase 1/2) brought no gain in walk-forward
  tests on two lakes
- Hour of day is a feature (`hour`, `hour_sin`, `hour_cos`), so one model covers the daily cycle
- Early stopping on every 10th training day; metrics on separate test days

What matters for the forecast and why: [FEATURE_RELEVANCE.md](research/03_FEATURE_ENGINEERING/FEATURE_RELEVANCE.md)

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

After download and feature engineering, the spot is moved to
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

**Move to the new layout, train and evaluate:**
```bash
uv run python cli.py migrate --from-dir data/lake_como/dervio --to-spot dervio
# convert data/spots/dervio/input/config.yaml to the simplified format
uv run python cli.py train --spot dervio
uv run python cli.py evaluate --spot dervio
```
*Time: about a minute*

**Generate Forecasts:**
```bash
uv run python cli.py predict --spot dervio
```

**Total pipeline time:** ~10-20 minutes (one-time setup, mostly the download)
**Daily forecast time:** under a minute

---

## Example Results: Lake Como (Lecco)

Windmojo successfully trained production wind forecasting models for Lecco on Lake Como, Italy.

> Historical results of the former Phase 2 models (feature selections, test weeks).
> Current models use all features; walk-forward results on Greifensee and Dervio are in
> [FEATURE_RELEVANCE.md](research/03_FEATURE_ENGINEERING/FEATURE_RELEVANCE.md).

**Input Data:**
- Location: Lecco, Lake Como (45.857°N, 9.384°E)
- Measurements: January - October 2024 (269 days, ~9 months)
- Training hours: 06:00-18:00 (13 daylight hours)
- Total samples: 2,332

**Performance on Held-Out Test Weeks** (measurements in km/h, converted to m/s):

| Model | MAE | vs. GFS | Improvement |
|-------|-----|---------|-------------|
| **gfs_balanced** (53 features) | **0.941 m/s** | 1.033 m/s | **+8.9%** ✅ |
| gfs_topn (75 features) | 0.944 m/s | 1.033 m/s | +8.6% ✅ |
| gfs_regime_indicators (40 features) | 0.988 m/s | 1.033 m/s | +4.4% ✅ |

**All 5 trained models beat GFS baseline** on unseen test weeks.

**Season Performance:**

| Season | GFS MAE | Best Model | Improvement |
|--------|---------|------------|-------------|
| Early (March) | 0.815 m/s | 0.902 m/s | Challenging |
| Mid (May-June) | 0.859 m/s | **0.759 m/s** | **12% better** ✅ |
| Late (August) | 1.425 m/s | **1.118 m/s** | **21% better** ✅ |

**Key Findings:**
- Summer thermal winds show biggest improvement (21%)
- GFS struggles with afternoon lake thermals
- Physics-based features alone (40 features) achieve 4.4% improvement

---

## Data & File Structure

Windmojo is migrating from an `area/spot` layout to a flat `spots/` layout. Download and
feature engineering use the legacy layout; `cli.py train`, `evaluate` and `predict` use
the new one (details in [data/README.md](data/README.md)):

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
│   │       └── features/        # Engineered features (Parquet)
│   └── spots/                   # New layout
│       └── [spot]/
│           ├── input/           # config.yaml + measurements.csv
│           ├── training/        # raw/, features/, validation/
│           ├── models/          # model.json + metadata.json
│           └── forecasts/       # forecast.json (cli.py predict)
├── web/index.html               # Forecast page for websites (see DEPLOYMENT.md)
└── windmojo/                    # Source code
    ├── data/                    # Data downloading
    ├── features/                # Feature engineering
    ├── training/                # Model training (cli.py train)
    └── prediction/              # Live forecasts (cli.py predict)
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

# Move the spot to the new layout (then convert its config to the simplified format)
uv run python cli.py migrate --from-dir data/area/spot --to-spot spot

# Train on all features and evaluate on the test days
uv run python cli.py train --spot spot
uv run python cli.py evaluate --spot spot

# Generate a live forecast as JSON (production use, new spot layout)
uv run python cli.py predict --spot spot --output path/to/spot.json
```

Running `predict` on a schedule and publishing the result on a website:
[DEPLOYMENT.md](DEPLOYMENT.md).

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
| < 1.1 | Excellent | High confidence for planning |
| 1.1 - 1.7 | Good | Reliable for wind sports |
| 1.7 - 2.2 | Fair | General guidance |
| > 2.2 | Poor | Consider more training data |

**Note:** Afternoon MAE matters most for wind sports. If afternoon MAE < 1.4 m/s, model is highly useful.

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
4. **Computational Cost**: The download takes 5-15 minutes per location, training under a minute
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
- **Phase 3: Forecasting Pipeline** — Scheduled forecast generation is available
  (`cli.py predict`, [DEPLOYMENT.md](DEPLOYMENT.md)); open: forecast verification against new measurements
- **Additional Locations** — Expand beyond Lake Como (Lecco, other lakes, coastal spots)
- **API Endpoint** — REST API for serving predictions

### Mid-Term
- **Multi-Location Training** — Share learned patterns across similar locations
- **Alternative Models** — LightGBM, neural networks for comparison
- **GUI** — Web interface for non-technical users

### Long-Term: Autonomous Research Loop

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — an autonomous AI agent that modifies code, runs experiments, keeps improvements, and repeats indefinitely. The pattern maps naturally to Windmojo:

- **Agent explores** XGBoost hyperparameters, new features, loss functions for strong wind, and training data weighting
- **XGBoost trains in seconds** (vs. 5 minutes for LLMs), enabling 1,000+ experiments overnight
- **Single metric** (MAE on held-out test weeks) provides clear optimization target
- **Feature selection is settled**: walk-forward tests showed that no subset beats all features ([FEATURE_RELEVANCE.md](research/03_FEATURE_ENGINEERING/FEATURE_RELEVANCE.md))

See [research/05_TOOLS_AND_METHODS/AUTORESEARCH.md](research/05_TOOLS_AND_METHODS/AUTORESEARCH.md) for full analysis.

---

## Version History

- **Unreleased**
  - Training uses all features; Phase 1/2 feature selection and `spatial_heavy` removed
  - Training and live forecasts get identical features (no outlier capping, capped look-backs)
  - GFS surface winds (10-180 m) are features again
  - Early stopping on validation days instead of the test days

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

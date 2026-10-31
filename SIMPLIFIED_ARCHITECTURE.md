# Windmojo Simplified Architecture

## Overview

Windmojo has been simplified to focus on a single, unified approach:

- **Single-level spot hierarchy** (removed area/spot duality)
- **Single model: one XGBoost model for all hours on all features** (removed Phase 1/Phase 2
  and feature selections)
- **Clear separation**: input vs training vs models
- **Unified CLI** at root level

## Directory Structure

```
windmojo/
├── cli.py                          # Main CLI entry point
├── windmojo/                       # Core library
│   ├── config.py                   # Configuration management
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py              # UnifiedTrainer
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # calculate_metrics()
│   │   └── diagrams.py             # create_validation_diagrams()
│   ├── download/                   # (TODO: implement)
│   └── prediction/
│       └── forecast.py             # Live forecast + JSON export (predict)
│
├── data/spots/{spot}/              # Per-spot data
│   ├── input/                      # User-provided inputs
│   │   ├── config.yaml             # Spot configuration
│   │   └── measurements.csv        # Historical measurements
│   ├── training/                   # Training artifacts
│   │   ├── raw/                    # Downloaded GFS data
│   │   ├── features/               # Engineered features
│   │   └── validation/             # Validation diagrams
│   ├── models/                     # Production-ready models
│   │   ├── model.json              # Trained XGBoost model
│   │   └── metadata.json           # Features, training medians, metrics, params
│   └── forecasts/
│       └── forecast.json           # Latest live forecast (predict)
│
├── web/index.html                  # Forecast page for websites
└── config/examples/                # Example configurations
    └── dervio.yaml
```

> **Transitional state:** download and feature engineering are not yet ported (see
> sections 4 and 5 below). They still write to the legacy `data/{area}/{spot}/` layout.
> New spots start in the legacy layout and are moved here with `migrate`.
> See [data/README.md](data/README.md).

## Measurement Data Format

`input/measurements.csv`:

```csv
timestamp,dervio speed,dervio direction
2025-05-01_0800,4.2,225
2025-05-01_0900,5.1,230
```

- `timestamp`: exact column name, format `YYYY-MM-DD_HHMM`, local time in the config's `timezone`
- Wind speed column: name set via `target.column`; unit set via `target.unit`
  (`m/s` default, `km/h`, `kn`, `mph`). Measurements are converted to m/s on loading,
  so models, metrics and forecasts are always in m/s.
- Direction column: optional, name set via `target.direction_column`

Measurements are joined to the features on exact timestamp match.

## Configuration Format

Simplified single-file YAML configuration:

```yaml
# Spot identity
spot: dervio
name: "Dervio, Lake Como"

# Location (center point for 9-point grid)
latitude: 46.083333
longitude: 9.3
timezone: "Europe/Rome"

# Grid configuration
grid:
  radius_km: 20

# Target variable
target:
  column: "dervio speed"
  unit: "km/h"            # m/s (default), km/h, kn or mph
  direction_column: "dervio direction"

# Training configuration
training:
  model: unified

  # Test date selection
  test_dates:
    strategy: first_n_days_every_nth_month
    n_days: 4
    nth_month: 2

  # XGBoost parameters
  xgboost:
    max_depth: 6
    learning_rate: 0.05
    n_estimators: 1000
    min_child_weight: 1
    subsample: 0.8
    colsample_bytree: 0.8
    reg_alpha: 0.1
    reg_lambda: 1.0
    objective: 'reg:squarederror'
    eval_metric: 'mae'
    random_state: 42
    tree_method: 'hist'
    early_stopping_rounds: 50

# Paths (relative to spot directory)
paths:
  measurements: "input/measurements.csv"
  config: "input/config.yaml"
  raw_data: "training/raw"
  features: "training/features"
  validation: "training/validation"
  model: "models/model.json"
  metadata: "models/metadata.json"
  forecast: "forecasts/forecast.json"

# Live forecast (predict); hours per day must match the training download
forecast:
  initialization_time: "06:00"   # local time in `timezone`
  horizon_hours: 18
  days_ahead: 7
```

## CLI Commands

### 1. Migration (from old structure)

```bash
# Migrate from old data/{area}/{spot}/ structure
python cli.py migrate \
  --from-dir data/lake_como/dervio \
  --to-spot dervio
```

**What it does:**
- Creates new directory structure
- Copies `measurements/wind_data.csv` → `input/measurements.csv`
- Copies raw data → `training/raw/`
- Copies features → `training/features/`
- Copies the legacy config `config/{area}/{spot}.yaml` → `input/config.yaml` (needs manual
  conversion to the simplified format below; `config/examples/dervio.yaml` is a reference)

### 2. Training

```bash
# Train the unified model
python cli.py train --spot dervio

# Or with custom config
python cli.py train --spot dervio --config /path/to/config.yaml
```

**What it does:**
1. Loads config and paths
2. Loads measurements and features
3. Uses all features of the pipeline (`feature_columns()`, including the hour encoding
   `hour`, `hour_sin`, `hour_cos`); taken before the merge, so measurement columns never
   become features
4. Splits data by test_dates strategy; every 10th remaining day is the early-stopping set
5. Trains unified XGBoost model
6. Saves model.json and metadata.json

**Example output:**
```
Training Complete!
Train MAE: 0.82 m/s
Test MAE: 1.49 m/s
Model saved to: data/spots/dervio/models/model.json
```

### 3. Evaluation

```bash
# Evaluate model and create validation diagrams
python cli.py evaluate --spot dervio
```

**What it does:**
1. Loads trained model
2. Makes predictions on test dates
3. Calculates metrics (MAE, RMSE, R², MBE)
4. Creates validation diagrams (one per month)
5. Compares: Our Model vs GFS Baseline vs Measurements

**Example output:**
```
Evaluation Complete!
Test MAE: 1.49 m/s
Diagrams: 5 created in data/spots/dervio/training/validation
  • validation_2025-01.png (MAE: 1.42, +4.3% improvement)
  • validation_2025-03.png (MAE: 1.17, +28.1% improvement)
  • validation_2025-05.png (MAE: 1.37, +2.9% improvement)
  • validation_2025-07.png (MAE: 1.93, +7.5% improvement)
  • validation_2025-09.png (MAE: 1.62, +9.4% improvement)

Overall: 1.49 m/s MAE (+10.7% over GFS)
```

### 4. Predict (live forecast)

```bash
python cli.py predict --spot dervio
python cli.py predict --spot dervio --output /path/to/web/dervio.json
```

**What it does:**
1. Loads the trained model and metadata
2. Downloads the current Open-Meteo forecast for the 9-point grid (plus 7 past days
   for lag/rolling features), restricted to the training hours (`forecast` section)
3. Runs the same feature pipeline as for training. No feature looks back more than 7 days
   and none uses statistics over the whole input, so live features equal training
   features (checked by `tests/test_pipeline.py::TestLiveParity`)
4. Predicts and writes JSON (default `data/spots/{spot}/forecasts/forecast.json`);
   `model.missing_features` counts model features missing in the forecast data
   (filled with training medians)

Scheduling, website integration and hosting variants (cron on the web host,
GitHub Actions + SFTP, own server): **[DEPLOYMENT.md](DEPLOYMENT.md)**.

### 5. Download (TODO)

```bash
# Download GFS data for date range
python cli.py download \
  --spot dervio \
  --start-date 2025-01-01 \
  --end-date 2025-12-31
```

### 6. Features (TODO)

```bash
# Engineer features from raw GFS data
python cli.py features --spot dervio
```

## Model Architecture: Unified Model

### What is it?

A **single XGBoost model** trained on all hours with hour encoding, using all features of
the pipeline.

### Key Characteristics:

1. **Unified across hours**: One model handles all hours
   - Uses hour encoding: `hour`, `hour_sin`, `hour_cos`
   - No per-hour models needed

2. **All features** (384 on Greifensee): GFS variables at the 9-point grid, spatial and
   vertical gradients, lags, rolling statistics, physics and snow indicators. A feature
   selection brought no gain in walk-forward tests on Greifensee and Dervio; the former
   packaged `spatial_heavy` list (164 features) was 0.024-0.035 m/s worse. Background:
   [FEATURE_RELEVANCE.md](research/03_FEATURE_ENGINEERING/FEATURE_RELEVANCE.md).

3. **XGBoost with regularization**:
   - Subsample: 0.8
   - Column sample: 0.8
   - L1/L2 regularization
   - Early stopping on every 10th training day (never on the test days)

### Performance:

Walk-forward, 15 test quarters, at least 2 years of training data:

| Spot | MAE | GFS 10 m, linearly calibrated | MAE at ≥ 4 m/s |
|------|-----|-------------------------------|----------------|
| Greifensee (2021-2026) | 0.69 m/s | 0.81 m/s | 1.20 m/s |

Dervio (9 months of data, monthly blocks): MAE 1.53-1.56 m/s, GFS calibrated 1.69-1.74 m/s.
All models underpredict strong wind (bias about −1 m/s at ≥ 4 m/s).

## Test Date Strategies

Three strategies for selecting validation/test days:

### 1. First N days every Nth month

```yaml
test_dates:
  strategy: first_n_days_every_nth_month
  n_days: 4
  nth_month: 2  # Every 2nd month: Jan, Mar, May, Jul, Sep, Nov
```

### 2. Last N days every Nth month

```yaml
test_dates:
  strategy: last_n_days_every_nth_month
  n_days: 4
  nth_month: 2
```

### 3. Custom dates

```yaml
test_dates:
  strategy: custom
  custom_dates:
    - "2025-05-01"
    - "2025-05-02"
    - "2025-05-15"
```

## Library API

### Configuration

```python
from windmojo.config import load_config, get_paths, get_test_dates

# Load config
config = load_config("data/spots/dervio/input/config.yaml")

# Get paths
paths = get_paths(config)
# Returns: measurements, config, raw_data, features,
#          validation, model, metadata

# Get test dates
import pandas as pd
all_dates = [...]  # List of available dates
test_dates = get_test_dates(config, all_dates)
```

### Training

```python
from windmojo.features.pipeline import feature_columns
from windmojo.training import UnifiedTrainer

# Initialize
trainer = UnifiedTrainer(config['training'])

# Train (early stopping on validation_df, metrics on test_df)
results = trainer.train(
    train_data=train_df,
    validation_data=validation_df,
    test_data=test_df,
    feature_columns=feature_columns(features)
)

# Save
trainer.save(
    model_path=paths['model'],
    metadata_path=paths['metadata']
)

# Load
trainer.load(
    model_path=paths['model'],
    metadata_path=paths['metadata']
)

# Predict
predictions = trainer.predict(X_test)

# Feature importance
importance_df = trainer.get_feature_importance()
```

### Evaluation

```python
from windmojo.evaluation import calculate_metrics, create_validation_diagrams

# Metrics
metrics = calculate_metrics(y_true, y_pred)
# Returns: {'mae': ..., 'rmse': ..., 'r2': ..., 'mbe': ...}

# Diagrams
diagram_files = create_validation_diagrams(
    predictions=predictions_df,
    measurements=measurements,
    features=features,
    test_dates=test_dates,
    target_col='dervio speed',
    output_dir=paths['validation'],
    spot_name='Dervio, Lake Como'
)
```

## Migration Guide

### From Old Structure

If you have data in the old `data/{area}/{spot}/` structure:

1. **Run migration**:
   ```bash
   python cli.py migrate \
     --from-dir data/lake_como/dervio \
     --to-spot dervio
   ```

2. **Simplify config**:
   - Copy example: `config/examples/dervio.yaml`
   - Remove `area` field
   - Update paths to new structure
   - Simplify training config (remove phase1/phase2)

3. **Train and validate**:
   ```bash
   python cli.py train --spot dervio
   python cli.py evaluate --spot dervio
   ```

## What Changed?

### Removed:

1. **Area/Spot duality**: Only `spot` now (single level)
2. **Phase 1/Phase 2**: Single unified training
3. **Multiple model types and feature selections**: Only the unified model on all features
4. **Per-hour models**: Unified model with hour encoding
5. **Complex config structure**: Simplified YAML
6. **Old CLI**: `windmojo/cli/` directory removed

### Simplified:

1. **Directory structure**: Clear input/training/models separation
2. **Configuration**: Single file, clear sections
3. **Training pipeline**: One command, one model type
4. **Evaluation**: Integrated metrics + diagrams

### Kept:

1. **Feature engineering**: Same feature extraction
2. **GFS data**: Same 9-point circular grid
3. **XGBoost**: Same core model
4. **Validation approach**: Same test date strategies
5. **Diagram format**: Same 3-way comparison plots

## Next Steps

### To Complete Implementation:

1. **Implement download command**: Wire up GFS data fetcher
2. **Implement features command**: Wire up feature engineering
3. **Add examples**: More sample configs

### For Production:

1. **Implement forecast command**: Real-time predictions
2. **Add API integration**: Fetch live GFS forecasts
3. **Deployment guide**: How to deploy trained model
4. **Monitoring**: Track prediction quality over time

## Performance Summary

Current results: see [Performance](#performance) above and
[FEATURE_RELEVANCE.md](research/03_FEATURE_ENGINEERING/FEATURE_RELEVANCE.md).

Earlier Dervio numbers (test MAE 1.49 m/s, +10.7 % over GFS, 166 `spatial_heavy`
features) are superseded: they came from a single test split with early stopping on the
test days, which made them optimistic.

## Questions?

For more details, see:
- Configuration examples: `config/examples/`
- Source code: `windmojo/`
- Original research: `research/` and `docs/`

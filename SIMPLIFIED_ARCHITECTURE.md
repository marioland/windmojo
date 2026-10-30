# Windmojo Simplified Architecture

## Overview

Windmojo has been simplified to focus on a single, unified approach:

- **Single-level spot hierarchy** (removed area/spot duality)
- **Single model: Spatial Heavy Unified** (removed Phase 1/Phase 2)
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
│   │   └── trainer.py              # SpatialHeavyTrainer
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # calculate_metrics()
│   │   └── diagrams.py             # create_validation_diagrams()
│   ├── download/                   # (TODO: implement)
│   └── prediction/                 # (TODO: implement)
│
├── data/spots/{spot}/              # Per-spot data
│   ├── input/                      # User-provided inputs
│   │   ├── config.yaml             # Spot configuration
│   │   └── measurements.csv        # Historical measurements
│   ├── training/                   # Training artifacts
│   │   ├── raw/                    # Downloaded GFS data
│   │   ├── features/               # Engineered features
│   │   └── validation/             # Validation diagrams
│   └── models/                     # Production-ready models
│       ├── model.json              # Trained XGBoost model
│       └── metadata.json           # Features, metrics, params
│
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
- Wind speed column: name set via `target.column`, in m/s
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
  direction_column: "dervio direction"

# Training configuration
training:
  model: spatial_heavy_unified

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
# Train Spatial Heavy Unified model
python cli.py train --spot dervio

# Or with custom config
python cli.py train --spot dervio --config /path/to/config.yaml
```

**What it does:**
1. Loads config and paths
2. Loads measurements and features
3. Loads Spatial Heavy feature selection (`windmojo/selections/spatial_heavy.json`, same for every spot)
4. Adds hour encoding (hour, hour_sin, hour_cos)
5. Splits data by test_dates strategy
6. Trains unified XGBoost model
7. Saves model.json and metadata.json

**Example output:**
```
Training Complete!
Train MAE: 2.94 m/s
Test MAE: 5.35 m/s
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
Test MAE: 5.35 m/s
Diagrams: 5 created in data/spots/dervio/training/validation
  • validation_2025-01.png (MAE: 5.12, +4.3% improvement)
  • validation_2025-03.png (MAE: 4.20, +28.1% improvement)
  • validation_2025-05.png (MAE: 4.93, +2.9% improvement)
  • validation_2025-07.png (MAE: 6.94, +7.5% improvement)
  • validation_2025-09.png (MAE: 5.83, +9.4% improvement)

Overall: 5.35 m/s MAE (+10.7% over GFS)
```

### 4. Download (TODO)

```bash
# Download GFS data for date range
python cli.py download \
  --spot dervio \
  --start-date 2025-01-01 \
  --end-date 2025-12-31
```

### 5. Features (TODO)

```bash
# Engineer features from raw GFS data
python cli.py features --spot dervio
```

## Model Architecture: Spatial Heavy Unified

### What is it?

A **single XGBoost model** trained on all hours with hour encoding, using a carefully selected subset of spatial features.

### Key Characteristics:

1. **Unified across hours**: One model handles all 24 hours
   - Uses hour encoding: `hour`, `hour_sin`, `hour_cos`
   - No per-hour models needed

2. **Spatial Heavy features**: packaged list of 724 feature names (`windmojo/selections/spatial_heavy.json`);
   features missing from the spot's feature file are skipped. Selected from:
   - GFS variables at 9-point circular grid
   - Synoptic-scale features (500mb heights, temperature gradients, etc.)
   - Wind shear features
   - Temporal features (lags, rolling statistics)

3. **No feature discovery phase**: Pre-selected features eliminate need for Phase 1/Phase 2

4. **XGBoost with regularization**:
   - Subsample: 0.8
   - Column sample: 0.8
   - L1/L2 regularization
   - Early stopping

### Performance:

- **Test MAE**: ~5.35 m/s
- **Improvement over GFS**: +10.7%
- **R² score**: 0.334

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
from windmojo.training import SpatialHeavyTrainer

# Initialize
trainer = SpatialHeavyTrainer(config['training'])

# Train
results = trainer.train(
    train_data=train_df,
    test_data=test_df,
    feature_columns=feature_list
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
3. **Multiple model types**: Only Spatial Heavy Unified
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
3. **Delete deprecated files**: Remove old phase-based code
4. **Update README**: Document new workflow
5. **Add examples**: More sample configs

### For Production:

1. **Implement forecast command**: Real-time predictions
2. **Add API integration**: Fetch live GFS forecasts
3. **Deployment guide**: How to deploy trained model
4. **Monitoring**: Track prediction quality over time

## Performance Summary

Based on Dervio spot (Lake Como):

- **Training samples**: ~6,600 hours
- **Test samples**: 318 hours (20 days, all hours)
- **Features**: 166 (Spatial Heavy selection)
- **Train MAE**: 2.94 m/s
- **Test MAE**: 5.35 m/s
- **Improvement over GFS**: +10.7%

**Per-month performance**:
- January: +4.3% improvement
- March: +28.1% improvement
- May: +2.9% improvement
- July: +7.5% improvement
- September: +9.4% improvement

**Metrics**:
- RMSE: 6.91 m/s
- R²: 0.334
- MBE: -0.27 m/s (slight underprediction bias)

## Questions?

For more details, see:
- Configuration examples: `config/examples/`
- Source code: `windmojo/`
- Original research: `research/` and `docs/`

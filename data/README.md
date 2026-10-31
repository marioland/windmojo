# Data Directory

This directory contains all location-specific data. It is gitignored (large files and
user-specific data); only `.gitkeep` and this README are tracked.

## Two Layouts (Transitional State)

Windmojo is migrating from an `area/spot` layout to a flat `spots/` layout. Both are
currently in use, by different pipeline steps:

| Layout | Used by |
|--------|---------|
| `data/[area]/[spot]/` (legacy) | download (`windmojo.data_fetcher`), feature engineering (`windmojo.feature_engineer`), Phase 1/2 training |
| `data/spots/[spot]/` (new) | `python cli.py train`, `evaluate` and `predict` |

A new spot therefore starts in the legacy layout and is moved to `spots/` with
`python cli.py migrate` once features exist.

### Legacy layout: `data/[area]/[spot]/`

```
data/
└── [area]/                          # Geographic area (e.g., lake_como)
    └── [spot]/                      # Specific measurement location (e.g., dervio)
        ├── measurements/
        │   └── wind_data.csv        # Your wind measurements (you provide this)
        ├── raw/                     # Downloaded forecast data (auto-created)
        ├── features/                # Engineered features (auto-created)
        │   └── [spot]_features_phase1.parquet
        ├── forecasts/               # Generated predictions (auto-created)
        └── models/                  # Phase 1/2 models
```

### New layout: `data/spots/[spot]/`

```
data/spots/[spot]/
├── input/
│   ├── config.yaml                  # Spot configuration (simplified format)
│   └── measurements.csv             # Your wind measurements
├── training/
│   ├── raw/                         # Downloaded forecast data
│   ├── features/                    # Engineered features
│   │   └── [spot]_features_phase1.parquet
│   └── validation/                  # Validation diagrams (created by evaluate)
├── models/
│   ├── model.json                   # Trained XGBoost model
│   └── metadata.json                # Features, training medians, metrics, params
└── forecasts/
    └── forecast.json                # Latest live forecast (created by predict)
```

For production (live forecasts on a website) only `input/config.yaml` and `models/` are
needed — see [DEPLOYMENT.md](../DEPLOYMENT.md).

`python cli.py migrate --from-dir data/[area]/[spot] --to-spot [spot]` creates this
structure and copies `measurements/wind_data.csv` → `input/measurements.csv`, the raw
and feature Parquet files, and the legacy config `config/[area]/[spot].yaml` →
`input/config.yaml`. The copied config is still in the legacy format; convert it to the
simplified format (see `config/examples/dervio.yaml`).

## Measurement File Format

Same format for `wind_data.csv` (legacy) and `measurements.csv` (new):

```csv
timestamp,dervio speed,dervio direction
2025-05-01_0800,4.2,225
2025-05-01_0900,5.1,230
```

| Column | Requirement |
|--------|-------------|
| `timestamp` | **Required, exact name.** Format `YYYY-MM-DD_HHMM` (underscore, no colon). Other formats such as `2025-05-01 08:00:00` fail in training. |
| wind speed | Any column name; must match `target.column` in the config. Unit: set `target.unit` to `m/s` (default), `km/h`, `kn` or `mph` (legacy configs: `measurement_unit`). The simplified pipeline converts to m/s on loading; all reported metrics are in m/s. |
| wind direction | Optional; any name, referenced by `target.direction_column`. Degrees. |

**Time zone:** timestamps must be local time in the zone set by `timezone` in the
config (e.g., `Europe/Zurich`). Forecast data is downloaded in that zone and joined
to the measurements on exact timestamp match.

**Resolution:** hourly values. Rows whose timestamp has no matching forecast hour
are dropped in the join; rows with an empty speed value are ignored.

## Getting Started (New Spot)

```bash
# 1. Measurements in the legacy layout (needed for download + features)
mkdir -p data/your_area/your_spot/measurements
cp your_wind_data.csv data/your_area/your_spot/measurements/wind_data.csv

# 2. Run download and feature engineering
#    (see "Run the Pipeline" in the main README)

# 3. Move to the new layout for training/evaluation
python cli.py migrate --from-dir data/your_area/your_spot --to-spot your_spot
# then convert data/spots/your_spot/input/config.yaml to the simplified format

# 4. Train and evaluate
python cli.py train --spot your_spot
python cli.py evaluate --spot your_spot
```

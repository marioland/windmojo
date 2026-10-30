# Configuration Files

Windmojo currently uses two configuration formats, matching the two data layouts
described in [`data/README.md`](../data/README.md).

| Format | Location | Used by |
|--------|----------|---------|
| Legacy | `config/[area]/[spot].yaml` (gitignored) | download, feature engineering, Phase 1/2 training |
| Simplified | `data/spots/[spot]/input/config.yaml` | `python cli.py train` / `evaluate` |

## Structure

```
config/
├── README.md
├── examples/                  # Example configurations (tracked in git)
│   └── dervio.yaml            # Simplified format
└── [area]/                    # Your legacy configurations (gitignored)
    └── [spot].yaml
```

## Simplified Format

Copy `config/examples/dervio.yaml` to `data/spots/[spot]/input/config.yaml` and edit.
Required fields: `spot`, `latitude`, `longitude`.

```yaml
spot: greifensee
name: "Greifensee"
area: "zurich"                     # Still needed: train reads the feature selection
                                   # from data/{area}/{spot}/models/phase2_selections/

latitude: 47.35
longitude: 8.68
timezone: "Europe/Zurich"          # Timestamps in measurements.csv must be in this zone

grid:
  radius_km: 20

target:
  column: "greifensee speed"       # Column name in measurements.csv
  direction_column: "greifensee direction"
```

All options (test date strategy, XGBoost parameters, paths) are documented in
`config/examples/dervio.yaml` and [SIMPLIFIED_ARCHITECTURE.md](../SIMPLIFIED_ARCHITECTURE.md).

## Legacy Format

Required fields: `area`, `spot`, `center.latitude`, `center.longitude`.

```yaml
location_name: "Your Spot Name"
area: "your_area"
spot: "your_spot"

center:
  latitude: 46.083333
  longitude: 9.3

measurement_file: "data/your_area/your_spot/measurements/wind_data.csv"
timezone: "Europe/Rome"

grid:
  radius_km: 20
```

Usage:

```bash
uv run python -m windmojo.data_fetcher --config config/your_area/your_spot.yaml
```

The measurement file format (timestamp `YYYY-MM-DD_HHMM`, local time) is described in
[`data/README.md`](../data/README.md#measurement-file-format).

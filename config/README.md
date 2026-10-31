# Configuration Files

Windmojo currently uses two configuration formats, matching the two data layouts
described in [`data/README.md`](../data/README.md).

| Format | Location | Used by |
|--------|----------|---------|
| Legacy | `config/[area]/[spot].yaml` (gitignored) | download, feature engineering |
| Simplified | `data/spots/[spot]/input/config.yaml` | `python cli.py train` / `evaluate` / `predict` / `serve` |

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
Required fields: `spot`, `latitude`, `longitude`. Minimal example (the top of
`dervio.yaml`):

```yaml
spot: dervio
name: "Dervio, Lake Como"
latitude: 46.083333
longitude: 9.3
timezone: "Europe/Rome"            # Timestamps in measurements.csv must be in this zone

grid:
  radius_km: 20

target:
  column: "dervio speed"           # Column name in measurements.csv
  unit: "km/h"                     # m/s (default), km/h, kn or mph; converted to m/s on loading
  direction_column: "dervio direction"
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
measurement_unit: "km/h"           # Unit of the speed column (default m/s); legacy models and
                                   # evaluations stay in this unit
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

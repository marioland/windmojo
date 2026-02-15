# Configuration Files

This directory contains configuration files for each location.

## Structure

```
config/
├── template.yaml              # Template configuration file (annotated)
├── examples/                  # Example configurations (tracked in git)
│   └── lake_como/
│       └── dervio.yaml
└── [area]/                    # Your configurations (gitignored)
    └── [spot].yaml
```

## Getting Started

### 1. Copy Template

```bash
cp config/template.yaml config/your_area/your_spot.yaml
```

### 2. Edit Configuration

Open `config/your_area/your_spot.yaml` and update:

- `location_name`: Name of your spot
- `area` and `spot`: Match your data directory structure
- `center`: GPS coordinates (latitude, longitude)
- `measurement_file`: Path to your wind data CSV
- `timezone`: Output timezone for forecasts

### 3. Use Configuration

```bash
uv run python -m windmojo.data_fetcher --config config/your_area/your_spot.yaml
```

## Configuration Structure

```yaml
# Location metadata
location_name: "Your Spot Name"
area: "your_area"
spot: "your_spot"

# GPS coordinates
center:
  latitude: 46.083333
  longitude: 9.3

# Measurement data
measurement_file: "data/your_area/your_spot/measurements/wind_data.csv"
timezone: "UTC"

# Training settings
training:
  test_split: 0.15

# Forecast settings
forecast:
  forecast_days: 7
```

See `template.yaml` for complete documentation of all options.

## Note

- Example configs in `examples/` are tracked in git
- Your personal configs in `[area]/[spot].yaml` are gitignored

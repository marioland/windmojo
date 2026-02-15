# Data Directory

This directory contains all location-specific data organized by area and spot.

## Structure

```
data/
└── [area]/                  # Geographic area (e.g., lake_como, san_francisco_bay)
    └── [spot]/              # Specific measurement location
        ├── measurements/    # Your wind measurement data (CSV)
        ├── raw/            # Downloaded GFS forecast data (auto-created)
        ├── features/       # Engineered features (auto-created)
        └── forecasts/      # Generated predictions (auto-created)
```

## Getting Started

1. Create your area/spot directory structure:
   ```bash
   mkdir -p data/your_area/your_spot/measurements
   ```

2. Copy your wind measurement CSV file:
   ```bash
   cp your_wind_data.csv data/your_area/your_spot/measurements/
   ```

3. The other directories (raw, features, forecasts) will be created automatically when you run the pipeline.

## Example

```
data/
└── lake_como/
    ├── dervio/
    │   ├── measurements/
    │   │   └── breva_guru_2024.csv
    │   ├── raw/
    │   ├── features/
    │   └── forecasts/
    ├── colico/
    │   └── ...
    └── domaso/
        └── ...
```

## Note

This directory is gitignored by default (contains large files and user-specific data).
Only the .gitkeep file is tracked to preserve the directory structure.

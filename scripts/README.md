# Utility Scripts

Development and research scripts used during model development. These are provided as reference examples, not as user-facing tools.

## Data Download

| Script | Description |
|--------|-------------|
| `download_may2025_full.py` | Download GFS data for May 2025 |
| `download_may_to_sept_2025.py` | Batch download May-September 2025 |
| `download_synoptic_grid_may2025.py` | Download synoptic grid data |
| `test_synoptic_download_single_day.py` | Test single-day synoptic download |
| `test_batch_download.py` | Test batch download functionality |

## Training & Analysis

| Script | Description |
|--------|-------------|
| `train_model1b_synoptic.py` | Train Model 1b with synoptic features |
| `train_model2_synoptic24.py` | Train Model 2 with 24-hour synoptic data |
| `run_phase2_all_selections.sh` | Run Phase 2 training for all feature selections |
| `compare_sorting_methods.py` | Compare feature sorting strategies |

## Visualization

| Script | Description |
|--------|-------------|
| `plot_feature_importance.py` | Generate feature importance charts |
| `plot_model2_diagram.py` | Plot Model 2 architecture diagram |
| `visualize_pressure_map.py` | Visualize pressure gradient maps |

## API Testing

| Script | Description |
|--------|-------------|
| `experiment_api.py` | Experiment with Open-Meteo API |
| `test_synoptic_api.py` | Test synoptic API endpoints |
| `test_synoptic_features.py` | Validate synoptic feature extraction |
| `test_new_downloader.py` | Test the new grid downloader |

## Note

These scripts were written for development and may contain location-specific defaults (e.g., Lake Como coordinates). Adapt paths and parameters for your own location.

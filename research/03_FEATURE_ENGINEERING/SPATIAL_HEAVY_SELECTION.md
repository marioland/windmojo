# Spatial Heavy Feature Selection

> **Superseded (2026-10-31):** `python cli.py train` uses all features; the selection and
> `windmojo/selections/` were removed. In walk-forward tests `spatial_heavy` was 0.024 m/s
> worse than all features on Greifensee and 0.030 m/s worse on Dervio. See
> [FEATURE_RELEVANCE.md](FEATURE_RELEVANCE.md). Kept for history.

**Date**: 2026-10-30
**Applied to**: `windmojo/selections/spatial_heavy.json` (removed), used by `python cli.py train`

---

## Summary

The production model uses the **Spatial Heavy** feature selection: **164 features** from the
feature pipeline, plus `hour` and `hour_cos` added by `train` = **166 model inputs**.

The selection originally came from the Phase 2 experiments (see
[FEATURE_SELECTION_LESSONS_LEARNED.md](../FEATURE_SELECTION_LESSONS_LEARNED.md)) as a list of
**724 feature names**. Only 164 of them exist in the current feature set; the packaged file
contains exactly these 164, in the order the trained Dervio model uses.

## Provenance

| Source | Features |
|--------|----------|
| Phase 2 selection `data/lake_como/dervio/models/phase2_selections/spatial_heavy.json` | 724 names ("All gradients + circle point raw values, 22 zero-impact features removed") |
| Dervio feature file `dervio_features_phase1.parquet` (2025-11-27) | 356 columns |
| Intersection | 164 |
| Trained Dervio model `data/spots/dervio/models/metadata.json` | 166 (= 164 + `hour`, `hour_cos`) |

Verified: filtering the 724-name list to the Dervio feature columns and appending the hour
features reproduces the feature list in `metadata.json` exactly, including order. Reducing the
packaged list to 164 therefore does not change the Dervio model.

No script in the repo generates the Phase 2 selections; the generator was not committed.

## Why 560 Names Were Missing

The 724-name list was built from an earlier download that fetched all variables at all
9 grid points. The current downloader (`windmojo/data/grid_downloader.py`, `CIRCLE_VARIABLES`)
fetches only **5 variables at the 8 circle points**:

- `pressure_msl`
- `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high`

All other variables are fetched at the center only. Circle-point names such as
`temperature_2m_n`, `wind_speed_850hPa_se` or `geopotential_height_500hPa_w` (and gradients
derived from them, e.g. `temperature_2m_grad_ns`) are no longer produced.

`train` now prints a warning when selected features are missing from the feature file, so a
mismatch like this is visible instead of being skipped silently.

## Composition (164 features)

Categories as reported in `dervio_feature_list_phase1.txt`:

| Category | Count | Content |
|----------|------:|---------|
| SPATIAL_GRADIENT | 99 | `pressure_msl_grad_*` (all 17), `cloud_cover*_grad_*`, magnitudes, directions, asymmetries |
| RAW | 45 | `pressure_msl_{n,ne,…}`, cloud cover at circle points, spatial contrasts, upper-level `wind_speed_*hPa_center`, a few indices |
| ROLLING_STATS | 6 | precipitation sums (12h, 3d, 7d), 3h std of precipitation and temperature, `is_summer` |
| BINARY | 6 | `is_night`, `is_evening`, `is_shoulder`, `is_winter`, `is_weekend`, `clear_sky_factor` |
| PHYSICS | 5 | `bulk_richardson_number`, `lake_thermal_effect`, `thermal_wind_strength_low/mid`, `gust_excess` |
| CYCLICAL | 3 | `hour_sin`, `month_sin`, `wind_direction_sin` |

## Available but Unused (192 features)

The Dervio feature file contains 192 features not in the selection. Candidates worth testing:

| Category | Count | Examples |
|----------|------:|----------|
| RAW (center values, physics indices) | 102 | `pressure_msl_center`, `temperature_2m_center`, `thickness_*`, `temp_lapse_*`, `total_totals_index`, `moisture_flux_*`, `wind_ratio_*` |
| TEMPORAL_LAG | 23 | `surface_pressure_center_lag_{1,3,6}h`, `temperature_2m_center_lag_*` |
| TIME_DERIVATIVE | 13 | `surface_pressure_center_change_{1,3}h`, `pressure_accel`, `temperature_2m_center_change_*` |
| BINARY | 16 | snow and precipitation state flags (mostly constant at Dervio) |
| SPATIAL_GRADIENT | 13 | remaining `cloud_cover_high_grad_*` variants |
| ROLLING_STATS | 10 | snowfall rolling sums, temperature rolling means |
| PHYSICS | 10 | `buoyancy_frequency`, `daytime_heating_potential`, `gust_factor`, soil moisture |
| CYCLICAL | 5 | `hour_cos` (added by train anyway), `month_cos`, `wind_direction_cos` |

Pressure tendency (`surface_pressure_center_change_*`, `pressure_accel`) and vertical structure
(`thickness_*`, `temp_lapse_*`) are physically plausible drivers of local wind and are the most
obvious candidates for an extended selection. Adding them changes the model, so it needs a
separate evaluation against the current MAE.

## Implications for New Spots

Feature names are spot-independent, so the same selection applies everywhere. If a spot's
feature file lacks some of the 164 features (different pipeline version, missing variables),
`train` warns and trains on the remaining ones.

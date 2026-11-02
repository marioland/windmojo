# Changelog

## Unreleased

**Training**
- One XGBoost model on all features; the Phase 1/2 feature selection and the packaged
  `spatial_heavy` selection were removed (no gain in walk-forward tests on two lakes)
- Training and live forecasts get identical features (no outlier capping, look-backs of
  at most 7 days); features that can never help were removed
- Weather model surface winds (10-180 m) are features again
- 2 m temperature at the 8 perimeter points, with 19 gradient features (land-sea and
  lake-land contrasts, cold air pools, in any direction). Neutral on Greifensee and Dervio,
  meant for coastal spots. Raw files downloaded earlier lack them; re-download or backfill
  the perimeter temperature before retraining
- Early stopping on validation days instead of the test days
- Timing-tolerant loss: a wind peak forecast up to ±3 h off is not punished twice
- `cli.py train --until` keeps later days unseen
- Fixed time-based features, target leakage and model selection

**Forecasts**
- `cli.py predict`: live forecast as JSON, with deployment guide ([DEPLOYMENT.md](DEPLOYMENT.md))
  and a ready-made forecast page (`web/index.html`)
- `cli.py serve`: local forecast page with update button, past days ("look back"),
  hindcast and CSV downloads
- Every live forecast is archived; the raw weather forecast for tomorrow and the day after
  is kept for training on longer lead times
- Measurement units m/s, km/h, kn and mph (`target.unit`)

**Clean-up**
- The command-line interface moved into the package (`windmojo/cli.py`); `windmojo` is
  now the same command as `python cli.py`. The old `windmojo download`/`cache` commands
  for the discontinued synoptic grid experiment were removed
- Removed the synoptic grid experiment (Model 1b: downloaders, pressure centers,
  teleconnections, regional aggregation) and the development scripts in `scripts/`
- Documentation corrected and updated (results, grid variables, download limits, lead time)

## v0.2.0 (2025-11-27)

- Unified model architecture (one model for all hours)
- 5 feature selection approaches
- Physics-driven regime discovery
- Enhanced feature engineering (498 features)
- Test week validation with interactive visualizations
- All models beat GFS baseline on held-out data
- Example: Lake Como Lecco (8.9% improvement)

## v0.1.0 (2025-11-02)

- Initial release
- Generic area/spot data structure
- XGBoost training pipeline
- Feature engineering and selection
- CLI for data fetching and forecasting

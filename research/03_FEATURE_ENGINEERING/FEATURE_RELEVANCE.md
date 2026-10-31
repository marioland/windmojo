# Feature Relevance

**Date**: 2026-10-31
**Applies to**: feature pipeline (`windmojo/features/`), `python cli.py train`
**Data**: Greifensee (Eawag platform, 2021-03 to 2026-10, 36,500 hours) and Dervio
(Lake Como, 2025-01 to 2025-10, 4,300 hours)

---

## Summary

1. **Use all features.** No feature selection beat the model with all features, neither
   the packaged `spatial_heavy` list nor the Phase 1 selections nor a selection made inside
   each training fold. XGBoost copes with ~380 features, including redundant and weak ones.
2. **The model corrects the GFS wind forecast.** GFS wind at 10 m, gusts and the lowest
   pressure level carry about half of the model's gain. Everything else (pressure gradients,
   clouds, stability, precipitation) adds smaller, real corrections.
3. **Keep the GFS surface winds of the center point.** The pipeline used to drop them
   (`wind_speed_10m_center` & co., a Phase 1 relic). Keeping them is the largest single gain
   found: −0.014 m/s MAE, −0.06 m/s at strong wind.
4. **Removing worthless features changes nothing measurable.** It still pays off: shorter
   feature list, readable importances, no features that can never work.
5. **A feature must be computable identically in training and live.** Two bugs violated
   this (outlier capping over the whole input, unbounded look-backs).
6. **Strong wind needs a probability, not a sharper point forecast.** The model is
   calibrated (when it forecasts ≥ 4 m/s, it is right on average), but a single value
   detects only half of the hours with ≥ 4 m/s.

## How it was tested

Same setup as `python cli.py train`: one XGBoost model for all hours (06–23), parameters
of `get_training_config`, NaN filled with training medians. Early stopping on every 10th
training day, never on the test days.

- **Walk-forward:** expanding training window with at least 2 years, test blocks of one
  quarter (15 folds, 2023-03 to 2026-10) or one year (4 folds), 7 days embargo between
  training and test
- **Block cross-validation** on 3 years (12 quarters) and on Dervio (9 months)
- 3 seeds per fold; differences are paired per fold, reported with a 95 % t-interval.
  A difference counts only if the interval excludes 0.
- MAE overall and for hours with measured wind ≥ 4 m/s (15 % of the hours at Greifensee,
  38 % at Dervio), because those are the hours that matter for water sports

## Feature selection does not help

Difference in MAE (m/s) to the model with all features of the same pipeline (370 features
on Greifensee). Negative = better.

| Feature set | Greifensee 5.6 y, 15 quarters | Greifensee 3 y, block CV | Dervio, block CV |
|---|---|---|---|
| All features (reference) | 0.706 | 0.688 | 1.547 |
| `spatial_heavy` (164, chosen on Dervio) | **+0.024 ± 0.011** | **+0.025 ± 0.013** | **+0.030 ± 0.027** |
| All minus worthless features | +0.001 ± 0.003 | +0.001 ± 0.002 | +0.003 ± 0.008 |
| Phase 1 selection `gfs_topn` (75) | +0.002 ± 0.006 | −0.002 ± 0.003 | – ¹ |
| Top 75 by gain, selected inside each fold | +0.002 ± 0.005 | −0.001 ± 0.003 | −0.005 ± 0.015 |
| GFS 10 m wind, linearly calibrated | +0.103 | +0.113 | +0.143 |

At ≥ 4 m/s, `spatial_heavy` is +0.09 m/s worse on Greifensee (in 0 of 15 quarters better).

¹ The stored Dervio Phase 1 list came from an older pipeline; only 17 of its 75 names
still exist.

**Lessons:**

- `spatial_heavy` lost even on Dervio, the spot it was chosen for: it misses the most
  important GFS wind features, and a list chosen once goes stale when the pipeline changes.
- A selection evaluated on the data it was chosen on looks better than it is. `gfs_topn`
  was 0.006 m/s better in a 3-year walk-forward, on exactly the 3 years it had been selected
  on. On 5.6 years the advantage vanished, also for the same test quarters.
- Only 164 of 369 features beat the strongest of 10 random noise columns in importance,
  yet keeping the other 205 costs nothing. Low importance is no reason to remove a feature.

## What drives the wind at Greifensee

From the gain of the model with all features and the correlation with the measured wind:

| Group | Features | Share of gain |
|---|---|---|
| GFS wind and shear | 48 | 52 % |
| Clouds | 132 | 19 % |
| Temperature and humidity | 42 | 7 % |
| Pressure and pressure gradients | 35 | 6 % |
| Precipitation and snow | 46 | 6 % |
| Stability, radiation, soil, time, other | 66 | 10 % |

- **GFS wind is the backbone.** `effective_wind_10m`, `wind_power_10m` and the gust lag
  each correlate with ρ ≈ 0.65 with the measured wind and together hold 36 % of the gain.
- **Greifensee is a frontal and gradient wind lake, not a thermal lake.** At ≥ 4 m/s the
  wind blows 47 % from W/NW (fronts), 27 % from S/SE and 11 % from NE (Bise). Solar radiation
  correlates negatively with the wind (ρ = −0.13), low clouds positively (+0.17). The daily
  thermal cycle is visible but weak (afternoon 2.8 m/s vs. morning 2.1 m/s).
- **The pressure gradient along the lake axis matters.** The strongest pressure feature is
  `pressure_msl_grad_nw_se` (NW–SE, the lake axis), the second `pressure_msl_grad_ne_sw`
  (Bise direction).
- **Upper air: 850/700 hPa yes, 250 hPa no.** Winds at 850/700 hPa help (mixed down in
  unstable conditions); everything at jet level (250 hPa) is below the noise level for a
  same-day forecast.
- **Clouds: many features, little effect.** 132 cloud features (a third of all) give 19 %
  of the gain, none is in the top 20. Gradients of a single cloud layer between points
  20 km apart act as rough weather-type indicators.

Other lakes differ (Dervio is a thermal lake with Breva/Tivano), which is one more reason
not to ship a fixed feature list: the model finds the relevant features per spot.

## What makes a feature worthless

| Cause | Examples | Action |
|---|---|---|
| Bug makes it constant | Outlier capping at 1/99 % turned rare events and 0/1 flags into 0 (`snowfall_center`, `is_snowing`, `precip_is_snow`); snow thresholds in cm while `snow_depth` comes in m (`is_snow_on_ground` and 8 dependents) | fixed |
| Differs between training and live | `hours_since_snowfall` counted back months in training, 7 days live (78 % of the rows differed); capping only in training (~200 features) | fixed: look-backs capped at 168 h, no capping |
| Same formula twice | `gust_intensity` = `gust_factor`, `saturation_deficit_2m` = 100 − RH, `clear_sky_factor`, `solar_heating_effective`, `temp_850_freezing_margin`, `pressure_accel`, `cloud_trend_3h`, `buoyancy_frequency` (a scaled temperature change, not a Brunt–Väisälä frequency) | removed |
| Physically meaningless | `day_of_week`, `is_weekend` | removed |
| API delivers nothing | `evapotranspiration` (always 0) | removed from download |
| Not computable live | `is_old_snow` (needs > 7 days of history) | removed |
| Information already in another form | absolute sea-level pressure of the 8 circle points (\|r\| ≥ 0.998 with the center; the gradients carry the information) | removed |
| Wrongly excluded | GFS surface winds of the center point, excluded as "measurement leak", but they are forecasts | restored: largest gain |

**Kept on purpose:** strongly correlated features (\|r\| 0.95–0.999), rare scenarios
(snow, showers), jet-level and soil features, features below the noise level. They do not
hurt, and they may matter at other spots (snow at alpine lakes, soil moisture at thermal
spots).

**Rules for new features:**

- Look back at most 7 days (`DEFAULT_HISTORY_DAYS` of the live forecast,
  `MAX_HOURS_SINCE_EVENT`). Longer windows cannot be computed live.
- No statistics over the whole input (percentiles, min/max normalization, z-scores): the
  pipeline sees years in training and 14 days live. `tests/test_pipeline.py::TestLiveParity`
  checks this.
- Check units against the Open-Meteo documentation (`snow_depth` in m, `snowfall` in cm,
  wind in km/h).
- Changing feature names breaks trained models (they select features by name);
  `tests/feature_names.txt` makes such changes deliberate, and models must be retrained.

## Robustness and cost

Greifensee, test year 2025-10 to 2026-10, live path simulated with weekly 14-day chunks:

| Scenario | All features | `spatial_heavy` | Phase 1 selection |
|---|---|---|---|
| One grid point missing (average of 8) | ±0.000 | ±0.000 | +0.001 |
| All pressure levels missing | +0.023 | +0.017 | +0.026 |
| Same at ≥ 4 m/s | +0.082 | +0.167 | +0.117 |

More features do not make the model more fragile: missing features are filled with
training medians, and redundant features stand in for each other. The forecast JSON
reports `model.missing_features`.

Cost of all features: training 3.0 s instead of 1.3 s, 26 ms instead of 13 ms per weekly
forecast. The model size (1.1–1.5 MB) depends on the number of trees, not of features.

## Open: strong wind

Hours with measured wind ≥ 4 m/s are about 1 m/s lower in the forecast (MAE 1.20 m/s
there, GFS calibrated 1.63 m/s). That is regression to the mean, not a model error:
selecting hours by the measured value picks those where the wind turned out stronger than
forecast. Conditioned on the forecast, the model is calibrated (test year 2025-10 to
2026-10, bias −0.07 m/s when it forecasts ≥ 4 m/s):

| Forecast | Hours | Mean forecast | Mean measured |
|---|---|---|---|
| 1–2 m/s | 3,033 | 1.49 m/s | 1.45 m/s |
| 2–3 m/s | 1,809 | 2.46 m/s | 2.50 m/s |
| 3–4 m/s | 876 | 3.43 m/s | 3.54 m/s |
| 4–5 m/s | 328 | 4.45 m/s | 4.55 m/s |
| > 5 m/s | 233 | 5.85 m/s | 5.87 m/s |

The limit is what a single value can say. Using "forecast ≥ 4 m/s" as decision detects
52 % of the hours with ≥ 4 m/s (21 % false alarms); "forecast ≥ 3 m/s" detects 84 % (50 %
false alarms). Weighting strong wind or changing the loss would only move along this
trade-off and break the calibration. The next lever is a probabilistic output (chance of
≥ 4 m/s per hour, or quantiles) plus a gust forecast. Feature work moved strong wind
little; the best step (GFS surface winds) reduced the MAE at ≥ 4 m/s by 0.06 m/s.

**Showers and thunderstorms.** The measuring platform (temperature, pressure, rain every
15 min) shows about 20 gust fronts per summer at Greifensee. In the hour of the gust front
the hourly mean is 5.2 m/s (best 15-min mean 7.3 m/s, gusts 12.3 m/s); the model forecasts
4.3 m/s and spreads the rise over three hours, because the weather model places cells about
an hour and a few kilometres off. When the weather model shows a thunderstorm code, the
model raises its forecast and is right on average. Gust fronts are 7 % of the summer hours
with ≥ 4 m/s and 4 % of the summer error. Candidate inputs: CAPE, lifted index and
convective inhibition (available from Open-Meteo, not downloaded yet) and precipitation at
the circle points (an approaching cell).

## Related

- [SPATIAL_HEAVY_SELECTION.md](SPATIAL_HEAVY_SELECTION.md) – the former packaged selection (superseded)
- [FEATURE_SELECTION_LESSONS_LEARNED.md](../FEATURE_SELECTION_LESSONS_LEARNED.md) – Phase 2 experiments 2025 (superseded)
- The analysis scripts (walk-forward comparison, noise test, live simulation) are spot
  tooling and not part of the repository.

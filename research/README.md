# Windmojo Research Notes

Background on wind physics, feature ideas and the experiments behind Windmojo. These are
working notes, not user documentation: several documents describe approaches that were
tested and dropped. For how to use Windmojo, see the [main README](../README.md).

**Start here:** [FEATURE_RELEVANCE.md](03_FEATURE_ENGINEERING/FEATURE_RELEVANCE.md) holds
the current findings (walk-forward tests on Greifensee and Dervio): which features matter,
why the model uses all of them, and where its limits are. The README section
[Theory vs. Practice](../README.md#theory-vs-practice) compares the notes below with what
the measurements show.

## Status of the Documents

- **Current**: describes the code and results as they are
- **Background**: physics and ideas that are still valid as background; rankings or
  feature counts in them were not confirmed
- **Historical**: approaches that were tested and dropped; the code is removed (see the
  git history)

| Document | Status | Content |
|----------|--------|---------|
| [03 FEATURE_RELEVANCE.md](03_FEATURE_ENGINEERING/FEATURE_RELEVANCE.md) | Current | Feature comparisons, what drives the wind at Greifensee, strong wind as the open weakness, rules for new features |
| [01 WIND_PHYSICS_REFERENCE.md](01_FUNDAMENTALS/WIND_PHYSICS_REFERENCE.md) | Background | Wind physics from pressure gradients to Alpine föhn and valley winds (~6,700 words) |
| [02 SYNOPTIC_PATTERNS_THEORY.md](02_SYNOPTIC_PATTERNS/SYNOPTIC_PATTERNS_THEORY.md) | Background | Large-scale European weather patterns and ways to encode them as features. In the model, synoptic features add nothing measurable |
| [03 FEATURE_ENGINEERING_COMPREHENSIVE_DE.md](03_FEATURE_ENGINEERING/FEATURE_ENGINEERING_COMPREHENSIVE_DE.md) | Background (German) | Detailed feature engineering guide. Its expected importance ranking assumed no weather model wind as input and was not confirmed |
| [04 OPEN_METEO_API_EVALUATION.md](04_DATA_SOURCES/OPEN_METEO_API_EVALUATION.md) | Background | Why the Historical Forecast API (and not the Ensemble API) is used for training |
| [04 openapi.yml](04_DATA_SOURCES/openapi.yml) | Reference | Copy of Open-Meteo's OpenAPI specification (CC BY 4.0, [open-meteo/open-meteo](https://github.com/open-meteo/open-meteo)). Official documentation: [Forecast API](https://open-meteo.com/en/docs), [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) |
| [05 AUTORESEARCH.md](05_TOOLS_AND_METHODS/AUTORESEARCH.md) | Idea | An autonomous experiment loop after karpathy/autoresearch (roadmap) |
| [FEATURE_SELECTION_LESSONS_LEARNED.md](FEATURE_SELECTION_LESSONS_LEARNED.md) | Historical | Phase 2 comparison of 24+ feature selections (single test splits; superseded) |
| [03 SPATIAL_HEAVY_SELECTION.md](03_FEATURE_ENGINEERING/SPATIAL_HEAVY_SELECTION.md) | Historical | The former packaged selection of 164 features (superseded) |
| [02 SYNOPTIC_FEATURES_IMPLEMENTATION.md](02_SYNOPTIC_PATTERNS/SYNOPTIC_FEATURES_IMPLEMENTATION.md) | Historical | Design of the synoptic grid experiment (Model 1b): European pressure grid, pressure centers, teleconnections |
| [02 FEATURE_SELECTION_STRATEGY.md](02_SYNOPTIC_PATTERNS/FEATURE_SELECTION_STRATEGY.md) | Historical | Atomic feature groups and pressure center sorting for Model 1b/2 |
| [02 SYNOPTIC_ONLY_LESSONS_LEARNED.md](02_SYNOPTIC_PATTERNS/SYNOPTIC_ONLY_LESSONS_LEARNED.md) | Historical | Post-mortem: why synoptic-only models (1b, 2) failed |

## Adding Documents

- Put a short header on top: date, data used, status
- State how a result was tested (walk-forward or a single split, which spot); untested
  rankings belong under "Background"
- When a finding is superseded, add a note at the top that links to the newer document
  instead of deleting the old one
- File names in `UPPER_SNAKE_CASE.md`, with `_DE` for German documents
- Add the document to the table above

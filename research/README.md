# Windmojo Research Documentation

**Comprehensive research materials for wind prediction using machine learning**

This directory contains theoretical background, implementation guides, and lessons learned from developing the Windmojo wind prediction system. The documentation is organized by topic to facilitate easy navigation and understanding.

---

## Quick Navigation

### [01 - Fundamentals](#01---fundamentals)
Core physics and atmospheric dynamics

### [02 - Synoptic Patterns](#02---synoptic-patterns)
Large-scale weather patterns and their encoding for ML

### [03 - Feature Engineering](#03---feature-engineering)
Comprehensive feature engineering approaches

### [04 - Data Sources](#04---data-sources)
API documentation and data source evaluation

---

## 01 - Fundamentals

### [WIND_PHYSICS_REFERENCE.md](01_FUNDAMENTALS/WIND_PHYSICS_REFERENCE.md)

**Complete guide to atmospheric wind physics, from global patterns to Alpine föhn**

**Topics covered:**
- Fundamental forces creating wind (pressure gradient, Coriolis, friction)
- How wind forces interact (vector addition, geostrophic balance, thermal wind)
- Global circulation and prevailing winds (Hadley cells, jet streams)
- Coastal winds and land-sea contrast (sea/land breezes, Mediterranean winds)
- Mountain and valley wind systems (anabatic/katabatic flows)
- Föhn winds in the Alpine region (mechanisms, locations, impacts)
- Thermal winds and convective phenomena (thermals, dust devils)
- Practical applications for various user groups

**Key insights:**
- Comprehensive consolidation of wind physics from basic principles to complex phenomena
- Detailed coverage of Alpine wind systems (critical for Lake Como predictions)
- Both accessible explanations and scientific depth
- ~30,000 words combining three previous documents

**When to use:**
- Understanding physical mechanisms behind wind patterns
- Interpreting model predictions in terms of atmospheric processes
- Designing physics-informed features for ML models
- Educational reference for team onboarding

---

## 02 - Synoptic Patterns

### [SYNOPTIC_PATTERNS_THEORY.md](02_SYNOPTIC_PATTERNS/SYNOPTIC_PATTERNS_THEORY.md)

**Understanding and encoding large-scale weather patterns for machine learning**

**Topics covered:**
- Top 10 European weather indicators (NAO, Azores High, Genoa Low, etc.)
- Top 5 Central European wind indicators ranked by direct influence
- Classic Central European wind patterns (westerly, föhn, easterly, calm)
- 10 approaches for encoding spatial weather patterns into ML features
- Detailed feature engineering for each top indicator
- Advanced composite features and interaction terms

**Key insights:**
- Synoptic patterns control large-scale wind forcing
- Multiple representation principle: position, intensity, geometry, motion, gradients
- Recommended feature count: ~40-50 features capture synoptic control
- Balance between meteorological meaningfulness and ML performance

**When to use:**
- Designing synoptic-scale features for wind prediction models
- Understanding which large-scale patterns affect local winds
- Choosing between different feature encoding approaches
- Interpreting model behavior in terms of weather patterns

### [SYNOPTIC_FEATURES_IMPLEMENTATION.md](02_SYNOPTIC_PATTERNS/SYNOPTIC_FEATURES_IMPLEMENTATION.md)

**Grid-to-features transformation pipeline for Model 1b**

**Topics covered:**
- Data parameters and coverage assessment (30°N-75°N, 25°W-40°E grid)
- Pressure center parameterization algorithm
- Regional aggregation approach with defined meteorological regions
- Teleconnection indices calculation (NAO, EA, SCAND, Alpine-specific)
- Complete feature set: ~83 features from 18,532 grid points
- Implementation roadmap and expected insights

**Key insights:**
- 0.4° resolution provides ~18,532 points per time step
- Single daily snapshot (12:00 UTC) limits dynamic event analysis
- Three complementary approaches extract different pattern aspects
- Feature importance analysis reveals which patterns matter most

**When to use:**
- Implementing synoptic feature extraction from pressure grids
- Understanding trade-offs in spatial vs. temporal resolution
- Building Model 1b (synoptic-only wind prediction)
- Designing grid-based feature pipelines

### [FEATURE_SELECTION_STRATEGY.md](02_SYNOPTIC_PATTERNS/FEATURE_SELECTION_STRATEGY.md)

**Atomic feature grouping and distance-based pressure center sorting**

**Topics covered:**
- Strategy for organizing features into atomic groups
- Distance-based sorting of pressure centers
- Systematic approach to feature selection

**When to use:**
- Organizing large feature sets systematically
- Implementing pressure center detection and ranking
- Designing feature selection experiments

### [SYNOPTIC_ONLY_LESSONS_LEARNED.md](02_SYNOPTIC_PATTERNS/SYNOPTIC_ONLY_LESSONS_LEARNED.md)

**Why synoptic-only approach failed (Models 1b & 2) - detailed post-mortem**

**Topics covered:**
- Detailed analysis of why synoptic-only models underperformed
- Comparison with local feature models
- Specific failure modes and their causes
- Recommendations for future approaches

**Key insights:**
- Critical lessons from discontinued approaches
- Importance of local vs. synoptic features
- Understanding model limitations

**When to use:**
- Learning from past failures before trying similar approaches
- Understanding why local features often dominate for lake wind prediction
- Designing hybrid models that balance synoptic and local features
- Avoiding known pitfalls in synoptic-scale modeling

---

## 03 - Feature Engineering

### [FEATURE_RELEVANCE.md](03_FEATURE_ENGINEERING/FEATURE_RELEVANCE.md)

**Which features matter, and why the model uses all of them**

- Walk-forward comparison of feature selections on Greifensee and Dervio: none beats all features
- What drives the wind at Greifensee (GFS wind, pressure gradient along the lake axis, clouds)
- What makes a feature worthless, which bugs caused it, and rules for new features
- Robustness against missing features; strong wind as the open weakness

### [SPATIAL_HEAVY_SELECTION.md](03_FEATURE_ENGINEERING/SPATIAL_HEAVY_SELECTION.md)

**Superseded:** origin and contents of the former packaged feature selection (164 features)

- Why the original 724-name Phase 2 list reduces to 164 existing features
- Composition by category
- Available but unused features (pressure tendency, thickness, lapse rates) as experiment candidates

### [FEATURE_ENGINEERING_COMPREHENSIVE_DE.md](03_FEATURE_ENGINEERING/FEATURE_ENGINEERING_COMPREHENSIVE_DE.md)

**2000-line comprehensive feature engineering guide (German language) for XGBoost wind prediction**

**Topics covered (in German):**
- 6 major feature categories
- Physics-based feature design
- Gradient calculations
- Temporal features
- Thermal features
- Comprehensive XGBoost-specific considerations

**Key insights:**
- Extremely detailed treatment of feature engineering
- Physics-based approach ensuring meteorological validity
- Specific to XGBoost wind prediction applications
- ~57KB of dense technical content

**When to use:**
- Deep dive into feature engineering methodology
- Understanding German-language feature engineering approaches
- Comprehensive reference for XGBoost wind modeling
- Advanced feature design patterns

**Note:** English summary/table of contents to be added in future version.

---

## 04 - Data Sources

### [OPEN_METEO_API_REFERENCE.md](04_DATA_SOURCES/OPEN_METEO_API_REFERENCE.md)

**Technical API documentation for Open-Meteo APIs**

**Topics covered:**
- Historical Forecast API documentation
- Ensemble API comparison
- Variables available (pressure levels, surface, derived)
- Rate limits and access patterns
- Grid download syntax
- API endpoints and parameters

**When to use:**
- Setting up data download pipelines
- Understanding available variables and their meanings
- Troubleshooting API access issues
- Planning data collection strategy

### [OPEN_METEO_API_EVALUATION.md](04_DATA_SOURCES/OPEN_METEO_API_EVALUATION.md)

**Historical Forecast API vs Ensemble API comparison for ML use cases**

**Topics covered:**
- Detailed comparison of two API options
- ML training data requirements
- Recommendations for different use cases
- Trade-offs between data sources
- GFS model characteristics

**Key insights:**
- Historical Forecast API recommended for ML training
- Ensemble API limitations for historical data
- Considerations for production forecasting vs. model training

**When to use:**
- Choosing between Open-Meteo API options
- Understanding data source trade-offs for ML
- Planning data collection strategy
- Evaluating whether Open-Meteo meets project requirements

### [openapi.yml](04_DATA_SOURCES/openapi.yml)

**OpenAPI specification for Open-Meteo weather forecast API**

**Standard OpenAPI format** - can be used with API client generators and documentation tools.

**When to use:**
- Generating API clients automatically
- API exploration with tools like Swagger UI
- Reference for exact endpoint specifications

---

## Document History

### Reorganization: 2025-11-27

**Previous structure:** 12 files, flat organization, significant redundancy

**Changes made:**
1. **Consolidated wind physics** (3 files → 1)
   - Merged: WINDS_WE_KNOW.md + WIND_FACTORS.md + UNDERSTANDING_WIND.md
   - Result: WIND_PHYSICS_REFERENCE.md (~30KB, eliminated ~40% duplication)

2. **Organized synoptic patterns** (3 files → 2 + 2 supporting)
   - Merged theory: EUROPEAN_GENERAL_WEATHER.md + GENERAL_WEATHER_SITUATIONS_FOR_ML.md
   - Result: SYNOPTIC_PATTERNS_THEORY.md
   - Renamed: ENGINEER_GENERAL_WEATHER.md → SYNOPTIC_FEATURES_IMPLEMENTATION.md
   - Renamed: ATOMIC_FEATURE_SELECTION.md → FEATURE_SELECTION_STRATEGY.md
   - Renamed: FAILED_SYNOPTIC_APPROACH.md → SYNOPTIC_ONLY_LESSONS_LEARNED.md

3. **Organized feature engineering** (1 file → 1 renamed)
   - Renamed: CLAUDE_FEATURE_ENGINEERING.md → FEATURE_ENGINEERING_COMPREHENSIVE_DE.md

4. **Organized data sources** (3 files → 3 renamed)
   - Renamed: OPEN_METEO_API.md → OPEN_METEO_API_REFERENCE.md
   - Renamed: DATA_SOURCE_EVAL.md → OPEN_METEO_API_EVALUATION.md
   - Kept: openapi.yml (standard format)

5. **Created folder hierarchy**
   - 01_FUNDAMENTALS/
   - 02_SYNOPTIC_PATTERNS/
   - 03_FEATURE_ENGINEERING/
   - 04_DATA_SOURCES/

**Current structure:** 9 files + README, organized by topic

**Benefits:**
- 25% fewer files
- ~40% less content redundancy
- Clear hierarchical organization
- Descriptive file names indicating actual content
- Single source of truth for each topic
- Easier navigation and onboarding
- Better maintainability

---

## Usage Recommendations

### For New Team Members

**Start here:**
1. [WIND_PHYSICS_REFERENCE.md](01_FUNDAMENTALS/WIND_PHYSICS_REFERENCE.md) - Understand the fundamentals
2. [SYNOPTIC_PATTERNS_THEORY.md](02_SYNOPTIC_PATTERNS/SYNOPTIC_PATTERNS_THEORY.md) - Learn about large-scale patterns
3. [OPEN_METEO_API_REFERENCE.md](04_DATA_SOURCES/OPEN_METEO_API_REFERENCE.md) - Understand data sources

### For Model Development

**Feature engineering:**
1. [SYNOPTIC_PATTERNS_THEORY.md](02_SYNOPTIC_PATTERNS/SYNOPTIC_PATTERNS_THEORY.md) - Design synoptic features
2. [FEATURE_ENGINEERING_COMPREHENSIVE_DE.md](03_FEATURE_ENGINEERING/FEATURE_ENGINEERING_COMPREHENSIVE_DE.md) - Deep dive (German)
3. [SYNOPTIC_FEATURES_IMPLEMENTATION.md](02_SYNOPTIC_PATTERNS/SYNOPTIC_FEATURES_IMPLEMENTATION.md) - Implementation details

**Avoiding pitfalls:**
1. [SYNOPTIC_ONLY_LESSONS_LEARNED.md](02_SYNOPTIC_PATTERNS/SYNOPTIC_ONLY_LESSONS_LEARNED.md) - Learn from failures

### For Data Collection

**API setup:**
1. [OPEN_METEO_API_EVALUATION.md](04_DATA_SOURCES/OPEN_METEO_API_EVALUATION.md) - Choose API
2. [OPEN_METEO_API_REFERENCE.md](04_DATA_SOURCES/OPEN_METEO_API_REFERENCE.md) - Implementation details
3. [openapi.yml](04_DATA_SOURCES/openapi.yml) - API specification

### For Interpretation

**Understanding predictions:**
1. [WIND_PHYSICS_REFERENCE.md](01_FUNDAMENTALS/WIND_PHYSICS_REFERENCE.md) - Physical mechanisms
2. [SYNOPTIC_PATTERNS_THEORY.md](02_SYNOPTIC_PATTERNS/SYNOPTIC_PATTERNS_THEORY.md) - Pattern controls

---

## Maintenance Guidelines

### When Adding New Documents

1. **Choose appropriate folder** based on topic
2. **Use descriptive names** that indicate actual content (not "GENERAL" or author names)
3. **Check for redundancy** with existing documents before creating new ones
4. **Update this README** with navigation links and descriptions

### When Updating Documents

1. **Check cross-references** to ensure they still work
2. **Update modification dates** at end of documents
3. **Note major changes** in document history
4. **Consolidate redundancy** if overlap is discovered

### Naming Conventions

- Use **UPPER_SNAKE_CASE.md** for research documents (following existing convention)
- Use **descriptive names** that indicate content (e.g., `WIND_PHYSICS_REFERENCE` not `WINDS_WE_KNOW`)
- Avoid **author names** in file names (e.g., not `CLAUDE_FEATURE_ENGINEERING`)
- Avoid **vague terms** like "GENERAL" or "UNDERSTANDING" without specificity
- Include **language suffix** if not English (e.g., `_DE` for German)

---

## Contributing

When adding research documentation:

1. **Determine if existing docs cover the topic** - avoid duplication
2. **Choose clear, descriptive file name**
3. **Place in appropriate folder** (01-04)
4. **Include table of contents** for documents >1000 words
5. **Add "When to use" section** explaining document's purpose
6. **Update this README** with navigation entry
7. **Cross-reference related documents** where relevant

---

## Contact

For questions about research documentation organization or content, contact the Windmojo development team.

---

**Last Updated:** 2025-11-27
**Documentation Version:** 2.0 (Post-reorganization)
**Total Documents:** 9 core + 1 README
**Total Size:** ~180 KB (down from ~220 KB pre-reorganization)

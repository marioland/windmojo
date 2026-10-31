# Feature Engineering für XGBoost Wind-Geschwindigkeits-Vorhersage

> **Hintergrund (2026-10-31):** Die Physik und die Feature-Ideen gelten weiterhin. Die hier
> erwartete Rangfolge (Druckgradienten 30–40 %, synoptische Indizes 20–30 %) hat sich nicht
> bestätigt: Der Leitfaden ging davon aus, dass der Bodenwind des Wettermodells kein Input
> ist. Gemessen trägt dieser rund die Hälfte, Druckgradienten 4–7 %. Aktueller Stand:
> [FEATURE_RELEVANCE.md](FEATURE_RELEVANCE.md).

## Strategische Analyse: Von historischen Forecast-Daten zu trainierbaren Features

**Autor:** Windmojo Contributors
**Datum:** 2025-11-15
**Kontext:** Windmojo ML-System für lokale Windvorhersage

---

## Executive Summary

Dieses Dokument analysiert die fundamentale Herausforderung des Windmojo-Projekts: **Wie bereiten wir historische Forecast-Daten auf, um einen XGBoost-Modell für `wind_speed_10m` zu trainieren, OHNE die forecast `wind_speed_10m` selbst als Feature zu verwenden?**

### Kernthese

**Wind-Geschwindigkeit ist keine isolierte Variable, sondern eine emergente Eigenschaft aus dem Zusammenspiel von:**
- Großräumigen europäischen Wetterlagen (synoptische Skala)
- Druckgradienten (lokal und regional)
- Thermischen Effekten (solar, topographisch, See/Land)
- Vertikalen atmosphärischen Profilen (Scherung, Stabilität)
- Zeitlichen Mustern (diurnal, saisonal)

### Strategischer Ansatz

1. **Zweistufiges Modell-Training** (300-500 Features → Top 50-100 Features)
2. **Physik-basierte Feature-Kategorien** kombiniert mit datengetriebener Selektion
3. **Räumliche Gradienten** zur Erfassung der Großwetterlage
4. **Lag- und Diff-Features** für zeitliche Dynamik
5. **Zyklische Kodierung** für wiederkehrende Muster
6. **Windrichtungs-Gradienten** als Proxy für Strömungsmuster

---

## 1. Die Herausforderung: Wind ohne Wind vorhersagen

### 1.1 Problem-Statement

**Gegeben:**
- Historische GFS-Forecast-Daten (2-5 Jahre)
- Messdaten von `wind_speed_10m` an einem spezifischen Ort in Europa
- 21 Gitterpunkte (5×5 Grid minus Ecken) mit 14+ atmosphärischen Variablen
- Zeitliche Auflösung: stündlich

**Gesucht:**
- XGBoost-Modell, das `wind_speed_10m` aus atmosphärischen Bedingungen vorhersagt
- **OHNE** die GFS-forecast `wind_speed_10m` direkt zu verwenden

**Später in Produktion:**
- Konsumieren eines aktuellen Weather Forecasts
- Ableitung der lokalen Windgeschwindigkeit aus atmosphärischen Parametern

### 1.2 Warum nicht einfach GFS wind_speed_10m verwenden?

**Drei fundamentale Gründe:**

1. **Räumliche Auflösung:** GFS hat ~25 km Gitterweite, lokale Effekte (See, Tal, Topographie) fehlen
2. **Modell-Bias:** GFS systematisch zu schwach bei thermischen Winden (Breva, Ora, Seebrise)
3. **Physikalisches Verständnis:** ML soll lernen, WIE Wind entsteht, nicht nur GFS-Fehler korrigieren

**Ziel:** Ein Modell, das die physikalischen Mechanismen der Windentstehung lernt und auf jeden Weather Forecast anwendbar ist.

---

## 2. Physikalische Grundlagen: Wie entsteht Wind?

### 2.1 Die fundamentale Kausalkette

```
Ungleiche solare Erwärmung
    ↓
Temperaturgradienten (horizontal & vertikal)
    ↓
Druckgradienten (Pressure Gradient Force)
    ↓
Luftbewegung (Wind)
    ↓
Modifikation durch:
  - Coriolis-Kraft (Ablenkung)
  - Reibung (Bodennähe)
  - Topographie (Kanalisierung, Blockierung)
  - Thermische Zirkulation (lokal)
```

### 2.2 Multi-Skalen-Interaktion

**Synoptische Skala (1000-3000 km):**
- Europäische Großwetterlagen
- Azorenhoch, Islandtief, Genuatief
- NAO (North Atlantic Oscillation)
- Jetstream-Position

**Mesoskale (10-100 km):**
- Regionale Druckgradienten
- Föhn-Situationen
- Konvergenz-/Divergenzzonen

**Lokale Skala (1-10 km):**
- See-/Landwind (Breva, Ora)
- Hang-/Talwind
- Topographische Kanalisierung

**Das ML-Modell muss ALLE Skalen erfassen!**

---

## 3. Feature-Engineering-Strategie: 6 Kategorien

### 3.1 Übersicht der Feature-Kategorien

| Kategorie | Features | Zweck | Beispiele |
|-----------|----------|-------|-----------|
| **A. Synoptische Wetterlagen** | ~50 | Großwetterlage Europa | Druck-Zentren, NAO-Proxies, Gradienten |
| **B. Räumliche Gradienten** | ~80 | Druckmuster, Strömung | N-S, E-W, diagonale Gradienten |
| **C. Thermische Indikatoren** | ~40 | Solare Heizung, Stabilität | Solar, T-Gradienten, Buoyancy |
| **D. Zeitliche Dynamik** | ~60 | Trends, Beschleunigung | Lags, Diffs, Rolling Stats |
| **E. Vertikalprofile** | ~30 | Windscherung, Stabilität | Scherung, Richardson-Zahl |
| **F. Zyklische/Binäre** | ~20 | Tageszeit, Saison | sin/cos, is_afternoon, is_summer |
| **G. Rohdaten** | ~120 | Basismessungen | T, P, clouds, radiation (alle Punkte) |

**Gesamt: ~400 Features für Model 1**

---

## 4. Kategorie A: Synoptische Wetterlagen (Großwetterlage)

### 4.1 Warum ist die europäische Großwetterlage entscheidend?

**Wind in Mitteleuropa wird zu 60-80% durch synoptische Muster bestimmt:**

1. **Westlage (NAO+):** Starke Westwinde, Atlantik-Stürme
2. **Hochdruckbrücke:** Schwache Winde, thermische Zirkulation dominant
3. **Föhn-Situation:** Süd-Nord Druckgradient, starke Alpensüdwinde
4. **Ostlage:** Kontinentale Strömung, oft schwach
5. **Genuatief (Vb-Track):** Starke Südwinde → Nordwestdrehung

### 4.2 Feature-Engineering-Ansätze

#### 4.2.1 Druckzentren-Parametrisierung

**Konzept:** Charakterisiere jedes Hoch/Tief als Feature-Set

```python
# Für jedes identifizierte Druckzentrum:
features = {
    # Position
    'high1_lat': 48.5,          # Breitengrad des Hochs
    'high1_lon': 15.2,          # Längengrad des Hochs
    'high1_distance_km': 650,   # Entfernung zum Messort
    'high1_bearing_deg': 85,    # Richtung zum Hoch (0-360°)

    # Intensität
    'high1_pressure_hPa': 1028,           # Zentraldruck
    'high1_anomaly_hPa': +12,             # Abweichung von Klimatologie
    'high1_gradient_max_hPa_100km': 2.5,  # Maximaler Gradient

    # Geometrie
    'high1_area_km2': 850000,    # Ausdehnung
    'high1_eccentricity': 0.6,   # Elliptizität
    'high1_orientation_deg': 45, # Hauptachse
}
```

**Für Mitteleuropa relevant:**
- 1-2 Hochs (Azoren, Skandinavien, Russland)
- 1-2 Tiefs (Island, Genua, Mitteleuropa)
- Gradienten zwischen ihnen

**Vorteil:** Kompakte Repräsentation (10-20 Features statt Tausender Gitterpunkte)

#### 4.2.2 Regionale Druck-Aggregation

**Konzept:** Europa in meteorologisch sinnvolle Regionen unterteilen

```python
# Definiere Regionen (Lat/Lon Boxen)
regions = {
    'iceland': (63-67°N, 15-25°W),
    'azores': (35-40°N, 25-30°W),
    'scandinavia': (58-65°N, 5-20°E),
    'biscay': (43-48°N, 5°W-2°E),
    'genoa': (42-46°N, 7-12°E),
    'central_europe': (47-52°N, 8-15°E),
    'mediterranean': (38-43°N, 8-18°E),
    'russia': (50-60°N, 30-50°E),
}

# Features pro Region
for region in regions:
    features[f'{region}_pressure_mean'] = mean(pressure)
    features[f'{region}_pressure_anomaly'] = pressure - climatology
    features[f'{region}_temp_mean'] = mean(temperature)

# Cross-regionale Gradienten (NAO-Proxy!)
features['nao_proxy'] = P_azores - P_iceland
features['genoa_gradient'] = P_biscay - P_genoa
features['east_west_gradient'] = P_central_europe - P_russia
```

**Diese Gradienten sind GOLD!** Sie erfassen die Großwetterlage in wenigen Features.

#### 4.2.3 Etablierte Teleconnection Indices

**Konzept:** Nachbildung bekannter meteorologischer Indices

```python
# NAO-Index (North Atlantic Oscillation)
NAO = pressure_azores - pressure_iceland
# Positiv: Starke Westwinde, mild/nass
# Negativ: Blockierung, kalt/trocken

# EA (East Atlantic Pattern)
EA = pressure_biscay - pressure_central_atlantic

# SCAND (Scandinavian Pattern)
SCAND = pressure_scandinavia - pressure_reference

# Custom: Föhn-Index für Alpenregion
Foehn_Index = pressure_genoa - pressure_north_alps
# Positiv: Südföhn-Situation
# Negativ: Nordföhn

# Custom: Thermal Wind Potential
Thermal_Potential = (T_south - T_north) * solar_radiation
```

**Vorteil:** Physikalisch interpretierbar, robuste Prädiktoren

### 4.3 Implementierungs-Roadmap

**Phase 1 (Einfach):**
1. Regionale Mittelwerte (8 Regionen × 3 Variablen = 24 Features)
2. Cross-regionale Differenzen (10 Gradienten)
3. **→ 34 Features, sofort umsetzbar**

**Phase 2 (Mittel):**
1. Teleconnection Proxies (NAO, EA, SCAND, Föhn) = 4 Features
2. 3×3 Gradienten-Matrix (Druck) = 9 Features
3. **→ +13 Features**

**Phase 3 (Komplex):**
1. Automatische Druckzentren-Detektion (lokal/regionale Maxima/Minima)
2. Parametrisierung (3-5 Zentren × 8 Parameter = 24-40 Features)
3. **→ +30 Features**

**Gesamt: ~80 synoptische Features**

---

## 5. Kategorie B: Räumliche Gradienten (Mesoskala)

### 5.1 Warum Gradienten essentiell sind

**Fundamentale Physik:**
- Pressure Gradient Force (PGF) = - (1/ρ) × ∇P
- Wind entsteht durch Druckgradienten!
- Je steiler der Gradient → desto stärker der Wind

**Das 17-Punkt-Diamant-Grid von Windmojo erfasst:**
- Nord-Süd-Gradienten (nah, mittel, fern)
- Ost-West-Gradienten (nah, mittel, fern)
- Diagonale Gradienten (NE-SW, NW-SE)
- Krümmung (zweite Ableitungen)

### 5.2 Druck-Gradienten (höchste Priorität)

```python
# Nord-Süd-Gradienten (3 Distanzen)
features['pressure_gradient_NS_near'] = (P_n - P_s) / distance_near
features['pressure_gradient_NS_mid'] = (P_n2 - P_s2) / distance_mid
features['pressure_gradient_NS_far'] = (P_n3 - P_s3) / distance_far

# Ost-West-Gradienten
features['pressure_gradient_EW_near'] = (P_e - P_w) / distance_near
features['pressure_gradient_EW_mid'] = (P_e2 - P_w2) / distance_mid
features['pressure_gradient_EW_far'] = (P_e3 - P_w3) / distance_far

# Diagonale Gradienten (für Föhn-Situationen wichtig!)
features['pressure_gradient_NE_SW'] = (P_ne - P_sw) / distance_diag
features['pressure_gradient_NW_SE'] = (P_nw - P_se) / distance_diag

# Gesamtgradient (Magnitude)
grad_x = (P_e - P_w) / (2 * dx)
grad_y = (P_n - P_s) / (2 * dy)
features['pressure_gradient_magnitude'] = sqrt(grad_x**2 + grad_y**2)
features['pressure_gradient_direction'] = atan2(grad_y, grad_x)

# Krümmung (Laplacian) - zeigt Konvergenz/Divergenz
features['pressure_curvature'] = (P_n + P_s + P_e + P_w - 4*P_center)

# Asymmetrie (zeigt Fronten an)
features['pressure_asymmetry_NS'] = abs(P_n - P_center) - abs(P_s - P_center)
```

**→ ~20 Druck-Gradienten-Features**

### 5.3 Temperatur-Gradienten (thermischer Wind!)

**Physik: Thermal Wind Relationship**
- Horizontale T-Gradienten → vertikale Windscherung
- Warm air to the right → westerly wind increases with height (NH)

```python
# Horizontale Temperatur-Gradienten (Thermal Wind!)
features['temp_gradient_NS'] = (T_n - T_s) / distance
features['temp_gradient_EW'] = (T_e - T_w) / distance

# Vertikale Temperatur-Gradienten (Stabilität)
# (siehe Kategorie E: Vertikalprofile)

# Land-See-Kontrast (für Seebrise essentiell!)
# Annahme: Wir kennen Land/Wasser-Verteilung im Grid
features['temp_land_water_diff'] = T_land_mean - T_water_mean

# Solar Heating Asymmetrie (Hangwind!)
features['temp_gradient_direction'] = atan2(dT_dy, dT_dx)
features['temp_gradient_solar_aligned'] = cos(sun_azimuth - temp_gradient_dir)
```

**→ ~15 Temperatur-Gradienten-Features**

### 5.4 Das Problem: Windrichtungs-Gradienten

**Herausforderung:** Windrichtung ist **zyklisch** (0° = 360°)
- Differenz zwischen 350° und 10° ist NICHT 340°, sondern 20°!
- Naive Subtraktion funktioniert nicht

**Lösung 1: Trigonometrische Komponenten**

```python
# Zerlege Windrichtung in u/v-Komponenten
u_center = sin(wind_dir_center * π/180)
v_center = cos(wind_dir_center * π/180)

u_north = sin(wind_dir_north * π/180)
v_north = cos(wind_dir_north * π/180)

# Gradienten der Komponenten (mathematisch sauber!)
features['wind_u_gradient_NS'] = (u_north - u_south) / distance
features['wind_v_gradient_NS'] = (v_north - v_south) / distance

# Divergenz (zeigt Konvergenz/Divergenzzonen)
du_dx = (u_east - u_west) / (2*dx)
dv_dy = (v_north - v_south) / (2*dy)
features['wind_divergence'] = du_dx + dv_dy

# Vorticity (Rotation der Strömung)
dv_dx = (v_east - v_west) / (2*dx)
du_dy = (u_north - u_south) / (2*dy)
features['wind_vorticity'] = dv_dx - du_dy
```

**Lösung 2: Richtungsänderungs-Features**

```python
def circular_diff(dir1, dir2):
    """Berechne kürzeste Winkeldifferenz (zyklisch)"""
    diff = (dir1 - dir2) % 360
    if diff > 180:
        diff -= 360
    return diff

# Backing/Veering (wichtig für Fronten!)
features['wind_dir_change_with_height'] = circular_diff(
    wind_dir_850hPa, wind_dir_10m
)

# Räumliche Richtungsänderung
features['wind_dir_change_NS'] = circular_diff(wind_dir_n, wind_dir_s)

# Richtungskohärenz (wie parallel ist die Strömung?)
dirs = [wind_dir_n, wind_dir_s, wind_dir_e, wind_dir_w]
features['wind_direction_spread'] = std(dirs)  # Achtung: circular std!
```

**→ ~15 Windrichtungs-Gradienten-Features**

**Gesamt Kategorie B: ~50 räumliche Gradienten**

---

## 6. Kategorie C: Thermische Indikatoren

### 6.1 Solare Einstrahlung (DER thermische Motor)

**Physik:**
- Solare Einstrahlung → differentielle Erwärmung → Druckgradienten → thermische Winde
- See-/Landwind, Hang-/Talwind sind direkt solar getrieben

```python
# Direkte solare Variablen
features['solar_radiation_direct'] = direct_radiation
features['solar_radiation_diffuse'] = diffuse_radiation
features['solar_radiation_total'] = direct + diffuse

# Clear-Sky-Index (Bewölkungseffekt)
features['clear_sky_index'] = radiation_actual / radiation_clear_sky

# Solar Heating Potential (kombinierter Indikator)
features['solar_heating_potential'] = (
    radiation_total * (1 - cloud_cover) * sin(solar_elevation)
)

# Zeitliche Ableitung (Heizrate!)
features['solar_heating_rate'] = d(solar_radiation) / dt

# Akkumulierte Heizung (thermische Trägheit)
features['accumulated_solar_3h'] = sum(solar_radiation[-3:])
features['accumulated_solar_12h'] = sum(solar_radiation[-12:])
```

**→ ~10 solare Features**

### 6.2 Thermische Stabilität (vertikale Durchmischung)

**Physik:**
- Stabile Schichtung → schwache Durchmischung → schwacher Wind
- Instabile Schichtung → starke Konvektion → thermischer Wind

```python
# Richardson Number (DER Stabilitäts-Indikator)
# Ri = (g/T) * (dT/dz) / (du/dz)^2
# Ri < 0.25: turbulent, stark durchmischt
# Ri > 1: stabil, schwach durchmischt

g = 9.81  # m/s²
T_mean = (T_surface + T_850hPa) / 2
dT_dz = (T_850hPa - T_surface) / height_diff
du_dz = (wind_850hPa - wind_10m) / height_diff

features['richardson_number'] = (g/T_mean) * dT_dz / (du_dz**2 + 1e-6)

# Buoyancy Frequency (Brunt-Väisälä)
features['buoyancy_frequency'] = sqrt((g/T_mean) * dT_dz)

# Thermische Stabilitäts-Index (vereinfacht)
features['thermal_stability_index'] = T_850hPa - T_surface - 6.5
# Wenn > 0: Inversion (stabil)
# Wenn < 0: Instabil
```

**→ ~5 Stabilitäts-Features**

### 6.3 See-/Land-Effekte (falls Gewässer im Grid)

```python
# Identifiziere Land/Wasser-Gitterpunkte (statisch)
land_points = [center, n, s, e, w, ...]  # basierend auf Landmaske
water_points = [...]  # See/Meer-Punkte

# Land-Wasser-Temperatur-Differenz (KERN der Seebrise!)
features['land_water_temp_diff'] = mean(T_land) - mean(T_water)

# Solar-verstärkte Land-Wasser-Differenz
features['land_water_thermal_contrast'] = (
    (mean(T_land) - mean(T_water)) * solar_radiation
)

# Zeitpunkt im Seebrise-Zyklus
features['seabreeze_potential'] = (
    land_water_temp_diff * is_daytime * (1 - cloud_cover)
)

# Nachtkühlung (Landwind-Potential)
features['landbreeze_potential'] = (
    (mean(T_water) - mean(T_land)) * (1 - is_daytime)
)
```

**→ ~10 See/Land-Features** (falls anwendbar)

**Gesamt Kategorie C: ~25 thermische Features**

---

## 7. Kategorie D: Zeitliche Dynamik (Lags, Diffs, Rolling)

### 7.1 Lag-Features (Persistenz und Trägheit)

**Physik:**
- Atmosphäre hat Trägheit
- Aktueller Zustand beeinflusst von vergangenen Zuständen
- Für XGBoost essentiell (tree-based models sehen nur "jetzt")

```python
# Kurzzeitlags (1-6h) - für Persistenz
for var in ['pressure', 'temperature', 'wind_speed_GFS', 'solar']:
    for lag in [1, 3, 6]:
        features[f'{var}_lag_{lag}h'] = value(t - lag*hours)

# Tageslags (für diurnale Zyklen)
for var in ['temperature', 'solar', 'wind_speed']:
    for lag in [24, 48]:
        features[f'{var}_lag_{lag}h'] = value(t - lag*hours)

# "Gleiche Stunde gestern" (sehr starker Prädiktor!)
features['wind_speed_same_hour_yesterday'] = wind_speed(t - 24h)
```

**WICHTIG:** Lags von `wind_speed_10m` Messdaten sind erlaubt!
- Das Modell lernt: "Wenn gestern um 14h Wind war, heute auch"
- Nicht forecast, sondern gemessene Historie

**→ ~30 Lag-Features**

### 7.2 Diff-Features (Trends und Beschleunigung)

**Erste Ableitungen (Tendenz):**

```python
# Drucktendenz (klassischer Wetterindikator!)
features['pressure_change_1h'] = P(t) - P(t-1)
features['pressure_change_3h'] = P(t) - P(t-3)
features['pressure_change_6h'] = P(t) - P(t-6)

# Temperaturtendenz (Heizrate)
features['temp_change_1h'] = T(t) - T(t-1)
features['temp_change_3h'] = T(t) - T(t-3)

# Windtendenz (Beschleunigung/Abschwächung)
features['wind_change_1h'] = W(t) - W(t-1)
features['wind_change_3h'] = W(t) - W(t-3)
```

**Zweite Ableitungen (Beschleunigung):**

```python
# Druckbeschleunigung (zeigt sich beschleunigende Systeme)
features['pressure_accel_3h'] = (
    P(t) - 2*P(t-3) + P(t-6)
)

# Temperaturbeschleunigung
features['temp_accel_3h'] = (
    T(t) - 2*T(t-3) + T(t-6)
)
```

**→ ~20 Diff-Features**

### 7.3 Rolling Statistics (Volatilität und Trends)

```python
# Kurzzeit-Volatilität (3h, 6h)
for var in ['wind_speed', 'temperature', 'pressure']:
    for window in [3, 6]:
        features[f'{var}_rolling_std_{window}h'] = std(values[-window:])
        features[f'{var}_rolling_min_{window}h'] = min(values[-window:])
        features[f'{var}_rolling_max_{window}h'] = max(values[-window:])

# Mittel-/Langfristtrends (3d, 7d)
for var in ['temperature', 'solar', 'pressure']:
    for window in [72, 168]:  # 3d, 7d in Stunden
        features[f'{var}_rolling_mean_{window}h'] = mean(values[-window:])

# Niederschlags-Akkumulation (Bodenfeuchte-Proxy)
features['precip_sum_3h'] = sum(precip[-3:])
features['precip_sum_12h'] = sum(precip[-12:])
features['precip_sum_3d'] = sum(precip[-72:])
```

**→ ~25 Rolling-Features**

**Gesamt Kategorie D: ~75 zeitliche Features**

---

## 8. Kategorie E: Vertikalprofile (Windscherung, Stabilität)

### 8.1 Windscherung (kritisch für Turbulenz und Durchmischung)

**Physik:**
- Vertikale Windscherung → Turbulenz → Durchmischung
- Boundary Layer Height hängt von Scherung ab
- Windgeschwindigkeit am Boden korreliert mit Scherung

```python
# Windscherung über verschiedene Layer
features['wind_shear_10m_100m'] = (
    (wind_100m - wind_10m) / (100 - 10)
)

features['wind_shear_10m_850hPa'] = (
    (wind_850hPa - wind_10m) / height_diff_850hPa
)

features['wind_shear_10m_500hPa'] = (
    (wind_500hPa - wind_10m) / height_diff_500hPa
)

# Richtungsscherung (Veering/Backing)
features['wind_dir_shear_10m_850hPa'] = circular_diff(
    wind_dir_850hPa, wind_dir_10m
)

# Totale integrierte Scherung
features['total_wind_shear'] = sqrt(
    (wind_500hPa - wind_10m)**2 + (wind_dir_shear)**2
)
```

**→ ~10 Scherung-Features**

### 8.2 Temperatur-Profile (Stabilität, Inversionen)

```python
# Lapse Rate (vertikaler Temperaturgradient)
features['lapse_rate_surface_850hPa'] = (
    (T_850hPa - T_surface) / height_diff
)

# Vergleich mit trockenadiabatischem Lapse Rate (9.8 K/km)
features['lapse_rate_deviation'] = lapse_rate - 9.8

# Inversion Detection
features['has_inversion'] = int(T_850hPa > T_surface)
features['inversion_strength'] = max(0, T_850hPa - T_surface)

# Boundary Layer Height (Schätzung)
# Definiert als Höhe, wo T < T_surface - 2K
features['boundary_layer_height_estimate'] = estimate_BLH(T_profile)
```

**→ ~8 Temperaturprofil-Features**

### 8.3 Feuchte-Profile (Konvektion, Wolken)

```python
# Vertikale Feuchte-Gradienten
features['humidity_gradient_surface_850hPa'] = (
    (RH_850hPa - RH_surface) / height_diff
)

# Lifted Condensation Level (LCL) - Wolkenbasis
features['LCL_height'] = calculate_LCL(T_surface, dewpoint_surface)

# CAPE (Convective Available Potential Energy) - wenn verfügbar
# Zeigt Gewitterpotential
features['CAPE'] = calculate_CAPE(T_profile, dewpoint_profile)
```

**→ ~8 Feuchte-Features**

**Gesamt Kategorie E: ~25 Vertikalprofil-Features**

---

## 9. Kategorie F: Zyklische & Binäre Features

### 9.1 Zyklische Kodierung (sin/cos)

**Warum sin/cos?**
- Stunde 23 und Stunde 0 sind NAH beieinander (nicht 23 Einheiten!)
- Monat 12 und Monat 1 sind NAH beieinander
- Tree-based models verstehen das nur mit sin/cos

```python
# Tageszeit (24h Zyklus)
hour = timestamp.hour
features['hour_sin'] = sin(2 * π * hour / 24)
features['hour_cos'] = cos(2 * π * hour / 24)

# Jahreszeit (12 Monate Zyklus)
month = timestamp.month
features['month_sin'] = sin(2 * π * month / 12)
features['month_cos'] = cos(2 * π * month / 12)

# Tag im Jahr (365 Tage, präziser für solare Effekte)
day_of_year = timestamp.dayofyear
features['dayofyear_sin'] = sin(2 * π * day_of_year / 365)
features['dayofyear_cos'] = cos(2 * π * day_of_year / 365)

# Windrichtung (bereits besprochen, für Vollständigkeit)
wind_dir = wind_direction_10m  # 0-360°
features['wind_dir_sin'] = sin(wind_dir * π / 180)
features['wind_dir_cos'] = cos(wind_dir * π / 180)
```

**→ ~8 zyklische Features**

### 9.2 Binäre Indikatoren (Regime-Switches)

```python
# Tageszeit-Phasen (thermische Regime!)
features['is_morning'] = int(6 <= hour < 10)
features['is_midday'] = int(10 <= hour < 14)
features['is_afternoon'] = int(14 <= hour < 18)
features['is_evening'] = int(18 <= hour < 22)
features['is_night'] = int(22 <= hour or hour < 6)

# Jahreszeiten
features['is_spring'] = int(month in [3, 4, 5])
features['is_summer'] = int(month in [6, 7, 8])
features['is_autumn'] = int(month in [9, 10, 11])
features['is_winter'] = int(month in [12, 1, 2])

# Wochenende (menschliche Aktivität, ggf. relevant)
features['is_weekend'] = int(timestamp.weekday() >= 5)

# Wetter-Bedingungen
features['is_raining'] = int(precipitation > 0.1)
features['is_cloudy'] = int(cloud_cover > 0.7)
features['is_clear_sky'] = int(cloud_cover < 0.3)
```

**→ ~15 binäre Features**

**Gesamt Kategorie F: ~23 zyklische/binäre Features**

---

## 10. Kategorie G: Rohdaten (Baseline-Features)

### 10.1 Welche Rohdaten einbeziehen?

**Alle 21 Gitterpunkte × relevante Variablen:**

```python
# Für jeden Gitterpunkt (center, n, s, e, w, ne, nw, se, sw, n2, s2, ...)
grid_points = ['center', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw',
               'n2', 's2', 'e2', 'w2', 'ne2', 'nw2', 'se2', 'sw2',
               'n3', 's3', 'e3', 'w3']

# Primäre atmosphärische Variablen
variables = [
    'temperature_2m',
    'pressure_msl',
    'cloud_cover',
    'precipitation',
    'solar_radiation',
    'wind_speed_10m',     # Falls doch verwendet - siehe Diskussion unten
    'wind_direction_10m',
    'temperature_850hPa',
    'geopotential_height_500hPa',
]

# Feature-Namen
for point in grid_points:
    for var in variables:
        features[f'{var}_{point}'] = value
```

**→ 21 Punkte × 9 Variablen = ~189 Rohdaten-Features**

### 10.2 Die GFS wind_speed_10m Frage

**Drei Strategien:**

**Strategie 1: OHNE GFS wind_speed**
- **Vorteil:** Modell lernt physikalische Mechanismen
- **Nachteil:** Schlechtere Performance initial
- **Verwendung:** Regime-basierte Modelle (siehe Phase 2)

**Strategie 2: MIT GFS wind_speed, aber nur Gradienten**
- **Kompromiss:** Nutze räumliche Muster, nicht absolute Werte
- **Features:** wind_gradient_NS, wind_divergence, wind_vorticity
- **Vorteil:** Balance zwischen Physik und Daten

**Strategie 3: MIT GFS wind_speed_10m als Feature**
- **Vorteil:** Beste Performance (GFS ist nicht schlecht!)
- **Verwendung:** "Balanced" Ansatz für Phase 2 Model 2
- **Interpretation:** ML lernt GFS-Bias zu korrigieren

**Empfehlung für 2-stufigen Ansatz:**
- **Model 1 (alle Features):** Inkludiere GFS wind (300-500 Features)
- **Model 2 Varianten:**
  - `gfs_balanced`: Mit GFS wind (beste Performance)
  - `gfs_no_wind`: Ohne GFS wind (physikalisches Verständnis)
  - `gfs_regime`: Nur atmosphärische Regime (Forschung)

**Gesamt Kategorie G: ~120-190 Rohdaten-Features**

---

## 11. Der Zweistufige Ansatz: Model 1 → Model 2

### 11.1 Phase 1: Feature Discovery mit Model 1

**Ziel:** Finde heraus, welche Features wichtig sind

**Strategie:**

```python
# Model 1 Konfiguration
model1_config = {
    'features': 'ALL',  # ~400-500 Features
    'samples': 'ALL',   # Alle verfügbaren Daten
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',

    # Absichtlich komplex (wir WOLLEN overfitting sehen!)
    'max_depth': 8,
    'n_estimators': 500,
    'learning_rate': 0.1,

    # Keine strikte Regularisierung
    'reg_alpha': 0,
    'reg_lambda': 1,
}

# Training
model1 = xgb.train(model1_config, dtrain)

# Feature Importance Extraktion
importance = model1.get_score(importance_type='gain')
# 'gain': Totaler Gewinn durch Splits mit diesem Feature
# 'weight': Anzahl der Splits
# 'cover': Durchschnittliche Abdeckung der Samples

# Ranking
feature_ranking = sorted(importance.items(),
                         key=lambda x: x[1],
                         reverse=True)
```

**Ausgabe:**

```
Top 50 Features (beispielhaft):
1. pressure_gradient_NS_mid        (gain: 15234)
2. nao_proxy                       (gain: 12456)
3. solar_radiation_center          (gain: 11234)
4. wind_speed_10m_center           (gain: 10987)  # Falls inkludiert
5. temperature_land_water_diff     (gain: 9876)
...
50. cloud_cover_ne                 (gain: 234)

Features 51-400: Gain < 200 (vermutlich noise)
```

**WICHTIG:** Analysiere auch Korrelationen innerhalb der Top-Features!

### 11.2 Phase 2: Feature Selection Strategien

**5 Ansätze (basierend auf Windmojo Lecco-Erfahrung):**

#### Ansatz 1: GFS Balanced (Empfohlen)

```python
# Ausgewogene Mischung: Top Features + physikalische Gruppen
features_gfs_balanced = {
    # Top 10 nach Gain (inkl. GFS wind)
    'top_gain': top_10_by_gain,

    # Synoptische Indikatoren (wichtigste)
    'synoptic': ['nao_proxy', 'pressure_gradient_NS', 'genoa_index'],

    # Thermische Indikatoren
    'thermal': ['solar_radiation', 'land_water_temp_diff',
                'richardson_number'],

    # Zeitliche Features
    'temporal': ['pressure_change_3h', 'wind_lag_24h',
                 'temp_rolling_mean_6h'],

    # Zyklische Features
    'cyclical': ['hour_sin', 'hour_cos', 'month_sin', 'month_cos'],
}

# Gesamt: ~50-70 Features
```

**Performance: Beste MAE** (z.B. Lecco: 3.388 m/s)

#### Ansatz 2: GFS Top-N (Datengetrieben)

```python
# Einfach: Nimm Top N nach Importance
features_gfs_topn = top_75_features_by_gain

# Gesamt: 75 Features
```

**Performance: Gut, aber ggf. redundant**

#### Ansatz 3: GFS No Wind (Physik-basiert)

```python
# Schließe ALLE GFS wind_speed/direction Features aus
features_no_wind = {
    'pressure': all_pressure_features,
    'temperature': all_temperature_features,
    'solar': all_solar_features,
    'stability': all_stability_features,
    'gradients': all_gradient_features,
    'temporal': all_temporal_features,
}

# Gesamt: ~75-100 Features (ohne Wind-Variablen)
```

**Performance: Schlechter als gfs_balanced, aber beweist Machbarkeit!**

#### Ansatz 4: Regime Indicators (Forschung)

```python
# NUR synoptische und thermische Regime-Indikatoren
features_regime = {
    'synoptic_indices': [
        'nao_proxy', 'ea_pattern', 'scand_index',
        'genoa_index', 'foehn_index'
    ],

    'pressure_patterns': [
        'pressure_gradient_NS', 'pressure_gradient_EW',
        'pressure_curvature', 'pressure_asymmetry'
    ],

    'thermal_regime': [
        'solar_heating_potential', 'land_water_thermal_contrast',
        'thermal_stability_index', 'richardson_number'
    ],

    'temporal_context': [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'is_afternoon', 'is_summer'
    ],
}

# Gesamt: ~35-45 Features
```

**Performance: Lecco 3.557 m/s (nur 40 Features!)**
**Beweis:** Wind kann aus atmosphärischen Regimen gelernt werden!

#### Ansatz 5: Hybrid (Korrelations-bereinigt)

```python
# Top Features, aber entferne hochkorrelierte (|r| > 0.95)
features_hybrid = remove_correlated(top_100_features, threshold=0.95)

# Gesamt: ~80-90 Features
```

**Performance: Ähnlich gfs_balanced**

### 11.3 Model 2 Training (Produktion)

```python
# Model 2 Konfiguration (regularisiert!)
model2_config = {
    'features': selected_features,  # 40-100 Features
    'samples': 'unified',  # Alle Stunden zusammen (nicht per-hour)

    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',

    # Regularisiert gegen Overfitting
    'max_depth': 5,  # Flacher als Model 1
    'n_estimators': 300,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,

    # L1/L2 Regularisierung
    'reg_alpha': 0.1,   # L1
    'reg_lambda': 1.0,  # L2

    # Early Stopping
    'early_stopping_rounds': 50,
}

# Training mit Validation
model2 = xgb.train(
    model2_config,
    dtrain,
    evals=[(dtest, 'test')],
    verbose_eval=10
)
```

**Ziel:** Generalisierung! Test MAE < Train MAE × 1.5

---

## 12. Gradienten-Berechnung: Technische Details

### 12.1 Finite Differenzen (Standard-Methode)

**Zentrale Differenz (2. Ordnung Genauigkeit):**

```python
def central_difference(f, point_plus, point_minus, distance):
    """
    Zentrale Differenz: df/dx ≈ (f(x+h) - f(x-h)) / (2h)

    Args:
        f: Wert der Variablen
        point_plus: Wert am Punkt x+h (z.B. Nord)
        point_minus: Wert am Punkt x-h (z.B. Süd)
        distance: 2*h (totale Distanz)
    """
    return (point_plus - point_minus) / distance

# Beispiel: Nord-Süd-Druckgradient
pressure_gradient_NS = central_difference(
    f=None,  # nicht benötigt für zentrale Diff
    point_plus=pressure_north,
    point_minus=pressure_south,
    distance=2 * grid_spacing
)

# Einheiten: hPa / km
# Typische Werte: 0.5-3 hPa/100km = 0.005-0.03 hPa/km
```

**Vorwärts-/Rückwärtsdifferenz (wenn Rand erreicht):**

```python
def forward_difference(f_center, f_plus, distance):
    """df/dx ≈ (f(x+h) - f(x)) / h"""
    return (f_plus - f_center) / distance

# Für äußere Gitterpunkte
pressure_gradient_at_edge = forward_difference(
    f_center=pressure_center,
    f_plus=pressure_n,
    distance=grid_spacing
)
```

### 12.2 Zweite Ableitungen (Krümmung, Laplacian)

**Krümmung entlang einer Achse:**

```python
def second_derivative(f_center, f_plus, f_minus, distance):
    """
    d²f/dx² ≈ (f(x+h) - 2*f(x) + f(x-h)) / h²
    """
    return (f_plus - 2*f_center + f_minus) / (distance**2)

# Beispiel: Druckkrümmung Nord-Süd
pressure_curvature_NS = second_derivative(
    f_center=pressure_center,
    f_plus=pressure_n,
    f_minus=pressure_s,
    distance=grid_spacing
)
```

**2D Laplacian (Divergenz des Gradienten):**

```python
def laplacian_2d(f_center, f_n, f_s, f_e, f_w, dx, dy):
    """
    ∇²f = d²f/dx² + d²f/dy²

    Physikalische Bedeutung:
    - Positiv: Lokales Minimum (Konvergenz)
    - Negativ: Lokales Maximum (Divergenz)
    """
    d2f_dx2 = (f_e - 2*f_center + f_w) / dx**2
    d2f_dy2 = (f_n - 2*f_center + f_s) / dy**2
    return d2f_dx2 + d2f_dy2

# Druck-Laplacian (zeigt Konvergenz/Divergenz)
pressure_laplacian = laplacian_2d(
    f_center=P_center,
    f_n=P_n, f_s=P_s, f_e=P_e, f_w=P_w,
    dx=grid_spacing_x,
    dy=grid_spacing_y
)

# Interpretation:
# > 0: Tief (Konvergenz, aufsteigende Luft)
# < 0: Hoch (Divergenz, absteigende Luft)
```

### 12.3 Gradientenrichtung und -magnitude

```python
def gradient_magnitude_direction(f, f_n, f_s, f_e, f_w, dx, dy):
    """
    Berechne Magnitude und Richtung des Gradienten

    Returns:
        magnitude: |∇f| = sqrt((df/dx)² + (df/dy)²)
        direction: atan2(df/dy, df/dx) in Grad [0-360°]
    """
    df_dx = (f_e - f_w) / (2 * dx)
    df_dy = (f_n - f_s) / (2 * dy)

    magnitude = sqrt(df_dx**2 + df_dy**2)
    direction_rad = atan2(df_dy, df_dx)
    direction_deg = (direction_rad * 180 / π) % 360

    return magnitude, direction_deg

# Beispiel: Druckgradient
pressure_grad_mag, pressure_grad_dir = gradient_magnitude_direction(
    f=P_center,
    f_n=P_n, f_s=P_s, f_e=P_e, f_w=P_w,
    dx=grid_spacing_lon,
    dy=grid_spacing_lat
)

# Geostrophischer Wind (Approximation)
# V_geo ∝ ∇P (senkrecht zum Gradienten, 90° gedreht)
geostrophic_wind_direction = (pressure_grad_dir + 90) % 360
```

### 12.4 Vektor-Gradienten (für Windrichtung)

**Divergenz (horizontale Ausdehnung):**

```python
def divergence(u_e, u_w, v_n, v_s, dx, dy):
    """
    Divergenz: ∇·V = du/dx + dv/dy

    > 0: Divergenz (Luft fließt auseinander)
    < 0: Konvergenz (Luft fließt zusammen)
    """
    du_dx = (u_e - u_w) / (2 * dx)
    dv_dy = (v_n - v_s) / (2 * dy)
    return du_dx + dv_dy
```

**Vorticity (Rotation):**

```python
def vorticity(u_n, u_s, v_e, v_w, dx, dy):
    """
    Vorticity (relative): ζ = dv/dx - du/dy

    > 0: Zyklonale Rotation (NH: counterclockwise)
    < 0: Antizyklonale Rotation (NH: clockwise)
    """
    dv_dx = (v_e - v_w) / (2 * dx)
    du_dy = (u_n - u_s) / (2 * dy)
    return dv_dx - du_dy
```

### 12.5 Grid-Spacing-Korrektur für Lat/Lon

**Problem:** Longitude-Abstände variieren mit Latitude!

```python
def latlon_to_meters(lat1, lon1, lat2, lon2):
    """
    Haversine-Formel für Distanz zwischen Lat/Lon-Punkten

    Returns: Distanz in Metern
    """
    R = 6371000  # Erdradius in Metern

    lat1_rad = lat1 * π / 180
    lat2_rad = lat2 * π / 180
    dlat = (lat2 - lat1) * π / 180
    dlon = (lon2 - lon1) * π / 180

    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c

# Korrektur für Longitude-Gradienten
lat_center = 46.0  # Beispiel: Lago di Como
dx_meters = latlon_to_meters(lat_center, lon_west, lat_center, lon_east)
dy_meters = latlon_to_meters(lat_south, lon_center, lat_north, lon_center)

# Gradient mit korrekten Distanzen
pressure_gradient_EW = (P_e - P_w) / (dx_meters / 1000)  # hPa/km
```

---

## 13. Praktische Implementierung: Code-Skizze

### 13.1 Feature-Engineering-Pipeline

```python
import pandas as pd
import numpy as np
from typing import Dict, List

class WindFeatureEngineer:
    """
    Feature Engineering für Wind-Geschwindigkeits-Vorhersage

    Transformiert GFS-Grid-Daten in ~400 ML-Features
    """

    def __init__(self, grid_config: Dict):
        self.grid = grid_config
        self.features_computed = []

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hauptfunktion: Berechne alle Feature-Kategorien

        Args:
            df: DataFrame mit Rohdaten (long format, 21 locations × N vars)

        Returns:
            df_features: Wide format, 1 row per timestamp, ~400 columns
        """

        # Stage 1: Pivot zu Wide Format
        df_wide = self._pivot_to_wide(df)

        # Stage 2: Synoptische Features
        df_wide = self._add_synoptic_features(df_wide)

        # Stage 3: Räumliche Gradienten
        df_wide = self._add_spatial_gradients(df_wide)

        # Stage 4: Thermische Indikatoren
        df_wide = self._add_thermal_features(df_wide)

        # Stage 5: Zeitliche Features
        df_wide = self._add_temporal_features(df_wide)

        # Stage 6: Vertikalprofile
        df_wide = self._add_vertical_features(df_wide)

        # Stage 7: Zyklische Features
        df_wide = self._add_cyclical_features(df_wide)

        return df_wide

    def _add_synoptic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kategorie A: Synoptische Wetterlagen"""

        # Regionale Mittelwerte
        df['pressure_iceland'] = df[[
            'pressure_msl_n3', 'pressure_msl_nw3', 'pressure_msl_ne3'
        ]].mean(axis=1)

        df['pressure_azores'] = df[[
            'pressure_msl_sw3', 'pressure_msl_s3', 'pressure_msl_se3'
        ]].mean(axis=1)

        # NAO-Proxy
        df['nao_proxy'] = df['pressure_azores'] - df['pressure_iceland']

        # Genua-Index (Föhn-Indikator)
        df['genoa_pressure'] = df['pressure_msl_s']
        df['alps_pressure'] = df['pressure_msl_n']
        df['foehn_index'] = df['genoa_pressure'] - df['alps_pressure']

        # ... weitere synoptische Features

        return df

    def _add_spatial_gradients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kategorie B: Räumliche Gradienten"""

        # Distanzen (in km, vorberechnet)
        dist_near = 25  # 0.25° ≈ 25 km
        dist_mid = 50   # 0.5° ≈ 50 km
        dist_far = 75   # 0.75° ≈ 75 km

        # Druck-Gradienten
        df['pressure_grad_NS_near'] = (
            (df['pressure_msl_n'] - df['pressure_msl_s']) / (2 * dist_near)
        )

        df['pressure_grad_NS_mid'] = (
            (df['pressure_msl_n2'] - df['pressure_msl_s2']) / (2 * dist_mid)
        )

        df['pressure_grad_EW_near'] = (
            (df['pressure_msl_e'] - df['pressure_msl_w']) / (2 * dist_near)
        )

        # Gradient Magnitude
        df['pressure_grad_magnitude'] = np.sqrt(
            df['pressure_grad_NS_near']**2 +
            df['pressure_grad_EW_near']**2
        )

        # Laplacian (Krümmung)
        df['pressure_laplacian'] = (
            df['pressure_msl_n'] + df['pressure_msl_s'] +
            df['pressure_msl_e'] + df['pressure_msl_w'] -
            4 * df['pressure_msl_center']
        ) / (dist_near**2)

        # ... weitere Gradienten (Temperatur, Wind, etc.)

        return df

    def _add_thermal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kategorie C: Thermische Indikatoren"""

        # Solar Heating Potential
        df['solar_heating_potential'] = (
            df['solar_radiation_center'] *
            (1 - df['cloud_cover_center']) *
            np.sin(df['solar_elevation_rad'])
        )

        # Richardson Number (Stabilität)
        g = 9.81
        T_mean = (df['temperature_2m_center'] + 273.15 +
                  df['temperature_850hPa_center'] + 273.15) / 2
        dT_dz = (df['temperature_850hPa_center'] -
                 df['temperature_2m_center']) / 1500  # m
        du_dz = (df['wind_speed_850hPa_center'] -
                 df['wind_speed_10m_center']) / 1500

        df['richardson_number'] = (
            (g / T_mean) * dT_dz / (du_dz**2 + 1e-6)
        )

        # ... weitere thermische Features

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kategorie D: Zeitliche Dynamik"""

        # Lags
        for var in ['pressure_msl_center', 'temperature_2m_center']:
            for lag in [1, 3, 6, 24]:
                df[f'{var}_lag_{lag}h'] = df[var].shift(lag)

        # Differenzen
        df['pressure_change_1h'] = df['pressure_msl_center'].diff(1)
        df['pressure_change_3h'] = df['pressure_msl_center'].diff(3)

        # Rolling Statistics
        for var in ['temperature_2m_center', 'wind_speed_10m_center']:
            for window in [3, 6, 24]:
                df[f'{var}_rolling_mean_{window}h'] = (
                    df[var].rolling(window).mean()
                )
                df[f'{var}_rolling_std_{window}h'] = (
                    df[var].rolling(window).std()
                )

        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kategorie F: Zyklische Features"""

        # Tageszeit
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

        # Jahreszeit
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

        # Tag im Jahr
        df['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

        # Binäre Indikatoren
        df['is_morning'] = (df.index.hour >= 6) & (df.index.hour < 10)
        df['is_afternoon'] = (df.index.hour >= 14) & (df.index.hour < 18)
        df['is_summer'] = df.index.month.isin([6, 7, 8])
        df['is_winter'] = df.index.month.isin([12, 1, 2])

        return df
```

### 13.2 Zweistufiges Training

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

class TwoStageWindModel:
    """
    Zweistufiges XGBoost-Training für Wind-Vorhersage

    Stage 1: Feature Discovery (alle 400 Features)
    Stage 2: Production Model (Top 50-100 Features)
    """

    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.selected_features = None

    def train_stage1(self, X: pd.DataFrame, y: pd.Series):
        """
        Stage 1: Kitchen Sink Approach

        Trainiere mit ALLEN Features, extrahiere Importance
        """

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, shuffle=False  # Zeitlich!
        )

        # Model 1 Config (komplex, wenig Regularisierung)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0,
            'reg_lambda': 1,
        }

        # Training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        self.model1 = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=50,
            verbose_eval=20
        )

        # Feature Importance
        importance = self.model1.get_score(importance_type='gain')
        self.feature_importance = pd.DataFrame({
            'feature': importance.keys(),
            'gain': importance.values()
        }).sort_values('gain', ascending=False)

        print(f"\nTop 20 Features:")
        print(self.feature_importance.head(20))

        return self.model1

    def select_features(self, strategy='balanced', top_n=50):
        """
        Feature Selection für Stage 2

        Strategies:
        - 'topn': Einfach Top N nach Gain
        - 'balanced': Physik-basierte Gruppen + Top Features
        - 'no_wind': Ohne GFS wind_speed Features
        - 'regime': Nur synoptische/thermische Regime
        """

        if strategy == 'topn':
            self.selected_features = (
                self.feature_importance.head(top_n)['feature'].tolist()
            )

        elif strategy == 'balanced':
            # Top 10
            top10 = self.feature_importance.head(10)['feature'].tolist()

            # Synoptische (wichtigste)
            synoptic = [f for f in self.feature_importance['feature']
                       if 'nao' in f or 'gradient' in f or 'foehn' in f][:10]

            # Thermische
            thermal = [f for f in self.feature_importance['feature']
                      if 'solar' in f or 'thermal' in f or 'richardson' in f][:8]

            # Zeitliche
            temporal = [f for f in self.feature_importance['feature']
                       if 'lag' in f or 'change' in f or 'rolling' in f][:10]

            # Zyklische
            cyclical = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                       'is_afternoon', 'is_summer']

            self.selected_features = list(set(
                top10 + synoptic + thermal + temporal + cyclical
            ))

        elif strategy == 'no_wind':
            # Alle Features OHNE wind_speed/wind_direction
            self.selected_features = [
                f for f in self.feature_importance['feature']
                if 'wind_speed' not in f and 'wind_direction' not in f
            ][:top_n]

        elif strategy == 'regime':
            # Nur synoptische Indices, Gradienten, thermische Regime
            regime_keywords = [
                'nao', 'gradient', 'foehn', 'genoa',
                'solar', 'thermal', 'richardson', 'stability',
                'hour_', 'month_', 'is_'
            ]

            self.selected_features = [
                f for f in self.feature_importance['feature']
                if any(kw in f for kw in regime_keywords)
            ][:top_n]

        print(f"\nSelected {len(self.selected_features)} features")
        print(f"Strategy: {strategy}")

        return self.selected_features

    def train_stage2(self, X: pd.DataFrame, y: pd.Series):
        """
        Stage 2: Production Model

        Trainiere mit selected Features, starke Regularisierung
        """

        # Nur selected features
        X_selected = X[self.selected_features]

        # Train/Test Split (gleich wie Stage 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.15, shuffle=False
        )

        # Model 2 Config (regularisiert!)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 5,  # Flacher!
            'learning_rate': 0.05,  # Langsamer!
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,  # L1 Regularisierung
            'reg_lambda': 1.0,  # L2 Regularisierung
        }

        # Training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        self.model2 = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=50,
            verbose_eval=20
        )

        # Evaluation
        y_pred_test = self.model2.predict(dtest)
        mae_test = np.mean(np.abs(y_test - y_pred_test))

        y_pred_train = self.model2.predict(dtrain)
        mae_train = np.mean(np.abs(y_train - y_pred_train))

        print(f"\nStage 2 Results:")
        print(f"Train MAE: {mae_train:.3f} m/s")
        print(f"Test MAE: {mae_test:.3f} m/s")
        print(f"Overfitting Gap: {mae_test/mae_train:.2f}x")

        return self.model2
```

---

## 14. Bewertung: Welche Features sind entscheidend?

### 14.1 Erwartete Feature-Importance (basierend auf Physik und Lecco-Daten)

**Top 10 (sehr wahrscheinlich):**

1. **Druck-Gradienten (NS, EW)**: ⭐⭐⭐⭐⭐
   - Direkteste Verbindung zu Wind
   - Pressure Gradient Force ist DER Windtreiber

2. **NAO-Proxy / Synoptische Indices**: ⭐⭐⭐⭐⭐
   - Erfasst Großwetterlage
   - Bestimmt grundlegendes Windregime

3. **Solar Radiation / Thermal Heating**: ⭐⭐⭐⭐⭐
   - Für thermische Winde ESSENTIELL
   - Breva, Ora, Seebrise direkt abhängig

4. **Temperature Land-Water Diff**: ⭐⭐⭐⭐⭐
   - Seebrise-Indikator #1
   - Nur falls See/Meer im Grid

5. **hour_sin / hour_cos**: ⭐⭐⭐⭐
   - Diurnaler Zyklus fundamental
   - Thermische Winde zeitabhängig

6. **Pressure Change (Tendenz)**: ⭐⭐⭐⭐
   - Klassischer Wetter-Indikator
   - Zeigt heranziehende Systeme

7. **Richardson Number / Stability**: ⭐⭐⭐⭐
   - Vertikale Durchmischung
   - Thermische Konvektion

8. **Wind Speed Lag (24h)**: ⭐⭐⭐⭐
   - Persistenz-Effekt
   - "Gestern um diese Zeit war auch Wind"

9. **Föhn-Index (Genua-Druck-Gradient)**: ⭐⭐⭐⭐
   - Für Alpenregion spezifisch
   - Südwind-Vorhersage

10. **month_sin / month_cos**: ⭐⭐⭐
    - Saisonale Muster
    - Sommer vs Winter unterschiedlich

### 14.2 Features mit Überraschungspotential

**Möglicherweise wichtiger als erwartet:**

- **Pressure Curvature (Laplacian)**: Konvergenz/Divergenz
- **Temp Gradient Direction**: Thermal Wind Alignment
- **Wind Divergence/Vorticity**: Strömungsmuster
- **Accumulated Solar (3h, 12h)**: Thermische Trägheit
- **Boundary Layer Height**: Durchmischungs-Tiefe

**Möglicherweise weniger wichtig:**

- **Einzelne Rohdaten** (außer Center-Punkt)
- **Entfernte Gitterpunkte** (n3, s3, e3, w3)
- **Hochkorrelierte Variablen** (redundant)

### 14.3 Feature-Kategorien nach erwarteter Wichtigkeit

| Rang | Kategorie | Erwartete Importance | Grund |
|------|-----------|---------------------|-------|
| 1 | Räumliche Gradienten | 30-40% | Direkter Wind-Treiber (PGF) |
| 2 | Synoptische Wetterlagen | 20-30% | Großwetterlage bestimmt Regime |
| 3 | Thermische Indikatoren | 15-25% | Lokale thermische Winde |
| 4 | Zeitliche Dynamik | 10-15% | Persistenz, Trends |
| 5 | Zyklische Features | 5-10% | Diurnal/Saisonal |
| 6 | Vertikalprofile | 5-10% | Stabilität, Scherung |
| 7 | Rohdaten | 5-10% | Einzelne wichtige Punkte |

**Gesamt: 100%**

---

## 15. Offene Fragen und Forschungsrichtungen

### Frage 1: Reichen atmosphärische Features ohne GFS Wind?

**Status:** Lecco beweist JA!
- `gfs_regime_indicators` erreicht 3.557 m/s MAE mit nur 40 Features
- Ohne jegliche GFS wind_speed/direction Features
- Nur Druck, Temperatur, Solar, Gradienten

**Implikation:** Wind IST ableitbar aus atmosphärischem Zustand

**Weitere Forschung:**
- Performance-Gap zu gfs_balanced (3.388 vs 3.557 = 5% schlechter)
- Ist dieser Gap akzeptabel für "Regime-Reinheit"?
- Können wir mit mehr Regime-Features auf 3.4 kommen?

### Frage 2: Optimal Feature Count?

**Bisherige Ergebnisse:**
- 40 Features (regime): 3.557 m/s
- 53 Features (balanced): **3.388 m/s** ← Winner
- 75 Features (topn): 3.412 m/s
- 84 Features (hybrid): 3.401 m/s

**Beobachtung:** Sweet Spot bei 50-60 Features

**Frage:** Gibt es ein theoretisches Optimum?
- Zu wenige: Unterfitting, wichtige Signale fehlen
- Zu viele: Overfitting, Noise dominiert

**Weiteres Experiment:**
- Systematisch 30, 40, 50, 60, 70, 80, 90, 100 Features testen
- Overfitting-Gap monitoren (Test/Train Ratio)

### Frage 3: Unified vs Per-Hour Models?

**Windmojo-Entscheidung:** Unified (alle Stunden zusammen)

**Vorteile Unified:**
- 13× mehr Trainingsdaten (2332 vs ~180 per hour)
- Bessere Generalisierung
- Geringerer Overfitting-Gap

**Nachteil:**
- Mögliche Verwischung hour-spezifischer Muster

**Frage:** Gibt es Fälle, wo per-hour besser wäre?
- Extrem unterschiedliche Regime (Nacht vs Tag)
- Bimodale Windverteilung (z.B. Breva vs Tivano)

**Kompromiss:** Hybrid
- Unified Model + `hour_sin/cos` Features
- Model lernt hour-spezifische Adjustments

### Frage 4: Wie berechnen wir Gradienten bei unklarer Windrichtung?

**Lösungsansätze:**

1. **U/V-Komponenten** (mathematisch sauber)
   - Zerlege in trigonometrische Komponenten
   - Berechne Gradienten von u, v separat
   - ✅ Implementiert

2. **Zirkular-Differenzen** (für Richtung selbst)
   - Shortest-angle difference
   - Nur für Richtungs-Features
   - ⚠️ Vorsicht bei 0°/360° Sprung

3. **Divergenz/Vorticity** (physikalisch)
   - Erfasst Strömungsmuster
   - Robuster als Richtungs-Gradienten
   - ✅ Empfohlen

**Empfehlung:**
- Primär: U/V-Gradienten, Divergenz, Vorticity
- Sekundär: Zirkulare Richtungs-Differenzen für spezifische Features

### Frage 5: Welche Preprocessing-Steps sind kritisch?

**XGBoost Besonderheiten:**
- ❌ KEINE Normalisierung nötig (tree-based)
- ❌ KEINE Outlier-Removal nötig (robust)
- ✅ Missing Values OK (XGBoost lernt Richtung)
- ✅ Aber: Zyklische Kodierung ESSENTIELL

**Kritische Preprocessing:**

1. **Zyklische Features:** sin/cos für Stunde, Monat, Richtung
2. **Lag-Features:** Zeitlich korrektes Shifting (kein Datenleak!)
3. **Grid-Alignment:** Korrekte Zuordnung Gitterpunkte ↔ Variablen
4. **Timestamp-Alignment:** Forecast ↔ Measurement exakt gleich

**Unkritisch:**
- Feature Scaling
- Log-Transformation
- PCA/Dimensionsreduktion

---

## 16. Zusammenfassung und Handlungsempfehlungen

### 16.1 Die 5 Kernerkenntisse

1. **Wind ist eine emergente Eigenschaft** aus Druck, Temperatur, Solar
   - Nicht isoliert vorhersagen
   - Multi-Skalen-Ansatz essentiell

2. **Gradienten > Absolutwerte**
   - Räumliche Gradienten erfassen Strömungsmuster
   - Pressure Gradient Force ist DER Treiber

3. **Synoptische Wetterlagen bestimmen 60-80%**
   - NAO, Föhn-Index, Druckzentren
   - Wenige Features mit hohem Signal

4. **Thermische Winde sind lernbar**
   - Solar + Land-Water-Diff + hour_sin/cos
   - Lecco regime-model beweist: keine GFS wind nötig

5. **Zweistufiger Ansatz funktioniert**
   - Model 1: Feature Discovery (400 Features)
   - Model 2: Production (50-70 Features, bessere Performance)

### 16.2 Feature-Engineering Roadmap

**Phase 1 (Quick Win):**
- ✅ Implementiere 7 Feature-Kategorien (Windmojo-Pipeline)
- ✅ ~400-500 Features
- ✅ Train Model 1, extrahiere Importance
- **Timeline:** 2-3 Tage Coding

**Phase 2 (Optimization):**
- ✅ 5 Feature-Selection-Strategien
- ✅ Train Model 2 Varianten
- ✅ Test-Week-Evaluation
- **Timeline:** 1-2 Wochen Experimente

**Phase 3 (Research):**
- 🔬 Automatische Druckzentren-Detektion
- 🔬 Erweiterte Regime-Klassifikation
- 🔬 Deep Learning Embeddings (Hybrid)
- **Timeline:** Forschungsprojekt

### 16.3 Konkrete nächste Schritte

**Für Ihr Projekt:**

1. **Daten-Check:**
   - ✅ Haben Sie 21-Punkt-Grid? (5×5 minus Ecken)
   - ✅ Welche Variablen verfügbar? (Pressure, T, Solar, Wind, Clouds)
   - ✅ Zeitspanne? (2+ Jahre empfohlen)

2. **Feature-Engineering:**
   - Start: Windmojo-Pipeline übernehmen (bereits 498 Features!)
   - Ergänzen: Synoptische Indices (NAO-Proxy, Föhn, etc.)
   - Custom: See-spezifische Features (falls Gewässer)

3. **Training:**
   - Model 1: Kitchen Sink (alle Features)
   - Analyse: Feature Importance, Korrelationen
   - Model 2: 5 Strategien parallel

4. **Evaluation:**
   - Test-Week-Ansatz (nicht random split!)
   - MAE pro Stunde analysieren (wann gut/schlecht?)
   - Vergleich mit GFS-Baseline

### 16.4 Erfolgskriterien

**Minimales Ziel:**
- Model 2 schlägt GFS-Baseline um 5-10%
- Test MAE < 4.0 m/s (für Windsurf-Spots)
- Overfitting-Gap < 2× (Test/Train)

**Ambitioniertes Ziel:**
- Model 2 schlägt GFS um 10-20%
- regime_indicators funktioniert (ohne GFS wind)
- MAE Nachmittags < 3.5 m/s

**Forschungsziel:**
- Physikalische Interpretierbarkeit
- Regime-basierte Vorhersage
- Transferierbarkeit auf andere Spots

---

## 17. Schlussfolgerung

Die Vorhersage von `wind_speed_10m` ohne direkte Verwendung von forecast wind_speed ist **machbar und wissenschaftlich fundiert**. Die Lecco-Resultate beweisen:

- ✅ 40 atmosphärische Features → 3.557 m/s MAE
- ✅ 53 balanced Features (inkl. GFS wind) → 3.388 m/s MAE
- ✅ Alle 5 Modelle schlagen GFS-Baseline

**Die Schlüssel zum Erfolg:**

1. **Multi-Skalen-Erfassung:** Synoptisch + Mesoskala + Lokal
2. **Physik-basierte Features:** Gradienten, Regime, Thermische Indikatoren
3. **Zweistufiges Training:** Discovery → Selection
4. **Unified Modell:** 13× mehr Daten, bessere Generalisierung

**Ihr Feature-Engineering sollte fokussieren auf:**

- 🎯 Druck-Gradienten (höchste Priorität)
- 🎯 Synoptische Indices (NAO, Föhn)
- 🎯 Thermische Indikatoren (Solar, Land-Water)
- 🎯 Zeitliche Features (Lags, Trends)
- 🎯 Zyklische Kodierung (Stunde, Monat)

**Der XGBoost-Ansatz wird:**
- Automatisch die wichtigsten Features selektieren
- Komplexe Interaktionen lernen
- Robuste Vorhersagen generieren

**Viel Erfolg mit Ihrem Windmojo-Modell!** 🌬️

---

## Appendix A: Literatur und Referenzen

### Meteorologische Grundlagen
- Holton, J. R. (2004). *An Introduction to Dynamic Meteorology*
- Wallace, J. M., & Hobbs, P. V. (2006). *Atmospheric Science: An Introductory Survey*

### Machine Learning für Wetter
- Rasp, S., et al. (2018). "Deep learning to represent subgrid processes in climate models"
- McGovern, A., et al. (2019). "Making the Black Box More Transparent"

### Wind-Forecasting
- Bauer, P., et al. (2015). "The quiet revolution of numerical weather prediction"
- Pinson, P. (2013). "Wind Energy: Forecasting Challenges"

### XGBoost
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- XGBoost Documentation: https://xgboost.readthedocs.io/

### Datenquellen
- Open-Meteo API: https://open-meteo.com/
- NOAA GFS: https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast

---

## Appendix B: Feature-Listen Template

### Synoptische Features (50)
```
nao_proxy
ea_pattern_index
scand_block_index
foehn_index_alps
genoa_low_pressure
azores_high_pressure
iceland_low_pressure
central_europe_pressure_mean
biscay_pressure_mean
mediterranean_pressure_mean
pressure_gradient_azores_iceland
pressure_gradient_genoa_alps
pressure_gradient_central_russia
...
```

### Räumliche Gradienten (80)
```
pressure_grad_NS_near
pressure_grad_NS_mid
pressure_grad_NS_far
pressure_grad_EW_near
pressure_grad_EW_mid
pressure_grad_EW_far
pressure_grad_NE_SW
pressure_grad_NW_SE
pressure_grad_magnitude
pressure_grad_direction
pressure_laplacian
pressure_curvature_NS
pressure_curvature_EW
temp_grad_NS
temp_grad_EW
wind_u_grad_NS
wind_v_grad_EW
wind_divergence
wind_vorticity
...
```

### Thermische Indikatoren (40)
```
solar_radiation_center
solar_heating_potential
solar_heating_rate
accumulated_solar_3h
accumulated_solar_12h
clear_sky_index
land_water_temp_diff
land_water_thermal_contrast
seabreeze_potential
landbreeze_potential
richardson_number
buoyancy_frequency
thermal_stability_index
...
```

### Zeitliche Dynamik (60)
```
pressure_lag_1h
pressure_lag_3h
pressure_lag_6h
pressure_lag_24h
temp_lag_1h
temp_lag_3h
wind_speed_lag_24h
pressure_change_1h
pressure_change_3h
pressure_change_6h
temp_change_1h
wind_change_1h
pressure_accel_3h
temp_accel_3h
pressure_rolling_std_3h
temp_rolling_mean_6h
wind_rolling_max_6h
precip_sum_3h
precip_sum_12h
...
```

### Vertikalprofile (30)
```
wind_shear_10m_100m
wind_shear_10m_850hPa
wind_shear_10m_500hPa
wind_dir_shear_10m_850hPa
total_wind_shear
lapse_rate_surface_850hPa
lapse_rate_deviation
has_inversion
inversion_strength
boundary_layer_height_estimate
humidity_gradient_surface_850hPa
...
```

### Zyklische & Binäre (20)
```
hour_sin
hour_cos
month_sin
month_cos
dayofyear_sin
dayofyear_cos
is_morning
is_afternoon
is_evening
is_night
is_summer
is_winter
is_weekend
is_raining
is_cloudy
...
```

**TOTAL: ~280 engineered + ~120 raw = 400 Features**

---

**Ende des Dokuments**

*Dieses Dokument wurde erstellt von Claude (Anthropic) basierend auf meteorologischer Fachliteratur, den Windmojo-Projektspezifikationen, und den Lecco-Trainingsergebnissen.*

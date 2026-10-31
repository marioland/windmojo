# Deployment: Live Forecasts on a Website

How to run a trained Windmojo model in production and show its forecasts on a
website — including ordinary shared web hosting (e.g. at a Swiss hosting provider)
that has no Python support.

**Contents**

1. [Architecture](#1-architecture)
2. [Prerequisites](#2-prerequisites)
3. [The `predict` command](#3-the-predict-command)
4. [JSON format](#4-json-format)
5. [Choosing a deployment variant](#5-choosing-a-deployment-variant)
6. [Variant A: on the web host (SSH + cron)](#6-variant-a-on-the-web-host-ssh--cron)
7. [Variant B: GitHub Actions + SFTP upload](#7-variant-b-github-actions--sftp-upload)
8. [Variant C: own server / VPS](#8-variant-c-own-server--vps)
9. [Website integration](#9-website-integration)
10. [Schedule](#10-schedule)
11. [Operations](#11-operations)
12. [Licensing and data protection](#12-licensing-and-data-protection)
13. [Troubleshooting](#13-troubleshooting)

Related: [README.md](README.md) (overview), [SIMPLIFIED_ARCHITECTURE.md](SIMPLIFIED_ARCHITECTURE.md)
(training pipeline, CLI), [data/README.md](data/README.md) (directory layout),
[`web/index.html`](web/index.html) (forecast page),
[`.github/workflows/forecast.yml`](.github/workflows/forecast.yml) (variant B).

---

## 1. Architecture

The model is **not** run by the website. A scheduled job computes the forecast a few
times per day and writes a small static JSON file; the website only displays that file.

```
 Scheduled job (cron / GitHub Actions / systemd timer), every 6 hours
 ┌──────────────────────────────────────────────────────────────────────┐
 │ python cli.py predict --spot greifensee --output .../greifensee.json │
 │   1. download Open-Meteo forecast for the 9-point grid               │
 │      (7 past days for lag/rolling features + 7 forecast days)        │
 │   2. same feature pipeline as for the training data                  │
 │   3. apply models/model.json                                         │
 │   4. write JSON (atomically)                                         │
 └──────────────────────────────┬───────────────────────────────────────┘
                                │ file on the web space (directly or via SFTP)
                                ▼
 Web host:  /wind/greifensee.json  +  /wind/index.html (web/index.html)
                                │
                                ▼
 Visitor's browser: loads the page, which loads the JSON (same domain)
```

Why this design:

- **Works with any hosting.** Shared hosting usually runs PHP, not Python with
  XGBoost/pandas. A static JSON file can be served by any host.
- **Matches the data.** The weather models behind Open-Meteo update a few times per
  day; computing the forecast per page view would only cost time and API calls.
- **Robust.** If the job fails once, the website still shows the last forecast
  (with a warning once it is older than 12 hours).
- **Fast and private.** Visitors only load two static files from your own domain.

Exporting the XGBoost model to JavaScript/PHP (e.g. via ONNX) is possible in principle,
but the ~160 features would have to be reimplemented outside Python — not worth it.

## 2. Prerequisites

1. **A trained model** in the new spot layout:

   ```
   data/spots/<spot>/
   ├── input/config.yaml       # simplified config format
   └── models/
       ├── model.json
       └── metadata.json
   ```

   Created by `python cli.py train --spot <spot>` (see
   [SIMPLIFIED_ARCHITECTURE.md](SIMPLIFIED_ARCHITECTURE.md)). Measurements, raw data and
   features are **not** needed in production — only `config.yaml` and `models/`.
   Forecasts are always in m/s. If your measurements are in km/h, knots or mph, set
   `target.unit` in the config **before training** (measurements are converted on
   loading); a model trained on unconverted km/h values would output km/h labelled as m/s.

2. **A `forecast` section in `config.yaml`** that matches the training download:

   ```yaml
   forecast:
     initialization_time: "06:00"   # first hour per day, local time (config `timezone`)
     horizon_hours: 18              # hours per day -> 06:00-23:00
     days_ahead: 7                  # forecast days (max. 16)
   ```

   The training data contains only these hours per day (default 06:00–23:00), so the
   forecast covers the same hours. Without the section, these defaults apply.

3. **Same code version for training and prediction.** `predict` recomputes the
   features with the current `windmojo/features/pipeline.py`. After changing the
   feature pipeline, regenerate the features and retrain before deploying.

   Models trained before `predict` existed lack the training medians in
   `metadata.json` (used to fill missing values). They still work, but retraining is
   recommended.

4. **Open-Meteo access.** The free API is for non-commercial use only. For a
   commercial website, get an API key (see [section 12](#12-licensing-and-data-protection)).

## 3. The `predict` command

```bash
python cli.py predict --spot greifensee
python cli.py predict --spot greifensee --output ~/public_html/wind/greifensee.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--spot` | (required) | Spot name |
| `--config`, `-c` | `data/spots/<spot>/input/config.yaml` | Spot configuration |
| `--output`, `-o` | `data/spots/<spot>/forecasts/forecast.json` (`paths.forecast` in the config) | JSON output file |
| `--history-days` | 7 | Past days downloaded for lag and rolling features (the longest model feature is a 7-day precipitation sum) |
| `--archive/--no-archive` | archive | Keep a copy of each forecast in `data/spots/<spot>/forecasts/archive/<YYYY-MM>/<time>.json` (`paths.forecast_archive`) for later verification against measurements. The first run of a day also keeps the raw weather forecast for tomorrow and the day after (`forecast.raw_archive_days`, default `[1, 2]`) in `forecasts/archive_raw/<YYYY-MM>/<date>.parquet`, about 45 MB per year: the Open-Meteo Previous Runs API lacks pressure levels, cloud layers and soil variables, so these files are the only complete old forecasts for training on longer lead times. |

What it does (`windmojo/prediction/forecast.py`):

1. Downloads the Open-Meteo forecast API for the 9 grid points — same variables, grid
   and time zone as the training download, one request per grid point (9 requests per
   run). The forecast API also serves the past days needed for lag features; for
   recent days its values are identical to the historical data used in training.
2. Keeps the hours of the forecast window (`initialization_time` + `horizon_hours`).
3. Runs the same feature pipeline as training. No feature looks back more than 7 days
   and none uses statistics over the whole input, so live features equal training
   features.
4. Predicts with the model; missing values are filled with the training medians.
5. Writes the JSON to a temporary file and renames it, so the web server never serves
   a half-written file, and keeps a copy in the forecast archive.

Resource use (measured for Greifensee, 7 forecast days): ~6 s runtime, ~210 MB RAM,
9 requests to Open-Meteo.
The Python environment with all dependencies needs roughly 0.7–1 GB of disk space.

If the Open-Meteo API key is needed, set `OPEN_METEO_API_KEY` in the environment or in a
`.env` file in the repository root (loaded automatically when running `cli.py` from the
checkout).

### Local forecast page: `cli.py serve`

For personal use no schedule and no web host are needed:

```bash
cd ~/windmojo                              # repository root
.venv/bin/python cli.py serve --spot myspot
```

Then open <http://localhost:8000/> in the browser (from WSL also in the Windows browser).
Stop with Ctrl+C. Options: `--port 8001` for another port, `--host 0.0.0.0` to reach the
page from other devices in the network (no password protection, only in trusted networks).

The page is `web/index.html`, served by a small server from the Python standard library.
Served this way it offers more than as a static file:

| Element | What it does |
|---------|--------------|
| Header | What the project is about at a glance: one chart with the past days the model was not trained on (measured, weather model, Windmojo) and the next days, the three data sources explained, and how many percent Windmojo was more accurate than the weather model on the measured hours. On a static website the header shows the forecast only. |
| Wingfoiling overview | Time windows of today and the next 2 days with Windmojo mean wind of 10–15 kn, 8–20 h (also on a static website; range in `SESSION` at the top of the page script). |
| Button "Update forecast and measurements" | Runs the measurement script of the spot (`measurements.update_script` in the config, e.g. a Datalakes download) and then `predict` (9 Open-Meteo requests, archived as usual). Each source is downloaded at most every 10 minutes; a second click within that time reuses the data. |
| Measured values | Hours of the forecast that are already measured show the measurement next to the forecast (table column "Measured", green line). Gusts and direction need `target.gust_column` and `target.direction_column`. |
| "Look back" | The past 7 days: the forecast as it was shown then (last archived run before the day, from `forecasts/archive/`; fills up with every update), the hindcast (dotted: what the model says for the day with the weather data of that day, from `forecasts/hindcast/`) and the measurement, with the mean deviation of each and how many percent Windmojo is more accurate than Open-Meteo (same hours only). Days up to the end of the model's training data are marked: their hindcast looks better than a real forecast. |
| CSV downloads | `download/forecast.csv` (current forecast with the measurements so far) and `download/history.csv?days=30` (past days: forecast shown, hindcast and measurement per hour), m/s, local time. |

Spot configuration for the update button:

```yaml
target:
  gust_column: "myspot gust"              # optional, measured gusts
forecast:
  days_ahead: 3                           # today + the next 2 days
  hindcast_days: 7                        # past days computed with their weather data
measurements:                             # optional, refreshes paths.measurements
  update_script: "data/myarea/myspot/download_measurements.py"   # relative to the repo root
  update_args: []
```

URL parameters (also on a static website): `?unit=kn` (or `m/s`, `km/h`) sets the unit,
`?view=header` shows only the header. The screenshot in the README is made that way, with any
Chrome or Edge:

```bash
chrome --headless --hide-scrollbars --window-size=1320,820 --screenshot=screenshot.png \
  "http://localhost:8000/?view=header&unit=kn"
```

The script runs with the same Python as the server. Without `measurements`, the button only
creates a new forecast. For the hindcast, `predict` downloads `hindcast_days` more past days
(same number of requests, longer time range); `hindcast_days: 0` switches it off.

## 4. JSON format

```json
{
  "spot": "greifensee",
  "name": "Greifensee",
  "latitude": 47.366,
  "longitude": 8.665,
  "timezone": "Europe/Zurich",
  "generated_at": "2026-10-31T10:10:05+01:00",
  "units": {"wind_speed": "m/s", "wind_direction": "deg"},
  "model": {"type": "unified", "test_mae": 0.71, "missing_features": 0},
  "source": "Weather data by Open-Meteo.com (CC BY 4.0)",
  "hours": [
    {
      "time": "2026-10-31T14:00",
      "wind_speed": 3.7,
      "openmeteo_wind_speed": 3.1,
      "openmeteo_gusts": 5.3,
      "wind_direction": 62
    }
  ]
}
```

| Field | Meaning |
|-------|---------|
| `generated_at` | Creation time with UTC offset. Use it to detect stale forecasts. |
| `model.test_mae` | Mean absolute error on the test days during training (m/s). |
| `model.missing_features` | Number of model features missing in the forecast data (filled with training medians). Should be 0; a value above 0 means the download lacked variables or grid points and the forecast may be less accurate. |
| `hours[].time` | Local time of the spot (`timezone`), without offset. Starts with today's first forecast hour; past hours of today are included. Night hours are not forecast. |
| `hours[].wind_speed` | **Windmojo forecast**, mean wind in m/s (what the model was trained on). |
| `hours[].openmeteo_wind_speed` | Raw weather model wind at 10 m, center point, m/s — for comparison. |
| `hours[].openmeteo_gusts` | Raw weather model gusts at 10 m, m/s. |
| `hours[].wind_direction` | Weather model wind direction at 10 m, degrees, direction the wind comes **from**. The model does not predict direction. |

Values may be `null` if Open-Meteo returns no value for an hour. Size: ~20 KB for 7 days.

## 5. Choosing a deployment variant

| | A: on the web host | B: GitHub Actions | C: own server / VPS |
|---|---|---|---|
| **Needs** | Hosting plan with SSH, cron and Python ≥ 3.10 | Any hosting with SFTP access, a GitHub account | A (virtual) server |
| **Runs where** | At your host | GitHub (USA), uploads to your host | Your server |
| **Cost** | Included in the plan | Free for public repos; private repos within the free minutes (4 runs/day ≈ 250–350 min/month) | Server rent |
| **Effort** | Low, if the plan allows it | Low, no installation at the host | Medium (updates, security) |
| **Good for** | Hosting plans with SSH | Classic web hosting (PHP only) | Several spots, own API, full control |

**Recommendation:** check first whether your hosting plan offers SSH access with
Python ≥ 3.10 and cron jobs (many Swiss providers such as Hostpoint, cyon, Infomaniak
or Metanet offer this on at least some plans — check your plan's documentation). If
yes, use **A**. If not, use **B**. Choose **C** only if you need more than a JSON file.

## 6. Variant A: on the web host (SSH + cron)

### 6.1 Check the hosting plan

Log in via SSH and check:

```bash
python3 --version        # needs 3.10 or newer (some hosts offer python3.11/python3.12 explicitly)
python3 -m venv --help   # venv module available?
crontab -l               # cron available? (some hosts only offer cron jobs in their control panel)
df -h ~ ; quota 2>/dev/null   # ~1 GB free for the Python environment
```

Also check the plan's limits for process runtime and memory (the job needs ~210 MB RAM
for a few seconds) and whether outgoing HTTPS connections are allowed (needed for
`api.open-meteo.com`).

### 6.2 Install

Keep the code, configuration and model **outside the web root**; only the JSON and
the page go into it.

```
~/windmojo/                     # repository checkout (not publicly reachable)
│   └── data/spots/greifensee/input/config.yaml, models/
~/public_html/wind/             # web root of your domain (name depends on the host)
    ├── index.html              # copy of web/index.html
    └── greifensee.json         # written by the cron job
```

```bash
cd ~
git clone https://github.com/marioland/windmojo.git windmojo   # or upload via SFTP
cd windmojo
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

# Copy config + model from your training machine, e.g.:
#   scp -r data/spots/greifensee/input data/spots/greifensee/models \
#       user@host:windmojo/data/spots/greifensee/

# Optional (commercial use): Open-Meteo API key
echo 'OPEN_METEO_API_KEY=...' > .env && chmod 600 .env

# Test run
.venv/bin/python cli.py predict --spot greifensee --output ~/public_html/wind/greifensee.json
```

If `pip install` fails for `xgboost` or `pyarrow` (no prebuilt wheel for the host's
platform, compiler missing, old glibc), the plan is not suitable: use variant B.

### 6.3 Cron job

`crontab -e` (or the cron job form in the host's control panel):

```cron
# Windmojo forecast, local server time; shortly after new weather model runs
15 0,6,12,18 * * *  cd $HOME/windmojo && .venv/bin/python cli.py predict --spot greifensee --output $HOME/public_html/wind/greifensee.json >> $HOME/windmojo/forecast.log 2>&1
```

- The `cd` matters: `cli.py` finds `data/spots/...` and `.env` relative to the checkout.
- Check `forecast.log` now and then, or set `MAILTO=you@example.ch` at the top of the
  crontab to get output by mail (then drop the `>> forecast.log` redirect, or redirect
  stdout only and keep errors: `> /dev/null`).
- Some control panels do not allow `cd ...&&` in the command field; then put the line
  into a small script (e.g. `~/windmojo/run_forecast.sh`, `chmod +x`) and call that.

Then set up the page: [section 9](#9-website-integration).

## 7. Variant B: GitHub Actions + SFTP upload

The workflow [`.github/workflows/forecast.yml`](.github/workflows/forecast.yml) runs
`predict` on GitHub's servers four times a day and uploads the JSON to your web space
via SFTP. Nothing needs to be installed at the host.

### 7.1 Put config and model into the repository

The workflow checks out the repository, so `config.yaml` and the model must be
committed. `data/` is ignored by git, so add them explicitly:

```bash
git add -f data/spots/greifensee/input/config.yaml \
           data/spots/greifensee/models/model.json \
           data/spots/greifensee/models/metadata.json
git commit -m "Add Greifensee model for forecast deployment"
```

Neither file contains measurement data (`model.json` ~1–2 MB, `metadata.json` holds
feature names, training medians and metrics). If you don't want to publish the model,
use a **private** repository (fork or copy) for the deployment.

The config must use relative default paths (no absolute `paths:` entries from your
machine).

### 7.2 SFTP access

You need from your host: SFTP host name, user, password or SSH key, and the target
directory inside the web root (e.g. `public_html/wind` or `web/wind`; the name depends
on the host).

An SSH key is preferable to a password where the host supports it:

```bash
ssh-keygen -t ed25519 -N "" -C windmojo-deploy -f windmojo_deploy
# Add windmojo_deploy.pub to the host (control panel or ~/.ssh/authorized_keys),
# store the content of windmojo_deploy as secret SFTP_PRIVATE_KEY, then delete the local file.
```

### 7.3 Configure the repository

GitHub → repository → *Settings → Secrets and variables → Actions*:

| Type | Name | Example | Required |
|------|------|---------|----------|
| Variable | `FORECAST_SPOT` | `greifensee` | yes — the workflow is inactive without it |
| Variable | `SFTP_HOST` | `sftp.example.ch` | yes |
| Variable | `SFTP_PORT` | `22` | no (default 22) |
| Variable | `SFTP_USER` | `myaccount` | yes |
| Variable | `SFTP_DIR` | `public_html/wind` | yes |
| Secret | `SFTP_PRIVATE_KEY` | content of the private key | one of the two |
| Secret | `SFTP_PASSWORD` | | one of the two |
| Secret | `OPEN_METEO_API_KEY` | | for commercial use |

The JSON is uploaded as `$SFTP_DIR/<spot>.json`.

### 7.4 Test

*Actions → Forecast → Run workflow*. Then open
`https://your-domain.ch/wind/greifensee.json` in the browser.

### 7.5 Things to know

- Scheduled runs only start on the **default branch** and may be delayed by several
  minutes when GitHub is busy.
- In **public** repositories, GitHub disables scheduled workflows after 60 days without
  repository activity. GitHub sends a notification; re-enable the workflow under
  *Actions*. Private repositories are not affected.
- GitHub emails the repository owner when a run fails.
- The workflow only processes public weather data and the model — no personal data.

## 8. Variant C: own server / VPS

Same installation as [variant A](#62-install), with full control:

- Swiss VPS/cloud offers exist (e.g. Infomaniak, Exoscale); any small Linux VM is
  enough (1 vCPU, 1 GB RAM).
- Schedule with cron as in [6.3](#63-cron-job) or a systemd timer.
- Serve the JSON with the existing web server (nginx/Apache), or upload it to the web
  host via SFTP as in the workflow.
- You are responsible for operating system updates and security.

Only worth it for several spots, an own API, or when no other variant fits.

## 9. Website integration

[`web/index.html`](web/index.html) is a ready-made forecast page: chart for the whole
week, one table per day, unit switch (m/s, knots, km/h), past hours greyed out, warning
for forecasts older than 12 hours, dark mode, mobile layout. It has **no external
dependencies** (no CDN, no web fonts, no tracking).

### 9.1 Standalone page

Copy `web/index.html` into the same directory as the JSON:

```
public_html/wind/index.html
public_html/wind/greifensee.json
```

→ `https://your-domain.ch/wind/`. If the JSON has a different name or location, change
`data-src` in the page:

```html
<div class="windmojo" id="windmojo" data-src="greifensee.json">
```

Texts are English; they are plain strings in the script and easy to translate.

### 9.2 Embedding in an existing site / CMS

- **iframe** (simplest, works in every CMS):

  ```html
  <iframe src="/wind/" title="Windprognose" style="width:100%;height:1200px;border:0"></iframe>
  ```

- **Inline** (e.g. WordPress "Custom HTML" block, Joomla/TYPO3 HTML element): copy the
  `<style>` block, the `<div class="windmojo" ...>` and the `<script>` block from
  `web/index.html`. All styles are scoped to `.windmojo`. Use an absolute
  `data-src` (e.g. `/wind/greifensee.json`). Some CMS strip `<script>` tags for
  non-admin users; then use the iframe.

- **Own design**: load the JSON yourself (`fetch('/wind/greifensee.json')`) — the format
  is described in [section 4](#4-json-format). Server-side PHP works as well:
  `json_decode(file_get_contents(__DIR__ . '/greifensee.json'), true)`.

### 9.3 JSON on a different domain

If the page and the JSON are on different domains, the browser needs a CORS header.
On Apache hosts (most shared hosting), add a `.htaccess` next to the JSON:

```apache
<Files "*.json">
  Header set Access-Control-Allow-Origin "https://www.your-domain.ch"
</Files>
```

### 9.4 Caching

The page appends a 10-minute cache-busting parameter to the JSON request. If your host
caches aggressively anyway, limit it with `.htaccess`:

```apache
<Files "*.json">
  Header set Cache-Control "max-age=600"
</Files>
```

## 10. Schedule

Open-Meteo's `best_match` combines several weather models. The global models (GFS,
ECMWF IFS) run at 00, 06, 12 and 18 UTC and become available a few hours later; the
high-resolution regional models used for Switzerland update more often. Four runs per
day are a good compromise:

- Variant B: `15 4,10,16,22 * * *` (UTC, preset in the workflow).
- Variant A/C: same idea in server time, e.g. `15 0,6,12,18 * * *` for Swiss time.

More frequent runs are possible. Each run makes 9 requests; since Open-Meteo weights
requests with many variables as several calls, a run counts as roughly 16 API calls
(7 history + 7 forecast days). The free tier allows 10,000 calls per day.

## 11. Operations

| Task | How |
|------|-----|
| Detect a stopped job | The page shows a warning when `generated_at` is older than 12 hours. Variant A: `forecast.log` / cron mail. Variant B: GitHub failure emails. |
| Deploy a retrained model | Replace `models/model.json` and `models/metadata.json` (A/C: copy to the host; B: commit). The next run uses it. |
| Update code | A/C: `git pull` in the checkout (+ `pip install -e .` if dependencies changed). B: automatic (workflow uses the repository state). After feature pipeline changes: retrain first (see [2.3](#2-prerequisites)). |
| Add a second spot | Second `data/spots/<spot>/`, second cron line or second workflow job with its own `FORECAST_SPOT`; the page can be copied with a different `data-src`. |
| Check plausibility | Compare `wind_speed` with `openmeteo_wind_speed` in the JSON; large systematic differences on many days indicate a problem (e.g. config/model mismatch). Long-term: compare the archived forecasts (`forecasts/archive/`) with new measurements per lead time and retrain periodically. |
| Retrain while forecasts run | Safe: `train` writes the model under temporary names and renames both files at the end. |

## 12. Licensing and data protection

- **Open-Meteo terms:** the free API (`api.open-meteo.com`) may only be used
  **non-commercially**. A commercial website (company site, ads, paid courses or
  rentals) needs an API subscription; set its key as `OPEN_METEO_API_KEY`. See
  <https://open-meteo.com/en/pricing> and <https://open-meteo.com/en/terms>.
- **Attribution:** Open-Meteo data is licensed under CC BY 4.0. The page shows
  "Weather data: Open-Meteo.com (CC BY 4.0)" in its footer — keep an equivalent notice
  when building your own page.
- **Measurement data:** production does not need the measurements. If you also want
  to publish live measurements on the website, check the data provider's terms of use
  separately (e.g. Datalakes for Greifensee).
- **Data protection (Swiss nDSG / GDPR):** the forecast is computed in the background;
  visitors' browsers only contact your own domain. `web/index.html` loads no external
  resources. The unit choice is stored in the visitor's `localStorage` (no cookie, no
  personal data). With variant B, only public weather data and the model are processed
  at GitHub.

## 13. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `FileNotFoundError: ... config.yaml` | Run from the repository root (`cd ~/windmojo`), or pass `--config`. |
| `... model.json` not found | `models/` missing on the host or not committed (variant B: `git add -f`, [7.1](#71-put-config-and-model-into-the-repository)). |
| `Missing required field: 'latitude'` | Config is still in the legacy format; convert it (see `config/examples/dervio.yaml`). |
| `N model features missing in forecast data` | Open-Meteo returned fewer variables than at training time, or model and pipeline versions differ. Missing values are filled with training medians; retrain if the warning persists. |
| `HTTP error 429` / `Rate limit exceeded` | Too many requests (also counts other users on a shared host IP). Run less often or use an API key. |
| `HTTP error 400` with API key | Key invalid or plan without access to the forecast API. |
| Page: "Could not load the forecast (HTTP 404)" | `data-src` does not match the JSON location. |
| Page shows old forecast without warning | Browser/host cache; see [9.4](#94-caching). |
| Times shifted by 1–2 hours | `timezone` missing in the config (defaults to UTC). |
| Workflow does not run on schedule | `FORECAST_SPOT` not set, not the default branch, or disabled after 60 days of inactivity ([7.5](#75-things-to-know)). |

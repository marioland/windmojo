"""
Local Web Server

`python cli.py serve` shows the forecast page (web/index.html) with what a
static website cannot offer: an update button (new forecast and new
measurements), the past days as forecast and as measured, and CSV downloads.
Standard library only; meant for the local machine, not for the internet.
"""

import json
import logging
import re
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytz

from windmojo.config import get_forecast_config
from windmojo.prediction.forecast import run_forecast
from windmojo.prediction.history import (
    history,
    load_archive,
    load_measurements,
    to_csv,
    with_measurements,
)

logger = logging.getLogger(__name__)

PAGE = Path(__file__).resolve().parents[1] / "web" / "index.html"

# Minimum time between two downloads of the same source (protects the free APIs)
MIN_UPDATE_INTERVAL = timedelta(minutes=10)


class ForecastApp:
    """Data and actions behind the web server (usable without HTTP)."""

    def __init__(
        self,
        config: Dict[str, Any],
        paths: Dict[str, Path],
        page: Path = PAGE,
        forecast_runner: Callable = run_forecast,
    ):
        """
        Args:
            config: Spot configuration (simplified format). Optional section
                `measurements: {update_script, update_args}`: script that
                refreshes paths["measurements"], run with this Python from
                the current directory
            paths: Paths from get_paths(config)
            page: Forecast page (HTML)
            forecast_runner: Creates a new forecast (run_forecast; replaceable in tests)
        """
        self.config = config
        self.paths = paths
        self.page = Path(page)
        self.forecast_runner = forecast_runner
        self.timezone = pytz.timezone(config.get("timezone", "UTC"))
        self.lock = threading.Lock()

    def now(self) -> datetime:
        return datetime.now(self.timezone)

    def page_html(self) -> str:
        """The forecast page, reading its JSON from this server."""
        html = self.page.read_text()
        return re.sub(r'data-src="[^"]*"', 'data-src="forecast.json"', html, count=1)

    def raw_forecast(self) -> Optional[Dict[str, Any]]:
        path = Path(self.paths["forecast"])
        return json.loads(path.read_text()) if path.exists() else None

    def measurements(self) -> pd.DataFrame:
        path = Path(self.paths["measurements"])
        if "target" not in self.config or not path.exists():
            return pd.DataFrame(columns=["speed"], index=pd.DatetimeIndex([]))
        return load_measurements(self.config, path)

    def forecast(self) -> Optional[Dict[str, Any]]:
        """Current forecast, each hour with the measurements available so far."""
        data = self.raw_forecast()
        if data is not None:
            data["hours"] = with_measurements(data, self.measurements())
        return data

    def status(self) -> Dict[str, Any]:
        forecast = self.raw_forecast()
        measured = self.measurements()["speed"].dropna()
        return {
            "spot": self.config["spot"],
            "name": self.config.get("name", self.config["spot"]),
            "forecast_generated_at": forecast["generated_at"] if forecast else None,
            "measurements_until": (
                measured.index.max().strftime("%Y-%m-%dT%H:%M")
                if len(measured)
                else None
            ),
            "measurements_update": bool(self._update_script()),
            "updating": self.lock.locked(),
        }

    def history(self, days: int) -> Dict[str, Any]:
        runs = load_archive(self.paths["forecast_archive"])
        window = get_forecast_config(self.config)
        start = int(window["initialization_time"].split(":")[0])
        hours = [(start + i) % 24 for i in range(window["horizon_hours"])]
        hindcast_dir = self.paths.get("forecast_hindcast")
        return {
            "days": history(
                runs, self.measurements(), self.now().date(), days, hours, hindcast_dir
            )
        }

    def forecast_csv(self) -> Tuple[str, str]:
        data = self.forecast()
        if data is None:
            return "", f"{self.config['spot']}_prognose.csv"
        name = f"{self.config['spot']}_prognose_{data['generated_at'][:10]}.csv"
        return to_csv(data["hours"], data["generated_at"]), name

    def history_csv(self, days: int) -> Tuple[str, str]:
        rows = [
            {**h, "forecast_generated_at": day["forecast_generated_at"]}
            for day in self.history(days)["days"]
            for h in day["hours"]
        ]
        name = f"{self.config['spot']}_rueckblick_{self.now():%Y-%m-%d}.csv"
        return to_csv(rows), name

    def update(self) -> Tuple[int, Dict[str, Any]]:
        """Download new measurements (if configured) and a new forecast."""
        if not self.lock.acquire(blocking=False):
            return 409, {"ok": False, "messages": ["An update is already running."]}
        try:
            messages = []
            ok = self._update_measurements(messages)
            ok = self._update_forecast(messages) and ok
        finally:
            self.lock.release()
        return 200, {"ok": ok, "messages": messages, **self.status()}

    def _update_script(self) -> Optional[str]:
        return self.config.get("measurements", {}).get("update_script")

    def _age(self, timestamp: Optional[datetime]) -> Optional[timedelta]:
        return None if timestamp is None else self.now() - timestamp

    def _update_measurements(self, messages) -> bool:
        script = self._update_script()
        if not script:
            return True
        path = Path(self.paths["measurements"])
        if path.exists():
            age = self._age(datetime.fromtimestamp(path.stat().st_mtime, self.timezone))
            if age < MIN_UPDATE_INTERVAL:
                messages.append(
                    f"Measurements are up to date (loaded {age.seconds // 60} min ago)."
                )
                return True
        args = [str(a) for a in self.config["measurements"].get("update_args", [])]
        logger.info(f"Updating measurements: {script} {' '.join(args)}")
        try:
            run = subprocess.run(
                [sys.executable, script, *args],
                capture_output=True,
                text=True,
                timeout=900,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            messages.append(f"Measurements: error ({e}).")
            return False
        if run.returncode != 0:
            last = (run.stderr or run.stdout).strip().splitlines()[-1:] or ["?"]
            messages.append(f"Measurements: error ({last[0]}).")
            return False
        messages.append("Measurements loaded.")
        return True

    def _update_forecast(self, messages) -> bool:
        current = self.raw_forecast()
        if current:
            age = self._age(datetime.fromisoformat(current["generated_at"]))
            if age < MIN_UPDATE_INTERVAL:
                messages.append(
                    f"Forecast is up to date (created {age.seconds // 60} min ago)."
                )
                return True
        try:
            self.forecast_runner(self.config, self.paths)
        except Exception as e:  # shown on the page instead of a server error
            logger.exception("Forecast failed")
            messages.append(f"Forecast: error ({e}).")
            return False
        messages.append("New forecast created.")
        return True


def make_handler(app: ForecastApp):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            days = min(max(int((query.get("days") or ["7"])[0]), 1), 60)
            if url.path in ("/", "/index.html"):
                self._send(200, app.page_html(), "text/html; charset=utf-8")
            elif url.path == "/forecast.json":
                data = app.forecast()
                if data is None:
                    self._json(
                        404, {"message": "No forecast yet. Please update."}
                    )
                else:
                    self._json(200, data)
            elif url.path == "/api/status":
                self._json(200, app.status())
            elif url.path == "/api/history":
                self._json(200, app.history(days))
            elif url.path == "/download/forecast.csv":
                self._csv(*app.forecast_csv())
            elif url.path == "/download/history.csv":
                self._csv(*app.history_csv(days))
            else:
                self._json(404, {"message": "Not found"})

        def do_POST(self):
            if urlparse(self.path).path == "/api/update":
                self._json(*app.update())
            else:
                self._json(404, {"message": "Not found"})

        def _send(self, status, body, content_type, headers=None):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status, body):
            self._send(
                status,
                json.dumps(body, ensure_ascii=False),
                "application/json; charset=utf-8",
            )

        def _csv(self, body, filename):
            self._send(
                200,
                body,
                "text/csv; charset=utf-8",
                {"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        def log_message(self, format, *args):
            logger.info("%s - %s", self.address_string(), format % args)

    return Handler


def serve(
    app: ForecastApp, host: str = "127.0.0.1", port: int = 8000
) -> ThreadingHTTPServer:
    """Create the HTTP server (call serve_forever() on the result)."""
    return ThreadingHTTPServer((host, port), make_handler(app))

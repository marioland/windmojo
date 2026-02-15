"""
Open-Meteo API Client

Interface with Open-Meteo Historical Forecast and Forecast APIs.
Handles requests, retries, rate limiting, and response parsing.
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class OpenMeteoError(Exception):
    """Raised when Open-Meteo API returns an error."""

    pass


class OpenMeteoClient:
    """
    Client for Open-Meteo Weather API.

    Supports both historical forecast (training data) and current forecast
    (production predictions) modes.
    """

    # API endpoints
    HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self, timeout: int = 30, max_retries: int = 3, retry_delay: float = 1.0
    ):
        """
        Initialize Open-Meteo API client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds, exponential backoff)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

        # Load API key from .env file (optional, for commercial tier)
        load_dotenv()
        self.api_key = os.getenv("OPEN_METEO_API_KEY") or os.getenv("apikey")

        if self.api_key:
            logger.info("Open-Meteo API key loaded (commercial tier)")
        else:
            logger.debug("No API key found, using free tier")

    def fetch_historical_forecast(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: List[str],
        timezone: str = "UTC",
        model: str = "best_match",
    ) -> pd.DataFrame:
        """
        Fetch historical forecast data.

        Use this for training data (past dates).

        Args:
            latitude: Location latitude (-90 to 90)
            longitude: Location longitude (-180 to 180)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            variables: List of variable names to fetch
            timezone: Timezone string (e.g., "Europe/Rome", "UTC")
            model: Model selection ("best_match" recommended)

        Returns:
            DataFrame with columns: time, [variables...]

        Raises:
            OpenMeteoError: If API request fails
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": timezone,
            "models": model,
        }

        # Add API key if available
        if self.api_key:
            params["apikey"] = self.api_key

        return self._fetch_with_retry(self.HISTORICAL_URL, params)

    def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: List[str],
        timezone: str = "UTC",
    ) -> pd.DataFrame:
        """
        Fetch current/future forecast data.

        Use this for production predictions (future dates, up to 16 days ahead).

        Args:
            latitude: Location latitude (-90 to 90)
            longitude: Location longitude (-180 to 180)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            variables: List of variable names to fetch
            timezone: Timezone string (e.g., "Europe/Rome", "UTC")

        Returns:
            DataFrame with columns: time, [variables...]

        Raises:
            OpenMeteoError: If API request fails
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": timezone,
        }

        # Add API key if available
        if self.api_key:
            params["apikey"] = self.api_key

        return self._fetch_with_retry(self.FORECAST_URL, params)

    def _fetch_with_retry(self, url: str, params: Dict) -> pd.DataFrame:
        """
        Fetch data with exponential backoff retry logic.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            DataFrame with fetched data

        Raises:
            OpenMeteoError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            is_rate_limit = False  # Track if this is a rate limit error
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)

                # Check for HTTP errors
                response.raise_for_status()

                # Parse JSON response
                data = response.json()

                # Check for API-level errors
                if "error" in data and data["error"]:
                    raise OpenMeteoError(
                        f"API error: {data.get('reason', 'Unknown error')}"
                    )

                # Parse into DataFrame
                if "hourly" not in data:
                    raise OpenMeteoError("Response missing 'hourly' data")

                df = pd.DataFrame(data["hourly"])

                # Convert time column to datetime
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])

                logger.debug(
                    f"Successfully fetched {len(df)} rows for "
                    f"({params['latitude']:.4f}, {params['longitude']:.4f})"
                )

                return df

            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout: {e}"
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries}: {last_error}"
                )
                is_rate_limit = False

            except requests.exceptions.HTTPError as e:
                # Check if this is a rate limit error (429)
                if e.response.status_code == 429:
                    last_error = "Rate limit exceeded"
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries}: {last_error}"
                    )
                    # For rate limit, use longer backoff
                    is_rate_limit = True

                # Server errors (5xx) are retryable
                elif 500 <= e.response.status_code < 600:
                    last_error = f"Server error: {e.response.status_code}"
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries}: {last_error}"
                    )
                    is_rate_limit = False

                # Client errors (4xx) are not retryable
                else:
                    raise OpenMeteoError(f"HTTP error {e.response.status_code}: {e}")

            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {e}"
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries}: {last_error}"
                )
                is_rate_limit = False

            except (KeyError, ValueError) as e:
                # JSON parsing or data structure errors - not retryable
                raise OpenMeteoError(f"Failed to parse response: {e}")

            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                if is_rate_limit:
                    # For rate limit errors, wait longer (10s, 20s, 40s)
                    delay = 10 * (2**attempt)
                else:
                    # Normal exponential backoff (1s, 2s, 4s)
                    delay = self.retry_delay * (2**attempt)
                logger.debug(f"Waiting {delay:.1f}s before retry...")
                time.sleep(delay)

        # All retries exhausted
        raise OpenMeteoError(f"Failed after {self.max_retries} attempts: {last_error}")

    def fetch_for_location(
        self,
        latitude: float,
        longitude: float,
        location_name: str,
        date: str,
        variables: List[str],
        mode: str,
        timezone: str = "UTC",
        forecast_time: str = "06:00",
        horizon_hours: int = 18,
    ) -> pd.DataFrame:
        """
        Fetch data for a single location and date with windmojo conventions.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            location_name: Human-readable location identifier (e.g., "center", "n_mid_c")
            date: Date string (YYYY-MM-DD)
            variables: List of variables to fetch
            mode: 'historical' or 'forecast'
            timezone: Timezone string
            forecast_time: Forecast initialization time (HH:MM)
            horizon_hours: Forecast horizon in hours

        Returns:
            DataFrame with columns: time, location, forecast_made, [variables...]
        """
        # Choose API based on mode
        if mode == "historical":
            df = self.fetch_historical_forecast(
                latitude, longitude, date, date, variables, timezone
            )
        else:
            df = self.fetch_forecast(
                latitude, longitude, date, date, variables, timezone
            )

        # Add metadata columns
        df["location"] = location_name

        # Parse forecast initialization time
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            hour, minute = map(int, forecast_time.split(":"))
            forecast_made = date_obj.replace(hour=hour, minute=minute)
            df["forecast_made"] = forecast_made

        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse forecast_time '{forecast_time}': {e}")
            df["forecast_made"] = pd.NaT

        # Add derived metadata
        if "time" in df.columns:
            df["month"] = df["time"].dt.month
            df["forecast_date"] = date

            # Filter to forecast window if specified
            if forecast_time and horizon_hours:
                try:
                    forecast_start = pd.Timestamp(forecast_made, tz=df["time"].dt.tz)
                    forecast_end = forecast_start + pd.Timedelta(hours=horizon_hours)

                    df = df[
                        (df["time"] >= forecast_start) & (df["time"] < forecast_end)
                    ].copy()

                except Exception as e:
                    logger.warning(f"Failed to filter forecast window: {e}")

        return df

    def test_connection(self) -> bool:
        """
        Test API connectivity with a simple request.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple test: fetch one day of temperature for a known location
            df = self.fetch_forecast(
                latitude=46.0,
                longitude=9.0,
                start_date=datetime.now().strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
                variables=["temperature_2m"],
                timezone="UTC",
            )

            return len(df) > 0

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

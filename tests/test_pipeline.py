"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from windmojo.data.grid_downloader import (
    CENTER_PRESSURE_LEVELS,
    CENTER_VARIABLES,
    CIRCLE_VARIABLES,
)
from windmojo.features.pipeline import FeatureEngineeringPipeline
from windmojo.features.snow_features import MAX_HOURS_SINCE_EVENT
from windmojo.prediction.forecast import apply_forecast_window

pytestmark = pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")

LOCATIONS = ["center", "n", "ne", "e", "se", "s", "sw", "w", "nw"]

# (name fragment, mean, standard deviation); first match wins
TYPICAL_VALUES = [
    ("geopotential_height_1000hPa", 110, 30),
    ("geopotential_height_850hPa", 1500, 30),
    ("geopotential_height_700hPa", 3000, 30),
    ("geopotential_height_500hPa", 5600, 50),
    ("geopotential_height_250hPa", 10400, 80),
    ("pressure", 1013, 4),
    ("temperature", 5, 4),
    ("dewpoint", 0, 4),
    ("humidity", 70, 10),
    ("cloud", 50, 30),
    ("direction", 180, 90),
    ("gusts", 20, 6),
    ("wind_speed", 10, 4),
    ("radiation", 200, 150),
    ("irradiance", 200, 150),
    ("soil_moisture", 0.3, 0.05),
    ("visibility", 20000, 5000),
]
PRECIPITATION = ["precipitation", "rain", "showers", "snowfall"]


def synthetic_value(name, rng, n):
    if name == "snow_depth":
        return np.zeros(n)
    if name in PRECIPITATION:
        return rng.exponential(1.0, n) * (rng.random(n) < 0.1)
    for fragment, mean, std in TYPICAL_VALUES:
        if fragment in name:
            return np.abs(rng.normal(mean, std, n))
    return np.abs(rng.normal(1.0, 0.2, n))


def make_raw(start="2026-01-01", days=21, seed=0):
    """Raw data in the downloaded format: 9 points, hours 06-23 local."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=days * 24, freq="h")
    frames = []
    for location in LOCATIONS:
        variables = (
            CENTER_VARIABLES + CENTER_PRESSURE_LEVELS
            if location == "center"
            else CIRCLE_VARIABLES
        )
        frame = pd.DataFrame({"time": times, "location": location})
        for variable in variables:
            frame[variable] = synthetic_value(variable, rng, len(times))
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    return apply_forecast_window(raw, "06:00", 18)


def set_center(raw, variable, times, value):
    mask = (raw["location"] == "center") & raw["time"].isin(pd.to_datetime(times))
    raw.loc[mask, variable] = value


@pytest.fixture(scope="module")
def raw():
    return make_raw()


@pytest.fixture(scope="module")
def features(raw):
    return FeatureEngineeringPipeline().transform(raw.copy())


class TestNoOutlierCapping:
    def test_rare_extreme_value_is_kept(self, raw):
        spiked = raw.copy()
        set_center(spiked, "wind_speed_10m", ["2026-01-10 14:00"], 150.0)
        features = FeatureEngineeringPipeline().transform(spiked)
        value = features.loc[features["time"] == "2026-01-10 14:00", "wind_speed_10m_center"]
        assert value.item() == 150.0


class TestSurfaceCenterWinds:
    def test_center_winds_are_features(self, features):
        for column in ["wind_speed_10m_center", "wind_gusts_10m_center", "wind_speed_120m_center"]:
            assert column in features.columns


class TestSnowFeatures:
    def test_snow_depth_in_metres(self, raw):
        snowy = raw.copy()
        set_center(snowy, "snow_depth", ["2026-01-10 12:00"], 0.05)  # 5 cm
        set_center(snowy, "snow_depth", ["2026-01-10 13:00"], 0.005)  # 0.5 cm
        features = FeatureEngineeringPipeline().transform(snowy).set_index("time")
        assert features.loc["2026-01-10 12:00", "is_snow_on_ground"] == 1
        assert features.loc["2026-01-10 13:00", "is_snow_on_ground"] == 0

    def test_hours_since_are_capped(self, features):
        for column in ["hours_since_snowfall", "hours_since_rain"]:
            assert features[column].between(0, MAX_HOURS_SINCE_EVENT).all()

    def test_hours_since_count_real_hours(self, raw):
        wet = raw.copy()
        set_center(wet, "precipitation", wet["time"].unique(), 0.0)
        set_center(wet, "temperature_2m", wet["time"].unique(), 10.0)
        set_center(wet, "precipitation", ["2026-01-10 20:00"], 2.0)
        features = FeatureEngineeringPipeline().transform(wet).set_index("time")
        # 10 hours later, across the night hours missing in the raw data
        assert features.loc["2026-01-11 06:00", "hours_since_rain"] == 10


def dry_except_early_events(raw):
    """No precipitation except one snowfall and one rain event on day 2."""
    raw = raw.copy()
    for variable in PRECIPITATION:
        set_center(raw, variable, raw["time"].unique(), 0.0)
    set_center(raw, "snowfall", ["2026-01-02 10:00"], 1.0)
    set_center(raw, "precipitation", ["2026-01-02 10:00", "2026-01-02 12:00"], 1.0)
    set_center(raw, "temperature_2m", ["2026-01-02 12:00"], 5.0)
    return raw


class TestLiveParity:
    """A live forecast (7 days history) must see the same features as training."""

    @pytest.mark.parametrize("scenario", ["random", "long_dry_spell"])
    def test_forecast_days_match_training_features(self, raw, scenario):
        if scenario == "long_dry_spell":
            raw = dry_except_early_events(raw)
        features = FeatureEngineeringPipeline().transform(raw.copy())

        forecast_start = pd.Timestamp("2026-01-15")
        live_raw = raw[raw["time"] >= forecast_start - pd.Timedelta(days=7)]
        live = FeatureEngineeringPipeline().transform(live_raw.copy())

        def forecast_days(df):
            df = df[df["time"] >= forecast_start].reset_index(drop=True)
            return df[sorted(df.columns)]

        pd.testing.assert_frame_equal(forecast_days(live), forecast_days(features))

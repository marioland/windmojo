"""Tests for windmojo.units (wind speed units)."""

import numpy as np
import pytest

from windmojo.units import kmh_to_unit, measurement_unit, to_ms


class TestMeasurementUnit:
    """Tests for measurement_unit."""

    def test_default_is_ms(self, valid_config):
        assert measurement_unit(valid_config) == "m/s"

    def test_simplified_config(self, valid_config):
        config = {**valid_config, "target": {"column": "x", "unit": "km/h"}}
        assert measurement_unit(config) == "km/h"

    def test_legacy_config(self):
        assert measurement_unit({"measurement_unit": "kn"}) == "kn"

    def test_unknown_unit_raises(self, valid_config):
        config = {**valid_config, "target": {"unit": "knots"}}
        with pytest.raises(ValueError, match="knots"):
            measurement_unit(config)


class TestConversions:
    """Tests for to_ms and kmh_to_unit."""

    @pytest.mark.parametrize(
        "value, unit, expected",
        [(10, "m/s", 10), (36, "km/h", 10), (10, "kn", 5.1444), (10, "mph", 4.4704)],
    )
    def test_to_ms(self, value, unit, expected):
        assert to_ms(value, unit) == pytest.approx(expected, abs=1e-4)

    def test_kmh_to_ms(self):
        assert kmh_to_unit(36, "m/s") == pytest.approx(10)

    def test_kmh_to_kmh_is_identity(self):
        np.testing.assert_allclose(
            kmh_to_unit(np.array([5.0, 12.9]), "km/h"), [5, 12.9]
        )

    def test_nan_stays_nan(self):
        assert np.isnan(kmh_to_unit(np.nan, "kn"))

"""Tests for windmojo.data.grid module."""

from math import sqrt

import pytest

from windmojo.data.grid import (
    calculate_9_point_circle,
    calculate_all_grids,
    calculate_distance_km,
    calculate_offset_coordinates,
    get_variable_locations,
    validate_coordinates,
)

# Lake Como center point for tests
LAKE_COMO_LAT = 46.083333
LAKE_COMO_LON = 9.3


class TestCalculateOffsetCoordinates:
    """Tests for calculate_offset_coordinates."""

    def test_zero_offset_returns_center(self):
        lat, lon = calculate_offset_coordinates(46.0, 9.0, 0, 0)
        assert lat == 46.0
        assert lon == 9.0

    def test_north_offset_increases_latitude(self):
        lat, lon = calculate_offset_coordinates(46.0, 9.0, 15, 0)
        assert lat > 46.0
        assert lon == 9.0

    def test_south_offset_decreases_latitude(self):
        lat, lon = calculate_offset_coordinates(46.0, 9.0, -15, 0)
        assert lat < 46.0
        assert lon == 9.0

    def test_east_offset_increases_longitude(self):
        lat, lon = calculate_offset_coordinates(46.0, 9.0, 0, 15)
        assert lat == 46.0
        assert lon > 9.0

    def test_west_offset_decreases_longitude(self):
        lat, lon = calculate_offset_coordinates(46.0, 9.0, 0, -15)
        assert lat == 46.0
        assert lon < 9.0

    def test_15km_north_offset_approximately_correct(self):
        """15 km north should add approximately 0.135 degrees latitude."""
        lat, lon = calculate_offset_coordinates(46.0, 9.0, 15, 0)
        delta_lat = lat - 46.0
        assert abs(delta_lat - 15 / 111.0) < 0.001

    def test_longitude_offset_accounts_for_latitude(self):
        """Longitude offset should be larger at higher latitudes (smaller cos)."""
        _, lon_equator = calculate_offset_coordinates(0, 9.0, 0, 15)
        _, lon_mid_lat = calculate_offset_coordinates(46.0, 9.0, 0, 15)
        # At higher lat, same km offset = larger degree offset
        assert (lon_mid_lat - 9.0) > (lon_equator - 9.0)


class TestCalculateDistanceKm:
    """Tests for calculate_distance_km."""

    def test_same_point_returns_zero(self):
        dist = calculate_distance_km(46.0, 9.0, 46.0, 9.0)
        assert dist == 0.0

    def test_symmetric(self):
        d1 = calculate_distance_km(46.0, 9.0, 46.1, 9.1)
        d2 = calculate_distance_km(46.1, 9.1, 46.0, 9.0)
        assert abs(d1 - d2) < 0.01

    def test_known_distance_north(self):
        """1 degree latitude is approximately 111 km."""
        dist = calculate_distance_km(46.0, 9.0, 47.0, 9.0)
        assert abs(dist - 111.0) < 1.0

    def test_positive_distance(self):
        dist = calculate_distance_km(46.0, 9.0, 46.1, 9.1)
        assert dist > 0

    def test_roundtrip_with_offset(self):
        """Offset by 20km, then measure distance should be ~20km."""
        new_lat, new_lon = calculate_offset_coordinates(46.0, 9.0, 20, 0)
        dist = calculate_distance_km(46.0, 9.0, new_lat, new_lon)
        assert abs(dist - 20.0) < 0.5


class TestValidateCoordinates:
    """Tests for validate_coordinates."""

    def test_valid_coordinates(self):
        assert validate_coordinates(46.0, 9.0) is True

    def test_equator_prime_meridian(self):
        assert validate_coordinates(0.0, 0.0) is True

    def test_boundary_values(self):
        assert validate_coordinates(90.0, 180.0) is True
        assert validate_coordinates(-90.0, -180.0) is True

    def test_invalid_latitude_too_high(self):
        with pytest.raises(ValueError, match="Invalid latitude"):
            validate_coordinates(91.0, 9.0)

    def test_invalid_latitude_too_low(self):
        with pytest.raises(ValueError, match="Invalid latitude"):
            validate_coordinates(-91.0, 9.0)

    def test_invalid_longitude_too_high(self):
        with pytest.raises(ValueError, match="Invalid longitude"):
            validate_coordinates(46.0, 181.0)

    def test_invalid_longitude_too_low(self):
        with pytest.raises(ValueError, match="Invalid longitude"):
            validate_coordinates(46.0, -181.0)


class TestCalculate9PointCircle:
    """Tests for calculate_9_point_circle."""

    def test_returns_9_points(self):
        grid = calculate_9_point_circle(LAKE_COMO_LAT, LAKE_COMO_LON, 20)
        assert len(grid) == 9

    def test_contains_all_compass_directions(self):
        grid = calculate_9_point_circle(LAKE_COMO_LAT, LAKE_COMO_LON, 20)
        expected_keys = {"center", "n", "ne", "e", "se", "s", "sw", "w", "nw"}
        assert set(grid.keys()) == expected_keys

    def test_center_is_input_coordinates(self):
        grid = calculate_9_point_circle(LAKE_COMO_LAT, LAKE_COMO_LON, 20)
        assert grid["center"] == (LAKE_COMO_LAT, LAKE_COMO_LON)

    def test_north_is_north_of_center(self):
        grid = calculate_9_point_circle(LAKE_COMO_LAT, LAKE_COMO_LON, 20)
        assert grid["n"][0] > grid["center"][0]  # higher latitude

    def test_south_is_south_of_center(self):
        grid = calculate_9_point_circle(LAKE_COMO_LAT, LAKE_COMO_LON, 20)
        assert grid["s"][0] < grid["center"][0]  # lower latitude

    def test_east_is_east_of_center(self):
        grid = calculate_9_point_circle(LAKE_COMO_LAT, LAKE_COMO_LON, 20)
        assert grid["e"][1] > grid["center"][1]  # higher longitude

    def test_west_is_west_of_center(self):
        grid = calculate_9_point_circle(LAKE_COMO_LAT, LAKE_COMO_LON, 20)
        assert grid["w"][1] < grid["center"][1]  # lower longitude

    def test_perimeter_points_equidistant(self):
        """All 8 perimeter points should be ~20km from center."""
        grid = calculate_9_point_circle(LAKE_COMO_LAT, LAKE_COMO_LON, 20)
        for direction in ["n", "ne", "e", "se", "s", "sw", "w", "nw"]:
            lat, lon = grid[direction]
            dist = calculate_distance_km(LAKE_COMO_LAT, LAKE_COMO_LON, lat, lon)
            assert (
                abs(dist - 20.0) < 1.0
            ), f"{direction}: expected ~20km, got {dist:.1f}km"

    def test_north_south_symmetry(self):
        grid = calculate_9_point_circle(LAKE_COMO_LAT, LAKE_COMO_LON, 20)
        n_offset = grid["n"][0] - LAKE_COMO_LAT
        s_offset = LAKE_COMO_LAT - grid["s"][0]
        assert abs(n_offset - s_offset) < 0.001


class TestCalculateAllGrids:
    """Tests for calculate_all_grids."""

    def test_returns_grid_and_center_keys(self):
        grids = calculate_all_grids(LAKE_COMO_LAT, LAKE_COMO_LON)
        assert "grid" in grids
        assert "center" in grids

    def test_grid_has_9_points(self):
        grids = calculate_all_grids(LAKE_COMO_LAT, LAKE_COMO_LON)
        assert len(grids["grid"]) == 9

    def test_center_has_1_point(self):
        grids = calculate_all_grids(LAKE_COMO_LAT, LAKE_COMO_LON)
        assert len(grids["center"]) == 1
        assert "center" in grids["center"]

    def test_default_radius_is_20(self):
        grids = calculate_all_grids(LAKE_COMO_LAT, LAKE_COMO_LON)
        lat, lon = grids["grid"]["n"]
        dist = calculate_distance_km(LAKE_COMO_LAT, LAKE_COMO_LON, lat, lon)
        assert abs(dist - 20.0) < 1.0

    def test_custom_radius(self):
        grids = calculate_all_grids(LAKE_COMO_LAT, LAKE_COMO_LON, radius_km=50)
        lat, lon = grids["grid"]["n"]
        dist = calculate_distance_km(LAKE_COMO_LAT, LAKE_COMO_LON, lat, lon)
        assert abs(dist - 50.0) < 2.0


class TestGetVariableLocations:
    """Tests for get_variable_locations."""

    def setup_method(self):
        self.grids = calculate_all_grids(LAKE_COMO_LAT, LAKE_COMO_LON)

    def test_grid_variable_returns_9_points(self):
        locs = get_variable_locations("temperature_2m", self.grids)
        assert len(locs) == 9

    def test_pressure_is_grid_variable(self):
        locs = get_variable_locations("pressure_msl", self.grids)
        assert len(locs) == 9

    def test_center_only_variable_returns_1_point(self):
        locs = get_variable_locations("wind_speed_100m", self.grids)
        assert len(locs) == 1

    def test_unknown_variable_defaults_to_center(self):
        locs = get_variable_locations("some_unknown_var", self.grids)
        assert len(locs) == 1
        assert "center" in locs

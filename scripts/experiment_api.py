#!/usr/bin/env python3
"""
Experimental API Testing Script

Quick script to test Open-Meteo API calls without full implementation.
Use this to validate API availability, response format, and rate limits.

Usage:
    uv run python scripts/experiment_api.py
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
from math import cos, radians


def calculate_grid_point(
    center_lat: float,
    center_lon: float,
    offset_lat_km: float,
    offset_lon_km: float
) -> tuple[float, float]:
    """
    Calculate GPS coordinates for a grid point.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        offset_lat_km: North/south offset in km (positive = north)
        offset_lon_km: East/west offset in km (positive = east)

    Returns:
        (latitude, longitude) tuple
    """
    # 1 degree latitude ≈ 111 km
    delta_lat = offset_lat_km / 111.0

    # 1 degree longitude ≈ 111 km * cos(latitude)
    delta_lon = offset_lon_km / (111.0 * cos(radians(center_lat)))

    return (center_lat + delta_lat, center_lon + delta_lon)


def test_api_single_point():
    """Test 1: Fetch data for single location and date."""
    print("\n" + "=" * 60)
    print("TEST 1: Single Point Fetch")
    print("=" * 60)

    # Lake Como, Dervio coordinates
    lat = 46.083333
    lon = 9.3

    # Test with yesterday's date (historical data)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': yesterday,
        'end_date': yesterday,
        'hourly': 'surface_pressure,cloud_cover,temperature_2m',
        'timezone': 'Europe/Rome',
        'models': 'best_match'
    }

    print(f"\nFetching data for:")
    print(f"  Location: {lat}, {lon} (Dervio, Lake Como)")
    print(f"  Date: {yesterday}")
    print(f"  Variables: surface_pressure, cloud_cover, temperature_2m")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        print(f"\n✓ API call successful!")
        print(f"  Response size: {len(response.content):,} bytes")
        print(f"  Data points returned: {len(data.get('hourly', {}).get('time', []))}")

        # Parse into DataFrame
        if 'hourly' in data:
            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'])

            print(f"\n  DataFrame shape: {df.shape}")
            print(f"\n  First 3 hours:")
            print(df.head(3).to_string())

            print(f"\n  Summary statistics:")
            print(df.describe().to_string())

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n✗ API call failed: {e}")
        return False


def test_api_all_variables():
    """Test 2: Fetch all windmojo variables for single location."""
    print("\n" + "=" * 60)
    print("TEST 2: All Variables Fetch")
    print("=" * 60)

    lat = 46.083333
    lon = 9.3
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    # All windmojo variables
    variables = [
        'surface_pressure',
        'cloud_cover',
        'temperature_2m',
        'soil_temperature_0cm',
        'shortwave_radiation',
        'boundary_layer_height',
        'relative_humidity_2m',
        'precipitation',
        'wind_speed_10m',
        'wind_direction_10m',
        'wind_gusts_10m',
        'wind_speed_100m',
        'wind_speed_850hPa',
        'wind_speed_250hPa'
    ]

    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': yesterday,
        'end_date': yesterday,
        'hourly': ','.join(variables),
        'timezone': 'Europe/Rome',
        'models': 'best_match'
    }

    print(f"\nFetching {len(variables)} variables for {yesterday}...")
    print(f"Variables: {', '.join(variables)}")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        print(f"\n✓ API call successful!")
        print(f"  Response size: {len(response.content):,} bytes")

        if 'hourly' in data:
            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'])

            print(f"\n  DataFrame shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")

            # Check which variables are available
            print(f"\n  Variable availability:")
            for var in variables:
                if var in df.columns:
                    missing = df[var].isna().sum()
                    print(f"    ✓ {var:30s} - {len(df) - missing}/{len(df)} values")
                else:
                    print(f"    ✗ {var:30s} - NOT AVAILABLE")

            print(f"\n  Sample data (first hour):")
            print(df.iloc[0].to_string())

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n✗ API call failed: {e}")
        return False


def test_grid_calculation():
    """Test 3: Calculate and display grid points."""
    print("\n" + "=" * 60)
    print("TEST 3: Grid Calculation")
    print("=" * 60)

    center_lat = 46.083333
    center_lon = 9.3

    print(f"\nCenter point: {center_lat}, {center_lon} (Dervio)")

    # Test 3×3 grid at 15 km
    print(f"\n3×3 Near Grid (15 km distance):")
    print(f"{'Direction':<15} {'Latitude':>12} {'Longitude':>12} {'Distance (km)':>15}")
    print("-" * 60)

    directions_3x3 = [
        ('NW', -15, -15),
        ('N', -15, 0),
        ('NE', -15, 15),
        ('W', 0, -15),
        ('CENTER', 0, 0),
        ('E', 0, 15),
        ('SW', 15, -15),
        ('S', 15, 0),
        ('SE', 15, 15),
    ]

    for name, lat_offset, lon_offset in directions_3x3:
        lat, lon = calculate_grid_point(center_lat, center_lon, lat_offset, lon_offset)
        # Approximate distance
        dist = (lat_offset**2 + lon_offset**2)**0.5
        print(f"{name:<15} {lat:>12.6f} {lon:>12.6f} {dist:>15.1f}")

    # Test 17-point diamond grid
    print(f"\n17-Point Diamond Grid (near=15km, mid=50km, far=150km):")
    print(f"{'Position':<20} {'Latitude':>12} {'Longitude':>12} {'Distance (km)':>15}")
    print("-" * 60)

    diamond_points = [
        # Center row (5 points)
        ('center', 0, 0),
        ('nw', -15, -15),
        ('ne', -15, 15),
        ('w', 0, -15),
        ('e', 0, 15),
        # Mid rows (6 points)
        ('n_mid_w', -50, -15),
        ('n_mid_c', -50, 0),
        ('n_mid_e', -50, 15),
        ('s_mid_w', 50, -15),
        ('s_mid_c', 50, 0),
        ('s_mid_e', 50, 15),
        # Far rows (6 points)
        ('n_far_w', -150, -50),
        ('n_far_c', -150, 0),
        ('n_far_e', -150, 50),
        ('s_far_w', 150, -50),
        ('s_far_c', 150, 0),
        ('s_far_e', 150, 50),
    ]

    for name, lat_offset, lon_offset in diamond_points:
        lat, lon = calculate_grid_point(center_lat, center_lon, lat_offset, lon_offset)
        dist = (lat_offset**2 + lon_offset**2)**0.5
        print(f"{name:<20} {lat:>12.6f} {lon:>12.6f} {dist:>15.1f}")


def test_multiple_locations():
    """Test 4: Fetch data for multiple grid points."""
    print("\n" + "=" * 60)
    print("TEST 4: Multiple Locations Fetch")
    print("=" * 60)

    center_lat = 46.083333
    center_lon = 9.3
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    # Test 3 points: center, north, south
    locations = {
        'center': (center_lat, center_lon),
        'north_15km': calculate_grid_point(center_lat, center_lon, -15, 0),
        'south_15km': calculate_grid_point(center_lat, center_lon, 15, 0),
    }

    print(f"\nFetching surface pressure for 3 locations:")
    for name, (lat, lon) in locations.items():
        print(f"  {name:15s}: {lat:.6f}, {lon:.6f}")

    results = {}

    for name, (lat, lon) in locations.items():
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': yesterday,
            'end_date': yesterday,
            'hourly': 'surface_pressure',
            'timezone': 'Europe/Rome',
            'models': 'best_match'
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'hourly' in data:
                df = pd.DataFrame(data['hourly'])
                df['time'] = pd.to_datetime(df['time'])
                df['location'] = name
                results[name] = df

                print(f"  ✓ {name:15s}: {len(df)} hours fetched")
            else:
                print(f"  ✗ {name:15s}: No data returned")

        except requests.exceptions.RequestException as e:
            print(f"  ✗ {name:15s}: {e}")

    # Combine and show pressure gradient
    if len(results) == 3:
        combined = pd.concat(results.values(), ignore_index=True)

        print(f"\n  Combined DataFrame shape: {combined.shape}")
        print(f"\n  Sample: First hour, all locations:")

        first_hour = combined[combined['time'] == combined['time'].min()]
        print(first_hour.to_string())

        # Calculate simple N-S gradient
        center_pressure = results['center']['surface_pressure'].iloc[0]
        north_pressure = results['north_15km']['surface_pressure'].iloc[0]
        south_pressure = results['south_15km']['surface_pressure'].iloc[0]

        ns_gradient = north_pressure - south_pressure

        print(f"\n  Pressure Gradient Analysis (first hour):")
        print(f"    North:  {north_pressure:.2f} hPa")
        print(f"    Center: {center_pressure:.2f} hPa")
        print(f"    South:  {south_pressure:.2f} hPa")
        print(f"    N-S gradient: {ns_gradient:.2f} hPa")

        return True

    return False


def test_date_range():
    """Test 5: Fetch multiple days of data."""
    print("\n" + "=" * 60)
    print("TEST 5: Date Range Fetch")
    print("=" * 60)

    lat = 46.083333
    lon = 9.3

    # Last 7 days
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=6)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_str,
        'end_date': end_str,
        'hourly': 'temperature_2m,wind_speed_10m',
        'timezone': 'Europe/Rome',
        'models': 'best_match'
    }

    print(f"\nFetching 7 days of data:")
    print(f"  Date range: {start_str} to {end_str}")
    print(f"  Variables: temperature_2m, wind_speed_10m")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        print(f"\n✓ API call successful!")
        print(f"  Response size: {len(response.content):,} bytes")

        if 'hourly' in data:
            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'])

            print(f"\n  DataFrame shape: {df.shape}")
            print(f"  Expected hours: {7 * 24} = 168")
            print(f"  Actual hours: {len(df)}")

            # Group by date
            df['date'] = df['time'].dt.date
            daily_stats = df.groupby('date').agg({
                'temperature_2m': ['mean', 'min', 'max'],
                'wind_speed_10m': ['mean', 'max']
            })

            print(f"\n  Daily summary:")
            print(daily_stats.to_string())

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n✗ API call failed: {e}")
        return False


def test_forecast_vs_historical():
    """Test 6: Compare forecast API vs historical API."""
    print("\n" + "=" * 60)
    print("TEST 6: Forecast vs Historical API")
    print("=" * 60)

    lat = 46.083333
    lon = 9.3

    # For forecast: tomorrow
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    # For historical: yesterday
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    variables = 'temperature_2m,wind_speed_10m,cloud_cover'

    # Test historical API
    print(f"\nTest A: Historical Forecast API")
    print(f"  Date: {yesterday} (yesterday)")

    hist_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    try:
        response = requests.get(hist_url, params={
            'latitude': lat,
            'longitude': lon,
            'start_date': yesterday,
            'end_date': yesterday,
            'hourly': variables,
            'timezone': 'Europe/Rome',
            'models': 'best_match'
        }, timeout=30)
        response.raise_for_status()

        data = response.json()
        print(f"  ✓ Historical API works")
        print(f"  Response size: {len(response.content):,} bytes")
        print(f"  Data points: {len(data.get('hourly', {}).get('time', []))}")

    except Exception as e:
        print(f"  ✗ Historical API failed: {e}")

    # Test forecast API
    print(f"\nTest B: Current Forecast API")
    print(f"  Date: {tomorrow} (tomorrow)")

    forecast_url = "https://api.open-meteo.com/v1/forecast"

    try:
        response = requests.get(forecast_url, params={
            'latitude': lat,
            'longitude': lon,
            'start_date': tomorrow,
            'end_date': tomorrow,
            'hourly': variables,
            'timezone': 'Europe/Rome',
        }, timeout=30)
        response.raise_for_status()

        data = response.json()
        print(f"  ✓ Forecast API works")
        print(f"  Response size: {len(response.content):,} bytes")
        print(f"  Data points: {len(data.get('hourly', {}).get('time', []))}")

    except Exception as e:
        print(f"  ✗ Forecast API failed: {e}")

    print(f"\n  Conclusion:")
    print(f"    - Use Historical API for training data (past dates)")
    print(f"    - Use Forecast API for production predictions (future dates)")


def main():
    """Run all API tests."""
    print("\n" + "=" * 60)
    print("windmojo Open-Meteo API Experimental Tests")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Purpose: Validate API availability and response format")

    tests = [
        ("Single Point Fetch", test_api_single_point),
        ("All Variables Fetch", test_api_all_variables),
        ("Grid Calculation", test_grid_calculation),
        ("Multiple Locations Fetch", test_multiple_locations),
        ("Date Range Fetch", test_date_range),
        ("Forecast vs Historical", test_forecast_vs_historical),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result if result is not None else True))
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! API is ready for use.")
        print("\nNext steps:")
        print("  1. Review grid coordinates in TEST 3")
        print("  2. Verify all variables available in TEST 2")
        print("  3. Proceed with full implementation")
    else:
        print("\n✗ Some tests failed. Review errors above.")


if __name__ == "__main__":
    main()

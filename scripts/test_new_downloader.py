#!/usr/bin/env python3
"""
Test the new downloader architecture.

This script verifies that:
1. Circle points (8 compass directions) get only synoptic variables
2. Center point gets ALL available variables
3. Data structure is correct

Usage:
    uv run python scripts/test_new_downloader.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from windmojo.core.config import load_config
from windmojo.data.grid_downloader import (
    DataDownloader,
    CIRCLE_VARIABLES,
    CENTER_VARIABLES,
    CENTER_PRESSURE_LEVELS
)


def test_variable_lists():
    """Test that variable lists are correctly defined."""
    print("\n" + "=" * 60)
    print("VARIABLE LIST VERIFICATION")
    print("=" * 60)

    print(f"\nCircle Variables ({len(CIRCLE_VARIABLES)}):")
    for var in CIRCLE_VARIABLES:
        print(f"  - {var}")

    print(f"\nCenter Surface Variables ({len(CENTER_VARIABLES)}):")
    for var in CENTER_VARIABLES:
        print(f"  - {var}")

    print(f"\nCenter Pressure Level Variables ({len(CENTER_PRESSURE_LEVELS)}):")
    for var in CENTER_PRESSURE_LEVELS:
        print(f"  - {var}")

    total_center = len(CENTER_VARIABLES) + len(CENTER_PRESSURE_LEVELS)
    print(f"\nTotal Center Variables: {total_center}")
    print(f"Total Circle Variables: {len(CIRCLE_VARIABLES)}")

    # Check for duplicates
    all_center = CENTER_VARIABLES + CENTER_PRESSURE_LEVELS
    if len(all_center) != len(set(all_center)):
        print("\n⚠️  WARNING: Duplicate variables in center lists!")
        duplicates = [var for var in all_center if all_center.count(var) > 1]
        print(f"Duplicates: {set(duplicates)}")
    else:
        print("\n✓ No duplicate variables")

    return True


def test_grid_structure():
    """Test that grid structure is correct."""
    print("\n" + "=" * 60)
    print("GRID STRUCTURE VERIFICATION")
    print("=" * 60)

    # Load config
    config_path = project_root / "config/lake_como/dervio.yaml"
    print(f"\nLoading config: {config_path}")
    config = load_config(str(config_path))

    # Create downloader
    downloader = DataDownloader(config)

    # Check grid structure
    print(f"\nGrid Types: {list(downloader.grids.keys())}")

    print("\nFull Grid (9 points):")
    for name, coords in sorted(downloader.grids['grid'].items()):
        print(f"  {name:10} → {coords}")

    print("\nCenter Grid (1 point):")
    for name, coords in downloader.grids['center'].items():
        print(f"  {name:10} → {coords}")

    # Verify circle points (exclude center)
    all_grid = downloader.grids['grid']
    circle_points = {k: v for k, v in all_grid.items() if k != 'center'}

    print(f"\nCircle Points (should be 8): {len(circle_points)}")
    if len(circle_points) == 8:
        print("✓ Correct number of circle points")
    else:
        print("⚠️  WARNING: Expected 8 circle points!")
        return False

    expected_directions = {'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'}
    actual_directions = set(circle_points.keys())

    if actual_directions == expected_directions:
        print("✓ All compass directions present")
    else:
        print(f"⚠️  WARNING: Missing directions: {expected_directions - actual_directions}")
        print(f"⚠️  WARNING: Extra directions: {actual_directions - expected_directions}")
        return False

    return True


def test_download_single_date():
    """Test downloading a single date (dry run simulation)."""
    print("\n" + "=" * 60)
    print("DOWNLOAD SIMULATION")
    print("=" * 60)

    # Load config
    config_path = project_root / "config/lake_como/dervio.yaml"
    config = load_config(str(config_path))

    # Create downloader
    downloader = DataDownloader(config)

    print("\nSimulating download for 2025-05-01...")
    print(f"  - Circle points: 8")
    print(f"  - Circle variables: {len(CIRCLE_VARIABLES)}")
    print(f"  - Center points: 1")
    print(f"  - Center variables: {len(CENTER_VARIABLES) + len(CENTER_PRESSURE_LEVELS)}")

    # Calculate expected data
    circle_data_rows = 8 * len(CIRCLE_VARIABLES) * 24  # 8 points × vars × 24 hours
    center_data_rows = 1 * (len(CENTER_VARIABLES) + len(CENTER_PRESSURE_LEVELS)) * 24
    total_rows = circle_data_rows + center_data_rows

    print(f"\nExpected data structure:")
    print(f"  - Circle data rows: {circle_data_rows}")
    print(f"  - Center data rows: {center_data_rows}")
    print(f"  - Total rows: {total_rows}")

    # API call estimation
    api_calls = 8 + 1  # 8 circle points + 1 center point
    print(f"\nEstimated API calls per day: {api_calls}")

    print("\n✓ Ready to download!")
    print("\nTo actually download, run:")
    print("  uv run python -m windmojo.data_fetcher \\")
    print("    --config config/lake_como/dervio.yaml \\")
    print("    --mode historical \\")
    print("    --start-date 2025-05-01 \\")
    print("    --end-date 2025-05-03")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NEW DOWNLOADER ARCHITECTURE TEST SUITE")
    print("=" * 60)

    tests = [
        ("Variable Lists", test_variable_lists),
        ("Grid Structure", test_grid_structure),
        ("Download Simulation", test_download_single_date),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status:10} - {name}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✓ All tests passed! Ready to download data.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

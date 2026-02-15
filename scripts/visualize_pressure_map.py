"""
Visualize Mean Sea Level Pressure Map

Creates a contour map of mean sea level pressure from synoptic data
with geographic context (country borders, cities).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata

# Try to import cartopy, fall back to basic matplotlib if not available
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed. Will create basic plot without country borders.")
    print("Install with: pip install cartopy")


def load_synoptic_data(date_str: str) -> pd.DataFrame:
    """Load synoptic grid data for a specific date."""
    data_dir = Path("data/lake_como/dervio/raw/synoptic")
    filename = f"synoptic_grid_{date_str}.parquet"
    filepath = data_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading: {filepath}")
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df)} grid points")
    print(f"Columns: {df.columns.tolist()}")

    return df


def create_pressure_map_cartopy(df: pd.DataFrame, date_str: str):
    """Create pressure map with cartopy (country borders, better projection)."""

    # Filter to 12:00 UTC if multiple times present
    if 'time' in df.columns:
        # Get unique times
        unique_times = df['time'].unique()
        print(f"Times in data: {unique_times}")
        # Filter to noon if available
        noon_mask = df['time'].dt.hour == 12
        if noon_mask.sum() > 0:
            df = df[noon_mask].copy()
            print(f"Filtered to 12:00 UTC: {len(df)} points")

    # Extract data
    lats = df['lat'].values
    lons = df['lon'].values
    pressure = df['pressure_msl'].values  # Already in hPa

    # Create regular grid for interpolation
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # Create meshgrid for plotting (regular grid)
    lat_grid = np.linspace(lat_min, lat_max, 100)
    lon_grid = np.linspace(lon_min, lon_max, 150)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Interpolate irregular data to regular grid
    print(f"Interpolating {len(pressure)} irregular points to regular grid...")
    points = np.column_stack((lons, lats))
    pressure_grid = griddata(points, pressure, (lon_mesh, lat_mesh), method='linear')

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set extent to match data
    ax.set_extent([-30, 35, 35, 72], crs=ccrs.PlateCarree())

    # Add country borders
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Plot pressure contours
    levels = np.arange(950, 1050, 4)  # 4 hPa intervals
    contours = ax.contour(lon_mesh, lat_mesh, pressure_grid,
                          levels=levels, colors='black', linewidths=1.0,
                          transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=9, fmt='%d')

    # City locations
    cities = {
        'Lisbon': (38.7223, -9.1393),
        'London': (51.5074, -0.1278),
        'Paris': (48.8566, 2.3522),
        'Bern': (46.9480, 7.4474),
        'Vienna': (48.2082, 16.3738),
        'Rome': (41.9028, 12.4964),
        'Dervio': (46.083333, 9.3)  # Lake Como target location
    }

    # Plot cities
    for city, (lat, lon) in cities.items():
        if city == 'Dervio':
            # Highlight target location with star
            ax.plot(lon, lat, marker='*', markersize=20, color='red',
                   transform=ccrs.PlateCarree(), zorder=5)
            ax.text(lon + 0.5, lat + 0.5, city, fontsize=11, fontweight='bold',
                   color='red', transform=ccrs.PlateCarree(), zorder=5)
        else:
            # Regular cities with dots
            ax.plot(lon, lat, marker='o', markersize=6, color='darkred',
                   transform=ccrs.PlateCarree(), zorder=4)
            ax.text(lon + 0.5, lat + 0.5, city, fontsize=9,
                   transform=ccrs.PlateCarree(), zorder=4)

    # Title and labels
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    ax.set_title(f'Mean Sea Level Pressure - {date_formatted} 12:00 UTC',
                fontsize=14, fontweight='bold', pad=20)

    # Add pressure statistics
    stats_text = f"Min: {pressure.min():.1f} hPa  |  Max: {pressure.max():.1f} hPa  |  Mean: {pressure.mean():.1f} hPa"
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    return fig


def create_pressure_map_basic(df: pd.DataFrame, date_str: str):
    """Create basic pressure map without cartopy."""

    # Filter to 12:00 UTC if multiple times present
    if 'time' in df.columns:
        # Get unique times
        unique_times = df['time'].unique()
        print(f"Times in data: {unique_times}")
        # Filter to noon if available
        noon_mask = df['time'].dt.hour == 12
        if noon_mask.sum() > 0:
            df = df[noon_mask].copy()
            print(f"Filtered to 12:00 UTC: {len(df)} points")

    # Extract data
    lats = df['lat'].values
    lons = df['lon'].values
    pressure = df['pressure_msl'].values  # Already in hPa

    # Create regular grid for interpolation
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # Create meshgrid for plotting (regular grid)
    lat_grid = np.linspace(lat_min, lat_max, 100)
    lon_grid = np.linspace(lon_min, lon_max, 150)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Interpolate irregular data to regular grid
    print(f"Interpolating {len(pressure)} irregular points to regular grid...")
    points = np.column_stack((lons, lats))
    pressure_grid = griddata(points, pressure, (lon_mesh, lat_mesh), method='linear')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot pressure contours
    levels = np.arange(950, 1050, 4)  # 4 hPa intervals
    contours = ax.contour(lon_mesh, lat_mesh, pressure_grid,
                          levels=levels, colors='black', linewidths=1.0)
    ax.clabel(contours, inline=True, fontsize=9, fmt='%d')

    # City locations
    cities = {
        'Lisbon': (38.7223, -9.1393),
        'London': (51.5074, -0.1278),
        'Paris': (48.8566, 2.3522),
        'Bern': (46.9480, 7.4474),
        'Vienna': (48.2082, 16.3738),
        'Rome': (41.9028, 12.4964),
        'Dervio': (46.083333, 9.3)
    }

    # Plot cities
    for city, (lat, lon) in cities.items():
        if city == 'Dervio':
            ax.plot(lon, lat, marker='*', markersize=20, color='red', zorder=5)
            ax.text(lon + 0.5, lat + 0.5, city, fontsize=11, fontweight='bold',
                   color='red', zorder=5)
        else:
            ax.plot(lon, lat, marker='o', markersize=6, color='darkred', zorder=4)
            ax.text(lon + 0.5, lat + 0.5, city, fontsize=9, zorder=4)

    # Labels and formatting
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_xlim(-30, 35)
    ax.set_ylim(35, 72)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    # Title
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    ax.set_title(f'Mean Sea Level Pressure - {date_formatted} 12:00 UTC',
                fontsize=14, fontweight='bold', pad=20)

    # Add pressure statistics
    stats_text = f"Min: {pressure.min():.1f} hPa  |  Max: {pressure.max():.1f} hPa  |  Mean: {pressure.mean():.1f} hPa"
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    return fig


def main():
    """Main execution."""

    # Configuration
    date_str = "20250501"  # May 1, 2025

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\n" + "="*60)
    print("PRESSURE MAP VISUALIZATION")
    print("="*60)
    df = load_synoptic_data(date_str)

    # Check for required columns
    if 'pressure_msl' not in df.columns:
        print("\nERROR: 'pressure_msl' column not found in data!")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Create map
    print(f"\nCreating pressure map...")
    if HAS_CARTOPY:
        print("Using cartopy for geographic features...")
        fig = create_pressure_map_cartopy(df, date_str)
    else:
        print("Using basic matplotlib (no country borders)...")
        fig = create_pressure_map_basic(df, date_str)

    # Save figure
    output_file = output_dir / f"pressure_map_{date_str}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    print(f"Resolution: 300 DPI")

    # Show statistics
    pressure = df['pressure_msl'].values / 100
    print(f"\nPressure Statistics:")
    print(f"  Min: {pressure.min():.1f} hPa")
    print(f"  Max: {pressure.max():.1f} hPa")
    print(f"  Mean: {pressure.mean():.1f} hPa")
    print(f"  Std: {pressure.std():.1f} hPa")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()

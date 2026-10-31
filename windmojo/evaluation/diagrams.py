"""Create validation diagrams comparing model predictions with measurements and GFS baseline."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import datetime
import logging
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from windmojo.units import KMH_TO_MS

logger = logging.getLogger(__name__)


def create_validation_diagrams(
    predictions: pd.DataFrame,
    measurements: pd.DataFrame,
    features: pd.DataFrame,
    test_dates: List[datetime.date],
    target_col: str,
    output_dir: Path,
    spot_name: str = "Wind Spot",
) -> Dict[str, Path]:
    """
    Create validation diagrams comparing model predictions with measurements and GFS baseline.

    Args:
        predictions: DataFrame with columns ['time', 'prediction']
        measurements: DataFrame with target wind measurements
        features: DataFrame with GFS baseline (effective_wind_10m, km/h)
        test_dates: List of test dates to create diagrams for
        target_col: Name of target column in measurements
        output_dir: Directory to save diagrams
        spot_name: Name of the spot for diagram titles

    Returns:
        Dictionary mapping month names to diagram file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare merged data
    logger.info("Merging predictions, measurements, and GFS baseline...")

    # Merge predictions with measurements
    merged = predictions.merge(
        measurements[["time", target_col]], on="time", how="inner"
    )

    # Merge with features to get GFS baseline
    merged = merged.merge(
        features[["time", "effective_wind_10m"]], on="time", how="inner"
    )
    merged["gfs_baseline"] = merged["effective_wind_10m"] * KMH_TO_MS

    merged["date"] = merged["time"].dt.date

    # Filter to test dates only
    merged = merged[merged["date"].isin(test_dates)]

    logger.info(f"Total samples for diagrams: {len(merged)}")

    # Group by month
    merged["month"] = merged["time"].dt.to_period("M")
    months = merged.groupby("month")

    diagram_files = {}

    for month_period, month_data in months:
        month_name = month_period.strftime("%B %Y")
        month_key = month_period.strftime("%Y-%m")
        month_dates = sorted(month_data["date"].unique())

        logger.info(f"Creating diagram for {month_name} ({len(month_dates)} days)...")

        # Sort by time
        month_data = month_data.sort_values("time")

        # Extract arrays
        times = month_data["time"].values
        measurements_arr = month_data[target_col].values
        predictions_arr = month_data["prediction"].values
        gfs_arr = month_data["gfs_baseline"].values

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))

        # Plot measurements - thick black line
        ax.plot(
            times,
            measurements_arr,
            color="black",
            linewidth=2.5,
            marker="o",
            markersize=8,
            label="Actual Measurements",
            zorder=10,
            alpha=0.9,
        )

        # Plot our model - blue line
        ax.plot(
            times,
            predictions_arr,
            color="#1f77b4",
            linewidth=2,
            marker="s",
            markersize=6,
            label="Our Model (Spatial Heavy Unified)",
            zorder=9,
            alpha=0.8,
            linestyle="--",
        )

        # Plot GFS - orange line
        ax.plot(
            times,
            gfs_arr,
            color="#ff7f0e",
            linewidth=2,
            marker="^",
            markersize=6,
            label="GFS Baseline",
            zorder=8,
            alpha=0.8,
            linestyle=":",
        )

        # Day separators at midnight
        for i in range(1, len(month_dates)):
            midnight = pd.Timestamp(
                datetime.datetime.combine(month_dates[i], datetime.time(0, 0))
            )
            ax.axvline(
                x=midnight,
                color="#333333",
                linewidth=3,
                linestyle="-",
                alpha=0.7,
                zorder=5,
            )

        # Calculate metrics
        our_mae = np.mean(np.abs(measurements_arr - predictions_arr))
        gfs_mae = np.mean(np.abs(measurements_arr - gfs_arr))
        improvement = ((gfs_mae - our_mae) / gfs_mae) * 100

        # Title
        date_range = f"{month_dates[0].strftime('%Y-%m-%d')} to {month_dates[-1].strftime('%Y-%m-%d')}"
        ax.set_title(
            f"{spot_name} - {month_name}\n"
            f"{date_range} | Our MAE: {our_mae:.2f} m/s | GFS MAE: {gfs_mae:.2f} m/s | "
            f"Improvement: {improvement:+.1f}% | {len(month_dates)} days | All hours (24h)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        # Format axes
        ax.set_xlabel("Date & Time", fontsize=12, fontweight="bold")
        ax.set_ylabel("Wind Speed (m/s)", fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Show every 6 hours
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(loc="best", fontsize=11, framealpha=0.95)

        # Y-axis limits
        all_values = np.concatenate([measurements_arr, predictions_arr, gfs_arr])
        ax.set_ylim(max(0, all_values.min() - 2), all_values.max() + 2)

        plt.tight_layout()

        # Save
        output_file = output_dir / f"validation_{month_key}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        diagram_files[month_name] = output_file
        logger.info(
            f"  Saved: {output_file.name} (MAE: {our_mae:.2f}, Improvement: {improvement:+.1f}%)"
        )

    # Overall statistics
    overall_our_mae = np.mean(np.abs(merged[target_col] - merged["prediction"]))
    overall_gfs_mae = np.mean(np.abs(merged[target_col] - merged["gfs_baseline"]))
    overall_improvement = ((overall_gfs_mae - overall_our_mae) / overall_gfs_mae) * 100

    logger.info(f"\nOverall Statistics:")
    logger.info(f"  Our Model MAE: {overall_our_mae:.2f} m/s")
    logger.info(f"  GFS MAE: {overall_gfs_mae:.2f} m/s")
    logger.info(f"  Improvement: {overall_improvement:+.1f}%")

    return diagram_files

"""
CLI command for creating validation diagrams.

Compare model predictions against GFS baseline and actual measurements.
"""

from pathlib import Path

import click
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import datetime
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from windmojo.core.config import get_test_dates, load_config
from windmojo.train.phase1_model1 import Model1Trainer


@click.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to training configuration YAML file",
)
@click.option(
    "--selection",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Path to feature selection JSON file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory for diagrams (default: data/{area}/{spot}/diagrams/validation)",
)
def diagram(config, selection, output):
    """
    Create validation diagrams comparing model predictions vs GFS vs measurements.

    Trains a unified model on non-test days and generates comparison diagrams
    for reserved test days configured in the YAML file.

    Example:
        windmojo diagram -c config/lake_como/dervio_training.yaml \\
                        -s data/lake_como/dervio/models/phase2_selections/spatial_heavy.json
    """
    # Load configuration
    config = load_config(config)
    area = config["area"]
    spot = config["spot"]

    # Determine output directory
    if output is None:
        output = Path(f"data/{area}/{spot}/diagrams/validation")
    else:
        output = Path(output)

    output.mkdir(parents=True, exist_ok=True)

    click.echo(
        f"\nCreating validation diagrams for {config['center'].get('name', spot)}..."
    )
    click.echo(f"Output directory: {output}")

    # Load feature selection
    with open(selection) as f:
        selection_data = json.load(f)

    selection_name = selection_data.get("name", "Unknown")
    selected_features = selection_data["features"]

    click.echo(f"Using selection: {selection_name} ({len(selected_features)} features)")

    # Load data
    click.echo("\n1. Loading measurements...")
    measurements_df = pd.read_csv(config["measurement_file"])
    measurements_df["time"] = pd.to_datetime(
        measurements_df["timestamp"], format="%Y-%m-%d_%H%M"
    )
    measurements_df["date"] = measurements_df["time"].dt.date

    click.echo("2. Loading features...")
    features_file = Path(f"data/{area}/{spot}/features/{spot}_features_phase1.parquet")
    features_df = pd.read_parquet(features_file)
    features_df["date"] = features_df["time"].dt.date

    target_column = config["target_column"]

    # Get available features
    available_features = set(features_df.columns)
    selected_features = [f for f in selected_features if f in available_features]

    # Add hour features for unified model
    hour_features = ["hour", "hour_sin", "hour_cos"]
    for hf in hour_features:
        if hf in available_features and hf not in selected_features:
            selected_features.append(hf)

    click.echo(f"   Using {len(selected_features)} features")

    # Merge measurements with features
    merged_df = features_df.merge(
        measurements_df[["time", target_column]], on="time", how="inner"
    ).dropna(subset=[target_column])

    click.echo(f"   Merged: {len(merged_df)} samples")

    # Get test dates from configuration
    all_dates = sorted(merged_df["date"].unique())
    test_dates = get_test_dates(config, all_dates)

    click.echo(
        f"\n3. Test dates strategy: {config.get('validation', {}).get('strategy', 'default')}"
    )
    click.echo(f"   Test dates: {len(test_dates)} days")

    # Show test dates grouped by month
    test_by_month = {}
    for date in test_dates:
        key = date.strftime("%Y-%m")
        if key not in test_by_month:
            test_by_month[key] = []
        test_by_month[key].append(date)

    for month, dates in sorted(test_by_month.items()):
        click.echo(f"   {month}: {dates}")

    # Split train and test
    train_mask = ~merged_df["date"].isin(test_dates)
    test_mask = merged_df["date"].isin(test_dates)

    train_df = merged_df[train_mask].copy()
    test_df = merged_df[test_mask].copy()

    click.echo(f"\n4. Training unified model on {len(train_df)} samples...")
    train_data = train_df[selected_features + [target_column]].copy()
    train_data = train_data.rename(columns={target_column: "target"})
    test_data_for_trainer = train_data.iloc[:10].copy()

    trainer = Model1Trainer()
    trainer.train(
        train_data=train_data,
        test_data=test_data_for_trainer,
        feature_columns=selected_features,
    )

    model = trainer.model

    click.echo("5. Generating predictions for test days...")
    X_test = test_df[selected_features]
    our_predictions = model.predict(X_test)

    # Add predictions to test dataframe
    test_df["our_predictions"] = our_predictions

    # Get data arrays
    times = test_df["time"].values
    dates = test_df["date"].values
    measurements = test_df[target_column].values
    our_preds = test_df["our_predictions"].values
    gfs_preds = test_df["effective_wind_10m"].values

    click.echo(f"   {len(test_df)} test samples")

    # Create diagrams for each month
    click.echo("\n6. Creating diagrams...")

    for month_key, month_dates in sorted(test_by_month.items()):
        month_name = datetime.datetime.strptime(month_key, "%Y-%m").strftime("%B %Y")
        month_mask = np.isin(dates, month_dates)

        if month_mask.sum() == 0:
            continue

        month_times = times[month_mask]
        month_measurements = measurements[month_mask]
        month_our = our_preds[month_mask]
        month_gfs = gfs_preds[month_mask]
        month_dates_array = dates[month_mask]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))

        # Plot measurements - thick black line
        ax.plot(
            month_times,
            month_measurements,
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
            month_times,
            month_our,
            color="#1f77b4",
            linewidth=2,
            marker="s",
            markersize=6,
            label=f"Our Model ({selection_name})",
            zorder=9,
            alpha=0.8,
            linestyle="--",
        )

        # Plot GFS - orange line
        ax.plot(
            month_times,
            month_gfs,
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
        unique_dates = sorted(set(month_dates_array))
        for i in range(1, len(unique_dates)):
            midnight = pd.Timestamp(
                datetime.datetime.combine(unique_dates[i], datetime.time(0, 0))
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
        our_mae = np.mean(np.abs(month_measurements - month_our))
        gfs_mae = np.mean(np.abs(month_measurements - month_gfs))
        improvement = ((gfs_mae - our_mae) / gfs_mae) * 100

        # Title
        date_range = f"{month_dates[0].strftime('%Y-%m-%d')} to {month_dates[-1].strftime('%Y-%m-%d')}"
        ax.set_title(
            f"{month_name} - Wind Speed Validation\n"
            f"{date_range} | Our MAE: {our_mae:.2f} m/s | GFS MAE: {gfs_mae:.2f} m/s | "
            f"Improvement: {improvement:+.1f}% | {len(unique_dates)} days",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        # Format
        ax.set_xlabel("Date & Time", fontsize=12, fontweight="bold")
        ax.set_ylabel("Wind Speed (m/s)", fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(loc="best", fontsize=11, framealpha=0.95)

        all_values = np.concatenate([month_measurements, month_our, month_gfs])
        ax.set_ylim(max(0, all_values.min() - 2), all_values.max() + 2)

        plt.tight_layout()

        # Save
        filename = f'validation_{month_key.replace("-", "_")}.png'
        output_file = output / filename
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        click.echo(f"   ✓ {filename}")

    # Overall statistics
    overall_our_mae = np.mean(np.abs(measurements - our_preds))
    overall_gfs_mae = np.mean(np.abs(measurements - gfs_preds))
    overall_improvement = ((overall_gfs_mae - overall_our_mae) / overall_gfs_mae) * 100

    click.echo(f"\n7. Overall Statistics:")
    click.echo(f"   Our Model MAE: {overall_our_mae:.2f} m/s")
    click.echo(f"   GFS MAE: {overall_gfs_mae:.2f} m/s")
    click.echo(f"   Improvement: {overall_improvement:+.1f}%")
    click.echo(f"\n✓ Diagrams created in: {output}")

"""
Generate visual diagram for Model 2 predictions with day separators

Shows actual vs predicted wind speeds over time with clear day boundaries.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def plot_predictions_with_days(pred_df, output_file, title="Model 2 - Synoptic 24 Features"):
    """
    Plot predictions with day separator lines.

    Args:
        pred_df: DataFrame with columns: valid_time, actual, predicted, dataset
        output_file: Path to save figure
        title: Plot title
    """
    # Sort by time
    pred_df = pred_df.sort_values('valid_time').copy()

    # Separate train and test
    train_df = pred_df[pred_df['dataset'] == 'train'].copy()
    test_df = pred_df[pred_df['dataset'] == 'test'].copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot training data
    ax.plot(train_df['valid_time'], train_df['actual'],
            'o', color='gray', alpha=0.3, markersize=3, label='Training (actual)')
    ax.plot(train_df['valid_time'], train_df['predicted'],
            '-', color='blue', alpha=0.5, linewidth=1, label='Training (predicted)')

    # Plot test data
    if len(test_df) > 0:
        ax.plot(test_df['valid_time'], test_df['actual'],
                'o', color='red', markersize=5, label='Test (actual)')
        ax.plot(test_df['valid_time'], test_df['predicted'],
                's', color='orange', markersize=4, label='Test (predicted)')

    # Add day separator lines
    all_times = pred_df['valid_time']
    if len(all_times) > 0:
        start_date = all_times.min().floor('D')
        end_date = all_times.max().ceil('D')

        current_date = start_date
        while current_date <= end_date:
            ax.axvline(current_date, color='black', linestyle='--', alpha=0.2, linewidth=0.5)
            current_date += timedelta(days=1)

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Wind Speed (kt)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Diagram saved: {output_file}")
    plt.close()


def main():
    """Generate diagram from predictions file."""
    # Load predictions
    pred_file = project_root / "data/lake_como/dervio/models/model2_synoptic24/predictions.parquet"

    if not pred_file.exists():
        print(f"Error: Predictions file not found: {pred_file}")
        print("Run train_model2_synoptic24.py first")
        sys.exit(1)

    pred_df = pd.read_parquet(pred_file)
    pred_df['valid_time'] = pd.to_datetime(pred_df['valid_time'])

    print(f"Loaded {len(pred_df)} predictions")
    print(f"  Train: {len(pred_df[pred_df['dataset']=='train'])}")
    print(f"  Test: {len(pred_df[pred_df['dataset']=='test'])}")

    # Generate diagram
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "model2_synoptic24_predictions.png"
    plot_predictions_with_days(pred_df, output_file)


if __name__ == '__main__':
    main()

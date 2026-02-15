"""
Generate feature importance visualization with color-coded categories.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Feature categorization based on previous analysis
def categorize_feature(feature_name):
    """Categorize features as location-agnostic, location-aware, or borderline."""

    # Temporal features
    if any(x in feature_name for x in ['hour_sin', 'hour_cos', 'day_of_year']):
        return 'temporal'

    # Location-aware features (use specific locations/regions)
    location_aware_patterns = [
        'alps_', 'alpine_', 'mediterranean_', 'scandinavia_',
        'atlantic_', 'azores_', 'iceland_', 'foehn', 'genoa_low',
        'breva_favorable'
    ]
    if any(pattern in feature_name.lower() for pattern in location_aware_patterns):
        return 'location_aware'

    # Borderline features (use Central Europe as region)
    if 'central_europe' in feature_name.lower():
        return 'borderline'

    # Teleconnection indices (location-agnostic)
    if any(x in feature_name.lower() for x in ['nao_', 'ea_', 'scand_']):
        return 'location_agnostic'

    # Pressure center features (location-agnostic)
    pressure_center_patterns = [
        'high1_', 'high2_', 'high3_', 'low1_', 'low2_', 'low3_',
        'zonal_flow', 'atlantic_ridge', 'scandi_blocking'
    ]
    if any(pattern in feature_name.lower() for pattern in pressure_center_patterns):
        return 'location_agnostic'

    # Default to location-agnostic for unmatched
    return 'location_agnostic'

# Parse feature importance from report
def parse_report(report_path):
    """Extract feature importance from markdown report."""
    with open(report_path, 'r') as f:
        content = f.read()

    # Find complete feature importance list
    pattern = r'\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|'
    matches = re.findall(pattern, content)

    features = []
    for rank, feature, importance in matches:
        features.append({
            'feature': feature,
            'importance': float(importance),
            'category': categorize_feature(feature)
        })

    return pd.DataFrame(features)

def plot_top_features(df, top_n=36, output_path='reports/model1b_feature_importance_top36.png'):
    """Create horizontal bar plot of top N features with color-coded categories."""

    # Get top N features
    df_top = df.head(top_n).copy()
    df_top = df_top.iloc[::-1]  # Reverse for bottom-to-top display

    # Define colors for each category
    category_colors = {
        'location_agnostic': '#2ecc71',  # Green
        'location_aware': '#e74c3c',      # Red
        'borderline': '#f39c12',          # Orange
        'temporal': '#3498db'             # Blue
    }

    # Create color list for bars
    colors = [category_colors[cat] for cat in df_top['category']]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 14))

    # Create horizontal bar plot
    y_pos = range(len(df_top))
    ax.barh(y_pos, df_top['importance'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Customize y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top['feature'], fontsize=9)

    # Customize x-axis
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, df_top['importance'].max() * 1.1)

    # Add title
    ax.set_title('Top 36 Feature Importance by Category\nModel 1b - Synoptic Features',
                 fontsize=14, fontweight='bold', pad=20)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=category_colors['location_agnostic'], label='Location-Agnostic', alpha=0.8),
        Patch(facecolor=category_colors['location_aware'], label='Location-Aware', alpha=0.8),
        Patch(facecolor=category_colors['borderline'], label='Borderline (Central Europe)', alpha=0.8),
        Patch(facecolor=category_colors['temporal'], label='Temporal', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature importance plot: {output_path}")

    # Print category breakdown
    print("\n=== Category Breakdown (Top 36) ===")
    category_counts = df_top['category'].value_counts()
    category_importance = df_top.groupby('category')['importance'].sum()

    for cat in ['location_aware', 'location_agnostic', 'borderline', 'temporal']:
        count = category_counts.get(cat, 0)
        total_imp = category_importance.get(cat, 0)
        pct = (count / len(df_top)) * 100
        print(f"{cat:20s}: {count:2d} features ({pct:5.1f}%), Total Gain: {total_imp:7.1f}")

    plt.close()

def main():
    """Main execution."""
    print("Generating feature importance visualization...")

    # Parse report
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    report_path = project_root / 'reports' / 'model1b_synoptic_report.md'
    df = parse_report(str(report_path))

    print(f"\nParsed {len(df)} features from report")
    print(f"Total categories: {df['category'].nunique()}")

    # Generate plot
    output_path = project_root / 'reports' / 'model1b_feature_importance_top36.png'
    plot_top_features(df, top_n=36, output_path=str(output_path))

    print("\nDone!")

if __name__ == '__main__':
    main()

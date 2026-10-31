#!/usr/bin/env python3
"""
Windmojo CLI - Wind Forecasting with Machine Learning

Unified command-line interface for the simplified windmojo pipeline.
"""

import click
import logging
from pathlib import Path


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.version_option(version="0.2.0", prog_name="windmojo")
def cli(verbose):
    """
    Windmojo - ML-powered wind forecasting.

    Simplified single-model pipeline: download → features → train → evaluate
    """
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@cli.command()
@click.option('--spot', required=True, help='Spot name (e.g., dervio)')
@click.option('--start-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='End date (YYYY-MM-DD)')
def download(spot, start_date, end_date):
    """Download GFS forecast data for a spot."""
    click.echo(f"Downloading data for {spot}: {start_date} to {end_date}")
    click.echo("Feature not yet implemented in simplified version.")
    click.echo("Use: python -m windmojo.data_fetcher for now")


@cli.command()
@click.option('--spot', required=True, help='Spot name')
def features(spot):
    """Engineer features from raw GFS data."""
    click.echo(f"Engineering features for {spot}")
    click.echo("Feature not yet implemented in simplified version.")
    click.echo("Use: python -m windmojo.feature_engineer for now")


@cli.command()
@click.option('--spot', required=True, help='Spot name')
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file (default: data/spots/{spot}/input/config.yaml)')
@click.option('--until', type=click.DateTime(formats=['%Y-%m-%d']),
              help='Train only on data up to this day (later days stay unseen, e.g. for comparisons)')
def train(spot, config, until):
    """Train the unified model on all features."""
    from windmojo.config import (load_config, get_paths, get_training_config, get_test_dates,
                                 get_validation_dates)
    from windmojo.features.pipeline import feature_columns
    from windmojo.training import UnifiedTrainer
    from windmojo.units import measurement_unit, to_ms
    import pandas as pd

    click.echo(f"\n{'='*60}")
    click.echo(f"Training Unified Model: {spot}")
    click.echo(f"{'='*60}\n")

    # Load config
    if config is None:
        config = Path(f"data/spots/{spot}/input/config.yaml")
    else:
        config = Path(config)

    cfg = load_config(config)
    paths = get_paths(cfg)
    training_cfg = get_training_config(cfg)

    # Load measurements
    click.echo("1. Loading measurements...")
    measurements = pd.read_csv(paths['measurements'])
    measurements['time'] = pd.to_datetime(measurements['timestamp'], format='%Y-%m-%d_%H%M')
    measurements['date'] = measurements['time'].dt.date
    target_col = cfg['target']['column']
    unit = measurement_unit(cfg)
    measurements[target_col] = to_ms(measurements[target_col], unit)
    if unit != 'm/s':
        click.echo(f"   Converted '{target_col}' from {unit} to m/s")

    # Load features
    click.echo("2. Loading features...")
    features_file = paths['features'] / f"{spot}_features_phase1.parquet"
    features = pd.read_parquet(features_file)

    # All features of the pipeline (taken before the merge, so measurement
    # columns can never become features)
    selected_features = feature_columns(features)
    features['date'] = features['time'].dt.date
    click.echo(f"3. Using all {len(selected_features)} features")

    # Merge
    merged = features.merge(
        measurements[['time', target_col]],
        on='time',
        how='inner'
    ).dropna(subset=[target_col])
    if until:
        merged = merged[merged['date'] <= until.date()]
        click.echo(f"   Using data up to {until:%Y-%m-%d}")

    # Split train/validation/test
    all_dates = sorted(merged['date'].unique())
    test_dates = get_test_dates(cfg, all_dates)
    validation_dates = get_validation_dates([d for d in all_dates if d not in set(test_dates)])

    click.echo(f"4. Splitting data ({len(test_dates)} test days, "
               f"{len(validation_dates)} early-stopping days)...")
    test_mask = merged['date'].isin(test_dates)
    validation_mask = merged['date'].isin(validation_dates)

    def prepare(mask):
        data = merged.loc[mask, ['time'] + selected_features + [target_col]]
        return data.rename(columns={target_col: 'target'})

    train_data = prepare(~test_mask & ~validation_mask)
    validation_data = prepare(validation_mask)
    test_data = prepare(test_mask)

    click.echo(f"   Train: {len(train_data)} samples")
    click.echo(f"   Validation: {len(validation_data)} samples")
    click.echo(f"   Test: {len(test_data)} samples")

    # Train
    click.echo("\n5. Training model...")
    trainer = UnifiedTrainer(training_cfg)
    results = trainer.train(train_data, validation_data, test_data, selected_features)
    # Days up to here are part of the training data (hindcasts for them look too good)
    results['data_until'] = str(max(all_dates))

    # Save
    click.echo("\n6. Saving model...")
    trainer.save(paths['model'], paths['metadata'])

    click.echo(f"\n{'='*60}")
    click.echo("Training Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"Train MAE: {results['train']['mae']:.2f} m/s")
    click.echo(f"Test MAE: {results['test']['mae']:.2f} m/s")
    click.echo(f"\nModel saved to: {paths['model']}")


@cli.command()
@click.option('--spot', required=True, help='Spot name')
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file (default: data/spots/{spot}/input/config.yaml)')
def evaluate(spot, config):
    """Evaluate trained model on test days and create diagrams."""
    from windmojo.config import load_config, get_paths, get_test_dates
    from windmojo.training import UnifiedTrainer
    from windmojo.evaluation import calculate_metrics, create_validation_diagrams
    from windmojo.units import measurement_unit, to_ms
    import pandas as pd
    import numpy as np

    click.echo(f"\n{'='*60}")
    click.echo(f"Evaluating Model: {spot}")
    click.echo(f"{'='*60}\n")

    # Load config
    if config is None:
        config = Path(f"data/spots/{spot}/input/config.yaml")
    else:
        config = Path(config)

    cfg = load_config(config)
    paths = get_paths(cfg)
    spot_name = cfg.get('name', spot)
    target_col = cfg['target']['column']

    # Load model
    click.echo("1. Loading trained model...")
    trainer = UnifiedTrainer(cfg.get('training', {}))
    trainer.load(paths['model'], paths['metadata'])

    # Load measurements
    click.echo("2. Loading measurements...")
    measurements = pd.read_csv(paths['measurements'])
    measurements['time'] = pd.to_datetime(measurements['timestamp'], format='%Y-%m-%d_%H%M')
    measurements['date'] = measurements['time'].dt.date
    unit = measurement_unit(cfg)
    measurements[target_col] = to_ms(measurements[target_col], unit)
    if unit != 'm/s':
        click.echo(f"   Converted '{target_col}' from {unit} to m/s")

    # Load features
    click.echo("3. Loading features...")
    features_file = paths['features'] / f"{spot}_features_phase1.parquet"
    features = pd.read_parquet(features_file)
    features['date'] = features['time'].dt.date

    # Get test dates
    all_dates = sorted(features['date'].unique())
    test_dates = get_test_dates(cfg, all_dates)

    click.echo(f"4. Evaluating on {len(test_dates)} test days...")

    # Merge and filter to test dates
    merged = features.merge(
        measurements[['time', target_col]],
        on='time',
        how='inner'
    ).dropna(subset=[target_col])

    test_df = merged[merged['date'].isin(test_dates)].copy()

    click.echo(f"   Test samples: {len(test_df)}")

    # Make predictions
    predictions = trainer.predict(test_df)

    # Calculate metrics
    y_true = test_df[target_col].values
    metrics = calculate_metrics(y_true, predictions)

    click.echo(f"\n5. Test Metrics:")
    click.echo(f"   MAE:  {metrics['mae']:.2f} m/s")
    click.echo(f"   RMSE: {metrics['rmse']:.2f} m/s")
    click.echo(f"   R²:   {metrics['r2']:.3f}")
    click.echo(f"   MBE:  {metrics['mbe']:.2f} m/s")

    # Create predictions dataframe for diagrams
    predictions_df = pd.DataFrame({
        'time': test_df['time'].values,
        'prediction': predictions
    })

    # Create diagrams
    click.echo(f"\n6. Creating validation diagrams...")
    diagram_files = create_validation_diagrams(
        predictions=predictions_df,
        measurements=measurements,
        features=features,
        test_dates=test_dates,
        target_col=target_col,
        output_dir=paths['validation'],
        spot_name=spot_name
    )

    click.echo(f"\n{'='*60}")
    click.echo("Evaluation Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"Test MAE: {metrics['mae']:.2f} m/s")
    click.echo(f"Diagrams: {len(diagram_files)} created in {paths['validation']}")
    for month_name, diagram_path in diagram_files.items():
        click.echo(f"  • {diagram_path.name}")


@cli.command()
@click.option('--spot', required=True, help='Spot name')
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file (default: data/spots/{spot}/input/config.yaml)')
@click.option('--output', '-o', type=click.Path(), help='JSON output file (default: data/spots/{spot}/forecasts/forecast.json)')
@click.option('--history-days', default=7, show_default=True, help='Past days downloaded for lag and rolling features')
@click.option('--archive/--no-archive', default=True, show_default=True,
              help='Keep the forecast JSON (paths.forecast_archive) and, once per day, the raw weather '
                   'forecast of forecast.raw_archive_days (paths.forecast_raw_archive)')
def predict(spot, config, output, history_days, archive):
    """Create a live forecast and export it as JSON (see DEPLOYMENT.md)."""
    from windmojo.config import load_config, get_paths
    from windmojo.prediction import run_forecast
    import warnings
    import pandas as pd

    # The feature pipeline adds columns one at a time; harmless, but it floods logs
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

    click.echo(f"\n{'='*60}")
    click.echo(f"Forecast: {spot}")
    click.echo(f"{'='*60}\n")

    # Load config
    if config is None:
        config = Path(f"data/spots/{spot}/input/config.yaml")
    else:
        config = Path(config)

    cfg = load_config(config)
    paths = get_paths(cfg)

    click.echo("Loading model, downloading forecast, predicting...")
    run = run_forecast(cfg, paths, history_days=history_days, archive=archive, output=output)
    if run['raw_archived']:
        click.echo(f"   Raw forecast archived: {run['raw_archived']}")
    if run['archived']:
        click.echo(f"   Archived: {run['archived']}")

    hours = run['hours']
    click.echo(f"\n✓ {len(hours)} hours ({hours['time'].min():%Y-%m-%d %H:%M} to "
               f"{hours['time'].max():%Y-%m-%d %H:%M}) written to {run['output']}")




@cli.command()
@click.option('--spot', required=True, help='Spot name')
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file (default: data/spots/{spot}/input/config.yaml)')
@click.option('--host', default='127.0.0.1', show_default=True,
              help='Address to listen on (0.0.0.0 = also reachable from other devices in the network)')
@click.option('--port', default=8000, show_default=True, help='Port')
def serve(spot, config, host, port):
    """Show the forecast page locally: update button, past days, CSV downloads."""
    from windmojo.config import load_config, get_paths
    from windmojo.web import ForecastApp, serve as make_server
    import warnings
    import pandas as pd

    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    if config is None:
        config = Path(f"data/spots/{spot}/input/config.yaml")
    cfg = load_config(Path(config))
    server = make_server(ForecastApp(cfg, get_paths(cfg)), host, port)
    click.echo(f"\nForecast page: http://localhost:{port}/   (stop with Ctrl+C)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopped.")
    finally:
        server.server_close()


@cli.command()
@click.option('--from-dir', required=True, type=click.Path(exists=True), help='Source directory (e.g., data/lake_como/dervio)')
@click.option('--to-spot', required=True, help='Target spot name (e.g., dervio)')
def migrate(from_dir, to_spot):
    """Migrate data from old structure to new spots/ structure."""
    import shutil

    from_dir = Path(from_dir)
    to_dir = Path(f"data/spots/{to_spot}")

    click.echo(f"\n{'='*60}")
    click.echo(f"Migrating: {from_dir} → {to_dir}")
    click.echo(f"{'='*60}\n")

    # Create structure
    (to_dir / "input").mkdir(parents=True, exist_ok=True)
    (to_dir / "training/raw").mkdir(parents=True, exist_ok=True)
    (to_dir / "training/features").mkdir(parents=True, exist_ok=True)
    (to_dir / "training/validation").mkdir(parents=True, exist_ok=True)
    (to_dir / "models").mkdir(parents=True, exist_ok=True)

    # Copy measurements
    meas_src = from_dir / "measurements/wind_data.csv"
    if meas_src.exists():
        shutil.copy(meas_src, to_dir / "input/measurements.csv")
        click.echo(f"✓ Copied measurements")

    # Copy raw data
    raw_src = from_dir / "raw"
    if raw_src.exists():
        for f in raw_src.glob("*.parquet"):
            shutil.copy(f, to_dir / "training/raw" / f.name)
        click.echo(f"✓ Copied {len(list(raw_src.glob('*.parquet')))} raw files")

    # Copy features
    feat_src = from_dir / "features"
    if feat_src.exists():
        for f in feat_src.glob("*.parquet"):
            shutil.copy(f, to_dir / "training/features" / f.name)
        click.echo(f"✓ Copied features")

    # Copy config (legacy location: config/{area}/{spot}.yaml)
    config_src = Path("config") / from_dir.parent.name / f"{from_dir.name}.yaml"
    if config_src.exists():
        shutil.copy(config_src, to_dir / "input/config.yaml")
        click.echo(f"✓ Copied config (needs manual simplification)")
    else:
        click.echo(f"⚠ No config found at {config_src}, create {to_dir}/input/config.yaml manually")

    click.echo(f"\n✓ Migration complete! Data in: {to_dir}")
    click.echo("\nNext steps:")
    click.echo(f"1. Edit {to_dir}/input/config.yaml to use simplified format")
    click.echo(f"2. Run: windmojo train --spot {to_spot}")


if __name__ == '__main__':
    cli()

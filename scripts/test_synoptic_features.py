"""
Test synoptic feature extraction pipeline.

Loads the downloaded synoptic grid and extracts all 123 features:
- Pressure center detection (highs/lows)
- Regional aggregation
- Teleconnection indices
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from windmojo.core.config import load_config
from windmojo.features.synoptic_pipeline import SynopticFeaturePipeline
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test synoptic feature extraction."""
    
    # Load experiment8 config
    config_path = project_root / "config" / "lake_como" / "dervio_experiment8.yaml"
    logger.info(f"Loading config from: {config_path}")
    
    config = load_config(str(config_path))
    
    # Create feature pipeline
    pipeline = SynopticFeaturePipeline(config)
    
    logger.info("=" * 70)
    logger.info("SYNOPTIC FEATURE EXTRACTION TEST")
    logger.info("=" * 70)
    logger.info(f"Expected features: {pipeline.get_n_features()}")
    logger.info(f"Feature names: {len(pipeline.get_feature_names())}")
    
    # Load the downloaded synoptic grid
    data_dir = project_root / "data" / "lake_como" / "dervio" / "raw" / "synoptic"
    grid_file = data_dir / "synoptic_grid_20250501.parquet"
    
    if not grid_file.exists():
        logger.error(f"Grid file not found: {grid_file}")
        return False
    
    logger.info(f"\nLoading synoptic grid: {grid_file.name}")
    grid_df = pd.read_parquet(grid_file)
    
    logger.info(f"Grid shape: {grid_df.shape}")
    logger.info(f"Grid columns: {list(grid_df.columns)}")
    logger.info(f"Unique locations: {grid_df['location'].nunique()}")
    logger.info(f"Time: {grid_df['time'].unique()[0]}")
    
    # Extract features
    logger.info("\n" + "=" * 70)
    logger.info("EXTRACTING FEATURES...")
    logger.info("=" * 70)
    
    try:
        features = pipeline.transform(grid_df)
        
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE EXTRACTION SUCCESSFUL!")
        logger.info("=" * 70)
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Total features: {len(features.columns)}")
        
        # Show feature breakdown
        logger.info("\nFeature breakdown:")
        
        # Pressure center features
        center_features = [c for c in features.columns if 'high' in c or 'low' in c]
        logger.info(f"  Pressure centers: {len(center_features)} features")
        
        # Regional features
        regional_features = [c for c in features.columns if any(r in c for r in 
            ['iceland', 'azores', 'atlantic', 'scandinavia', 'central_europe', 
             'mediterranean', 'alps_north', 'alps_south'])]
        logger.info(f"  Regional aggregation: {len(regional_features)} features")
        
        # Teleconnection features
        tele_features = [c for c in features.columns if any(t in c for t in 
            ['nao', 'ea', 'scand', 'genoa', 'foehn', 'alpine', 'breva'])]
        logger.info(f"  Teleconnection indices: {len(tele_features)} features")
        
        # Display sample features
        logger.info("\n" + "=" * 70)
        logger.info("SAMPLE FEATURES:")
        logger.info("=" * 70)
        
        # Show highs
        logger.info("\nPressure Centers (Highs):")
        for i in range(1, 4):
            prefix = f'high{i}'
            if f'{prefix}_lat' in features.columns:
                logger.info(f"  High {i}:")
                logger.info(f"    Position: {features[f'{prefix}_lat'].iloc[0]:.1f}°N, "
                          f"{features[f'{prefix}_lon'].iloc[0]:.1f}°E")
                logger.info(f"    Z500: {features[f'{prefix}_z500'].iloc[0]:.0f} m")
                logger.info(f"    Distance: {features[f'{prefix}_dist_km'].iloc[0]:.0f} km")
                logger.info(f"    Exists: {features[f'{prefix}_exists'].iloc[0]}")
        
        # Show lows
        logger.info("\nPressure Centers (Lows):")
        for i in range(1, 4):
            prefix = f'low{i}'
            if f'{prefix}_lat' in features.columns:
                logger.info(f"  Low {i}:")
                logger.info(f"    Position: {features[f'{prefix}_lat'].iloc[0]:.1f}°N, "
                          f"{features[f'{prefix}_lon'].iloc[0]:.1f}°E")
                logger.info(f"    Z500: {features[f'{prefix}_z500'].iloc[0]:.0f} m")
                logger.info(f"    Distance: {features[f'{prefix}_dist_km'].iloc[0]:.0f} km")
                logger.info(f"    Exists: {features[f'{prefix}_exists'].iloc[0]}")
        
        # Show teleconnection indices
        logger.info("\nTeleconnection Indices:")
        if 'nao_index' in features.columns:
            logger.info(f"  NAO index: {features['nao_index'].iloc[0]:.2f}")
        if 'ea_index' in features.columns:
            logger.info(f"  EA index: {features['ea_index'].iloc[0]:.2f}")
        if 'scand_index' in features.columns:
            logger.info(f"  SCAND index: {features['scand_index'].iloc[0]:.2f}")
        if 'foehn_gradient' in features.columns:
            logger.info(f"  Foehn gradient: {features['foehn_gradient'].iloc[0]:.1f} hPa")
        
        # Show all feature names
        logger.info("\n" + "=" * 70)
        logger.info("ALL FEATURES:")
        logger.info("=" * 70)
        for i, feature in enumerate(sorted(features.columns), 1):
            value = features[feature].iloc[0]
            logger.info(f"{i:3d}. {feature:40s} = {value:12.3f}")
        
        logger.info("\n" + "=" * 70)
        logger.info("TEST PASSED!")
        logger.info("=" * 70)
        logger.info(f"Successfully extracted {len(features.columns)} synoptic features")
        logger.info(f"Pressure centers detected: {sum(features[f'high{i}_exists'].iloc[0] for i in range(1, 4))} highs, "
                   f"{sum(features[f'low{i}_exists'].iloc[0] for i in range(1, 4))} lows")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

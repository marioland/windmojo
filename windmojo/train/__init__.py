"""
Training Module for windmojo

Phase 1: Model 1 - Feature discovery with all 433 features
Phase 2: Model 2 - Production model with selected features (future)
"""

from windmojo.train.data_loader import load_training_data, prepare_datasets
from windmojo.train.importance_extractor import (
    calculate_shap_values,
    extract_importance,
)
from windmojo.train.phase1_model1 import Model1Trainer, train_model1
from windmojo.train.phase1_report import generate_phase1_report

__all__ = [
    "load_training_data",
    "prepare_datasets",
    "Model1Trainer",
    "train_model1",
    "extract_importance",
    "calculate_shap_values",
    "generate_phase1_report",
]

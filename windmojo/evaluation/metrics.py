"""Evaluation metrics for wind forecasting."""

from typing import Dict

import numpy as np


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE, RMSE, R2, etc.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Mean Bias Error
    mbe = np.mean(y_pred - y_true)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mbe": float(mbe),
        "samples": len(y_true),
    }

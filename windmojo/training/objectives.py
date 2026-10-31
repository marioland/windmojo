"""
Training Objectives

Custom XGBoost objectives for the unified model.
"""

from typing import Callable, Tuple

import numpy as np
import pandas as pd


def time_tolerant_objective(
    times: pd.Series,
    target: pd.Series,
    hours: int = 2,
    shift_penalty: float = 0.25,
    exact_weight: float = 0.2,
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Squared error that does not punish a timing error twice.

    Plain squared error punishes a wind peak forecast one hour too early twice:
    as a false alarm at the forecast hour and as a miss at the measured hour.
    The model learns to flatten such peaks. This objective matches in both
    directions instead:
      - each forecast hour with the best-fitting measurement within +-hours
      - each measured hour with the best-fitting forecast within +-hours
    Each match costs shift_penalty per squared hour of shift. Matching both
    ways keeps a missed peak as expensive as before (the measured peak finds no
    fitting forecast). exact_weight keeps part of the plain squared error at
    the exact hour:

        loss = exact_weight * (p_t - y_t)^2 + (1 - exact_weight) / 2 *
               [min_s ((p_t - y_t+s)^2 + c s^2) + min_s ((p_t+s - y_t)^2 + c s^2)]

    Args:
        times: Timestamps of the training rows (hourly; gaps such as nights
            are no neighbours)
        target: Measured values of the training rows, same order
        hours: Tolerance in hours (0 = plain squared error)
        shift_penalty: Cost per squared hour of shift in (m/s)^2 (0.25: one
            hour of shift costs as much as 0.5 m/s error)
        exact_weight: Share of the plain squared error at the exact hour

    Returns:
        Objective for xgboost.XGBRegressor: (y_true, y_pred) -> (grad, hess),
        for the rows in the order given here
    """
    times = pd.DatetimeIndex(times)
    y = np.asarray(target, dtype=float)
    position = pd.Series(np.arange(len(times)), index=times)
    position = position[~position.index.duplicated()]
    shifts = np.arange(-hours, hours + 1)

    # Row of t+s for every row t and shift s (NaN if t+s is no training row)
    neighbour = np.column_stack(
        [(times + pd.Timedelta(hours=int(s))).map(position).to_numpy() for s in shifts]
    ).astype(float)
    valid = ~np.isnan(neighbour)
    rows = np.where(valid, neighbour, 0).astype(int)
    y_neighbour = np.where(valid, y[rows], np.nan)
    penalty = shift_penalty * shifts**2
    index = np.arange(len(y))
    weight = 1 - exact_weight

    def objective(y_true: np.ndarray, y_pred: np.ndarray):
        grad = 2 * exact_weight * (y_pred - y)
        hess = np.full_like(y_pred, 2 * exact_weight)

        # Forecast -> best-fitting measurement
        cost = np.where(valid, (y_pred[:, None] - y_neighbour) ** 2 + penalty, np.inf)
        best = y_neighbour[index, np.argmin(cost, axis=1)]
        grad += weight * (y_pred - best)
        hess += weight

        # Measurement -> best-fitting forecast (gradient on the matched row)
        p_neighbour = np.where(valid, y_pred[rows], np.nan)
        cost = np.where(valid, (p_neighbour - y[:, None]) ** 2 + penalty, np.inf)
        matched = rows[index, np.argmin(cost, axis=1)]
        np.add.at(grad, matched, weight * (y_pred[matched] - y))
        np.add.at(hess, matched, weight)

        return grad, hess

    return objective

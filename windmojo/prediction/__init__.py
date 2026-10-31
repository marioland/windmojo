"""
Prediction: live forecasts from a trained model.
"""

from windmojo.prediction.forecast import (
    download_forecast,
    forecast_to_dict,
    predict_forecast,
    save_forecast_json,
)

__all__ = [
    "download_forecast",
    "forecast_to_dict",
    "predict_forecast",
    "save_forecast_json",
]

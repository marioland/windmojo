"""
Prediction: live forecasts from a trained model.
"""

from windmojo.prediction.forecast import (
    archive_forecast,
    archive_raw_forecast,
    download_forecast,
    forecast_to_dict,
    predict_forecast,
    save_forecast_json,
)

__all__ = [
    "archive_forecast",
    "archive_raw_forecast",
    "download_forecast",
    "forecast_to_dict",
    "predict_forecast",
    "save_forecast_json",
]

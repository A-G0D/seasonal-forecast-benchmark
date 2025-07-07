"""Time-series forecasting toolkit.

A small, dependency-light toolkit for generating synthetic seasonal series,
fitting a handful of forecasters, and backtesting them on a holdout window.

The layers are deliberately thin:

    series      -> synthetic data generation + validation
    forecasters -> the models (naive, seasonal-naive, Holt-Winters, regression)
    base        -> the common Forecaster interface they all implement
    metrics     -> point-error metrics (MAE / RMSE / MAPE / sMAPE / MASE)
    backtest    -> holdout split + benchmark harness that ties it together
"""
from .base import Forecaster
from .series import SeriesSpec, make_series, SERIES_LIBRARY

__all__ = ["Forecaster", "SeriesSpec", "make_series", "SERIES_LIBRARY"]

__version__ = "0.4.0"

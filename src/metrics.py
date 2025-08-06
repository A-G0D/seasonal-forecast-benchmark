"""Point-forecast error metrics.

All functions take ``(y_true, y_pred)`` 1-D arrays of equal length and return a
single float. MASE additionally needs the training series so it can compute the
naive in-sample scale.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _align(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("empty arrays")
    return y_true, y_pred


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, as a percentage."""
    y_true, y_pred = _align(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), _EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE, as a percentage (range 0..200)."""
    y_true, y_pred = _align(y_true, y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, _EPS)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray,
         period: int = 1) -> float:
    """Mean absolute scaled error.

    Scales the forecast MAE by the in-sample MAE of a (seasonal) naive model
    on the training series. A value below 1 beats that naive baseline.
    """
    y_true, y_pred = _align(y_true, y_pred)
    y_train = np.asarray(y_train, dtype=np.float64)
    if period < 1:
        raise ValueError("period must be >= 1")
    if y_train.size <= period:
        raise ValueError("training series too short for the chosen period")

    naive_errors = np.abs(y_train[period:] - y_train[:-period])
    scale = float(np.mean(naive_errors))
    if scale < _EPS:
        scale = _EPS
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray,
                period: int = 1) -> dict[str, float]:
    """Compute every metric at once and return a dict keyed by metric name."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train, period=period),
    }


# all of ours are lower-is-better, but reports iterate this so keep it explicit
LOWER_IS_BETTER = ("mae", "rmse", "mape", "smape", "mase")

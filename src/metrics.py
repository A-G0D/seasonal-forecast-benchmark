"""Point-forecast error metrics."""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _align(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("shape mismatch")
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

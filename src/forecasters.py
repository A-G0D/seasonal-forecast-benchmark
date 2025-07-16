"""Baseline forecasters."""
from __future__ import annotations

import numpy as np

from .base import Forecaster


class NaiveForecaster(Forecaster):
    """Forecast every future step as the last observed value."""

    name = "naive"

    def __init__(self) -> None:
        super().__init__()
        self._last: float = 0.0

    def fit(self, y: np.ndarray) -> "NaiveForecaster":
        y = self._check_train(y)
        self._last = float(y[-1])
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        return np.full(horizon, self._last, dtype=np.float64)


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the last full season forward."""

    name = "seasonal_naive"

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = int(period)
        self._last_season: np.ndarray = np.empty(0)

    def fit(self, y: np.ndarray) -> "SeasonalNaiveForecaster":
        y = self._check_train(y)
        self._last_season = y[-self.period:].copy()
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        idx = np.arange(horizon) % self.period
        return self._last_season[idx].astype(np.float64)

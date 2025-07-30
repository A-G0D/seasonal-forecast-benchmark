"""Two baselines (naive, seasonal-naive) and additive Holt-Winters on numpy."""
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
        horizon = self._check_horizon(horizon)
        return np.full(horizon, self._last, dtype=np.float64)


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the last full season forward."""

    name = "seasonal_naive"

    def __init__(self, period: int) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = int(period)
        self._last_season: np.ndarray = np.empty(0)

    def fit(self, y: np.ndarray) -> "SeasonalNaiveForecaster":
        y = self._check_train(y)
        if y.size < self.period:
            raise ValueError(
                f"need at least {self.period} points for seasonal-naive, "
                f"got {y.size}"
            )
        self._last_season = y[-self.period:].copy()
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        horizon = self._check_horizon(horizon)
        idx = np.arange(horizon) % self.period
        return self._last_season[idx].astype(np.float64)


class HoltWinters(Forecaster):
    """Additive triple exponential smoothing. One recursive pass updates the
    level, trend and per-step seasonal components; predict extends the final
    level/trend and cycles the seasonal component. alpha/beta/gamma are the
    level/trend/seasonal smoothing weights.
    """

    name = "holt_winters"

    def __init__(
        self,
        period: int,
        alpha: float = 0.3,
        beta: float = 0.05,
        gamma: float = 0.2,
    ) -> None:
        super().__init__()
        if period < 2:
            raise ValueError("Holt-Winters needs a period of at least 2")
        for nm, v in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{nm} must be in [0, 1]")
        if alpha == 0.0:
            raise ValueError("alpha must be > 0")
        self.period = int(period)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self._level = 0.0
        self._trend = 0.0
        self._season = np.zeros(period)

    def _initial_components(self, y: np.ndarray) -> tuple[float, float, np.ndarray]:
        m = self.period
        first = y[:m]
        second = y[m:2 * m] if y.size >= 2 * m else first
        level0 = float(first.mean())
        trend0 = float((second.mean() - first.mean()) / m)
        season0 = first - level0
        return level0, trend0, season0

    def fit(self, y: np.ndarray) -> "HoltWinters":
        y = self._check_train(y)
        m = self.period
        if y.size < 2 * m:
            raise ValueError(
                f"Holt-Winters needs at least two seasons ({2 * m}) of data"
            )

        level, trend, season = self._initial_components(y)
        season = season.copy()

        for i in range(y.size):
            s_idx = i % m
            prev_level = level
            seasonal = season[s_idx]
            level = self.alpha * (y[i] - seasonal) + (1 - self.alpha) * (prev_level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
            season[s_idx] = self.gamma * (y[i] - level) + (1 - self.gamma) * seasonal

        self._level = level
        self._trend = trend
        self._season = season
        self._n_seen = y.size
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        horizon = self._check_horizon(horizon)
        m = self.period
        steps = np.arange(1, horizon + 1)
        base = self._level + steps * self._trend
        season_idx = (self._n_seen + steps - 1) % m
        return base + self._season[season_idx]

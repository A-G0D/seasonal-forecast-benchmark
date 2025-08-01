"""The common forecaster contract: fit(train) -> self, predict(horizon) -> a
1-D float array of `horizon` values. fit returns self so calls chain. Every
model in src.forecasters honours this so the backtest can swap them freely.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    name: str = "forecaster"  # label used in reports and logs

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, y: np.ndarray) -> "Forecaster":
        """Fit on a 1-D training series and return self."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return ``horizon`` forecasted values as a 1-D float array."""

    @staticmethod
    def _check_train(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError("training series must be 1-D")
        if y.size == 0:
            raise ValueError("training series is empty")
        if not np.all(np.isfinite(y)):
            raise ValueError("training series contains non-finite values")
        return y

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

    @staticmethod
    def _check_horizon(horizon: int) -> int:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        return int(horizon)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self._fitted else "unfitted"
        return f"<{type(self).__name__} name={self.name!r} {state}>"

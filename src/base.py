"""The common forecaster contract: fit(train) -> self, predict(horizon)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    name: str = "forecaster"

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, y: np.ndarray) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        ...

    @staticmethod
    def _check_train(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError("training series must be 1-D")
        if y.size == 0:
            raise ValueError("training series is empty")
        return y

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

"""The model line-up, in one place so eval and tests agree on what counts as
baseline vs improved."""
from __future__ import annotations

from typing import Dict

from .backtest import ModelFactory
from .forecasters import (
    HoltWinters,
    NaiveForecaster,
    RegressionForecaster,
    SeasonalNaiveForecaster,
)
from .series import SeriesSpec

BASELINE_MODELS: Dict[str, ModelFactory] = {
    "naive": lambda spec: NaiveForecaster(),
    "seasonal_naive": lambda spec: SeasonalNaiveForecaster(spec.period),
}

IMPROVED_MODELS: Dict[str, ModelFactory] = {
    "holt_winters": lambda spec: HoltWinters(spec.period),
    "regression": lambda spec: RegressionForecaster(
        spec.period, lags=(1, spec.period)
    ),
}


def all_models() -> Dict[str, ModelFactory]:
    """Baseline + improved factories in one dict."""
    return {**BASELINE_MODELS, **IMPROVED_MODELS}

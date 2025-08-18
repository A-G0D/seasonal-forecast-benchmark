"""Backtest harness: hold out the tail of a series, fit on the prefix, score
the forecast against the holdout. Benchmark runs a set of model factories over
a set of series. Each fit/score emits one trace event through the Observer, so
a full run leaves a JSONL trail under logs/.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from shared.obs import Observer

from . import metrics as M
from .base import Forecaster
from .series import SeriesSpec, make_series, train_test_split


@dataclass
class BacktestResult:
    series: str
    model: str
    metrics: Dict[str, float]
    horizon: int
    forecast: List[float] = field(default_factory=list)

    def as_record(self, include_forecast: bool = False) -> dict:
        rec = {
            "series": self.series,
            "model": self.model,
            "horizon": self.horizon,
            "metrics": self.metrics,
        }
        if include_forecast:
            rec["forecast"] = self.forecast
        return rec


# takes a spec (so it can read the period) and returns a fresh unfitted model
ModelFactory = Callable[[SeriesSpec], Forecaster]


def backtest_one(
    spec: SeriesSpec,
    factory: ModelFactory,
    horizon: int,
    *,
    seed: int = 137,
    observer: Optional[Observer] = None,
) -> BacktestResult:
    """Fit one model on the training prefix of ``spec`` and score the holdout."""
    y = make_series(spec, seed=seed)
    train, test = train_test_split(y, horizon)

    model = factory(spec)
    start = time.perf_counter()
    model.fit(train)
    forecast = model.predict(horizon)
    latency_ms = (time.perf_counter() - start) * 1000.0

    scores = M.all_metrics(test, forecast, train, period=spec.period)
    result = BacktestResult(
        series=spec.name,
        model=model.name,
        metrics=scores,
        horizon=horizon,
        forecast=[float(v) for v in forecast],
    )

    if observer is not None:
        observer.emit(
            input={"series": spec.name, "model": model.name, "horizon": horizon,
                   "train_len": int(train.size)},
            output={"metrics": scores},
            latency_ms=latency_ms,
        )
    return result


@dataclass
class Benchmark:
    """Run a set of model factories across a set of series at a fixed holdout."""

    specs: List[SeriesSpec]
    models: Dict[str, ModelFactory]
    horizon: int = 28
    seed: int = 137

    def run(self, observer: Optional[Observer] = None) -> List[BacktestResult]:
        results: List[BacktestResult] = []
        for spec in self.specs:
            for factory in self.models.values():
                results.append(
                    backtest_one(
                        spec, factory, self.horizon,
                        seed=self.seed, observer=observer,
                    )
                )
        return results

    @staticmethod
    def aggregate(results: List[BacktestResult]) -> Dict[str, Dict[str, float]]:
        """Average each metric per model across all series.

        Returns a mapping ``model -> {metric: mean_value}``.
        """
        by_model: Dict[str, List[Dict[str, float]]] = {}
        for r in results:
            by_model.setdefault(r.model, []).append(r.metrics)

        out: Dict[str, Dict[str, float]] = {}
        for model, rows in by_model.items():
            keys = rows[0].keys()
            out[model] = {
                k: float(np.mean([row[k] for row in rows])) for k in keys
            }
        return out

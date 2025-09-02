import io

import numpy as np
import pytest

from shared.obs import Observer

from src.backtest import Benchmark, backtest_one
from src.registry import BASELINE_MODELS, IMPROVED_MODELS, all_models
from src.series import SERIES_LIBRARY, SeriesSpec


def test_backtest_one_returns_all_metrics():
    spec = SERIES_LIBRARY["retail_daily"]
    factory = IMPROVED_MODELS["holt_winters"]
    res = backtest_one(spec, factory, horizon=28, seed=137)
    assert res.series == "retail_daily"
    assert res.model == "holt_winters"
    assert set(res.metrics) == {"mae", "rmse", "mape", "smape", "mase"}
    assert len(res.forecast) == 28


def test_backtest_emits_one_event():
    spec = SERIES_LIBRARY["energy_load"]
    obs = Observer("test", sink=io.StringIO(), deterministic=True)
    backtest_one(spec, BASELINE_MODELS["naive"], horizon=14, seed=137, observer=obs)
    assert len(obs.events) == 1
    ev = obs.events[0].canonical()
    assert ev["module"] == "test"
    assert ev["input"]["series"] == "energy_load"


def test_benchmark_runs_all_combinations():
    specs = [SERIES_LIBRARY["retail_daily"], SERIES_LIBRARY["web_traffic"]]
    bench = Benchmark(specs, all_models(), horizon=21, seed=137)
    results = bench.run()
    assert len(results) == len(specs) * len(all_models())


def test_aggregate_averages_per_model():
    specs = list(SERIES_LIBRARY.values())
    bench = Benchmark(specs, BASELINE_MODELS, horizon=28, seed=137)
    agg = Benchmark.aggregate(bench.run())
    assert set(agg) == set(BASELINE_MODELS)
    for scores in agg.values():
        assert set(scores) == {"mae", "rmse", "mape", "smape", "mase"}


def test_end_to_end_improved_beats_baseline():
    specs = list(SERIES_LIBRARY.values())
    base = Benchmark.aggregate(
        Benchmark(specs, BASELINE_MODELS, horizon=28, seed=137).run()
    )
    imp = Benchmark.aggregate(
        Benchmark(specs, IMPROVED_MODELS, horizon=28, seed=137).run()
    )

    metrics = ["mae", "rmse", "mape", "smape", "mase"]
    wins = 0
    for m in metrics:
        best_base = min(base[k][m] for k in base)
        best_imp = min(imp[k][m] for k in imp)
        if best_imp < best_base:
            wins += 1
    assert wins >= 1

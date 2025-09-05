import numpy as np

from shared.determinism import set_seed

from src.backtest import Benchmark
from src.forecasters import HoltWinters, RegressionForecaster
from src.registry import all_models
from src.series import SERIES_LIBRARY, make_series


def _run():
    set_seed(137)
    specs = list(SERIES_LIBRARY.values())
    return Benchmark(specs, all_models(), horizon=28, seed=137).run()


def test_series_generation_repeatable():
    spec = SERIES_LIBRARY["web_traffic"]
    a = make_series(spec, seed=137)
    b = make_series(spec, seed=137)
    assert np.array_equal(a, b)


def test_whole_benchmark_repeats():
    a, b = _run(), _run()
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.series == rb.series and ra.model == rb.model
        assert np.array_equal(np.array(ra.forecast), np.array(rb.forecast))
        assert ra.metrics == rb.metrics


def test_individual_models_deterministic():
    spec = SERIES_LIBRARY["energy_load"]
    train = make_series(spec, seed=137)[:-28]
    for ctor in (lambda: HoltWinters(spec.period),
                 lambda: RegressionForecaster(spec.period, lags=(1, 7))):
        a = ctor().fit(train).predict(28)
        b = ctor().fit(train).predict(28)
        assert np.array_equal(a, b)

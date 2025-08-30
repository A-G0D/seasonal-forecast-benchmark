import numpy as np
import pytest

from src.forecasters import (
    HoltWinters,
    NaiveForecaster,
    RegressionForecaster,
    SeasonalNaiveForecaster,
)
from src.series import SeriesSpec, make_series


@pytest.fixture
def clean_seasonal():
    # noise-free seasonal series so models with a seasonal component can
    # recover the structure closely
    spec = SeriesSpec(name="c", n=210, period=7, trend=0.2, noise=0.0,
                      season_amp=15.0)
    return make_series(spec), spec


def test_naive_repeats_last_value():
    f = NaiveForecaster().fit(np.array([1.0, 2.0, 3.0]))
    out = f.predict(4)
    assert np.allclose(out, 3.0)
    assert out.shape == (4,)


def test_seasonal_naive_cycles_last_season():
    y = np.array([10, 20, 30, 40, 11, 21, 31, 41], dtype=float)
    f = SeasonalNaiveForecaster(period=4).fit(y)
    out = f.predict(6)
    assert np.allclose(out[:4], [11, 21, 31, 41])
    assert np.allclose(out[4:], [11, 21])


def test_holt_winters_recovers_clean_series(clean_seasonal):
    y, spec = clean_seasonal
    train, test = y[:-14], y[-14:]
    f = HoltWinters(spec.period).fit(train)
    pred = f.predict(14)
    err = np.mean(np.abs(pred - test))
    assert err < 2.0  # near-perfect on a noise-free series


def test_holt_winters_beats_naive_on_trended_seasonal(clean_seasonal):
    y, spec = clean_seasonal
    train, test = y[:-14], y[-14:]
    hw = HoltWinters(spec.period).fit(train).predict(14)
    naive = NaiveForecaster().fit(train).predict(14)
    assert np.mean(np.abs(hw - test)) < np.mean(np.abs(naive - test))


def test_regression_runs_and_returns_horizon(clean_seasonal):
    y, spec = clean_seasonal
    train = y[:-14]
    f = RegressionForecaster(spec.period, lags=(1, 7)).fit(train)
    out = f.predict(14)
    assert out.shape == (14,)
    assert np.all(np.isfinite(out))


def test_regression_tracks_trend(clean_seasonal):
    y, spec = clean_seasonal
    train, test = y[:-14], y[-14:]
    pred = RegressionForecaster(spec.period, lags=(1, 7)).fit(train).predict(14)
    # forecast should land in the neighbourhood of the held-out level
    assert abs(pred.mean() - test.mean()) < 8.0


def test_predict_before_fit_raises():
    for model in (NaiveForecaster(), SeasonalNaiveForecaster(7)):
        with pytest.raises(RuntimeError):
            model.predict(3)


def test_horizon_must_be_positive():
    f = NaiveForecaster().fit(np.arange(10.0))
    with pytest.raises(ValueError):
        f.predict(0)


def test_holt_winters_param_validation():
    with pytest.raises(ValueError):
        HoltWinters(period=1)
    with pytest.raises(ValueError):
        HoltWinters(period=7, alpha=0.0)
    with pytest.raises(ValueError):
        HoltWinters(period=7, gamma=1.5)


def test_holt_winters_needs_two_seasons():
    f = HoltWinters(period=7)
    with pytest.raises(ValueError):
        f.fit(np.arange(10.0))


def test_regression_rejects_bad_lags():
    with pytest.raises(ValueError):
        RegressionForecaster(7, lags=())
    with pytest.raises(ValueError):
        RegressionForecaster(7, lags=(0, 3))


def test_fit_rejects_non_finite():
    f = NaiveForecaster()
    with pytest.raises(ValueError):
        f.fit(np.array([1.0, np.nan, 3.0]))

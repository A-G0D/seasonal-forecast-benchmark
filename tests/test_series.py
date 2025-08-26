import numpy as np
import pytest

from src.series import (
    SeriesSpec,
    SERIES_LIBRARY,
    list_series,
    make_series,
    train_test_split,
)


def test_make_series_length_and_dtype():
    spec = SeriesSpec(name="t", n=100, period=7)
    y = make_series(spec, seed=137)
    assert y.shape == (100,)
    assert y.dtype == np.float64


def test_make_series_is_deterministic():
    spec = SERIES_LIBRARY["retail_daily"]
    a = make_series(spec, seed=137)
    b = make_series(spec, seed=137)
    assert np.array_equal(a, b)


def test_different_seed_changes_noise_but_not_shape():
    spec = SeriesSpec(name="t", n=140, period=7, noise=5.0)
    a = make_series(spec, seed=1)
    b = make_series(spec, seed=2)
    assert not np.array_equal(a, b)
    # means stay close because only the noise term differs
    assert abs(a.mean() - b.mean()) < 2.0


def test_trend_increases_level():
    up = make_series(SeriesSpec(name="u", n=200, period=7, trend=0.5, noise=0.0))
    assert up[-7:].mean() > up[:7].mean()


def test_seasonal_profile_repeats():
    spec = SeriesSpec(name="s", n=70, period=7, noise=0.0, trend=0.0)
    y = make_series(spec)
    # with no trend/noise the series is exactly periodic
    assert np.allclose(y[:7], y[7:14])


def test_spiky_shape_has_a_peak():
    spec = SeriesSpec(name="s", n=70, period=7, noise=0.0, trend=0.0,
                      season_shape="spiky", season_amp=20.0)
    y = make_series(spec)
    period0 = y[:7]
    assert period0.max() - period0.min() > 10.0


@pytest.mark.parametrize("bad", [
    dict(n=0),
    dict(n=5, period=7),        # shorter than two periods
    dict(noise=-1.0),
    dict(season_shape="zigzag"),
    dict(period=0),
])
def test_invalid_specs_raise(bad):
    params = dict(name="bad", n=100, period=7)
    params.update(bad)
    with pytest.raises(ValueError):
        SeriesSpec(**params)


def test_train_test_split_shapes():
    y = np.arange(50.0)
    train, test = train_test_split(y, horizon=10)
    assert train.shape == (40,)
    assert test.shape == (10,)
    assert np.array_equal(test, np.arange(40.0, 50.0))


def test_train_test_split_rejects_bad_horizon():
    y = np.arange(10.0)
    with pytest.raises(ValueError):
        train_test_split(y, horizon=10)
    with pytest.raises(ValueError):
        train_test_split(y, horizon=0)


def test_library_names_sorted():
    assert list_series() == sorted(SERIES_LIBRARY)
    assert "retail_daily" in list_series()

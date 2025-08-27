import numpy as np
import pytest

from src import metrics as M


def test_perfect_forecast_is_zero_error():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.mae(y, y) == 0.0
    assert M.rmse(y, y) == 0.0
    assert M.mape(y, y) == pytest.approx(0.0, abs=1e-6)
    assert M.smape(y, y) == pytest.approx(0.0, abs=1e-6)


def test_mae_known_value():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([2.0, 2.0, 5.0])
    assert M.mae(yt, yp) == pytest.approx((1 + 0 + 2) / 3)


def test_rmse_known_value():
    yt = np.array([0.0, 0.0])
    yp = np.array([3.0, 4.0])
    assert M.rmse(yt, yp) == pytest.approx(np.sqrt((9 + 16) / 2))


def test_rmse_at_least_mae():
    rng = np.random.default_rng(0)
    yt = rng.normal(size=50)
    yp = rng.normal(size=50)
    assert M.rmse(yt, yp) >= M.mae(yt, yp) - 1e-9


def test_mape_known_value():
    yt = np.array([100.0, 200.0])
    yp = np.array([110.0, 180.0])
    # |10/100| + |20/200| = 0.1 + 0.1 -> 10%
    assert M.mape(yt, yp) == pytest.approx(10.0)


def test_smape_is_symmetric():
    a = np.array([10.0, 20.0])
    b = np.array([12.0, 18.0])
    assert M.smape(a, b) == pytest.approx(M.smape(b, a))


def test_smape_bounded():
    yt = np.array([1.0, 1.0])
    yp = np.array([-100.0, 200.0])
    assert 0.0 <= M.smape(yt, yp) <= 200.0


def test_mase_below_one_when_better_than_naive():
    # training series with strong step-to-step movement -> large naive scale
    train = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
    yt = np.array([5.0, 5.0])
    yp = np.array([5.1, 4.9])  # almost perfect
    assert M.mase(yt, yp, train, period=1) < 1.0


def test_mase_seasonal_scale():
    train = np.arange(20.0)
    yt = np.array([20.0, 21.0])
    yp = np.array([20.0, 21.0])
    assert M.mase(yt, yp, train, period=7) == pytest.approx(0.0)


def test_all_metrics_keys():
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([1.5, 2.5, 2.5, 4.5])
    train = np.arange(20.0)
    out = M.all_metrics(yt, yp, train, period=1)
    assert set(out) == {"mae", "rmse", "mape", "smape", "mase"}
    assert all(isinstance(v, float) for v in out.values())


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.mae(np.array([1.0, 2.0]), np.array([1.0]))


def test_empty_raises():
    with pytest.raises(ValueError):
        M.mae(np.array([]), np.array([]))

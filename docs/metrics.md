# Metric reference

All metrics are lower-is-better and operate on a forecast aligned to the
holdout window.

## MAE (mean absolute error)

```
mean(|y_true - y_pred|)
```

Same units as the series. Robust to the occasional large miss, and easy to read.

## RMSE (root mean squared error)

```
sqrt(mean((y_true - y_pred)^2))
```

Same units as the series, but penalises large errors more than MAE. Always
`>= MAE`.

## MAPE (mean absolute percentage error)

```
mean(|(y_true - y_pred) / y_true|) * 100
```

Scale-free percentage. Blows up when the truth is near zero, so a small floor is
applied to the denominator. Fine here because all series sit well above zero.

## sMAPE (symmetric MAPE)

```
mean(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)) * 100
```

Bounded in `[0, 200]` and symmetric in the two arguments, which avoids MAPE's
asymmetric over/under-forecast bias.

## MASE (mean absolute scaled error)

```
mean(|y_true - y_pred|) / mean(|y_train[t] - y_train[t-m]|)
```

Scales the forecast MAE by the in-sample MAE of an `m`-step naive model on the
training series, so below 1 means the forecast beats that naive baseline. With
`m` set to the seasonal period it's scaled against seasonal-naive, which is the
bar that actually matters here.

# Baseline vs improved forecasters

Holdout horizon: **28** steps. Seed: 137. Averaged over series: retail_daily, energy_load, web_traffic.

## Per-model averages

| model | MAE | RMSE | MAPE | SMAPE | MASE |
|---|---|---|---|---|---|
| naive | 77.598 | 94.698 | 8.026 | 7.825 | 2.582 |
| seasonal_naive | 36.290 | 45.221 | 3.486 | 3.482 | 1.113 |
| holt_winters | 39.408 | 46.799 | 3.237 | 3.189 | 1.002 |
| regression | 25.914 | 32.186 | 2.531 | 2.519 | 0.805 |

## Best on each side

| metric | best baseline | best improved | winner |
|---|---|---|---|
| MAE | 36.290 | 25.914 | improved |
| RMSE | 45.221 | 32.186 | improved |
| MAPE | 3.486 | 2.531 | improved |
| SMAPE | 3.482 | 2.519 | improved |
| MASE | 1.113 | 0.805 | improved |

The improved models win on 5 of 5 metrics against the strongest baseline.

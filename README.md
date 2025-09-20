# forecast-toolkit

A small, dependency-light toolkit for forecasting seasonal time series. It
generates deterministic synthetic series, fits a handful of forecasters behind
a common interface, and backtests them on a holdout window.

## Quick start

```bash
python eval/run_benchmark.py
```

This writes the comparison artifacts under `eval/` and a JSONL trace of every
fit to `logs/trace.jsonl`.

## Tests

```bash
python -m pytest
```

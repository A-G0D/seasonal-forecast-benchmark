# Design notes

## How it's split up

Each module is small enough to test on its own:

- `series` turns a `SeriesSpec` into a numpy array, and owns the validation
  (length vs period, noise sign, known season shape) so nothing downstream
  has to re-check.
- `base` is the `Forecaster` contract (`fit(y) -> self`,
  `predict(horizon) -> np.ndarray`). The shared input/horizon checks sit here
  as static helpers so the models validate the same way.
- `forecasters` is the four models. They don't import backtest or eval, which
  keeps the dependencies pointing one way.
- `metrics` is just functions over arrays, no model knowledge.
- `backtest` is the only place that deals with splits, scoring and logging.
- `registry` is where the baseline/improved split is decided.

## Why these models

The two baselines are the obvious reference points for seasonal data: a flat
naive carry-forward, and a seasonal-naive that repeats the last full cycle. Any
"improved" model has to clear the seasonal-naive bar to be worth anything.

On the improved side:

- Additive Holt-Winters tracks level, trend and a per-step seasonal component
  with three smoothing weights, in one recursive numpy pass (no stats library).
  Additive works here because the seasonal swing doesn't grow with the level.
- Ridge on calendar + lag features gets an explicit trend term (normalised time
  index), a one-hot of the position in the cycle, and a couple of lags (1 and a
  full period). It rolls forward at predict time, feeding its own output back in
  as the lags. The ridge penalty stops the one-hot + lag design from overfitting
  on short series.

## Determinism

I wanted runs to be reproducible, so series generation uses
`np.random.default_rng(seed)` rather than global numpy state, the seed is set
once at the top via `set_seed`, and none of the models carry hidden randomness
(Holt-Winters is deterministic and ridge has a closed-form solution). The
determinism tests just run the benchmark twice and check the forecasts and
metrics match.

## Logging

Each fit/score in the backtest emits one event through the `Observer`
(`event_id, timestamp, module, input, output, latency_ms`). In deterministic
mode the clock and ids are stable, so the JSONL trace diffs cleanly run to run.

"""Synthetic seasonal series, built additively as level + trend + seasonality
+ noise. Deterministic given a seed, so the forecasters can be checked against
structure we actually know. The named presets in SERIES_LIBRARY are what the
eval harness refers to by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class SeriesSpec:
    # season_shape: "sine" = smooth wave, "spiky" = weekday/weekend profile
    name: str
    n: int = 365
    period: int = 7
    base: float = 100.0
    trend: float = 0.05
    season_amp: float = 12.0
    noise: float = 3.0
    season_shape: str = "sine"

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.period < 1:
            raise ValueError("period must be >= 1")
        if self.n < 2 * self.period:
            raise ValueError(
                f"series '{self.name}' too short: need at least two full "
                f"periods ({2 * self.period}), got n={self.n}"
            )
        if self.noise < 0:
            raise ValueError("noise std must be non-negative")
        if self.season_shape not in ("sine", "spiky"):
            raise ValueError(f"unknown season_shape: {self.season_shape!r}")


def _seasonal_profile(spec: SeriesSpec) -> np.ndarray:
    """One period's worth of seasonal offsets, mean-centred."""
    idx = np.arange(spec.period)
    if spec.season_shape == "sine":
        prof = spec.season_amp * np.sin(2.0 * np.pi * idx / spec.period)
    else:
        # A blunt weekday/weekend style shape: most steps near zero, a couple
        # of pronounced peaks. Works for period 7 and degrades gracefully.
        prof = np.full(spec.period, -spec.season_amp * 0.4)
        peak = spec.period // 2
        prof[peak] = spec.season_amp
        if peak + 1 < spec.period:
            prof[peak + 1] = spec.season_amp * 0.7
    return prof - prof.mean()


def make_series(spec: SeriesSpec, seed: int = 137) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(spec.n, dtype=np.float64)

    level = spec.base + spec.trend * t
    profile = _seasonal_profile(spec)
    seasonal = np.take(profile, np.arange(spec.n) % spec.period)
    eps = rng.normal(0.0, spec.noise, size=spec.n)

    return level + seasonal + eps


def train_test_split(y: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a series into a training prefix and a holdout of ``horizon`` steps."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if horizon >= len(y):
        raise ValueError("horizon must be shorter than the series")
    return y[:-horizon], y[-horizon:]


SERIES_LIBRARY: Dict[str, SeriesSpec] = {
    "retail_daily": SeriesSpec(
        name="retail_daily",
        n=420,
        period=7,
        base=240.0,
        trend=0.18,
        season_amp=35.0,
        noise=8.0,
        season_shape="spiky",
    ),
    "energy_load": SeriesSpec(
        name="energy_load",
        n=480,
        period=7,
        base=900.0,
        trend=-0.10,
        season_amp=140.0,
        noise=22.0,
        season_shape="sine",
    ),
    "web_traffic": SeriesSpec(
        name="web_traffic",
        n=365,
        period=7,
        base=1500.0,
        trend=1.2,
        season_amp=220.0,
        noise=60.0,
        season_shape="spiky",
    ),
}


def list_series() -> list[str]:
    """Return the names of the bundled series presets."""
    return sorted(SERIES_LIBRARY)

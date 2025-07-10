"""Synthetic seasonal series for testing forecasters."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    n: int = 365
    period: int = 7
    base: float = 100.0
    trend: float = 0.05
    season_amp: float = 12.0
    noise: float = 3.0


def make_series(spec: SeriesSpec, seed: int = 137) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(spec.n, dtype=np.float64)
    level = spec.base + spec.trend * t
    idx = np.arange(spec.period)
    profile = spec.season_amp * np.sin(2.0 * np.pi * idx / spec.period)
    profile = profile - profile.mean()
    seasonal = np.take(profile, np.arange(spec.n) % spec.period)
    eps = rng.normal(0.0, spec.noise, size=spec.n)
    return level + seasonal + eps


def train_test_split(y: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    return y[:-horizon], y[-horizon:]

"""Seeding helpers so a run is reproducible. Returns the seed back so callers
can stash it in their logs."""
from __future__ import annotations

import os
import random
from typing import Optional


def set_seed(seed: int = 137) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    return seed


def seeded_rng(seed: Optional[int] = None) -> random.Random:
    """An isolated RNG so we don't stomp on the global random state."""
    return random.Random(137 if seed is None else seed)

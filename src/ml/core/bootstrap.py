# src/ml/core/bootstrap.py
"""Moving-block bootstrap for autocorrelated daily rows.

The 5-day target overlaps itself (lag-1 autocorrelation 0.92), so resampling single
rows would treat a few hundred dependent days as independent and report intervals that
are far too narrow. Whole blocks of consecutive days are resampled instead.
"""
from __future__ import annotations

import numpy as np

BLOCK = 20
REPLICATES = 2000
SEED = 20261011
# Below this share of usable replicates an interval describes the resampling, not the model.
MIN_VALID_FRACTION = 0.9


def block_indices(n: int, *, block: int = BLOCK, replicates: int = REPLICATES, seed: int = SEED) -> np.ndarray:
    """(replicates, n) row indices; each row of the result is a chain of whole blocks."""
    if n < 1 or not 1 <= block <= n or replicates < 1:
        raise ValueError("Need 1 <= block <= n and at least one replicate")
    rng = np.random.RandomState(seed)
    blocks_needed = -(-n // block)
    starts = rng.randint(0, n - block + 1, size=(replicates, blocks_needed))
    chains = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(replicates, -1)
    return chains[:, :n]


def replicate(statistic, arrays, indices) -> dict[str, np.ndarray]:
    """Evaluate statistic(*resampled arrays) -> {name: float | None} on every replicate.

    An undefined value (a resample with no positive day, say) is stored as NaN so the
    caller can count it instead of silently averaging over fewer replicates.
    """
    arrays = [np.asarray(array) for array in arrays]
    names = list(statistic(*arrays))
    values = {name: np.full(len(indices), np.nan) for name in names}
    for row, index in enumerate(indices):
        result = statistic(*(array[index] for array in arrays))
        for name in names:
            if result[name] is not None:
                values[name][row] = result[name]
    return values


def interval(point, values, *, level: float = 0.95) -> dict:
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    summary = {"point": point, "low": None, "high": None, "std": None,
               "n_valid": int(len(valid)), "n_replicates": int(len(values)), "level": level}
    if point is None or len(valid) < MIN_VALID_FRACTION * len(values):
        return summary
    tail = (1 - level) / 2
    low, high = np.quantile(valid, [tail, 1 - tail])
    summary.update(low=float(low), high=float(high), std=float(valid.std(ddof=1)))
    return summary


def difference(point_a, values_a, point_b, values_b, *, level: float = 0.95) -> dict:
    """Interval for b - a when both were scored on the same resampled rows."""
    point = None if point_a is None or point_b is None else point_b - point_a
    summary = interval(point, np.asarray(values_b) - np.asarray(values_a), level=level)
    summary["excludes_zero"] = (
        None if summary["low"] is None else bool(summary["low"] > 0 or summary["high"] < 0)
    )
    return summary

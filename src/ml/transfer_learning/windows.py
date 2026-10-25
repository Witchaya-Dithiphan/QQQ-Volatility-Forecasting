"""Causal lookback windows: the input for target day t is the feature rows t-L+1 .. t.

Context rows may come from before the split that owns day t (purge days included),
because those features were known at t's close; only *labels* are restricted by split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.contracts import FEATURE_COLUMNS


def timeline(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(dates, features) of the rows where all eight features exist, oldest first."""
    frame = frame.loc[frame[list(FEATURE_COLUMNS)].notna().all(axis=1)]
    dates = pd.to_datetime(frame["Date"]).to_numpy()
    if (np.diff(dates.astype("int64")) <= 0).any():
        raise ValueError("Timeline dates must be strictly increasing")
    return dates, frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)


def make_windows(dates: np.ndarray, features: np.ndarray, target_dates: np.ndarray,
                 lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(kept, positions, X): X[k] is features[pos-L+1 : pos+1] for target_dates[kept[k]].

    Target days without a full lookback of history are dropped (kept says which survived);
    a target day missing from the timeline is an error, never a silent skip.
    """
    positions = np.searchsorted(dates, target_dates)
    if (positions >= len(dates)).any() or not np.array_equal(dates[positions], target_dates):
        raise ValueError("A target date is not on the feature timeline")
    kept = np.flatnonzero(positions >= lookback - 1)
    pos = positions[kept]
    X = np.stack([features[p - lookback + 1:p + 1] for p in pos]) if len(pos) else np.empty((0, lookback, features.shape[1]))
    return kept, pos, X


def contiguous_windows(features: np.ndarray, lookback: int) -> np.ndarray:
    """Every full window of one series (for self-supervised pretraining)."""
    return np.stack([features[end - lookback + 1:end + 1] for end in range(lookback - 1, len(features))])

"""Pure Phase 2 affected-window primitives and detector compatibility exports.

Affected-window and split-clipping logic lives only in this module. Direct
detector calculation lives in :mod:`src.detect_spikes`; its established Series
APIs are re-exported here so existing callers do not duplicate or break logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_bool_dtype

from config import (
    SPIKE_FEATURE_FORWARD_REACH,
    SPIKE_TARGET_BACKWARD_REACH,
)
from src.detect_spikes import (
    ExtremeIQRThreshold,
    fit_extreme_iqr_threshold,
    flag_direct_spikes,
)

__all__ = [
    "AffectedWindow",
    "ExtremeIQRThreshold",
    "affected_windows",
    "build_affected_mask",
    "build_split_affected_masks",
    "fit_extreme_iqr_threshold",
    "flag_direct_spikes",
]


@dataclass(frozen=True, slots=True)
class AffectedWindow:
    """Requested and split-clipped positional bounds for one direct spike."""

    spike_position: int
    requested_start: int
    requested_end: int
    clipped_start: int
    clipped_end: int
    left_clipped: bool
    right_clipped: bool


def affected_windows(
    direct_spikes: pd.Series,
    *,
    rows_before: int = SPIKE_TARGET_BACKWARD_REACH,
    rows_after: int = SPIKE_FEATURE_FORWARD_REACH,
) -> tuple[AffectedWindow, ...]:
    """Describe inclusive affected windows clipped within one supplied split.

    Positions always refer to the original chronological split passed by the
    caller. Callers must build this metadata before filtering any rows.
    """
    if not isinstance(direct_spikes, pd.Series):
        raise TypeError("direct_spikes must be a pandas Series")
    if not is_bool_dtype(direct_spikes):
        raise ValueError("direct_spikes must have a boolean dtype")
    if direct_spikes.isna().any():
        raise ValueError("direct_spikes must not contain missing values")
    for name, value in (("rows_before", rows_before), ("rows_after", rows_after)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    split_length = len(direct_spikes)
    windows: list[AffectedWindow] = []
    for spike_position in np.flatnonzero(direct_spikes.to_numpy(dtype=bool)):
        requested_start = int(spike_position - rows_before)
        requested_end = int(spike_position + rows_after)
        windows.append(
            AffectedWindow(
                spike_position=int(spike_position),
                requested_start=requested_start,
                requested_end=requested_end,
                clipped_start=max(0, requested_start),
                clipped_end=min(split_length - 1, requested_end),
                left_clipped=requested_start < 0,
                right_clipped=requested_end >= split_length,
            )
        )
    return tuple(windows)


def build_affected_mask(
    direct_spikes: pd.Series,
    *,
    rows_before: int = SPIKE_TARGET_BACKWARD_REACH,
    rows_after: int = SPIKE_FEATURE_FORWARD_REACH,
) -> pd.Series:
    """Build the union of inclusive affected windows within one split."""
    windows = affected_windows(
        direct_spikes,
        rows_before=rows_before,
        rows_after=rows_after,
    )
    difference: NDArray[np.int64] = np.zeros(
        len(direct_spikes) + 1, dtype=np.int64
    )
    for window in windows:
        difference[window.clipped_start] += 1
        difference[window.clipped_end + 1] -= 1
    mask = np.cumsum(difference[:-1]) > 0
    return pd.Series(mask, index=direct_spikes.index, name="is_spike_affected")


def build_split_affected_masks(
    direct_spikes_by_split: Mapping[str, pd.Series],
) -> dict[str, pd.Series]:
    """Build independent masks so windows never propagate across split gaps."""
    return {
        split_name: build_affected_mask(direct_spikes)
        for split_name, direct_spikes in direct_spikes_by_split.items()
    }

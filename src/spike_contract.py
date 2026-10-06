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
    "AffectedMaskResult",
    "AffectedMaskSummary",
    "AffectedWindow",
    "ExtremeIQRThreshold",
    "affected_windows",
    "build_affected_mask",
    "build_affected_result",
    "build_split_affected_masks",
    "build_split_affected_results",
    "fit_extreme_iqr_threshold",
    "flag_direct_spikes",
]


BOUNDARY_LIMITATION = (
    "Features and targets were computed before splitting, so rows at a split "
    "boundary may still reference pre-split timeline data; within-split clipping "
    "is an operational experiment policy, not proof of zero cross-boundary influence."
)
RSI_LIMITATION = (
    "Wilder RSI is recursive, so spike influence may persist after s+19; the "
    "inclusive [s-5, s+19] window is the frozen operational definition."
)


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
    split_name: str = "unspecified"
    event_date: str | None = None
    clipped_start_date: str | None = None
    clipped_end_date: str | None = None


@dataclass(frozen=True, slots=True)
class AffectedMaskSummary:
    """Strict-JSON-safe aggregate metadata for one split-local mask.

    ``overlap_position_count`` counts split positions covered by more than one
    event window. It does not count window pairs or duplicate flag operations.
    """

    split_name: str
    direct_count: int
    affected_union_count: int
    left_clipped_window_count: int
    right_clipped_window_count: int
    overlap_position_count: int
    mask_scope: str = "within_split"
    cross_split_propagation: bool = False
    boundary_limitation: str = BOUNDARY_LIMITATION
    rsi_limitation: str = RSI_LIMITATION


@dataclass(frozen=True, slots=True)
class AffectedMaskResult:
    """Pure split-local flags plus event and aggregate boundary metadata."""

    split_name: str
    direct_flags: pd.Series
    affected_flags: pd.Series
    windows: tuple[AffectedWindow, ...]
    summary: AffectedMaskSummary


def _validated_split_name(split_name: str) -> str:
    """Validate and return a non-empty split identifier."""
    if not isinstance(split_name, str):
        raise TypeError("split_name must be a non-empty string")
    if not split_name.strip():
        raise ValueError("split_name must be a non-empty string")
    return split_name


def _validated_direct_flags(direct_spikes: pd.Series) -> pd.Series:
    """Return an independent boolean copy preserving positional order/index."""
    if not isinstance(direct_spikes, pd.Series):
        raise TypeError("direct_spikes must be a pandas Series")
    if not is_bool_dtype(direct_spikes):
        raise ValueError("direct_spikes must have a boolean dtype")
    if direct_spikes.isna().any():
        raise ValueError("direct_spikes must not contain missing values")
    return direct_spikes.astype(bool).copy().rename("is_spike")


def _validated_dates(
    dates: pd.Series | None,
    direct_spikes: pd.Series,
) -> pd.Series | None:
    """Return normalized dates after exact length and index-alignment checks."""
    if dates is None:
        return None
    if not isinstance(dates, pd.Series):
        raise TypeError("dates must be a pandas Series or None")
    if len(dates) != len(direct_spikes):
        raise ValueError("dates and direct_spikes must have the same length")
    if not dates.index.equals(direct_spikes.index):
        raise ValueError("dates index must exactly match direct_spikes index")
    if dates.isna().any():
        raise ValueError("dates must contain only valid dates")
    try:
        normalized = pd.to_datetime(dates.copy(), errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("dates must contain only valid dates") from error
    return normalized


def _validated_reaches(rows_before: int, rows_after: int) -> tuple[int, int]:
    """Validate the non-negative positional reach on each side of a spike."""
    for name, value in (("rows_before", rows_before), ("rows_after", rows_after)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    return int(rows_before), int(rows_after)


def _iso_date_at(dates: pd.Series | None, position: int) -> str | None:
    """Return one positional date as an ISO calendar-date string."""
    if dates is None:
        return None
    return pd.Timestamp(dates.iloc[position]).date().isoformat()


def _coverage_counts(
    split_length: int,
    windows: tuple[AffectedWindow, ...],
) -> NDArray[np.int64]:
    """Count how many inclusive windows cover each split-local position."""
    difference: NDArray[np.int64] = np.zeros(split_length + 1, dtype=np.int64)
    for window in windows:
        difference[window.clipped_start] += 1
        difference[window.clipped_end + 1] -= 1
    return np.cumsum(difference[:-1])


def affected_windows(
    direct_spikes: pd.Series,
    *,
    split_name: str = "unspecified",
    dates: pd.Series | None = None,
    rows_before: int = SPIKE_TARGET_BACKWARD_REACH,
    rows_after: int = SPIKE_FEATURE_FORWARD_REACH,
) -> tuple[AffectedWindow, ...]:
    """Describe inclusive affected windows clipped within one supplied split.

    Positions always refer to the original chronological split passed by the
    caller. Callers must build this metadata before filtering any rows.
    """
    selected_split = _validated_split_name(split_name)
    flags = _validated_direct_flags(direct_spikes)
    selected_dates = _validated_dates(dates, flags)
    selected_before, selected_after = _validated_reaches(rows_before, rows_after)

    split_length = len(flags)
    windows: list[AffectedWindow] = []
    for spike_position in np.flatnonzero(flags.to_numpy(dtype=bool)):
        requested_start = int(spike_position - selected_before)
        requested_end = int(spike_position + selected_after)
        clipped_start = max(0, requested_start)
        clipped_end = min(split_length - 1, requested_end)
        windows.append(
            AffectedWindow(
                spike_position=int(spike_position),
                requested_start=requested_start,
                requested_end=requested_end,
                clipped_start=clipped_start,
                clipped_end=clipped_end,
                left_clipped=requested_start < 0,
                right_clipped=requested_end >= split_length,
                split_name=selected_split,
                event_date=_iso_date_at(selected_dates, int(spike_position)),
                clipped_start_date=_iso_date_at(selected_dates, clipped_start),
                clipped_end_date=_iso_date_at(selected_dates, clipped_end),
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
    mask = _coverage_counts(len(direct_spikes), windows) > 0
    return pd.Series(mask, index=direct_spikes.index, name="is_spike_affected")


def build_affected_result(
    direct_spikes: pd.Series,
    *,
    split_name: str,
    dates: pd.Series | None = None,
    rows_before: int = SPIKE_TARGET_BACKWARD_REACH,
    rows_after: int = SPIKE_FEATURE_FORWARD_REACH,
) -> AffectedMaskResult:
    """Build immutable metadata and independent split-local boolean flags.

    Positions are always ``iloc``-style offsets into the supplied Original
    split. ``dates`` is optional, but when provided its length and index must
    exactly match ``direct_spikes``; no automatic index alignment occurs.

    The overlap metric is the number of positions covered by more than one
    clipped event window. This pure API performs no filesystem I/O and does not
    filter, concatenate, or mutate any split.
    """
    selected_split = _validated_split_name(split_name)
    flags = _validated_direct_flags(direct_spikes)
    windows = affected_windows(
        flags,
        split_name=selected_split,
        dates=dates,
        rows_before=rows_before,
        rows_after=rows_after,
    )
    coverage = _coverage_counts(len(flags), windows)
    affected = pd.Series(
        coverage > 0,
        index=flags.index,
        name="is_spike_affected",
    )
    summary = AffectedMaskSummary(
        split_name=selected_split,
        direct_count=int(flags.sum()),
        affected_union_count=int(affected.sum()),
        left_clipped_window_count=sum(window.left_clipped for window in windows),
        right_clipped_window_count=sum(window.right_clipped for window in windows),
        overlap_position_count=int((coverage > 1).sum()),
    )
    return AffectedMaskResult(
        split_name=selected_split,
        direct_flags=flags,
        affected_flags=affected,
        windows=windows,
        summary=summary,
    )


def build_split_affected_masks(
    direct_spikes_by_split: Mapping[str, pd.Series],
) -> dict[str, pd.Series]:
    """Build independent masks so windows never propagate across split gaps."""
    return {
        split_name: build_affected_mask(direct_spikes)
        for split_name, direct_spikes in direct_spikes_by_split.items()
    }


def build_split_affected_results(
    direct_spikes_by_split: Mapping[str, pd.Series],
    *,
    dates_by_split: Mapping[str, pd.Series] | None = None,
) -> dict[str, AffectedMaskResult]:
    """Build rich results independently so no window crosses a split or gap."""
    if not isinstance(direct_spikes_by_split, Mapping):
        raise TypeError("direct_spikes_by_split must be a mapping")
    if dates_by_split is not None:
        if not isinstance(dates_by_split, Mapping):
            raise TypeError("dates_by_split must be a mapping or None")
        if set(dates_by_split) != set(direct_spikes_by_split):
            raise ValueError(
                "dates_by_split and direct_spikes_by_split must use the same split names"
            )
    return {
        split_name: build_affected_result(
            direct_spikes,
            split_name=split_name,
            dates=None if dates_by_split is None else dates_by_split[split_name],
        )
        for split_name, direct_spikes in direct_spikes_by_split.items()
    }

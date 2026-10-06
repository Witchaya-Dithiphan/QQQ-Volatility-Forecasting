"""Pure Phase 2 spike-rule primitives shared by tests and future pipelines.

This module freezes the primary rule mechanics without creating experiment
datasets or writing artifacts. Thresholds remain fitted from caller-supplied
Original Train returns and must never be hard-coded from an audit result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from config import (
    PRIMARY_SPIKE_IQR_MULTIPLIER,
    SPIKE_AFFECTED_ROWS_AFTER,
    SPIKE_AFFECTED_ROWS_BEFORE,
)


@dataclass(frozen=True, slots=True)
class ExtremeIQRThreshold:
    """Train-fitted metadata for one strict Extreme-IQR threshold."""

    q1: float
    q3: float
    iqr: float
    multiplier: float
    threshold: float
    quantile_method: str = "linear"
    comparison_rule: str = "abs(return_1d) > threshold"
    source_split: str = "original_train"


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


def _finite_numeric_series(values: pd.Series, *, name: str) -> pd.Series:
    """Return a float copy after validating one non-empty numeric Series."""
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if values.empty:
        raise ValueError(f"{name} must not be empty")
    if not is_numeric_dtype(values):
        raise ValueError(f"{name} must have a numeric dtype")
    numeric = values.astype(float).copy()
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values")
    return numeric


def fit_extreme_iqr_threshold(
    original_train_returns: pd.Series,
    *,
    multiplier: float = PRIMARY_SPIKE_IQR_MULTIPLIER,
) -> ExtremeIQRThreshold:
    """Fit the frozen Extreme-IQR rule from Original Train returns only.

    Args:
        original_train_returns: ``return_1d`` from the complete Original Train
            split, before any spike-affected rows are filtered.
        multiplier: Positive IQR multiplier. Phase 2 primary analysis uses 3.0.

    Returns:
        Immutable fitted threshold metadata.

    Raises:
        TypeError: If the returns are not a Series.
        ValueError: If returns or the multiplier violate the contract.
    """
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, Real)
        or not np.isfinite(multiplier)
        or multiplier <= 0
    ):
        raise ValueError("multiplier must be a finite positive number")

    abs_return = _finite_numeric_series(
        original_train_returns, name="original_train_returns"
    ).abs()
    q1 = float(abs_return.quantile(0.25, interpolation="linear"))
    q3 = float(abs_return.quantile(0.75, interpolation="linear"))
    iqr = q3 - q1
    threshold = q3 + float(multiplier) * iqr
    return ExtremeIQRThreshold(
        q1=q1,
        q3=q3,
        iqr=iqr,
        multiplier=float(multiplier),
        threshold=threshold,
    )


def flag_direct_spikes(
    returns: pd.Series,
    fitted: ExtremeIQRThreshold,
) -> pd.Series:
    """Flag direct spikes with the fitted strict ``>`` comparison."""
    numeric = _finite_numeric_series(returns, name="returns")
    return numeric.abs().gt(fitted.threshold).rename("is_spike")


def affected_windows(
    direct_spikes: pd.Series,
    *,
    rows_before: int = SPIKE_AFFECTED_ROWS_BEFORE,
    rows_after: int = SPIKE_AFFECTED_ROWS_AFTER,
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
    rows_before: int = SPIKE_AFFECTED_ROWS_BEFORE,
    rows_after: int = SPIKE_AFFECTED_ROWS_AFTER,
) -> pd.Series:
    """Build the union of inclusive affected windows within one split."""
    windows = affected_windows(
        direct_spikes,
        rows_before=rows_before,
        rows_after=rows_after,
    )
    difference = np.zeros(len(direct_spikes) + 1, dtype=np.int64)
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

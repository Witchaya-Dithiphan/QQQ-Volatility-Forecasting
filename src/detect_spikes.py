"""Pure, Train-fitted direct-spike detection for Phase 2.

This module is the single owner of the Extreme-IQR calculation and strict
direct-spike comparison. It performs no filesystem I/O and never creates the
affected windows owned by :mod:`src.spike_contract`.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from config import (
    PRIMARY_SPIKE_IQR_MULTIPLIER,
    SPIKE_COMPARISON_FORMULA,
    SPIKE_QUANTILE_METHOD,
    SPIKE_RETURN_COLUMN,
)


@dataclass(frozen=True, slots=True)
class ExtremeIQRThreshold:
    """Immutable metadata for one Original-Train-fitted detector."""

    q1: float
    q3: float
    iqr: float
    multiplier: float
    threshold: float
    quantile_method: str = SPIKE_QUANTILE_METHOD
    comparison_rule: str = SPIKE_COMPARISON_FORMULA
    source_split: str = "original_train"
    return_column: str = SPIKE_RETURN_COLUMN


FittedSpikeDetector = ExtremeIQRThreshold


def _validated_return_copy(data: pd.DataFrame, *, name: str) -> pd.Series:
    """Return a finite float copy of the configured return column.

    Args:
        data: Candidate detector input.
        name: Human-readable input name used in validation errors.

    Returns:
        An independent float Series preserving the input index.

    Raises:
        TypeError: If ``data`` is not a DataFrame.
        ValueError: If the input is empty, lacks ``return_1d``, or contains an
            invalid return dtype/value.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if SPIKE_RETURN_COLUMN not in data.columns:
        raise ValueError(f"{name} is missing required column: {SPIKE_RETURN_COLUMN}")
    if data.empty:
        raise ValueError(f"{name} must not be empty")
    returns = data[SPIKE_RETURN_COLUMN]
    if not is_numeric_dtype(returns):
        raise ValueError(f"{name}.{SPIKE_RETURN_COLUMN} must have a numeric dtype")
    numeric = returns.astype(float).copy()
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(
            f"{name}.{SPIKE_RETURN_COLUMN} must contain only finite values"
        )
    return numeric


def _validated_multiplier(multiplier: float) -> float:
    """Return a float multiplier after enforcing the frozen numeric contract."""
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, Real)
        or not np.isfinite(multiplier)
        or multiplier <= 0
    ):
        raise ValueError("multiplier must be a finite positive number")
    return float(multiplier)


def fit_primary_spike_detector(
    original_train: pd.DataFrame,
    *,
    multiplier: float = PRIMARY_SPIKE_IQR_MULTIPLIER,
) -> ExtremeIQRThreshold:
    """Fit the primary Extreme-IQR detector from verified Original Train.

    The caller is responsible for supplying the M1-verified Original Train;
    provenance cannot be inferred from an in-memory DataFrame. Validation and
    Test data must never be passed to this fit API.

    Args:
        original_train: Complete labeled Original Train before row filtering.
        multiplier: Finite positive IQR multiplier; the primary rule uses 3.0.

    Returns:
        Immutable fitted detector metadata.

    Raises:
        TypeError: If the input is not a DataFrame.
        ValueError: If the input, multiplier, or derived metadata is invalid.
    """
    selected_multiplier = _validated_multiplier(multiplier)
    absolute_returns = _validated_return_copy(
        original_train, name="original_train"
    ).abs()
    q1 = float(
        absolute_returns.quantile(0.25, interpolation=SPIKE_QUANTILE_METHOD)
    )
    q3 = float(
        absolute_returns.quantile(0.75, interpolation=SPIKE_QUANTILE_METHOD)
    )
    iqr = q3 - q1
    with np.errstate(over="ignore", invalid="ignore"):
        threshold = float(np.add(q3, np.multiply(selected_multiplier, iqr)))
    if not np.isfinite([q1, q3, iqr, threshold]).all():
        raise ValueError("fit produced non-finite detector metadata")
    return ExtremeIQRThreshold(
        q1=q1,
        q3=q3,
        iqr=iqr,
        multiplier=selected_multiplier,
        threshold=threshold,
    )


def apply_spike_detector(
    data: pd.DataFrame,
    fitted: ExtremeIQRThreshold,
) -> pd.Series:
    """Apply one fitted Train threshold without refitting on the supplied split.

    Args:
        data: Train, Validation, or Test rows to flag without modification.
        fitted: Immutable metadata previously fitted from Original Train.

    Returns:
        Boolean ``is_spike`` Series preserving the input index and order.

    Raises:
        TypeError: If ``data`` or ``fitted`` has the wrong type.
        ValueError: If return values or the fitted threshold are invalid.
    """
    if not isinstance(fitted, ExtremeIQRThreshold):
        raise TypeError("fitted must be an ExtremeIQRThreshold")
    if not np.isfinite(fitted.threshold):
        raise ValueError("fitted threshold must be finite")
    returns = _validated_return_copy(data, name="data")
    return returns.abs().gt(fitted.threshold).rename("is_spike")


def fit_extreme_iqr_threshold(
    original_train_returns: pd.Series,
    *,
    multiplier: float = PRIMARY_SPIKE_IQR_MULTIPLIER,
) -> ExtremeIQRThreshold:
    """Compatibility API delegating Series input to the canonical fit API."""
    if not isinstance(original_train_returns, pd.Series):
        raise TypeError("original_train_returns must be a pandas Series")
    frame = original_train_returns.to_frame(name=SPIKE_RETURN_COLUMN)
    return fit_primary_spike_detector(frame, multiplier=multiplier)


def flag_direct_spikes(
    returns: pd.Series,
    fitted: ExtremeIQRThreshold,
) -> pd.Series:
    """Compatibility API delegating Series input to the canonical apply API."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    frame = returns.to_frame(name=SPIKE_RETURN_COLUMN)
    return apply_spike_detector(frame, fitted)

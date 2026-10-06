"""Synthetic and read-only audit tests for Phase 2 affected-window policies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from config import SPIKE_INPUT_CONTRACT_REPORT_PATH, TRAIN_LABELED_DATA_PATH
from src.detect_spikes import apply_spike_detector, fit_primary_spike_detector
from src.spike_contract import (
    affected_windows,
    build_affected_mask,
    build_split_affected_masks,
    fit_extreme_iqr_threshold,
    flag_direct_spikes,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _direct_mask(length: int, *spike_positions: int) -> pd.Series:
    """Create one chronological boolean fixture with direct spike positions."""
    mask = pd.Series(False, index=pd.RangeIndex(length), dtype=bool)
    mask.iloc[list(spike_positions)] = True
    return mask


def test_primary_threshold_is_fitted_from_supplied_original_train_only() -> None:
    train = pd.Series([0.00, 0.01, -0.02, 0.03, -0.04, 0.05])
    fitted = fit_extreme_iqr_threshold(train)
    assert fitted.source_split == "original_train"
    assert fitted.multiplier == 3.0
    assert fitted.threshold == pytest.approx(
        fitted.q3 + fitted.multiplier * fitted.iqr
    )


def test_direct_spike_comparison_is_strict_and_uses_absolute_returns() -> None:
    fitted = fit_extreme_iqr_threshold(pd.Series([0.0, 0.1, 0.2, 0.3]))
    values = pd.Series(
        [fitted.threshold, -fitted.threshold, fitted.threshold + 1e-12]
    )
    assert flag_direct_spikes(values, fitted).tolist() == [False, False, True]


def test_spike_at_split_start_is_left_clipped() -> None:
    direct = _direct_mask(30, 0)
    windows = affected_windows(direct)
    affected = build_affected_mask(direct)
    assert windows[0].requested_start == -5
    assert windows[0].clipped_start == 0
    assert windows[0].clipped_end == 19
    assert windows[0].left_clipped
    assert not windows[0].right_clipped
    assert affected.iloc[:20].all()
    assert not affected.iloc[20:].any()


def test_spike_at_split_end_is_right_clipped() -> None:
    direct = _direct_mask(30, 29)
    windows = affected_windows(direct)
    affected = build_affected_mask(direct)
    assert windows[0].requested_start == 24
    assert windows[0].requested_end == 48
    assert windows[0].clipped_end == 29
    assert windows[0].right_clipped
    assert not affected.iloc[:24].any()
    assert affected.iloc[24:].all()


def test_overlapping_windows_are_counted_once() -> None:
    direct = _direct_mask(40, 10, 15)
    affected = build_affected_mask(direct)
    assert affected.iloc[5:35].all()
    assert int(affected.sum()) == 30


def test_every_direct_spike_is_affected() -> None:
    direct = _direct_mask(50, 0, 10, 25, 49)
    affected = build_affected_mask(direct)
    assert affected.loc[direct].all()


def test_primary_train_audit_has_213_affected_rows_and_contains_all_spikes() -> None:
    if not TRAIN_LABELED_DATA_PATH.is_file():
        pytest.skip("Ignored production Train CSV is unavailable")
    if not SPIKE_INPUT_CONTRACT_REPORT_PATH.is_file():
        pytest.skip("Generated M1 input contract is unavailable")

    contract = json.loads(SPIKE_INPUT_CONTRACT_REPORT_PATH.read_text(encoding="utf-8"))
    expected_hash = contract["inputs"]["train"]["sha256"]
    assert _sha256(TRAIN_LABELED_DATA_PATH) == expected_hash
    train = pd.read_csv(TRAIN_LABELED_DATA_PATH)

    fitted = fit_primary_spike_detector(train)
    direct = apply_spike_detector(train, fitted)
    affected = build_affected_mask(direct)

    assert int(affected.sum()) == 213
    assert affected.loc[direct].all()


def test_split_masks_do_not_propagate_across_gaps_or_boundaries() -> None:
    train = pd.Series(
        False,
        index=pd.date_range("2024-01-01", periods=10, freq="B"),
        dtype=bool,
    )
    train.iloc[-1] = True
    validation = pd.Series(
        False,
        index=pd.date_range("2024-03-01", periods=10, freq="B"),
        dtype=bool,
    )
    test = pd.Series(
        False,
        index=pd.date_range("2024-06-03", periods=10, freq="B"),
        dtype=bool,
    )
    direct_by_split = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    masks = build_split_affected_masks(direct_by_split)
    assert int(masks["train"].sum()) == 6
    assert not masks["validation"].any()
    assert not masks["test"].any()
    pd.testing.assert_index_equal(masks["validation"].index, validation.index)
    pd.testing.assert_index_equal(masks["test"].index, test.index)


def test_original_positions_are_used_before_filtering() -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=35, freq="B"))
    direct = _direct_mask(35, 10)
    affected = build_affected_mask(direct)
    kept_dates = dates.loc[~affected].reset_index(drop=True)
    assert dates.loc[5:29].tolist() == dates.loc[affected].tolist()
    assert kept_dates.tolist() == [*dates.iloc[:5], *dates.iloc[30:]]


@pytest.mark.parametrize(
    "direct_spikes",
    [
        pd.Series([0, 1], dtype="int64"),
        pd.Series([True, pd.NA], dtype="boolean"),
    ],
)
def test_affected_mask_rejects_invalid_direct_flags(
    direct_spikes: pd.Series,
) -> None:
    with pytest.raises(ValueError):
        build_affected_mask(direct_spikes)


@pytest.mark.parametrize(
    "returns",
    [
        pd.Series(dtype=float),
        pd.Series([0.0, float("nan")]),
        pd.Series([0.0, float("inf")]),
        pd.Series(["not", "numeric"]),
    ],
)
def test_threshold_fit_rejects_invalid_train_returns(returns: pd.Series) -> None:
    with pytest.raises((TypeError, ValueError)):
        fit_extreme_iqr_threshold(returns)

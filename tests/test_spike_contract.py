"""Synthetic and read-only audit tests for Phase 2 affected-window policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from config import (
    SPIKE_INPUT_CONTRACT_REPORT_PATH,
    TEST_LABELED_DATA_PATH,
    TRAIN_LABELED_DATA_PATH,
    VALIDATION_LABELED_DATA_PATH,
)
from src.detect_spikes import apply_spike_detector, fit_primary_spike_detector
from src.spike_contract import (
    affected_windows,
    build_affected_mask,
    build_affected_result,
    build_split_affected_masks,
    build_split_affected_results,
    fit_extreme_iqr_threshold,
    flag_direct_spikes,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load_m1_verified_split(split_name: str, path: Path) -> pd.DataFrame:
    if not path.is_file():
        pytest.skip(f"Ignored production {split_name} CSV is unavailable")
    if not SPIKE_INPUT_CONTRACT_REPORT_PATH.is_file():
        pytest.skip("Generated M1 input contract is unavailable")
    contract = json.loads(SPIKE_INPUT_CONTRACT_REPORT_PATH.read_text(encoding="utf-8"))
    assert _sha256(path) == contract["inputs"][split_name]["sha256"]
    return pd.read_csv(path)


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
    assert fitted.threshold == pytest.approx(fitted.q3 + fitted.multiplier * fitted.iqr)


def test_direct_spike_comparison_is_strict_and_uses_absolute_returns() -> None:
    fitted = fit_extreme_iqr_threshold(pd.Series([0.0, 0.1, 0.2, 0.3]))
    values = pd.Series([fitted.threshold, -fitted.threshold, fitted.threshold + 1e-12])
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


def test_overlap_metadata_counts_positions_covered_by_multiple_windows() -> None:
    result = build_affected_result(_direct_mask(40, 10, 15), split_name="train")

    assert result.summary.direct_count == 2
    assert result.summary.affected_union_count == 30
    assert result.summary.overlap_position_count == 20


def test_non_overlapping_windows_have_zero_overlap_positions() -> None:
    result = build_affected_result(_direct_mask(60, 5, 30), split_name="train")

    assert result.summary.affected_union_count == 50
    assert result.summary.overlap_position_count == 0


def test_unclipped_window_is_inclusive_and_has_25_positions() -> None:
    direct = _direct_mask(40, 10)
    window = affected_windows(direct, split_name="train")[0]
    result = build_affected_result(direct, split_name="train")

    assert (window.clipped_start, window.clipped_end) == (5, 29)
    assert result.summary.affected_union_count == 25


@pytest.mark.parametrize(
    ("spike_position", "left_clipped", "right_clipped"),
    [
        pytest.param(4, True, False, id="one-before-left-safe-boundary"),
        pytest.param(5, False, False, id="exact-left-safe-boundary"),
        pytest.param(20, False, False, id="exact-right-safe-boundary"),
        pytest.param(21, False, True, id="one-after-right-safe-boundary"),
    ],
)
def test_exact_and_exceeded_clip_boundaries(
    spike_position: int,
    left_clipped: bool,
    right_clipped: bool,
) -> None:
    window = affected_windows(_direct_mask(40, spike_position))[0]

    assert window.left_clipped is left_clipped
    assert window.right_clipped is right_clipped


def test_every_direct_spike_is_affected() -> None:
    direct = _direct_mask(50, 0, 10, 25, 49)
    affected = build_affected_mask(direct)
    assert affected.loc[direct].all()


def test_empty_direct_flags_return_empty_mask_and_zero_metadata() -> None:
    direct = pd.Series(dtype=bool, name="is_spike")

    result = build_affected_result(direct, split_name="validation")

    assert result.direct_flags.empty
    assert result.affected_flags.empty
    assert result.windows == ()
    assert result.summary.direct_count == 0
    assert result.summary.affected_union_count == 0
    assert result.summary.overlap_position_count == 0


def test_primary_train_audit_has_213_affected_rows_and_contains_all_spikes() -> None:
    train = _load_m1_verified_split("train", TRAIN_LABELED_DATA_PATH)

    fitted = fit_primary_spike_detector(train)
    direct = apply_spike_detector(train, fitted)
    result = build_affected_result(
        direct,
        split_name="train",
        dates=train["Date"],
    )

    assert result.summary.direct_count == 19
    assert result.summary.affected_union_count == 213
    assert result.summary.left_clipped_window_count == 0
    assert result.summary.right_clipped_window_count == 0
    assert result.affected_flags.loc[result.direct_flags].all()


def test_primary_validation_and_test_diagnostics_use_train_threshold() -> None:
    train = _load_m1_verified_split("train", TRAIN_LABELED_DATA_PATH)
    validation = _load_m1_verified_split("validation", VALIDATION_LABELED_DATA_PATH)
    test = _load_m1_verified_split("test", TEST_LABELED_DATA_PATH)
    validation_before = validation.copy(deep=True)
    test_before = test.copy(deep=True)
    fitted = fit_primary_spike_detector(train)

    validation_result = build_affected_result(
        apply_spike_detector(validation, fitted),
        split_name="validation",
        dates=validation["Date"],
    )
    test_result = build_affected_result(
        apply_spike_detector(test, fitted),
        split_name="test",
        dates=test["Date"],
    )

    assert (
        validation_result.summary.direct_count,
        validation_result.summary.affected_union_count,
    ) == (0, 0)
    assert (
        test_result.summary.direct_count,
        test_result.summary.affected_union_count,
    ) == (4, 54)
    assert validation_result.summary.left_clipped_window_count == 0
    assert validation_result.summary.right_clipped_window_count == 0
    assert test_result.summary.left_clipped_window_count == 0
    assert test_result.summary.right_clipped_window_count == 0
    pd.testing.assert_frame_equal(validation, validation_before)
    pd.testing.assert_frame_equal(test, test_before)


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


def test_split_results_keep_metadata_and_masks_isolated() -> None:
    direct_by_split = {
        "train": _direct_mask(10, 9),
        "validation": _direct_mask(10),
        "test": _direct_mask(10, 0),
    }

    results = build_split_affected_results(direct_by_split)

    assert results["train"].summary.affected_union_count == 6
    assert results["validation"].summary.affected_union_count == 0
    assert results["test"].summary.affected_union_count == 10
    assert results["train"].summary.right_clipped_window_count == 1
    assert results["test"].summary.left_clipped_window_count == 1


def test_original_positions_are_used_before_filtering() -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=35, freq="B"))
    direct = _direct_mask(35, 10)
    affected = build_affected_mask(direct)
    kept_dates = dates.loc[~affected].reset_index(drop=True)
    assert dates.loc[5:29].tolist() == dates.loc[affected].tolist()
    assert kept_dates.tolist() == [*dates.iloc[:5], *dates.iloc[30:]]


def test_non_range_index_uses_positions_and_preserves_alignment() -> None:
    index = pd.Index(["a", "b", "c", "d", "e", "f", "g"])
    direct = pd.Series(
        [False, False, False, True, False, False, False],
        index=index,
        dtype=bool,
        name="source_flags",
    )

    result = build_affected_result(
        direct,
        split_name="train",
        rows_before=1,
        rows_after=2,
    )

    assert result.affected_flags.tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    pd.testing.assert_index_equal(result.direct_flags.index, index)
    pd.testing.assert_index_equal(result.affected_flags.index, index)
    assert result.direct_flags.name == "is_spike"
    assert result.affected_flags.name == "is_spike_affected"


def test_event_metadata_contains_bounds_dates_and_json_safe_values() -> None:
    index = pd.Index([10, 20, 30, 40, 50, 60])
    direct = pd.Series([True, False, False, False, False, True], index=index)
    dates = pd.Series(
        pd.date_range("2024-01-01", periods=6, freq="B"),
        index=index,
        name="Date",
    )

    result = build_affected_result(
        direct,
        split_name="test",
        dates=dates,
        rows_before=1,
        rows_after=2,
    )

    first, last = result.windows
    assert first.split_name == "test"
    assert first.spike_position == 0
    assert first.event_date == "2024-01-01"
    assert first.requested_start == -1
    assert first.clipped_start == 0
    assert first.clipped_end == 2
    assert first.clipped_start_date == "2024-01-01"
    assert first.clipped_end_date == "2024-01-03"
    assert last.event_date == "2024-01-08"
    assert last.clipped_start_date == "2024-01-05"
    assert last.clipped_end_date == "2024-01-08"
    assert result.summary.mask_scope == "within_split"
    assert result.summary.cross_split_propagation is False
    assert "pre-split" in result.summary.boundary_limitation
    assert "Wilder RSI" in result.summary.rsi_limitation
    json.dumps(
        {
            "windows": [asdict(window) for window in result.windows],
            "summary": asdict(result.summary),
        },
        allow_nan=False,
    )


def test_flags_and_dates_are_not_mutated() -> None:
    direct = _direct_mask(8, 3)
    dates = pd.Series(pd.date_range("2024-01-01", periods=8, freq="B"))
    direct_before = direct.copy(deep=True)
    dates_before = dates.copy(deep=True)

    build_affected_result(direct, split_name="train", dates=dates)

    pd.testing.assert_series_equal(direct, direct_before)
    pd.testing.assert_series_equal(dates, dates_before)


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        (pd.Series(pd.date_range("2024-01-01", periods=2)), "same length"),
        (
            pd.Series(pd.date_range("2024-01-01", periods=3), index=[1, 2, 3]),
            "index must exactly match",
        ),
        (pd.Series(["2024-01-01", "bad-date", "2024-01-03"]), "valid dates"),
    ],
)
def test_dates_must_match_flags_and_be_valid(
    dates: pd.Series,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_affected_result(_direct_mask(3, 1), split_name="train", dates=dates)


def test_split_dates_must_have_exactly_the_same_keys() -> None:
    with pytest.raises(ValueError, match="same split names"):
        build_split_affected_results(
            {"train": _direct_mask(3, 1)},
            dates_by_split={
                "validation": pd.Series(pd.date_range("2024-01-01", periods=3))
            },
        )


@pytest.mark.parametrize("split_name", ["", " ", 1, None])
def test_split_name_must_be_a_non_empty_string(split_name: object) -> None:
    with pytest.raises((TypeError, ValueError), match="split_name"):
        build_affected_result(
            _direct_mask(3, 1),
            split_name=split_name,  # type: ignore[arg-type]
        )


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

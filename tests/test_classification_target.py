"""Tests for train-only high-volatility classification targets."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.build_targets as target_module
from src.build_features import FEATURE_COLUMNS
from src.build_targets import (
    CLASSIFICATION_TARGET,
    HIGH_VOLATILITY_QUANTILE,
    TARGET_COLUMN,
    add_classification_target,
    calculate_high_volatility_threshold,
    run_classification_target_pipeline,
    validate_classification_splits,
)


def _split_frame(start: str, targets: list[float]) -> pd.DataFrame:
    """Create a complete processed split with deterministic values."""
    rows = len(targets)
    positions = np.arange(rows, dtype=float)
    close = 100 + positions
    frame = pd.DataFrame(
        {
            "Date": pd.date_range(start, periods=rows, freq="B"),
            "Open": close,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": 1_000_000 + positions,
        }
    )
    for number, column in enumerate(FEATURE_COLUMNS, start=1):
        frame[column] = positions / 100 + number
    frame[TARGET_COLUMN] = targets
    return frame


def _three_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ordered, non-overlapping train, validation, and test frames."""
    train = _split_frame("2020-01-01", [0.10, 0.20, 0.30, 0.40, 0.50])
    validation = _split_frame("2021-01-01", [10.0, 20.0])
    test = _split_frame("2022-01-01", [100.0, 200.0])
    return train, validation, test


def test_threshold_is_train_75th_percentile() -> None:
    train, _, _ = _three_splits()
    expected = float(np.quantile(train[TARGET_COLUMN].to_numpy(), 0.75, method="linear"))
    assert calculate_high_volatility_threshold(train) == pytest.approx(expected)
    assert HIGH_VOLATILITY_QUANTILE == 0.75


def test_threshold_uses_linear_interpolation() -> None:
    train = _split_frame("2020-01-01", [0.0, 10.0])
    assert calculate_high_volatility_threshold(train) == pytest.approx(7.5)


def test_validation_values_do_not_affect_threshold() -> None:
    train, validation, _ = _three_splits()
    baseline = calculate_high_volatility_threshold(train)
    validation[TARGET_COLUMN] *= 1_000_000
    assert calculate_high_volatility_threshold(train) == baseline


def test_test_values_do_not_affect_threshold() -> None:
    train, _, test = _three_splits()
    baseline = calculate_high_volatility_threshold(train)
    test[TARGET_COLUMN] *= 1_000_000
    assert calculate_high_volatility_threshold(train) == baseline


def test_threshold_is_calculated_from_train_only() -> None:
    train, validation, test = _three_splits()
    threshold = calculate_high_volatility_threshold(train)
    combined_threshold = pd.concat([train, validation, test])[TARGET_COLUMN].quantile(0.75)
    assert threshold == pytest.approx(0.40)
    assert threshold != pytest.approx(combined_threshold)


@pytest.mark.parametrize(
    ("value", "expected_label"),
    [(0.51, 1), (0.49, 0), (0.50, 0)],
)
def test_classification_comparison_rule(value: float, expected_label: int) -> None:
    frame = _split_frame("2020-01-01", [value])
    result = add_classification_target(frame, threshold=0.50)
    assert result[CLASSIFICATION_TARGET].iloc[0] == expected_label


def test_classification_target_contains_only_zero_one_and_no_missing() -> None:
    train, _, _ = _three_splits()
    result = add_classification_target(train, threshold=0.30)
    assert set(result[CLASSIFICATION_TARGET].unique()).issubset({0, 1})
    assert not result[CLASSIFICATION_TARGET].isna().any()
    assert result[CLASSIFICATION_TARGET].dtype == "int8"


def test_add_classification_target_does_not_mutate_input() -> None:
    train, _, _ = _three_splits()
    snapshot = train.copy(deep=True)
    add_classification_target(train, 0.30)
    pd.testing.assert_frame_equal(train, snapshot)


def test_row_order_count_and_existing_columns_are_preserved() -> None:
    train, _, _ = _three_splits()
    result = add_classification_target(train, 0.30)
    assert len(result) == len(train)
    assert list(result.columns) == list(train.columns) + [CLASSIFICATION_TARGET]
    pd.testing.assert_frame_equal(result[train.columns], train)


def test_same_threshold_is_used_for_all_splits() -> None:
    train, validation, test = _three_splits()
    threshold = calculate_high_volatility_threshold(train)
    for frame in (train, validation, test):
        labeled = add_classification_target(frame, threshold)
        expected = (frame[TARGET_COLUMN] > threshold).astype("int8")
        pd.testing.assert_series_equal(labeled[CLASSIFICATION_TARGET], expected, check_names=False)


def test_missing_regression_target_is_rejected() -> None:
    train, _, _ = _three_splits()
    with pytest.raises(ValueError, match="Missing required column"):
        calculate_high_volatility_threshold(train.drop(columns=[TARGET_COLUMN]))


def test_nan_regression_target_is_rejected() -> None:
    train, _, _ = _three_splits()
    train.loc[0, TARGET_COLUMN] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        calculate_high_volatility_threshold(train)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_regression_target_is_rejected(value: float) -> None:
    train, _, _ = _three_splits()
    train.loc[0, TARGET_COLUMN] = value
    with pytest.raises(ValueError, match="Infinity"):
        add_classification_target(train, 0.30)


def test_existing_classification_target_is_rejected() -> None:
    train, _, _ = _three_splits()
    train[CLASSIFICATION_TARGET] = 0
    with pytest.raises(ValueError, match="already exists"):
        add_classification_target(train, 0.30)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf, "0.3", True])
def test_invalid_threshold_is_rejected(threshold: object) -> None:
    train, _, _ = _three_splits()
    with pytest.raises(ValueError, match="finite numeric"):
        add_classification_target(train, threshold)  # type: ignore[arg-type]


def test_empty_dataframe_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_high_volatility_threshold(pd.DataFrame(columns=[TARGET_COLUMN]))


def test_overlapping_split_dates_are_rejected() -> None:
    train, validation, test = _three_splits()
    validation["Date"] = pd.date_range("2020-01-05", periods=len(validation), freq="B")
    with pytest.raises(ValueError, match="overlap|out of order"):
        validate_classification_splits(train, validation, test)


def test_unsorted_dates_are_rejected() -> None:
    train, validation, test = _three_splits()
    train = train.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        validate_classification_splits(train, validation, test)


def test_adjusted_close_is_rejected() -> None:
    train, validation, test = _three_splits()
    train["Adj Close"] = train["Close"]
    with pytest.raises(ValueError, match="adjusted-close"):
        validate_classification_splits(train, validation, test)


def _write_split(frame: pd.DataFrame, path: Path) -> None:
    """Write one test split in the pipeline's source format."""
    frame.to_csv(path, index=False, date_format="%Y-%m-%d")


def test_pipeline_does_not_overwrite_source_files(tmp_path: Path) -> None:
    train, validation, test = _three_splits()
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    for frame, path in ((train, train_path), (validation, validation_path), (test, test_path)):
        _write_split(frame, path)
    with pytest.raises(ValueError, match="must not overwrite"):
        run_classification_target_pipeline(
            train_path,
            validation_path,
            test_path,
            train_path,
            tmp_path / "validation_labeled.csv",
            tmp_path / "test_labeled.csv",
            tmp_path / "report.json",
        )


def test_pipeline_raises_if_source_checksum_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, validation, test = _three_splits()
    source_paths = [tmp_path / f"{name}.csv" for name in ("train", "validation", "test")]
    for frame, path in zip((train, validation, test), source_paths, strict=True):
        _write_split(frame, path)

    original_hash = target_module._classification_sha256
    calls = 0

    def changed_hash(path: str | Path) -> str:
        nonlocal calls
        calls += 1
        value = original_hash(path)
        return value if calls <= 3 else "0" * len(value)

    monkeypatch.setattr(target_module, "_classification_sha256", changed_hash)
    with pytest.raises(RuntimeError, match="changed source files"):
        run_classification_target_pipeline(
            *source_paths,
            tmp_path / "train_labeled.csv",
            tmp_path / "validation_labeled.csv",
            tmp_path / "test_labeled.csv",
            tmp_path / "report.json",
        )


def test_saved_files_match_and_report_is_strict_json(tmp_path: Path) -> None:
    train, validation, test = _three_splits()
    source_paths = [tmp_path / f"{name}.csv" for name in ("train", "validation", "test")]
    for frame, path in zip((train, validation, test), source_paths, strict=True):
        _write_split(frame, path)
    output_paths = [tmp_path / f"{name}_labeled.csv" for name in ("train", "validation", "test")]
    report_path = tmp_path / "classification_threshold.json"

    labeled_train, labeled_validation, labeled_test, threshold = (
        run_classification_target_pipeline(
            *source_paths,
            *output_paths,
            report_path,
        )
    )

    assert threshold == pytest.approx(0.40)
    for expected, path in zip(
        (labeled_train, labeled_validation, labeled_test), output_paths, strict=True
    ):
        saved = pd.read_csv(path)
        assert len(saved) == len(expected)
        assert list(saved.columns) == list(expected.columns)

    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(
        report_text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert "NaN" not in report_text
    assert report["threshold_source"] == "train_only"
    assert all(
        values["before"] == values["after"] and values["unchanged"]
        for values in report["source_checksums"].values()
    )

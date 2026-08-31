"""Tests for causal QQQ feature engineering."""

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import src.build_features as feature_module
from src.build_features import (
    ANNUALIZATION_FACTOR,
    FEATURE_COLUMNS,
    add_rsi_feature,
    build_all_features,
    create_feature_report,
    run_feature_pipeline,
    save_feature_data,
    save_feature_report,
    validate_feature_input,
)


def _valid_data(rows: int = 40) -> pd.DataFrame:
    """Create sorted, valid QQQ-like data with deterministic values."""
    close = pd.Series(100 + np.arange(rows) + np.sin(np.arange(rows)), dtype=float)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=rows, freq="B"),
            "Open": close,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": 1_000_000 + np.arange(rows) * 10_000,
        }
    )


def test_builds_all_eight_features() -> None:
    result = build_all_features(_valid_data())
    assert all(column in result.columns for column in FEATURE_COLUMNS)


def test_does_not_modify_original_dataframe() -> None:
    original = _valid_data()
    snapshot = original.copy(deep=True)
    build_all_features(original)
    pd.testing.assert_frame_equal(original, snapshot)


def test_preserves_rows_original_columns_and_appends_features() -> None:
    original = _valid_data()
    result = build_all_features(original)
    assert len(result) == len(original)
    assert list(result.columns) == list(original.columns) + FEATURE_COLUMNS
    pd.testing.assert_frame_equal(result[original.columns], original)


def test_return_features_are_correct() -> None:
    data = _valid_data()
    result = build_all_features(data)
    expected_1d = data["Close"].pct_change(1, fill_method=None)
    expected_5d = data["Close"].pct_change(5, fill_method=None)
    pd.testing.assert_series_equal(result["return_1d"], expected_1d, check_names=False)
    pd.testing.assert_series_equal(result["return_5d"], expected_5d, check_names=False)


@pytest.mark.parametrize("window", [5, 20])
def test_historical_volatility_is_correct_and_annualized(window: int) -> None:
    data = _valid_data()
    result = build_all_features(data)
    expected = (
        data["Close"]
        .pct_change(fill_method=None)
        .rolling(window, min_periods=window)
        .std(ddof=1)
        * np.sqrt(ANNUALIZATION_FACTOR)
    )
    pd.testing.assert_series_equal(
        result[f"historical_volatility_{window}d"], expected, check_names=False
    )


def test_intraday_range_is_correct() -> None:
    data = _valid_data()
    result = build_all_features(data)
    expected = (data["High"] - data["Low"]) / data["Close"]
    pd.testing.assert_series_equal(result["intraday_range"], expected, check_names=False)


def test_sma_ratio_is_correct() -> None:
    data = _valid_data()
    result = build_all_features(data)
    expected = (
        data["Close"].rolling(5, min_periods=5).mean()
        / data["Close"].rolling(20, min_periods=20).mean()
    )
    pd.testing.assert_series_equal(result["sma_ratio_5_20"], expected, check_names=False)


def test_wilder_rsi_uses_simple_average_seed_then_recursion() -> None:
    data = _valid_data()
    positions = np.arange(len(data), dtype=float)
    close = 100 + positions * 0.2 + (positions % 2) * 2.0
    data["Close"] = close
    result = add_rsi_feature(data)
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    seed_gain = gain.iloc[1:15].mean()
    seed_loss = loss.iloc[1:15].mean()
    expected_seed = 100 - 100 / (1 + seed_gain / seed_loss)
    assert result["rsi_14"].iloc[14] == pytest.approx(expected_seed)

    next_gain = (seed_gain * 13 + gain.iloc[15]) / 14
    next_loss = (seed_loss * 13 + loss.iloc[15]) / 14
    expected_next = 100 - 100 / (1 + next_gain / next_loss)
    assert result["rsi_14"].iloc[15] == pytest.approx(expected_next)


def test_wilder_rsi_supports_fewer_than_fifteen_rows() -> None:
    result = add_rsi_feature(_valid_data(rows=14))
    assert result["rsi_14"].isna().all()


@pytest.mark.parametrize(
    ("close", "expected"),
    [
        (np.arange(100.0, 120.0), 100.0),
        (np.arange(120.0, 100.0, -1.0), 0.0),
        (np.full(20, 100.0), 50.0),
    ],
)
def test_rsi_edge_cases(close: np.ndarray, expected: float) -> None:
    data = _valid_data(rows=20)
    data["Close"] = close
    data["Open"] = close
    data["High"] = close + 1
    data["Low"] = close - 1
    result = build_all_features(data)
    valid_rsi = result["rsi_14"].dropna()
    assert valid_rsi.between(0, 100).all()
    assert valid_rsi.iloc[-1] == pytest.approx(expected)


def test_volume_zscore_is_correct() -> None:
    data = _valid_data()
    result = build_all_features(data)
    mean = data["Volume"].rolling(20, min_periods=20).mean()
    std = data["Volume"].rolling(20, min_periods=20).std(ddof=1)
    expected = (data["Volume"] - mean) / std
    pd.testing.assert_series_equal(result["volume_zscore_20"], expected, check_names=False)


def test_constant_volume_produces_nan_not_infinity() -> None:
    data = _valid_data()
    data["Volume"] = 1_000_000
    result = build_all_features(data)
    assert result["volume_zscore_20"].isna().all()
    assert not np.isinf(result["volume_zscore_20"]).any()


def test_initial_rolling_nans_are_preserved() -> None:
    result = build_all_features(_valid_data())
    assert result["return_1d"].iloc[:1].isna().all()
    assert result["return_5d"].iloc[:5].isna().all()
    assert result["historical_volatility_20d"].iloc[:20].isna().all()
    assert result["sma_ratio_5_20"].iloc[:19].isna().all()
    assert result["rsi_14"].iloc[:14].isna().all()
    assert result["volume_zscore_20"].iloc[:19].isna().all()


def test_implementation_has_no_negative_shift_or_centered_rolling() -> None:
    source = inspect.getsource(feature_module)
    compact_source = "".join(source.split())
    assert "shift(-" not in compact_source
    assert "center=True" not in compact_source


def test_date_order_does_not_change() -> None:
    data = _valid_data()
    result = build_all_features(data)
    pd.testing.assert_series_equal(result["Date"], data["Date"])


def test_build_rejects_non_datetime_date() -> None:
    data = _valid_data()
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    with pytest.raises(ValueError, match="datetime"):
        build_all_features(data)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda df: df.drop(columns=["Close"]), "Missing required columns"),
        (lambda df: df.assign(Date=df["Date"].where(df.index != 1, df["Date"].iloc[0])), "duplicate"),
        (lambda df: df.iloc[::-1].reset_index(drop=True), "sorted"),
        (lambda df: df.assign(Close=df["Close"].where(df.index != 0, 0)), "greater than zero"),
        (lambda df: df.assign(Volume=df["Volume"].where(df.index != 0, -1)), "Volume"),
        (lambda df: df.assign(Close=df["Close"].where(df.index != 0, np.nan)), "Missing values"),
    ],
)
def test_invalid_clean_input_is_rejected(mutation: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_feature_input(mutation(_valid_data()))


@pytest.mark.parametrize("protected_path", ["data/raw/qqq_daily.csv", "data/interim/qqq_clean.csv"])
def test_feature_save_does_not_overwrite_source_data(protected_path: str) -> None:
    with pytest.raises(ValueError, match="source data"):
        save_feature_data(build_all_features(_valid_data()), protected_path)


def test_pipeline_parses_date_and_saves_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "clean.csv"
    output_path = tmp_path / "features" / "features.csv"
    report_path = tmp_path / "reports" / "features.json"
    _valid_data().to_csv(input_path, index=False, date_format="%Y-%m-%d")

    result, report = run_feature_pipeline(input_path, output_path, report_path)

    assert str(result["Date"].dtype) == "datetime64[ns]"
    assert output_path.is_file()
    assert report_path.is_file()
    assert report["price_column"] == "Close"


def test_infinity_counts_are_reported_before_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    original_add_volume = feature_module.add_volume_feature

    def add_infinite_volume(df: pd.DataFrame) -> pd.DataFrame:
        result = original_add_volume(df)
        result.loc[result.index[-1], "volume_zscore_20"] = np.inf
        return result

    monkeypatch.setattr(feature_module, "add_volume_feature", add_infinite_volume)
    result, counts = feature_module._build_all_features_with_diagnostics(_valid_data())
    report = create_feature_report(result, counts)
    assert counts["volume_zscore_20"] == 1
    assert pd.isna(result["volume_zscore_20"].iloc[-1])
    assert report["infinite_values_replaced"]["volume_zscore_20"] == 1


def test_report_converts_all_nan_statistics_to_none_and_saves_strict_json(
    tmp_path: Path,
) -> None:
    result = build_all_features(_valid_data(rows=10))
    report = create_feature_report(result)
    assert report["first_valid_date"]["historical_volatility_20d"] is None
    assert report["min_values"]["historical_volatility_20d"] is None
    assert report["max_values"]["historical_volatility_20d"] is None

    report_path = tmp_path / "feature_report.json"
    save_feature_report(report, report_path)
    saved_text = report_path.read_text(encoding="utf-8")
    assert "NaN" not in saved_text
    assert json.loads(saved_text)["min_values"]["historical_volatility_20d"] is None

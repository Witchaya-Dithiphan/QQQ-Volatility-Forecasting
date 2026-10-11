"""Audit guards pinned to the accepted data artifacts (2026-10-11 data-pipeline audit).

Every test recomputes a property independently of the pipeline code and compares it
with the files on disk, so a data refresh or a code change that breaks causality,
the chronological split or the train-only thresholds fails loudly.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import (
    CLASSIFICATION_THRESHOLD_REPORT_PATH,
    CLEAN_DATA_PATH,
    FEATURE_DATA_PATH,
    NON_SPIKE_TRAIN_PATH,
    PRIMARY_SPIKE_IQR_MULTIPLIER,
    PROJECT_ROOT,
    REGRESSION_TARGET_DATA_PATH,
    SPIKE_FEATURE_FORWARD_REACH,
    SPIKE_TARGET_BACKWARD_REACH,
    TEST_FLAGGED_PATH,
    TEST_LABELED_DATA_PATH,
    TRAIN_LABELED_DATA_PATH,
    VALIDATION_FLAGGED_PATH,
    VALIDATION_LABELED_DATA_PATH,
    WITH_SPIKES_TRAIN_PATH,
)
from src.build_features import FEATURE_COLUMNS, build_all_features
from src.build_targets import CLASSIFICATION_TARGET, FORWARD_HORIZON, TARGET_COLUMN

PURGE_GAP_ROWS = 5
EXPECTED_SPLIT_ROWS = {"train": 1733, "validation": 371, "test": 373}
ACCEPTED_Q75 = 0.2530580184684854
ACCEPTED_SPIKE_THRESHOLD = 0.0465436445413787

REQUIRED_DATA = [
    CLEAN_DATA_PATH,
    FEATURE_DATA_PATH,
    REGRESSION_TARGET_DATA_PATH,
    TRAIN_LABELED_DATA_PATH,
    VALIDATION_LABELED_DATA_PATH,
    TEST_LABELED_DATA_PATH,
    WITH_SPIKES_TRAIN_PATH,
    NON_SPIKE_TRAIN_PATH,
    VALIDATION_FLAGGED_PATH,
    TEST_FLAGGED_PATH,
]

needs_data = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED_DATA),
    reason="ignored production inputs unavailable",
)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"], float_precision="round_trip")


@pytest.fixture(scope="module")
def full_table() -> pd.DataFrame:
    """The 2512-row calendar with features and the regression target."""
    return _read(REGRESSION_TARGET_DATA_PATH)


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    return {
        "train": _read(TRAIN_LABELED_DATA_PATH),
        "validation": _read(VALIDATION_LABELED_DATA_PATH),
        "test": _read(TEST_LABELED_DATA_PATH),
    }


def _positions(full_table: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Row positions of a split inside the NaN-dropped modelling table."""
    clean = full_table.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN]).reset_index(drop=True)
    lookup = pd.Series(clean.index, index=clean["Date"])
    return lookup.loc[frame["Date"]].to_numpy()


def _spike_threshold(train: pd.DataFrame) -> float:
    magnitude = train["return_1d"].abs()
    q1, q3 = magnitude.quantile(0.25), magnitude.quantile(0.75)
    return float(q3 + PRIMARY_SPIKE_IQR_MULTIPLIER * (q3 - q1))


# --- features and target: causality on the real data ---------------------------------


@needs_data
@pytest.mark.parametrize("rows", [21, 30, 100, 500, 1000, 1753, 2000, 2511])
def test_features_on_a_truncated_history_equal_the_full_build(rows: int) -> None:
    clean = _read(CLEAN_DATA_PATH)
    full = build_all_features(clean)
    truncated = build_all_features(clean.iloc[:rows].copy())
    pd.testing.assert_frame_equal(
        truncated[FEATURE_COLUMNS].reset_index(drop=True),
        full[FEATURE_COLUMNS].iloc[:rows].reset_index(drop=True),
        check_exact=True,
    )


@needs_data
def test_stored_features_match_an_independent_numpy_recompute() -> None:
    stored = _read(FEATURE_DATA_PATH)
    close = stored["Close"].to_numpy()
    rows = len(close)
    simple_return = np.full(rows, np.nan)
    simple_return[1:] = close[1:] / close[:-1] - 1.0
    volatility_20d = np.full(rows, np.nan)
    for row in range(20, rows):
        volatility_20d[row] = np.std(simple_return[row - 19 : row + 1], ddof=1) * np.sqrt(252)
    intraday_range = (stored["High"] - stored["Low"]) / stored["Close"]

    np.testing.assert_allclose(stored["return_1d"], simple_return, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        stored["historical_volatility_20d"], volatility_20d, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(stored["intraday_range"], intraday_range, rtol=0, atol=1e-12)


@needs_data
def test_stored_target_uses_only_the_five_returns_after_each_row(
    full_table: pd.DataFrame,
) -> None:
    close = full_table["Close"].to_numpy()
    rows = len(close)
    simple_return = np.full(rows, np.nan)
    simple_return[1:] = close[1:] / close[:-1] - 1.0
    expected = np.full(rows, np.nan)
    for row in range(rows - FORWARD_HORIZON):
        future = simple_return[row + 1 : row + 1 + FORWARD_HORIZON]
        expected[row] = np.std(future, ddof=1) * np.sqrt(252)

    stored = full_table[TARGET_COLUMN].to_numpy()
    assert np.isnan(stored[-FORWARD_HORIZON:]).all()
    assert not np.isnan(stored[:-FORWARD_HORIZON]).any()
    np.testing.assert_allclose(stored, expected, rtol=0, atol=1e-12)


# --- chronological split ---------------------------------------------------------------


@needs_data
def test_committed_splits_sit_at_the_expected_positions(
    full_table: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> None:
    assert {name: len(frame) for name, frame in splits.items()} == EXPECTED_SPLIT_ROWS
    start = 0
    for name in ("train", "validation", "test"):
        positions = _positions(full_table, splits[name])
        expected = np.arange(start, start + EXPECTED_SPLIT_ROWS[name])
        np.testing.assert_array_equal(positions, expected)
        start += EXPECTED_SPLIT_ROWS[name] + PURGE_GAP_ROWS


@needs_data
def test_purge_gap_covers_the_forward_target_horizon(
    full_table: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> None:
    assert PURGE_GAP_ROWS >= FORWARD_HORIZON
    calendar = full_table["Date"].reset_index(drop=True)
    lookup = pd.Series(calendar.index, index=calendar)
    for earlier, later in (("train", "validation"), ("validation", "test")):
        last_row = lookup.loc[splits[earlier]["Date"].iloc[-1]]
        target_window_end = calendar.iloc[last_row + FORWARD_HORIZON]
        assert target_window_end < splits[later]["Date"].iloc[0]


@needs_data
def test_no_date_is_shared_across_splits(splits: dict[str, pd.DataFrame]) -> None:
    dates = pd.concat([frame["Date"] for frame in splits.values()])
    assert dates.is_unique
    assert dates.is_monotonic_increasing


# --- train-only thresholds -------------------------------------------------------------


@needs_data
def test_q75_threshold_is_fitted_on_train_only(splits: dict[str, pd.DataFrame]) -> None:
    train_quantile = float(splits["train"][TARGET_COLUMN].quantile(0.75))
    pooled_quantile = float(pd.concat(splits.values())[TARGET_COLUMN].quantile(0.75))
    stored = json.loads(CLASSIFICATION_THRESHOLD_REPORT_PATH.read_text(encoding="utf-8"))

    assert train_quantile == ACCEPTED_Q75
    assert stored["threshold_decimal"] == train_quantile
    assert abs(pooled_quantile - train_quantile) > 0.01


@needs_data
@pytest.mark.parametrize(
    "path",
    [
        TRAIN_LABELED_DATA_PATH,
        VALIDATION_LABELED_DATA_PATH,
        TEST_LABELED_DATA_PATH,
        WITH_SPIKES_TRAIN_PATH,
        NON_SPIKE_TRAIN_PATH,
        VALIDATION_FLAGGED_PATH,
        TEST_FLAGGED_PATH,
    ],
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_labels_are_strictly_above_the_train_threshold(path: Path) -> None:
    frame = _read(path)
    expected = (frame[TARGET_COLUMN] > ACCEPTED_Q75).astype(int)
    np.testing.assert_array_equal(frame[CLASSIFICATION_TARGET].to_numpy(), expected.to_numpy())


@needs_data
def test_spike_threshold_is_fitted_on_train_only(splits: dict[str, pd.DataFrame]) -> None:
    assert _spike_threshold(splits["train"]) == pytest.approx(ACCEPTED_SPIKE_THRESHOLD, abs=1e-15)


# --- experiment datasets ---------------------------------------------------------------


@needs_data
def test_with_spikes_train_is_the_original_train_byte_for_byte() -> None:
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert digest(WITH_SPIKES_TRAIN_PATH) == digest(TRAIN_LABELED_DATA_PATH)


@needs_data
def test_spike_mask_on_the_full_calendar_equals_the_split_local_mask(
    full_table: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> None:
    """A spike in a purge gap or across a split edge would make the two masks differ."""
    threshold = _spike_threshold(splits["train"])
    direct = (full_table["return_1d"].abs() > threshold).to_numpy()
    affected = np.zeros(len(full_table), dtype=bool)
    for position in np.flatnonzero(direct):
        start = max(position - SPIKE_TARGET_BACKWARD_REACH, 0)
        affected[start : position + SPIKE_FEATURE_FORWARD_REACH + 1] = True
    on_calendar = pd.Series(affected, index=full_table["Date"])

    non_spike = _read(NON_SPIKE_TRAIN_PATH)
    train = splits["train"]
    stored_train_mask = ~train["Date"].isin(non_spike["Date"]).to_numpy()
    np.testing.assert_array_equal(on_calendar.loc[train["Date"]].to_numpy(), stored_train_mask)
    pd.testing.assert_frame_equal(
        train.loc[~stored_train_mask].reset_index(drop=True), non_spike, check_exact=True
    )

    for path in (VALIDATION_FLAGGED_PATH, TEST_FLAGGED_PATH):
        flagged = _read(path)
        np.testing.assert_array_equal(
            on_calendar.loc[flagged["Date"]].to_numpy(),
            flagged["is_spike_affected"].astype(bool).to_numpy(),
        )


# --- static scans ----------------------------------------------------------------------

SHUFFLING_PATTERNS = ("train_test_split", "shuffle=True", "permutation(", "default_rng")
TEST_SPLIT_PATTERNS = (
    "load_test(",
    "TEST_DATA_PATH",
    "TEST_LABELED_DATA_PATH",
    "TEST_FLAGGED_PATH",
    "test_labeled",
    "test_flagged",
)
TEST_SPLIT_ALLOWED = {"core/data.py", "core/contracts.py", "run.py"}


def test_src_never_shuffles_or_splits_at_random() -> None:
    offenders = [
        f"{path.relative_to(PROJECT_ROOT).as_posix()}: {pattern}"
        for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
        for pattern in SHUFFLING_PATTERNS
        if pattern in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_only_the_data_module_and_finalize_reference_the_test_split() -> None:
    ml_root = PROJECT_ROOT / "src" / "ml"
    offenders = [
        f"{path.relative_to(ml_root).as_posix()}: {pattern}"
        for path in sorted(ml_root.rglob("*.py"))
        if path.relative_to(ml_root).as_posix() not in TEST_SPLIT_ALLOWED
        for pattern in TEST_SPLIT_PATTERNS
        if pattern in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


FINALIZE_RECEIPT = PROJECT_ROOT / "outputs" / "modeling" / "finalize_receipt.json"
FINALIZE_ONLY_CSV = "test_predictions.csv"


@needs_data
def test_only_finalize_outputs_contain_test_period_dates(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Test-period rows may appear in one file name only, and only once finalize left a receipt."""
    first_test_date = splits["test"]["Date"].iloc[0].strftime("%Y-%m-%d")
    date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}),", re.MULTILINE)
    finalized = FINALIZE_RECEIPT.is_file()
    offenders = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in sorted((PROJECT_ROOT / "outputs" / "modeling").rglob("*.csv"))
        if not (finalized and path.name == FINALIZE_ONLY_CSV)
        if any(
            date >= first_test_date
            for date in date_pattern.findall(path.read_text(encoding="utf-8"))
        )
    ]
    assert offenders == []


def test_finalize_outputs_exist_only_with_a_receipt() -> None:
    produced = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for name in ("test_metrics.json", FINALIZE_ONLY_CSV)
        for path in (PROJECT_ROOT / "outputs" / "modeling").rglob(name)
    )
    assert FINALIZE_RECEIPT.is_file() or produced == []

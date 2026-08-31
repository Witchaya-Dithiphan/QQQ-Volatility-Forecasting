"""Tests that guard against temporal feature leakage."""

import numpy as np
import pandas as pd

from src.build_features import FEATURE_COLUMNS, build_all_features


def _leakage_data(rows: int = 50) -> pd.DataFrame:
    """Create deterministic cleaned data for temporal-invariance testing."""
    close = pd.Series(100 + np.arange(rows) * 0.5 + np.sin(np.arange(rows)))
    return pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-01", periods=rows, freq="B"),
            "Open": close,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": 1_000_000 + np.arange(rows) * 5_000,
        }
    )


def test_future_values_do_not_change_features_at_or_before_cutoff() -> None:
    data = _leakage_data()
    cutoff = 29
    original_features = build_all_features(data)

    modified = data.copy(deep=True)
    future_mask = modified.index > cutoff
    modified.loc[future_mask, ["Open", "High", "Low", "Close"]] *= 10
    modified.loc[future_mask, "Volume"] *= 10
    modified_features = build_all_features(modified)

    pd.testing.assert_frame_equal(
        original_features.loc[:cutoff, FEATURE_COLUMNS],
        modified_features.loc[:cutoff, FEATURE_COLUMNS],
    )

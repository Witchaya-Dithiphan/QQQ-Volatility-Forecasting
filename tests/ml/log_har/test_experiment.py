"""Selection rule, QLIKE and the real ETF prior used by the log-HAR experiment."""
import numpy as np
import pandas as pd
import pytest

from src.ml.log_har.experiment import etf_frames, qlike, select_kappa
from src.ml.log_har.model import pooled_prior
from src.ml.transfer_learning.sources import MANIFEST_PATH, SOURCE_CUTOFF, TICKERS, load_source


def test_qlike_is_zero_for_a_perfect_forecast_and_follows_patton():
    y = np.array([0.1, 0.2, 0.4])
    assert qlike(y, y) == 0.0
    prediction = np.array([0.2, 0.2, 0.2])
    ratio = (y / prediction) ** 2
    assert np.isclose(qlike(y, prediction), np.mean(ratio - np.log(ratio) - 1))
    assert qlike(y, np.array([0.1, 0.0, 0.2])) is None  # undefined for a zero forecast


def test_kappa_selection_takes_the_lowest_rmse_and_breaks_ties_towards_less_shrinkage():
    path = [{"kappa": 0.0, "validation_rmse": 0.07}, {"kappa": 0.3, "validation_rmse": 0.065},
            {"kappa": 1.0, "validation_rmse": 0.065}, {"kappa": np.inf, "validation_rmse": 0.08}]
    assert select_kappa(path) == 0.3


@pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="source snapshot not downloaded")
def test_real_prior_uses_only_pre_cutoff_rows_and_is_finite():
    frames, rows = etf_frames()
    assert set(rows) == set(TICKERS) and all(n > 1500 for n in rows.values())
    for ticker in TICKERS:
        assert pd.to_datetime(load_source(ticker)["label_end_date"]).max() <= pd.Timestamp(SOURCE_CUTOFF)
    beta, per_asset = pooled_prior(frames, "X")
    assert beta.shape == (5,) and np.isfinite(beta).all() and len(per_asset) == len(TICKERS)

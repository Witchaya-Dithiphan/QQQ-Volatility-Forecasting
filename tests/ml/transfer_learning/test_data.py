"""Causality and cutoff checks for the transfer-learning inputs."""
import numpy as np
import pandas as pd
import pytest

from src.ml.transfer_learning.nn import Model
from src.ml.transfer_learning.sources import MANIFEST_PATH, SOURCE_CUTOFF, featurize, load_source
from src.ml.transfer_learning.train import PURGE, finetune, span_mask, tail_split
from src.ml.transfer_learning.windows import contiguous_windows, make_windows

needs_sources = pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="source snapshot not downloaded")


def test_window_holds_t_minus_l_plus_one_through_t_and_drops_short_history():
    dates = np.arange("2020-01-01", "2020-02-10", dtype="datetime64[D]")
    features = np.arange(len(dates), dtype=float)[:, None] * np.ones((1, 8))
    targets = dates[[2, 10, 25]]
    kept, positions, X = make_windows(dates, features, targets, lookback=5)
    assert kept.tolist() == [1, 2]  # day 2 has only 3 rows of history
    assert X[0, :, 0].tolist() == [6, 7, 8, 9, 10] and X[1, -1, 0] == 25
    assert positions.tolist() == [10, 25]
    with pytest.raises(ValueError):
        make_windows(dates, features, np.array(["2021-01-01"], dtype="datetime64[D]"), 5)


def test_contiguous_windows_cover_every_full_window():
    features = np.arange(10, dtype=float)[:, None]
    windows = contiguous_windows(features, 4)
    assert windows.shape == (7, 4, 1) and windows[-1, :, 0].tolist() == [6, 7, 8, 9]


def test_tail_split_purges_five_trading_days_before_the_tail():
    positions = np.arange(100)
    sub, tail = tail_split(positions)
    assert tail[0] == 85 and sub.max() == 85 - PURGE - 1
    assert np.intersect1d(sub, tail).size == 0


def test_span_mask_hides_two_weekly_spans():
    mask = span_mask(np.random.RandomState(0), 200, 60)
    assert mask.shape == (200, 60)
    assert 5 <= mask.sum(axis=1).min() and mask.sum(axis=1).max() == 10


def synthetic_ohlcv(n=80):
    rng = np.random.RandomState(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    days = pd.bdate_range("2023-05-01", periods=n)
    return pd.DataFrame({"Date": days.strftime("%Y-%m-%d"), "Close": close, "Volume": rng.randint(1e6, 2e6, n),
                         "Open": close, "High": close * 1.01, "Low": close * 0.99}).astype(str)


def test_featurize_cuts_before_target_so_no_label_crosses_the_cutoff():
    raw = synthetic_ohlcv()
    cutoff = np.datetime64(raw["Date"].iloc[60])
    frame, _ = featurize(raw, cutoff)
    assert frame["Date"].max() == pd.Timestamp(cutoff)
    assert frame["target_volatility_5d"].tail(5).isna().all()
    labeled = frame.loc[frame["target_volatility_5d"].notna()]
    assert (labeled["label_end_date"] <= pd.Timestamp(cutoff)).all()
    # Same rows, uncut: the labeled rows' targets agree, so the cut changed nothing else.
    full, _ = featurize(raw)
    merged = labeled.merge(full, on="Date", suffixes=("", "_full"))
    assert np.allclose(merged["target_volatility_5d"], merged["target_volatility_5d_full"])


@needs_sources
def test_real_sources_end_at_the_qqq_train_cutoff():
    for ticker in ("SPY", "EEM"):
        frame = load_source(ticker)
        assert frame["label_end_date"].max() <= pd.Timestamp(SOURCE_CUTOFF)
        assert frame["target_volatility_5d"].notna().all()


def test_linear_probe_never_moves_the_encoder():
    model = Model("transformer", lookback=10, d_model=8, n_heads=2, n_layers=1, d_ff=8, dropout=0.0, seed=0)
    rng = np.random.RandomState(1)
    X = rng.normal(size=(60, 10, 8)).astype(np.float32)
    y = (rng.uniform(size=60) > 0.6).astype(float)
    before = {n: t.data.copy() for n, t in model.params.items()}
    finetune(model, X, y, np.arange(60), seed=0, encoder_lr=None, head_lr=1e-2, max_epochs=3, patience=3, batch=16)
    for name in model.params:
        if name.startswith("enc.") or name.startswith("recon."):
            assert np.array_equal(model.params[name].data, before[name]), name

# tests/ml/core/test_bootstrap.py
import numpy as np
import pytest

from src.ml.core.bootstrap import block_indices, difference, interval, replicate


def test_indices_are_reproducible_chains_of_whole_blocks():
    first = block_indices(371, block=20, replicates=50, seed=7)
    assert first.shape == (50, 371)
    assert first.min() >= 0 and first.max() <= 370
    np.testing.assert_array_equal(first, block_indices(371, block=20, replicates=50, seed=7))
    steps = np.diff(first, axis=1)
    inside_a_block = np.arange(370) % 20 != 19
    assert (steps[:, inside_a_block] == 1).all()


def test_a_different_seed_draws_different_blocks():
    assert not np.array_equal(
        block_indices(100, replicates=5, seed=1), block_indices(100, replicates=5, seed=2)
    )


@pytest.mark.parametrize("n, block", [(0, 1), (10, 11), (10, 0)])
def test_impossible_block_lengths_are_rejected(n, block):
    with pytest.raises(ValueError):
        block_indices(n, block=block)


def test_blocks_widen_the_interval_of_an_autocorrelated_mean():
    rng = np.random.RandomState(0)
    series = np.zeros(400)
    for t in range(1, 400):
        series[t] = 0.9 * series[t - 1] + rng.normal()

    def mean(values):
        return {"mean": float(values.mean())}

    blocked = replicate(mean, (series,), block_indices(400, block=20, replicates=500))["mean"].std()
    rowwise = replicate(mean, (series,), block_indices(400, block=1, replicates=500))["mean"].std()
    assert blocked > 1.5 * rowwise


def test_undefined_replicates_are_counted_not_hidden():
    y = np.r_[np.ones(3), np.zeros(97)]

    def rate(values):
        return {"rate": float(values.mean()) if values.any() else None}

    values = replicate(rate, (y,), block_indices(100, block=10, replicates=200))["rate"]
    summary = interval(0.03, values)
    assert summary["n_replicates"] == 200
    assert summary["n_valid"] == int(np.isfinite(values).sum()) < 180
    assert summary["low"] is None and summary["high"] is None


def test_interval_is_the_percentile_range_of_the_replicates():
    summary = interval(0.5, np.linspace(0.0, 1.0, 1001))
    assert summary["low"] == pytest.approx(0.025) and summary["high"] == pytest.approx(0.975)
    assert summary["n_valid"] == summary["n_replicates"] == 1001


def test_paired_difference_of_identical_scores_is_exactly_zero():
    values = np.random.RandomState(1).rand(300)
    gap = difference(0.4, values, 0.4, values)
    assert gap["point"] == 0.0 and gap["low"] == 0.0 and gap["high"] == 0.0
    assert gap["excludes_zero"] is False


def test_paired_difference_detects_a_consistent_gain():
    base = np.random.RandomState(2).rand(300)
    gap = difference(0.4, base, 0.5, base + 0.1)
    assert gap["excludes_zero"] is True
    assert gap["low"] == pytest.approx(0.1)

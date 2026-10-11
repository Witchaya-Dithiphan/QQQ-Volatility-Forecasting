# tests/ml/core/test_importance.py
import numpy as np
import pytest

from src.ml.core.contracts import FEATURE_COLUMNS
from src.ml.core.importance import (
    block_permutation_importance,
    block_shuffled,
    feature_importance,
    linear_coefficients,
)


def test_block_shuffle_moves_whole_blocks_and_keeps_every_value():
    column = np.arange(100.0)
    shuffled = block_shuffled(column, block=20, rng=np.random.RandomState(3))
    assert sorted(shuffled) == sorted(column)
    assert not np.array_equal(shuffled, column)
    assert (np.diff(shuffled.reshape(5, 20), axis=1) == 1).all()


def test_only_the_feature_the_score_depends_on_is_important():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 3))
    X.setflags(write=False)
    y = X[:, 0].copy()

    def score(matrix):
        return -float(np.mean((matrix[:, 0] - y) ** 2))

    result = block_permutation_importance(score, X, block=20, repeats=5, seed=1)
    assert result["baseline"] == 0.0
    assert result["mean"][0] > 0.5
    assert result["mean"][1] == 0.0 and result["mean"][2] == 0.0
    assert result["std"].shape == (3,)


def test_importance_is_reproducible_for_a_fixed_seed():
    X = np.random.RandomState(0).normal(size=(120, 2))

    def score(matrix):
        return -float(np.abs(matrix[:, 1] - X[:, 1]).mean())

    first = block_permutation_importance(score, X, repeats=4, seed=9)
    second = block_permutation_importance(score, X, repeats=4, seed=9)
    np.testing.assert_array_equal(first["mean"], second["mean"])


@pytest.fixture(scope="module")
def importance_rows():
    return feature_importance("with_spike", repeats=3)


def test_real_importance_covers_every_supervised_model_and_feature(importance_rows):
    assert len(importance_rows) == 17 * 8
    assert {row["feature"] for row in importance_rows} == set(FEATURE_COLUMNS)
    assert all(row["metric"] in ("average_precision", "negative_rmse") for row in importance_rows)


def test_a_single_feature_model_ignores_the_other_seven(importance_rows):
    rows = [row for row in importance_rows if row["model"] == "simple_linear"]
    unused = [row for row in rows if row["feature"] != "historical_volatility_20d"]
    assert all(row["importance_mean"] == 0.0 for row in unused)


def test_linear_coefficients_are_reported_per_original_feature():
    rows = linear_coefficients("with_spike")
    assert len(rows) == 6 * 8
    svm = [row for row in rows if row["model"] == "svm"]
    assert [row["feature"] for row in svm] == list(FEATURE_COLUMNS)
    assert all(np.isfinite(row["coefficient"]) for row in rows)

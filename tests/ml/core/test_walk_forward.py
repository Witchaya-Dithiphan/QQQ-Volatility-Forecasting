# tests/ml/core/test_walk_forward.py
import json

import numpy as np
import pytest

from src.ml.core.artifacts import run_directory
from src.ml.core.base import BaseModel
from src.ml.core.trainer import OUTPUT_ROOT
from src.ml.core.walk_forward import spearman, walk_forward
from src.ml.registry import MODELS, models_for_task


class Recorder(BaseModel):
    """Stands in for a model that asks for trading-day positions."""

    name = "gaussian_nb"
    task = "classification"
    calls: list = []

    def get_params(self):
        return {}

    def fit(self, X, y=None, positions=None, n_positions=None):
        type(self).calls.append({"rows": len(X), "last": int(positions[-1]), "n_positions": n_positions,
                                 "mean": np.asarray(X).mean(axis=0)})
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.rate_)

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int64)

    def state_dict(self):
        return {}

    def load_state(self, arrays, metadata):
        return self


@pytest.fixture()
def recorded():
    Recorder.calls = []
    rows = walk_forward("gaussian_nb", variant="with_spike", model_class=Recorder, replicates=20)
    return rows, Recorder.calls


def test_four_folds_and_one_pooled_row(recorded):
    rows, _ = recorded
    assert [row["fold"] for row in rows] == [0, 1, 2, 3, "pooled"]
    assert rows[-1]["block_rows"] == sum(row["block_rows"] for row in rows[:4])
    assert rows[-1]["positives"] == sum(row["positives"] for row in rows[:4])


def test_training_history_grows_and_stops_before_the_purge_gap(recorded):
    rows, calls = recorded
    assert [call["rows"] for call in calls] == sorted(call["rows"] for call in calls)
    for row, call in zip(rows[:4], calls):
        assert call["last"] < row["block_start_position"] - 5
        assert call["n_positions"] == call["last"] + 1


def test_blocks_are_consecutive_and_cover_the_last_sixty_percent(recorded):
    rows, _ = recorded
    assert rows[0]["block_start_position"] == int(1733 * 0.4)
    for earlier, later in zip(rows[:3], rows[1:4]):
        assert earlier["block_stop_position"] == later["block_start_position"]
    assert rows[3]["block_stop_position"] == 1733


def test_non_spike_counts_days_on_the_original_timeline():
    Recorder.calls = []
    rows = walk_forward("gaussian_nb", variant="non_spike", model_class=Recorder, replicates=20)
    assert rows[0]["block_start_position"] == int(1733 * 0.4)
    assert rows[-1]["block_rows"] < 1040


def test_real_model_returns_metrics_with_a_pooled_interval():
    rows = walk_forward("gaussian_nb", variant="with_spike", replicates=50)
    pooled = rows[-1]
    assert 0 < pooled["average_precision"] <= 1
    assert pooled["average_precision_low"] <= pooled["average_precision"] <= pooled["average_precision_high"]
    assert pooled["positives"] > 300


@pytest.mark.parametrize("name", models_for_task("classification"))
def test_saved_winner_params_rebuild_every_classifier(name):
    config = json.loads((run_directory(OUTPUT_ROOT, "with_spike", name) / "config.json").read_text(encoding="utf-8"))
    MODELS[name](**config["params"])


def test_spearman_of_reversed_ranks_is_minus_one():
    assert spearman([0.1, 0.2, 0.3, 0.4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([0.1, 0.2, 0.3], [10, 20, 30]) == pytest.approx(1.0)


class RefusesTheFirstFold(Recorder):
    """A model that cannot be fitted on the shortest history (stacking on real data:
    the first 275 trading days of Train hold no high-volatility day at all)."""

    def fit(self, X, y=None, positions=None, n_positions=None):
        if len(X) < 800:
            raise ValueError("single-class history")
        return super().fit(X, y, positions=positions, n_positions=n_positions)


def test_a_fold_the_model_cannot_fit_is_reported_with_its_reason_not_fatal():
    rows = walk_forward("gaussian_nb", variant="with_spike", model_class=RefusesTheFirstFold, replicates=20)
    assert [row["fold"] for row in rows] == [0, 1, 2, 3, "pooled"]
    assert rows[0]["average_precision"] is None
    assert rows[0]["reason"] == "single-class history"
    assert rows[1]["reason"] is None and rows[1]["average_precision"] is not None
    assert rows[-1]["folds_used"] == 3
    assert rows[-1]["block_rows"] == sum(row["block_rows"] for row in rows[1:4])


def test_pooled_row_counts_every_fold_when_all_of_them_fit(recorded):
    rows, _ = recorded
    assert rows[-1]["folds_used"] == 4 and rows[-1]["reason"] is None


def test_real_stacking_survives_the_calm_opening_year_of_train():
    rows = walk_forward("stacking", variant="with_spike", replicates=20)
    assert rows[0]["reason"] is not None and rows[-1]["folds_used"] == 3

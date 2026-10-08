"""Scratch Gradient Boosting: binary logistic regression trees (M6, with-spike)."""
import numpy as np
import pytest

from src.modeling.ensemble._tree import tree_depth
from src.modeling.ensemble.gradient_boosting import GradientBoosting
from src.modeling.metrics import classification_metrics, loss, select_threshold
from tests.modeling.ensemble_helpers import ENSEMBLE_DIR, imported_modules, real_q75, strict_roundtrip, synthetic


def _logloss(y, p):
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))


def test_no_sklearn_in_production():
    assert "sklearn" not in imported_modules(ENSEMBLE_DIR / "gradient_boosting.py")


def test_frozen_defaults():
    m = GradientBoosting()
    assert (m.n_estimators, m.learning_rate, m.max_depth, m.random_state) == (100, 0.1, 2, 42)


@pytest.mark.parametrize("kwargs", [{"n_estimators": 0}, {"learning_rate": 0}, {"learning_rate": float("nan")},
                                    {"max_depth": 0}, {"max_depth": 1.5}, {"tol": -1}])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        GradientBoosting(**kwargs)


def test_input_validation_binary_labels_and_unfitted():
    X, y = synthetic()
    with pytest.raises(ValueError, match="not fitted"):
        GradientBoosting().predict_proba(X)
    for bad_X, bad_y in [(X[:, 0], y), (X, y[:-1]), (np.where(X > 0, np.inf, X), y), (X, y + 1), (X, np.zeros_like(y))]:
        with pytest.raises(ValueError):
            GradientBoosting(n_estimators=3).fit(bad_X, bad_y)
    model = GradientBoosting(n_estimators=3).fit(X, y)
    with pytest.raises(ValueError, match="feature"):
        model.predict(X[:, :2])


def test_initial_score_is_prior_log_odds_and_loss_is_tracked():
    X, y = synthetic()
    model = GradientBoosting(n_estimators=20, learning_rate=0.1, max_depth=1).fit(X, y)
    assert model.init_score_ == pytest.approx(np.log(y.mean() / (1 - y.mean())))
    assert len(model.trees_) == len(model.loss_history_) == 20
    assert np.all(np.diff(model.loss_history_) < 0)  # small shrinkage: monotone descent
    p = model.predict_proba(X)[:, 1]
    assert model.loss_history_[-1] == pytest.approx(_logloss(y, p))
    assert isinstance(model.converged_, bool)


def test_trees_respect_depth_and_probabilities_stay_finite():
    X, y = synthetic()
    sep = (X[:, 0] > 0).astype(int)  # perfectly separable: raw scores grow large
    model = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=2).fit(X, sep)
    assert all(tree_depth(t) <= 2 for t in model.trees_)
    proba = model.predict_proba(X)
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.mean(model.predict(X) == sep) > 0.97


def test_deterministic():
    X, y = synthetic()
    a = GradientBoosting(n_estimators=10).fit(X, y)
    b = GradientBoosting(n_estimators=10).fit(X, y)
    assert a.trees_ == b.trees_
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


@pytest.mark.parametrize("depth", [1, 2])
def test_tracks_sklearn_reference(depth):
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = synthetic(n=500)
    ours = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=depth).fit(X[:350], y[:350])
    ref = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=depth, random_state=42).fit(X[:350], y[:350])
    po, pr = ours.predict_proba(X[350:])[:, 1], ref.predict_proba(X[350:])[:, 1]
    assert np.corrcoef(po, pr)[0, 1] > 0.99
    assert abs(_logloss(y[350:], po) - _logloss(y[350:], pr)) < 0.03
    assert np.mean((po >= 0.5) == (pr >= 0.5)) > 0.95


def test_serialization_strict_json_exact_reload():
    X, y = synthetic()
    model = GradientBoosting(n_estimators=15, learning_rate=0.03, max_depth=2).fit(X, y)
    state = strict_roundtrip(model.to_dict())
    loaded = GradientBoosting.from_dict(state)
    np.testing.assert_array_equal(model.predict_proba(X), loaded.predict_proba(X))
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    assert loaded.to_dict() == model.to_dict()
    with pytest.raises(ValueError):
        GradientBoosting().to_dict()
    with pytest.raises(ValueError):
        GradientBoosting.from_dict({**state, "trees": []})


def test_q75_real_data_smoke():
    Xtr, ytr, Xva, yva = real_q75()
    model = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=1).fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:, 1]
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert classification_metrics(yva, p)["roc_auc"]["value"] > 0.5
    assert select_threshold(yva, p)["selection_split"] == "validation"
    scores = np.log(p / (1 - p))
    assert loss("binary_logistic_loss", yva, scores=scores)["value"] is not None
    loaded = GradientBoosting.from_dict(strict_roundtrip(model.to_dict()))
    np.testing.assert_array_equal(loaded.predict_proba(Xva), model.predict_proba(Xva))

"""Scratch Random Forest (M6, with-spike)."""
import numpy as np
import pytest

from src.modeling.ensemble._tree import apply_tree, tree_depth
from src.modeling.ensemble.random_forest import RandomForest
from src.modeling.metrics import classification_metrics, loss, select_threshold
from tests.modeling.ensemble_helpers import ENSEMBLE_DIR, imported_modules, real_q75, strict_roundtrip, synthetic


def test_no_sklearn_in_production_and_wrapper_removed():
    assert not (ENSEMBLE_DIR / "sklearn_wrapper.py").exists()
    for name in ("random_forest.py", "_tree.py"):
        assert "sklearn" not in imported_modules(ENSEMBLE_DIR / name)


def test_frozen_defaults_and_hyperparameters():
    m = RandomForest()
    assert (m.n_estimators, m.max_depth, m.max_features, m.min_samples_leaf, m.random_state) == (100, 5, "sqrt", 5, 42)


@pytest.mark.parametrize("kwargs", [{"n_estimators": 0}, {"n_estimators": 2.5}, {"max_depth": 0}, {"min_samples_leaf": 0},
                                    {"max_features": "log2"}, {"random_state": -1}])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        RandomForest(**kwargs)


def test_input_validation_and_unfitted():
    X, y = synthetic()
    with pytest.raises(ValueError, match="not fitted"):
        RandomForest().predict(X)
    for bad_X, bad_y in [(X[:, 0], y), (X, y[:-1]), (np.where(X > 0, np.nan, X), y), (X, y + 1), (X, y.astype(float) + 0.5)]:
        with pytest.raises(ValueError):
            RandomForest(n_estimators=3).fit(bad_X, bad_y)
    model = RandomForest(n_estimators=3).fit(X, y)
    with pytest.raises(ValueError, match="feature"):
        model.predict(X[:, :3])
    with pytest.raises(ValueError):
        model.predict_proba(np.full((2, 5), np.inf))


def test_probabilities_are_genuine_vote_fractions():
    X, y = synthetic()
    model = RandomForest(n_estimators=25, max_depth=3).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    votes = np.sum([model.predict_tree_votes(i, X) for i in range(25)], axis=0)
    np.testing.assert_allclose(proba[:, 1], votes / 25)
    assert set(np.unique(np.round(proba[:, 1] * 25, 9))) <= set(range(26))
    np.testing.assert_array_equal(model.predict(X), (proba[:, 1] >= 0.5).astype(int))


def test_bootstrap_and_feature_randomness_reproducible():
    X, y = synthetic(d=8)
    a = RandomForest(n_estimators=10, max_depth=3, random_state=7).fit(X, y)
    b = RandomForest(n_estimators=10, max_depth=3, random_state=7).fit(X, y)
    c = RandomForest(n_estimators=10, max_depth=3, random_state=8).fit(X, y)
    assert a.max_features_ == 2
    assert a.bootstrap_indices_.shape == (10, len(X))
    np.testing.assert_array_equal(a.bootstrap_indices_, b.bootstrap_indices_)
    assert not np.array_equal(a.bootstrap_indices_, c.bootstrap_indices_)
    assert a.trees_ == b.trees_
    assert len(np.unique(a.bootstrap_indices_[0])) < len(X)  # sampled with replacement


def test_tree_constraints_hold_on_bootstrap_sample():
    X, y = synthetic(d=8)
    model = RandomForest(n_estimators=12, max_depth=3, min_samples_leaf=5).fit(X, y)
    for tree, idx in zip(model.trees_, model.bootstrap_indices_):
        assert tree_depth(tree) <= 3
        counts = np.bincount(apply_tree(tree, X[idx]))
        assert counts[counts > 0].min() >= 5


def test_learns_signal_and_tracks_sklearn_reference():
    from sklearn.ensemble import RandomForestClassifier
    X, y = synthetic(n=600)
    Xtr, ytr, Xte, yte = X[:400], y[:400], X[400:], y[400:]
    ours = RandomForest(n_estimators=50, max_depth=5).fit(Xtr, ytr)
    ref = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=5, random_state=42).fit(Xtr, ytr)
    acc, ref_acc = np.mean(ours.predict(Xte) == yte), np.mean(ref.predict(Xte) == yte)
    assert acc > 0.8 and abs(acc - ref_acc) < 0.06
    assert np.corrcoef(ours.predict_proba(Xte)[:, 1], ref.predict_proba(Xte)[:, 1])[0, 1] > 0.9


def test_serialization_strict_json_exact_reload():
    X, y = synthetic(d=8)
    model = RandomForest(n_estimators=8, max_depth=4, random_state=3).fit(X, y)
    state = strict_roundtrip(model.to_dict())
    loaded = RandomForest.from_dict(state)
    np.testing.assert_array_equal(model.predict_proba(X), loaded.predict_proba(X))
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    np.testing.assert_array_equal(model.bootstrap_indices_, loaded.bootstrap_indices_)
    assert loaded.to_dict() == model.to_dict()
    with pytest.raises(ValueError):
        RandomForest().to_dict()
    with pytest.raises(ValueError):
        RandomForest.from_dict({**state, "n_features": -1})


def test_q75_real_data_smoke():
    Xtr, ytr, Xva, yva = real_q75()
    model = RandomForest(n_estimators=50, max_depth=3).fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert classification_metrics(yva, p)["roc_auc"]["value"] > 0.5
    assert select_threshold(yva, p)["selection_split"] == "validation"
    assert loss("log_loss", yva, probabilities=np.clip(p, 1e-15, 1 - 1e-15))["value"] is not None
    loaded = RandomForest.from_dict(strict_roundtrip(model.to_dict()))
    np.testing.assert_array_equal(loaded.predict_proba(Xva), model.predict_proba(Xva))

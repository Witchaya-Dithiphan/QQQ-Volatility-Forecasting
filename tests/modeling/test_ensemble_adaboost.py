"""Scratch discrete AdaBoost with weighted decision stumps (M6, with-spike)."""
import numpy as np
import pytest

from src.modeling.ensemble.adaboost import AdaBoost, AdaBoostFailure
from src.modeling.metrics import classification_metrics, loss, select_threshold
from tests.modeling.ensemble_helpers import ENSEMBLE_DIR, imported_modules, real_q75, strict_roundtrip, synthetic

# Found by exhaustive stump search: round-1 error .25, then every stump has weighted error >= .5.
EARLY_X = np.array([[2, 0], [0, 0], [0, 0], [2, 0], [0, 0], [2, 0], [0, 0], [2, 0]], dtype=float)
EARLY_Y = np.array([0, 0, 0, 1, 1, 1, 0, 1])


def test_no_sklearn_in_production():
    assert "sklearn" not in imported_modules(ENSEMBLE_DIR / "adaboost.py")


def test_frozen_constants_and_defaults():
    m = AdaBoost()
    assert (m.n_estimators, m.learning_rate, m.alpha_factor, m.epsilon) == (50, 1.0, 0.5, 1e-12)


@pytest.mark.parametrize("kwargs", [{"n_estimators": 0}, {"learning_rate": 0}, {"learning_rate": float("inf")}])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        AdaBoost(**kwargs)


def test_input_validation_binary_labels_and_unfitted():
    X, y = synthetic()
    with pytest.raises(ValueError, match="not fitted"):
        AdaBoost().predict(X)
    for bad_X, bad_y in [(X[:, 0], y), (X, y[:-1]), (np.where(X > 0, np.nan, X), y), (X, y + 1)]:
        with pytest.raises(ValueError):
            AdaBoost(n_estimators=3).fit(bad_X, bad_y)
    model = AdaBoost(n_estimators=3).fit(X, y)
    with pytest.raises(ValueError, match="feature"):
        model.predict(X[:, :2])


def test_stump_searches_both_polarities():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    flipped = AdaBoost(n_estimators=5).fit(X, np.array([1, 1, 0, 0]))  # high x -> label -1
    assert flipped.stumps_[0] == {"feature": 0, "threshold": 2.5, "polarity": -1}
    assert flipped.status_ == "completed_perfect"
    np.testing.assert_array_equal(flipped.predict(X), [1, 1, 0, 0])
    normal = AdaBoost(n_estimators=5).fit(X, np.array([0, 0, 1, 1]))
    assert normal.stumps_[0]["polarity"] == 1


def test_stump_ties_prefer_feature_then_threshold_then_normal_polarity():
    # x=[1,2,3], y=[+,-,+]: (1.5, flipped) and (2.5, normal) tie at 1/3 -> lower threshold wins.
    model = AdaBoost(n_estimators=1).fit(np.array([[1.0], [2.0], [3.0]]), np.array([1, 0, 1]))
    assert model.stumps_[0] == {"feature": 0, "threshold": 1.5, "polarity": -1}
    # duplicate columns tie exactly -> lowest feature index
    col = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    dup = AdaBoost(n_estimators=1).fit(np.hstack([col, col]), np.array([0, 0, 0, 1, 1, 1]))
    assert dup.stumps_[0]["feature"] == 0


def test_first_weak_learner_failure_is_honest():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # XOR: best stump error == .5
    model = AdaBoost(n_estimators=5)
    with pytest.raises(AdaBoostFailure):
        model.fit(X, np.array([0, 1, 1, 0]))
    assert model.status_ == "failed" and model.stumps_ == []
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X)


def test_later_weak_learner_failure_completes_early():
    model = AdaBoost(n_estimators=10).fit(EARLY_X, EARLY_Y)
    assert model.status_ == "completed_early"
    assert len(model.stumps_) == 1 and model.estimator_errors_ == [pytest.approx(0.25)]
    assert model.stop_error_ >= 0.5
    assert model.predict(EARLY_X).shape == (8,)


def test_perfect_stump_completes_perfect_with_finite_alpha():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    model = AdaBoost(n_estimators=10, learning_rate=0.5).fit(X, np.array([0, 0, 1, 1]))
    assert model.status_ == "completed_perfect" and len(model.stumps_) == 1
    assert model.estimator_weights_[0] == pytest.approx(0.5 * 0.5 * np.log((1 - 1e-12) / 1e-12))
    assert np.isfinite(model.estimator_weights_).all()


def test_alpha_and_weight_update_match_closed_form():
    X, y = synthetic(n=60)
    model = AdaBoost(n_estimators=2, learning_rate=1.0).fit(X, y)
    e = model.estimator_errors_[0]
    assert model.estimator_weights_[0] == pytest.approx(0.5 * np.log((1 - e) / e))
    s = model.stumps_[0]
    h = s["polarity"] * np.where(X[:, s["feature"]] > s["threshold"], 1, -1)
    ysg = 2 * y - 1
    w = np.exp(-model.estimator_weights_[0] * ysg * h) / len(y)
    w /= w.sum()
    s2 = model.stumps_[1]
    h2 = s2["polarity"] * np.where(X[:, s2["feature"]] > s2["threshold"], 1, -1)
    assert model.estimator_errors_[1] == pytest.approx(w[h2 != ysg].sum())


def test_zero_vote_predicts_label_one_and_scores_are_stable():
    X, y = synthetic()
    model = AdaBoost(n_estimators=5).fit(X, y)
    model.estimator_weights_ = [0.0] * len(model.stumps_)
    assert np.all(model.decision_function(X) == 0) and np.all(model.predict(X) == 1)
    big = AdaBoost(n_estimators=100).fit(X, (X[:, 0] > 0).astype(int))
    assert np.isfinite(big.decision_function(X)).all() and np.isfinite(big.predict_proba(X)).all()


def test_weights_stay_normalized_and_finite_over_many_rounds():
    X, y = synthetic(n=300)
    noisy = np.where(np.random.default_rng(1).random(300) < 0.2, 1 - y, y)
    model = AdaBoost(n_estimators=100).fit(X, noisy)
    assert np.isfinite(model.log_sample_weights_).all()
    assert np.exp(model.log_sample_weights_).sum() == pytest.approx(1.0)


def test_boosting_effect_and_loss_history():
    X, y = synthetic(n=300)
    model = AdaBoost(n_estimators=30).fit(X, y)
    assert len(model.loss_history_) == len(model.stumps_)
    assert model.loss_history_[-1] < model.loss_history_[0]
    proba = model.predict_proba(X)
    assert proba.shape == (300, 2) and ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_array_equal(model.predict(X), (model.decision_function(X) >= 0).astype(int))


def test_deterministic_and_tracks_sklearn_reference():
    from sklearn.ensemble import AdaBoostClassifier
    X, y = synthetic(n=600)
    a = AdaBoost(n_estimators=50).fit(X[:400], y[:400])
    assert a.stumps_ == AdaBoost(n_estimators=50).fit(X[:400], y[:400]).stumps_
    ref = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X[:400], y[:400])
    acc, ref_acc = np.mean(a.predict(X[400:]) == y[400:]), np.mean(ref.predict(X[400:]) == y[400:])
    assert acc > 0.8 and abs(acc - ref_acc) < 0.06


@pytest.mark.parametrize("fit_args", ["synthetic", "early"])
def test_serialization_strict_json_exact_reload(fit_args):
    X, y = synthetic() if fit_args == "synthetic" else (EARLY_X, EARLY_Y)
    model = AdaBoost(n_estimators=20, learning_rate=0.5).fit(X, y)
    state = strict_roundtrip(model.to_dict())
    loaded = AdaBoost.from_dict(state)
    np.testing.assert_array_equal(model.decision_function(X), loaded.decision_function(X))
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    assert loaded.status_ == model.status_ and loaded.to_dict() == model.to_dict()
    with pytest.raises(ValueError):
        AdaBoost().to_dict()


def test_q75_real_data_smoke():
    Xtr, ytr, Xva, yva = real_q75()
    model = AdaBoost(n_estimators=25, learning_rate=0.5).fit(Xtr, ytr)
    assert model.status_ in ("completed", "completed_early", "completed_perfect")
    p = model.predict_proba(Xva)[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert classification_metrics(yva, p)["roc_auc"]["value"] > 0.5
    assert select_threshold(yva, model.decision_function(Xva), score_kind="decision")["selection_split"] == "validation"
    assert loss("exponential_loss", yva, scores=model.decision_function(Xva))["value"] is not None
    loaded = AdaBoost.from_dict(strict_roundtrip(model.to_dict()))
    np.testing.assert_array_equal(loaded.predict(Xva), model.predict(Xva))


@pytest.mark.parametrize("constant", [True, False])
def test_constant_prediction_is_minimum_and_survives_strict_reload(constant):
    X = np.ones((5, 1)) if constant else np.arange(5.0).reshape(-1, 1)
    y = np.array([0, 0, 1, 0, 0])
    model = AdaBoost(n_estimators=1).fit(X, y)
    assert model.estimator_errors_ == [pytest.approx(0.2)]
    np.testing.assert_array_equal(model.predict(X), np.zeros(5))
    loaded = AdaBoost.from_dict(strict_roundtrip(model.to_dict()))
    np.testing.assert_array_equal(loaded.decision_function(X), model.decision_function(X))


def test_boundary_ties_and_unnormalized_weights():
    from src.modeling.ensemble.adaboost import _best_stump
    X = np.ones((4, 2))
    err, stump = _best_stump(X, np.array([-1, 1, -1, 1]), np.full(4, 3.0))
    assert err == pytest.approx(0.5)
    assert stump == {"feature": 0, "threshold": 1.0, "polarity": 1}
    # An internal threshold ties with the constant prediction: lower threshold wins.
    err, stump = _best_stump(np.arange(3.0).reshape(-1, 1), np.array([1, -1, 1]), np.ones(3))
    assert err == pytest.approx(1 / 3)
    assert stump == {"feature": 0, "threshold": 0.5, "polarity": -1}


@pytest.mark.parametrize("epsilon", [0, 0.5, 0.6])
def test_epsilon_range_at_construction_and_fit(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        AdaBoost(epsilon=epsilon)
    model = AdaBoost()
    model.epsilon = epsilon
    with pytest.raises(ValueError, match="epsilon"):
        model.fit(np.arange(5.0).reshape(-1, 1), np.array([0, 0, 1, 0, 0]))


@pytest.mark.parametrize("later", [False, True])
def test_no_candidate_has_honest_status(monkeypatch, later):
    import src.modeling.ensemble.adaboost as module
    candidates = iter([(0.2, {"feature": 0, "threshold": 4.0, "polarity": 1}), None] if later else [None])
    monkeypatch.setattr(module, "_best_stump", lambda *args: next(candidates))
    model = AdaBoost(n_estimators=2)
    X, y = np.arange(5.0).reshape(-1, 1), np.array([0, 0, 1, 0, 0])
    if later:
        model.fit(X, y)
        assert model.status_ == "completed_early" and len(model.stumps_) == 1
    else:
        with pytest.raises(AdaBoostFailure):
            model.fit(X, y)
        assert model.status_ == "failed" and model.stumps_ == []
    assert model.stop_error_ == 0.5
    assert np.isfinite(model.log_sample_weights_).all()

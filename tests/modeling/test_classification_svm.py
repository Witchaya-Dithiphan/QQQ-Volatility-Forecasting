"""Behavioral tests for the scratch linear soft-margin SVM (M6)."""
import json

import numpy as np
import pytest

from src.modeling.classification.svm import LinearSVM, objective_and_subgradient
from src.modeling.configuration import load_config
from src.modeling.datasets import load_train_validation
from src.modeling.metrics import classification_metrics, loss, select_threshold
from src.modeling.preprocessing import PCA, Standardizer


def blobs(n=200, d=4, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    X = rng.normal(size=(n, d)) * noise + (2 * y[:, None] - 1) * np.array([1.0, 0.5, 0.0, 0.0][:d])
    return X * np.array([5.0, 1.0, 0.1, 30.0][:d]) + 3.0, y  # unstandardized scales


def objective(w, b, X, y, C):
    return 0.5 * w @ w + C * np.maximum(0, 1 - (2 * y - 1) * (X @ w + b)).sum()


# ---- mathematics -----------------------------------------------------------
def test_objective_and_subgradient_hand_example():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 3.0]])
    y = np.array([1, 0, 1])  # signed: +1, -1, +1
    obj, gw, gb = objective_and_subgradient(np.zeros(2), 0.0, X, y, C=2.0)
    # margins 0, 0, 0 -> every hinge is 1
    assert obj == pytest.approx(2.0 * 3)
    np.testing.assert_allclose(gw, -2.0 * (X[0] - X[1] + X[2]))
    assert gb == pytest.approx(-2.0 * (1 - 1 + 1))


def test_margin_exactly_one_contributes_no_subgradient():
    X, y = np.array([[1.0]]), np.array([1])
    obj, gw, gb = objective_and_subgradient(np.array([1.0]), 0.0, X, y, C=5.0)
    assert obj == pytest.approx(0.5)
    np.testing.assert_allclose(gw, [1.0])  # only the ||w||^2/2 term
    assert gb == 0.0


def test_subgradient_matches_finite_differences_off_kink():
    X, y = blobs(30, 3, seed=1)
    rng = np.random.default_rng(2)
    w, b = rng.normal(size=3) * 0.1, 0.3
    _, gw, gb = objective_and_subgradient(w, b, X, y, C=0.7)
    eps = 1e-6
    for j in range(3):
        d = np.zeros(3); d[j] = eps
        fd = (objective(w + d, b, X, y, 0.7) - objective(w - d, b, X, y, 0.7)) / (2 * eps)
        assert gw[j] == pytest.approx(fd, rel=1e-5, abs=1e-6)
    fd_b = (objective(w, b + eps, X, y, 0.7) - objective(w, b - eps, X, y, 0.7)) / (2 * eps)
    assert gb == pytest.approx(fd_b, rel=1e-5, abs=1e-6)


def test_intercept_is_not_penalized():
    X, y = blobs(20, 2, seed=3)
    w = np.array([0.2, -0.1])
    for b in (0.0, 50.0):
        obj, _, gb = objective_and_subgradient(w, b, X, y, C=1.0)
        hinge = np.maximum(0, 1 - (2 * y - 1) * (X @ w + b)).sum()
        assert obj == pytest.approx(0.5 * w @ w + hinge)
    # every sample violates the margin -> gradient wrt b is exactly -C * sum(signed y)
    _, _, gb = objective_and_subgradient(np.zeros(2), 0.0, X, y, C=1.0)
    assert gb == pytest.approx(-(2 * y - 1).sum())


# ---- behaviour vs reference ------------------------------------------------
@pytest.mark.parametrize("C", load_config()["models"]["svm"]["grid"]["C"])
def test_matches_linearsvc_reference_objective_and_labels(C):
    svm_mod = pytest.importorskip("sklearn.svm")
    X, y = blobs(300, 4, seed=4)
    ours = LinearSVM(C=C).fit(X, y)
    Z = ours.scaler_.transform(X)
    ref = svm_mod.LinearSVC(loss="hinge", C=C, dual=True, max_iter=200000, tol=1e-8, random_state=42).fit(Z, y)
    ref_obj = objective(ref.coef_[0], ref.intercept_[0], Z, y, C)
    our_obj = objective(ours.coef_, ours.intercept_, Z, y, C)
    assert our_obj <= ref_obj * 1.02  # unpenalized intercept can only help
    agree = np.mean(ours.predict(X) == ref.predict(Z))
    assert agree >= 0.95


# ---- preprocessing / leakage ----------------------------------------------
def test_scaler_fit_on_train_only_and_predict_does_not_mutate_state():
    X, y = blobs(150, 4, seed=5)
    model = LinearSVM(C=1.0).fit(X, y)
    np.testing.assert_allclose(model.scaler_.mean_, X.mean(axis=0))
    before = model.to_json()
    extreme = X[:20] * 1e6 + 1e6  # "validation" far from Train
    model.decision_function(extreme)
    model.predict(extreme)
    assert model.to_json() == before


def test_pca_variant_uses_m2_standardizer_and_pca_on_train_only():
    X, y = blobs(200, 4, seed=6)
    model = LinearSVM(C=1.0, pca_components=2).fit(X, y)
    Z = Standardizer().fit_transform(X)
    P = PCA(2).fit(Z)
    np.testing.assert_allclose(model.pca_.components_, P.components_)
    assert model.coef_.shape == (2,)
    manual = P.transform(Standardizer().fit(X).transform(X[:7])) @ model.coef_ + model.intercept_
    np.testing.assert_allclose(model.decision_function(X[:7]), manual)


def test_no_pca_keeps_all_features():
    X, y = blobs(100, 4, seed=7)
    model = LinearSVM().fit(X, y)
    assert model.pca_ is None and model.coef_.shape == (4,)


# ---- output semantics ------------------------------------------------------
def test_decision_function_is_finite_scores_not_probabilities():
    X, y = blobs(120, 4, seed=8)
    model = LinearSVM(C=10).fit(X, y)
    s = model.decision_function(X)
    assert s.shape == (120,) and np.isfinite(s).all()
    assert s.min() < 0 < s.max() and (np.abs(s) > 1).any()  # not squashed into [0, 1]
    assert not hasattr(model, "predict_proba")


def test_threshold_boundary_zero_is_positive_class():
    X, y = blobs(30, 2, seed=15)
    model = LinearSVM().fit(X, y)
    model.coef_, model.intercept_ = np.zeros(2), 0.0
    assert model.decision_function(X[:3]).tolist() == [0.0, 0.0, 0.0]
    assert model.predict(X[:3]).tolist() == [1, 1, 1]
    model.intercept_ = -1e-12
    assert model.predict(X[:3]).tolist() == [0, 0, 0]


# ---- validation of inputs --------------------------------------------------
@pytest.mark.parametrize("kwargs", [dict(C=0), dict(C=-1), dict(C=float("nan")), dict(max_iter=0), dict(tol=-1.0),
                                    dict(pca_components=0), dict(pca_components=2.5)])
def test_bad_hyperparameters_rejected(kwargs):
    with pytest.raises(ValueError):
        LinearSVM(**kwargs)


def test_bad_fit_inputs_rejected():
    X, y = blobs(30, 3, seed=9)
    with pytest.raises(ValueError):
        LinearSVM().fit(X, np.zeros(30, dtype=int))  # one class
    with pytest.raises(ValueError):
        LinearSVM().fit(X, np.where(y == 1, 2, 0))  # not {0, 1}
    with pytest.raises(ValueError):
        LinearSVM().fit(X, y[:-1])
    with pytest.raises(ValueError):
        LinearSVM().fit(X[:, 0], y)
    bad = X.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        LinearSVM().fit(bad, y)
    with pytest.raises(ValueError):
        LinearSVM(pca_components=4).fit(X, y)  # more components than features


def test_predict_dimension_and_fit_state_checks():
    X, y = blobs(40, 3, seed=10)
    with pytest.raises(ValueError):
        LinearSVM().decision_function(X)
    model = LinearSVM().fit(X, y)
    with pytest.raises(ValueError):
        model.decision_function(X[:, :2])


# ---- determinism, convergence, persistence --------------------------------
def test_fit_is_deterministic():
    X, y = blobs(150, 4, seed=11)
    assert LinearSVM(C=1).fit(X, y).to_json() == LinearSVM(C=1).fit(X, y).to_json()


def test_convergence_status_is_honest():
    X, y = blobs(150, 4, seed=12)
    short = LinearSVM(max_iter=3).fit(X, y)
    assert short.status_ == "max_iter" and short.n_iter_ == 3
    long = LinearSVM(max_iter=20000, tol=1e-6).fit(X, y)
    assert long.status_ in ("converged", "max_iter") and 1 <= long.n_iter_ <= 20000
    assert long.objective_ <= short.objective_
    assert np.isfinite(long.objective_)


def test_strict_json_roundtrip_is_exact():
    X, y = blobs(100, 4, seed=13)
    model = LinearSVM(C=0.1, pca_components=2).fit(X, y)
    text = model.to_json()
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))  # no NaN/Infinity tokens
    clone = LinearSVM.from_json(text)
    assert clone.to_json() == text
    np.testing.assert_array_equal(clone.decision_function(X), model.decision_function(X))
    assert (clone.status_, clone.n_iter_, clone.objective_) == (model.status_, model.n_iter_, model.objective_)


def test_from_json_rejects_nan_and_unfitted_serialization():
    with pytest.raises(ValueError):
        LinearSVM().to_json()
    X, y = blobs(50, 3, seed=14)
    text = LinearSVM().fit(X, y).to_json().replace('"intercept": ', '"intercept": NaN, "x": ', 1)
    with pytest.raises(ValueError):
        LinearSVM.from_json(text)


def test_compat_alias_is_the_scratch_class():
    from src.modeling import advanced
    assert advanced.SVM is LinearSVM


# ---- real-data smoke (Q75 label contract, Train-only fit) ------------------
def test_real_data_q75_smoke_with_validation_threshold_m2():
    data = load_train_validation("with_spike")
    model = LinearSVM(C=1.0, max_iter=2000).fit(data.train.X, data.train.y_classification)
    train_scaler_mean = data.train.X.mean(axis=0)
    np.testing.assert_allclose(model.scaler_.mean_, train_scaler_mean)  # never fit on Validation
    scores = model.decision_function(data.validation.X)
    assert scores.shape == (len(data.validation.X),) and np.isfinite(scores).all()
    yv = data.validation.y_classification
    metrics = classification_metrics(yv, scores, threshold=0.0)
    assert metrics["roc_auc"]["value"] is not None
    assert select_threshold(yv, scores, score_kind="decision")["selection_split"] == "validation"
    assert loss("hinge_loss", yv, scores=scores)["value"] >= 0
    np.testing.assert_array_equal(model.predict(data.validation.X), (scores >= 0).astype(int))

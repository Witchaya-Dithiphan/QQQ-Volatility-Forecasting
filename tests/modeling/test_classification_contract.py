"""M5 contract regression tests: scratch-only package, frozen M2 target, plan Section 14 scopes."""
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SKLogistic, Perceptron as SKPerceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as SKTree

import src.modeling.classification as pkg
from src.modeling.classification import LogisticRegression
from src.modeling.classification.decision_tree import DecisionTreeClassifier
from src.modeling.classification.knn import KNN
from src.modeling.classification.naive_bayes import GaussianNaiveBayes
from src.modeling.classification.perceptron import Perceptron
from src.modeling.classification.slp import SLP
from src.modeling.classification import logistic as logistic_module
from src.modeling.datasets import load_train_validation
from src.modeling.metrics import classification_metrics, loss, select_threshold
from src.modeling.preprocessing import Standardizer

PACKAGE_DIR = Path(pkg.__file__).parent
TEST_DIR = Path(__file__).parent

FACTORIES = {
    "logistic": lambda: LogisticRegression(max_iter=300),
    "naive_bayes": GaussianNaiveBayes,
    "knn": lambda: KNN(n_neighbors=3),
    "perceptron": lambda: Perceptron(max_iter=20),
    "slp": lambda: SLP(max_iter=20),
    "tree": lambda: DecisionTreeClassifier(max_depth=3),
}
PROBA = [n for n in FACTORIES if n not in ("perceptron",)]


def data(n=120, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    return X, (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n) > 0).astype(int)


# ---- package / source contract -------------------------------------------------------------

def test_package_exports_scratch_logistic():
    assert pkg.LogisticRegression is logistic_module.LogisticRegression
    assert set(pkg.__all__) >= {"LogisticRegression", "GaussianNaiveBayes", "KNN", "Perceptron", "SLP", "DecisionTreeClassifier"}


@pytest.mark.parametrize("path", sorted(PACKAGE_DIR.glob("*.py")), ids=lambda p: p.name)
def test_no_sklearn_in_production(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf8"))):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        assert not any(n.split(".")[0] == "sklearn" for n in names), path.name


def test_tests_use_frozen_classification_target():
    for path in TEST_DIR.glob("test_classification_*.py"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf8")
        assert "np.median" not in text and "y_regression" not in text, f"{path.name} recomputes target"


# ---- shared input / state validation -------------------------------------------------------

@pytest.mark.parametrize("name", FACTORIES)
def test_rejects_bad_inputs(name):
    X, y = data()
    with pytest.raises(ValueError):
        FACTORIES[name]().fit(X, np.where(y == 1, 2, 0))
    with pytest.raises(ValueError):
        FACTORIES[name]().fit(X, np.zeros(len(y), dtype=int))
    with pytest.raises(ValueError):
        FACTORIES[name]().fit(X[:-1], y)
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        FACTORIES[name]().fit(bad, y)


@pytest.mark.parametrize("name", FACTORIES)
def test_fitted_state_and_dimensions(name):
    X, y = data()
    with pytest.raises(ValueError, match="not fitted"):
        FACTORIES[name]().predict(X)
    with pytest.raises(ValueError, match="not fitted"):
        FACTORIES[name]().to_dict()
    model = FACTORIES[name]().fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X[:, :2])
    with pytest.raises(ValueError):
        model.predict(np.where(np.arange(X.size).reshape(X.shape) == 0, np.inf, X))


@pytest.mark.parametrize("make", [
    lambda: LogisticRegression(learning_rate=0), lambda: LogisticRegression(l2=-1), lambda: LogisticRegression(max_iter=0),
    lambda: LogisticRegression(tol=0), lambda: LogisticRegression(class_weight="bad"),
    lambda: GaussianNaiveBayes(var_smoothing=-1), lambda: KNN(n_neighbors=0), lambda: KNN(weights="bad"),
    lambda: Perceptron(learning_rate=0), lambda: Perceptron(max_iter=0),
    lambda: SLP(learning_rate=0), lambda: SLP(batch_size=0), lambda: SLP(patience=0), lambda: SLP(l2=-1),
    lambda: DecisionTreeClassifier(max_depth=0), lambda: DecisionTreeClassifier(min_samples_leaf=0),
])
def test_rejects_bad_hyperparameters(make):
    X, y = data()
    with pytest.raises(ValueError):
        make().fit(X, y)


@pytest.mark.parametrize("name", PROBA)
def test_probability_contract(name):
    X, y = data()
    proba = FACTORIES[name]().fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2) and proba.dtype == np.float64
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", FACTORIES)
def test_json_roundtrip_is_exact(name):
    X, y = data()
    model = FACTORIES[name]().fit(X, y)
    loaded = type(model).from_dict(json.loads(json.dumps(model.to_dict())))
    Xt = data(30, seed=9)[0]
    np.testing.assert_array_equal(model.predict(Xt), loaded.predict(Xt))
    if name in PROBA:
        np.testing.assert_array_equal(model.predict_proba(Xt), loaded.predict_proba(Xt))
    if name == "perceptron":
        np.testing.assert_array_equal(model.decision_function(Xt), loaded.decision_function(Xt))


# ---- frozen M2 data contract / metrics reuse -----------------------------------------------

@pytest.mark.parametrize("name", FACTORIES)
def test_real_data_frozen_target_and_m2_metrics(name):
    ds = load_train_validation("with_spike")
    tr, va = ds.train, ds.validation
    assert set(np.unique(tr.y_classification)) == {0, 1}
    model = FACTORIES[name]().fit(tr.X, tr.y_classification)
    if name == "perceptron":
        score, kind, thr = model.decision_function(va.X), "decision", 0.0
    else:
        score, kind, thr = model.predict_proba(va.X)[:, 1], "probability", 0.5
    metrics = classification_metrics(va.y_classification, score, threshold=thr)
    assert np.isfinite(score).all() and np.array(metrics["confusion_matrix"]).sum() == len(va.X)
    assert select_threshold(va.y_classification, score, score_kind=kind)["selection_split"] == "validation"
    if name == "perceptron":
        assert 0 <= loss("perceptron_criterion", va.y_classification, scores=score)["mistake_rate"]["value"] <= 1
    else:
        assert loss("log_loss", va.y_classification, probabilities=score)["value"] > 0


# ---- Logistic ------------------------------------------------------------------------------

@pytest.mark.parametrize("cw", [None, "balanced"])
def test_logistic_matches_sklearn_objective(cw):
    X, y = data(400, seed=3)
    y = (X[:, 0] + X[:, 2] > 1.0).astype(int)  # imbalanced
    Xs = Standardizer().fit_transform(X)
    l2 = 0.01
    ours = LogisticRegression(l2=l2, class_weight=cw, max_iter=100000, standardize=False).fit(Xs, y)
    ref = SKLogistic(C=1 / (len(y) * l2), class_weight=cw, tol=1e-12, max_iter=10000).fit(Xs, y)
    assert ours.converged_
    np.testing.assert_allclose(ours.coefficients_, ref.coef_[0], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_[0], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(ours.predict_proba(Xs), ref.predict_proba(Xs), atol=1e-4)


def test_logistic_balanced_raises_recall_on_imbalance():
    X, y = data(600, seed=4)
    y = (X[:, 0] > 1.0).astype(int)
    plain = LogisticRegression().fit(X, y).predict(X)
    balanced = LogisticRegression(class_weight="balanced").fit(X, y).predict(X)
    assert balanced.sum() > plain.sum()


def test_logistic_scaler_is_train_only_and_applied_at_predict():
    X, y = data()
    model = LogisticRegression().fit(X, y)
    np.testing.assert_allclose(model.scaler_.mean_, X.mean(axis=0))
    shifted = X + 100
    manual = LogisticRegression(standardize=False).fit(Standardizer().fit_transform(X), y)
    np.testing.assert_allclose(model.predict_proba(shifted), manual.predict_proba(model.scaler_.transform(shifted)))


def test_logistic_intercept_not_penalized_and_convergence_reported():
    X, y = data(300)
    y = (np.arange(300) % 5 == 0).astype(int)
    model = LogisticRegression(l2=100, learning_rate=0.005, max_iter=50000, standardize=False).fit(X, y)
    assert abs(model.intercept_ - np.log(0.2 / 0.8)) < 1e-3
    assert np.abs(model.coefficients_).max() < 1e-3
    with pytest.raises(ValueError, match="diverged"):
        LogisticRegression(l2=1e6, standardize=False).fit(X, y)
    tiny = LogisticRegression(max_iter=2).fit(X, y)
    assert tiny.converged_ is False and tiny.n_iter_ == 2
    assert all(b <= a + 1e-12 for a, b in zip(model.loss_history_, model.loss_history_[1:]))


# ---- Gaussian NB ---------------------------------------------------------------------------

@pytest.mark.parametrize("vs", [1e-9, 1e-2])
def test_nb_matches_sklearn_var_smoothing(vs):
    X, y = data(200, seed=5)
    Xt = data(50, seed=6)[0] * 3
    ours = GaussianNaiveBayes(var_smoothing=vs).fit(X, y)
    ref = GaussianNB(var_smoothing=vs).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(Xt), ref.predict_proba(Xt), atol=1e-10)
    np.testing.assert_allclose(ours.class_priors_, ref.class_prior_)


def test_nb_extreme_inputs_stay_finite():
    X, y = data()
    proba = GaussianNaiveBayes().fit(X, y).predict_proba(np.full((2, 4), 1e6))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


# ---- k-NN ----------------------------------------------------------------------------------

def test_knn_uses_float64_distances():
    X = np.array([[1.0], [1.0 + 1e-9]])
    model = KNN(n_neighbors=1, standardize=False).fit(X, [0, 1])
    assert model.predict(np.array([[1.0 + 1e-9]]))[0] == 1


def test_knn_ties_are_deterministic_by_train_order():
    q = np.array([[0.0]])
    assert KNN(1, standardize=False).fit([[-1.0], [1.0]], [0, 1]).predict(q)[0] == 0
    assert KNN(1, standardize=False).fit([[1.0], [-1.0]], [1, 0]).predict(q)[0] == 1
    even = KNN(2, standardize=False).fit([[-1.0], [1.0]], [1, 0])
    np.testing.assert_array_equal(even.predict_proba(q), [[0.5, 0.5]])
    assert even.predict(q)[0] == 1  # vote tie -> probability 0.5 -> positive (M2: score >= threshold)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("k", [3, 15])
def test_knn_matches_sklearn(weights, k):
    X, y = data(150, seed=7)
    Xt = data(40, seed=8)[0]
    ours = KNN(k, weights=weights, standardize=False).fit(X, y)
    ref = KNeighborsClassifier(k, weights=weights).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(Xt), ref.predict_proba(Xt), atol=1e-12)


def test_knn_distance_weight_exact_match_dominates():
    X, y = data(50)
    model = KNN(5, weights="distance", standardize=False).fit(X, y)
    np.testing.assert_array_equal(model.predict_proba(X[:1])[0], [1 - y[0], y[0]])


def test_knn_k_larger_than_train_rejected_and_scaler_train_only():
    X, y = data(20)
    with pytest.raises(ValueError):
        KNN(21).fit(X, y)
    np.testing.assert_allclose(KNN(3).fit(X, y).scaler_.mean_, X.mean(axis=0))


# ---- Perceptron ----------------------------------------------------------------------------

def test_perceptron_is_decision_score_not_probability():
    X, y = data()
    model = Perceptron().fit(X, y)
    assert not hasattr(model, "predict_proba")
    score = model.decision_function(X)
    assert score.dtype == np.float64 and (score < 0).any() and (score > 1).any()
    np.testing.assert_array_equal(model.predict(X), (score >= 0).astype(int))
    assert model.mistake_rate_history_[0] > 0 and all(0 <= r <= 1 for r in model.mistake_rate_history_)
    sep = (X[:, 0] > 0).astype(int)
    assert Perceptron(max_iter=1000).fit(X, sep).mistake_rate_history_[-1] == 0


@pytest.mark.parametrize("lr", [0.001, 0.1])
def test_perceptron_matches_sklearn_online_update(lr):
    X, y = data(100, seed=11)
    y = (X[:, 0] + 2 * X[:, 1] + 0.8 * np.random.default_rng(1).standard_normal(100) > 0).astype(int)  # not separable
    Xs = Standardizer().fit_transform(X)
    ours = Perceptron(learning_rate=lr, max_iter=5, standardize=False).fit(Xs, y)
    ref = SKPerceptron(eta0=lr, shuffle=False, max_iter=5, tol=None, penalty=None).fit(Xs, y)
    np.testing.assert_allclose(ours.coefficients_, ref.coef_[0], atol=1e-12)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_[0], atol=1e-12)


# ---- SLP -----------------------------------------------------------------------------------

def test_slp_gradient_matches_finite_differences():
    X, y = data(30, d=3)
    w, b, l2 = np.array([0.3, -0.2, 0.1]), 0.05, 0.1
    _, gw, gb = SLP._loss_grad(X, y, w, b, l2)
    eps = 1e-6
    for i in range(3):
        step = np.zeros(3)
        step[i] = eps
        num = (SLP._loss_grad(X, y, w + step, b, l2)[0] - SLP._loss_grad(X, y, w - step, b, l2)[0]) / (2 * eps)
        assert abs(num - gw[i]) < 1e-6
    num_b = (SLP._loss_grad(X, y, w, b + eps, l2)[0] - SLP._loss_grad(X, y, w, b - eps, l2)[0]) / (2 * eps)
    assert abs(num_b - gb) < 1e-6


def test_slp_is_sigmoid_bce_not_tanh_mse():
    X, y = data()
    model = SLP(max_iter=50, standardize=False).fit(X, y)
    z = X @ model.coefficients_ + model.intercept_
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], 1 / (1 + np.exp(-z)))
    final = loss("binary_cross_entropy", y, probabilities=model.predict_proba(X)[:, 1])["value"]
    assert abs(model.loss_history_[model.best_epoch_ - 1] - final) < 1e-9
    assert model.loss_history_[0] == pytest.approx(np.log(2), abs=0.05)


def test_slp_minibatch_sequence_without_shuffle():
    X, y = data(5, d=2)
    model = SLP(learning_rate=0.1, batch_size=2, max_iter=1, standardize=False, random_state=0).fit(X, [0, 1, 0, 1, 1])
    w, b = SLP(max_iter=1)._init(2, 0)
    yy = np.array([0, 1, 0, 1, 1])
    for lo, hi in [(0, 2), (2, 4), (4, 5)]:
        _, gw, gb = SLP._loss_grad(X[lo:hi], yy[lo:hi], w, b, 0.0)
        w, b = w - 0.1 * gw, b - 0.1 * gb
    np.testing.assert_allclose(model.coefficients_, w)
    full = SLP(learning_rate=0.1, batch_size=None, max_iter=1, standardize=False, random_state=0).fit(X, yy)
    assert not np.allclose(full.coefficients_, model.coefficients_)


def test_slp_l2_shrinks_and_patience_stops_on_monitored_loss():
    X, y = data(200)
    free = SLP(l2=0.0, max_iter=100, standardize=False).fit(X, y)
    reg = SLP(l2=1.0, max_iter=100, standardize=False).fit(X, y)
    assert np.linalg.norm(reg.coefficients_) < np.linalg.norm(free.coefficients_)
    model = SLP(learning_rate=0.1, patience=3, max_iter=100).fit(X, y, eval_set=(X, 1 - y))
    assert model.best_epoch_ == 1 and model.epochs_run_ == 4


def test_slp_is_distinct_from_logistic_and_perceptron():
    assert not issubclass(SLP, LogisticRegression) and not issubclass(SLP, Perceptron)
    assert "class_weight" not in inspect.signature(SLP).parameters
    assert {"batch_size", "patience"} <= set(inspect.signature(SLP).parameters)
    assert not {"batch_size", "patience"} & set(inspect.signature(LogisticRegression).parameters)
    assert hasattr(SLP(), "predict_proba") and not hasattr(Perceptron(), "predict_proba")


# ---- Decision tree -------------------------------------------------------------------------

def test_tree_pure_leaf_probability_uses_correct_class():
    tree = DecisionTreeClassifier(max_depth=1).fit([[0.0], [0.1], [1.0], [1.1]], [0, 0, 1, 1])
    np.testing.assert_array_equal(tree.predict_proba([[0.0], [1.0]]), [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(tree.predict([[0.0], [1.0]]), [0, 1])


def _leaves(node):
    return [node] if node.is_leaf() else _leaves(node.left) + _leaves(node.right)


def _depth(node):
    return 0 if node.is_leaf() else 1 + max(_depth(node.left), _depth(node.right))


@pytest.mark.parametrize("msl", [1, 5, 10])
def test_tree_min_samples_leaf_and_depth(msl):
    X, y = data(300, seed=12)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=msl).fit(X, y)
    assert min(leaf.value.sum() for leaf in _leaves(tree.tree_)) >= msl
    assert _depth(tree.tree_) <= 4


@pytest.mark.parametrize("depth,msl", [(2, 5), (3, 10), (3, 25)])
def test_tree_matches_sklearn_gini(depth, msl):
    X, y = data(600, seed=13)
    Xt = data(100, seed=14)[0]
    ours = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl).fit(X, y)
    ref = SKTree(max_depth=depth, min_samples_leaf=msl, criterion="gini").fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(Xt), ref.predict_proba(Xt))


def test_tree_tie_breaking_and_zero_gain_split():
    X, y = data(100)
    dup = np.column_stack([X[:, 0], X[:, 0]])
    assert DecisionTreeClassifier(max_depth=1).fit(dup, y).tree_.feature == 0
    xor_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, dtype=float)
    xor_y = np.array([0, 1, 1, 0] * 5)
    assert (DecisionTreeClassifier(max_depth=2).fit(xor_X, xor_y).predict(xor_X) == xor_y).all()


def test_tree_full_state_roundtrip():
    X, y = data(200, seed=15)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5).fit(X, y)
    state = json.loads(json.dumps(tree.to_dict()))
    loaded = DecisionTreeClassifier.from_dict(state)
    assert loaded.to_dict() == tree.to_dict()
    assert (loaded.max_depth, loaded.min_samples_leaf, loaded.n_features_) == (4, 5, 4)
    np.testing.assert_array_equal(loaded.classes_, [0, 1])

"""M5 repair tests: M2-consistent thresholds (score >= threshold), float64-extreme safety, warning-free divergence."""
import warnings

import numpy as np
import pytest

from src.modeling.classification import DecisionTreeClassifier, GaussianNaiveBayes, KNN, LogisticRegression, Perceptron, SLP
from src.modeling.metrics import classification_metrics

BIG = 1.7e308


@pytest.fixture(autouse=True)
def _no_numpy_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _data(n=60, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    return X, (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(int)


# ---- 1. threshold semantics: probability >= 0.5, decision >= 0.0 ------------------------------

def test_nb_exact_half_probability_predicts_positive():
    X, y = np.array([[-1.5], [-0.5], [0.5], [1.5]]), np.array([0, 0, 1, 1])
    model = GaussianNaiveBayes().fit(X, y)
    q = np.array([[0.0]])  # equidistant from both class means, equal variances and priors
    assert model.predict_proba(q)[0, 1] == 0.5
    assert model.predict(q)[0] == 1


def test_tree_exact_half_leaf_predicts_positive():
    tree = DecisionTreeClassifier().fit(np.zeros((4, 1)), [0, 1, 0, 1])  # unsplittable -> leaf counts [2, 2]
    assert tree.predict_proba([[0.0]])[0, 1] == 0.5
    assert tree.predict([[0.0]])[0] == 1


def test_knn_vote_tie_predicts_positive():
    model = KNN(2, standardize=False).fit([[-1.0], [1.0]], [1, 0])
    assert model.predict_proba([[0.0]])[0, 1] == 0.5
    assert model.predict([[0.0]])[0] == 1


def test_perceptron_zero_score_predicts_positive():
    model = Perceptron(standardize=False)
    model.coefficients_, model.intercept_ = np.array([1.0]), 0.0
    assert model.decision_function([[0.0]])[0] == 0.0
    assert model.predict([[0.0]])[0] == 1


@pytest.mark.parametrize("cls", [LogisticRegression, SLP])
def test_sigmoid_zero_logit_predicts_positive(cls):
    model = cls(standardize=False)
    model.coefficients_, model.intercept_ = np.array([1.0]), 0.0
    assert model.predict_proba([[0.0]])[0, 1] == 0.5
    assert model.predict([[0.0]])[0] == 1


@pytest.mark.parametrize("make,score", [
    (lambda: LogisticRegression(max_iter=200), "proba"), (lambda: GaussianNaiveBayes(), "proba"),
    (lambda: KNN(3), "proba"), (lambda: DecisionTreeClassifier(max_depth=3), "proba"),
    (lambda: SLP(max_iter=30), "proba"), (lambda: Perceptron(max_iter=20), "decision"),
])
def test_predict_agrees_with_m2_labels(make, score):
    X, y = _data()
    model = make().fit(X, y)
    if score == "proba":
        s, thr = model.predict_proba(X)[:, 1], 0.5
    else:
        s, thr = model.decision_function(X), 0.0
    cm = np.array(classification_metrics(y, s, threshold=thr)["confusion_matrix"])
    pred = model.predict(X)
    np.testing.assert_array_equal(pred, (s >= thr).astype(int))
    assert cm[:, 1].sum() == pred.sum()


# ---- 2. overflow safety near float64 extremes -------------------------------------------------

@pytest.mark.parametrize("lo,hi", [(1.7e308, 1.79e308), (-1.79e308, -1.7e308), (-BIG, BIG)])
def test_tree_midpoint_does_not_overflow(lo, hi):
    X, y = np.array([[lo], [lo], [hi], [hi]]), np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    assert np.isfinite(tree.tree_.threshold) and lo <= tree.tree_.threshold < hi
    np.testing.assert_array_equal(tree.predict(X), y)
    np.testing.assert_array_equal(DecisionTreeClassifier.from_dict(tree.to_dict()).predict(X), y)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_extreme_values_give_finite_normalized_probabilities(weights):
    X = np.array([[-BIG, -BIG], [BIG, BIG], [1.6e308, 1.6e308], [-1.6e308, 1.6e308]])
    y = np.array([0, 1, 1, 0])
    q = np.array([[1.65e308, 1.65e308], [-BIG, BIG], [0.0, 0.0]])
    proba = KNN(3, weights=weights, standardize=False).fit(X, y).predict_proba(q)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert KNN(1, weights=weights, standardize=False).fit(X, y).predict(q[:1])[0] == 1


def test_knn_extreme_query_with_standardization():
    X, y = _data()
    proba = KNN(5, weights="distance").fit(X, y).predict_proba(np.full((2, 3), 1e300))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_knn_subnormal_distances_keep_finite_distance_weights():
    X = np.array([[0.0], [5e-324], [1.0]])
    proba = KNN(2, weights="distance", standardize=False).fit(X, [0, 1, 1]).predict_proba([[3e-324]])
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_nb_extreme_queries_give_finite_normalized_probabilities():
    X, y = _data()
    model = GaussianNaiveBayes().fit(X, y)
    q = np.array([[BIG, BIG, BIG], [-BIG, BIG, -BIG], [1e200, 0, 0], [0.0, 0.0, 0.0]])
    proba = model.predict_proba(q)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.predict(q), (proba[:, 1] >= 0.5).astype(int))


def test_nb_saturated_rows_follow_the_larger_variance_class_like_moderate_rows():
    # Gaussian NB with equal means: far from the mean the wider class (1) must win, for any sign and magnitude.
    X, y = np.array([[-1.0], [1.0], [-3.0], [3.0]]), np.array([0, 0, 1, 1])
    model = GaussianNaiveBayes().fit(X, y)
    moderate = model.predict_proba(np.array([[50.0], [-50.0]]))
    extreme = model.predict_proba(np.array([[1e250], [-1e250], [BIG], [-BIG]]))
    np.testing.assert_array_equal(moderate.argmax(axis=1), [1, 1])
    np.testing.assert_array_equal(extreme, [[0.0, 1.0]] * 4)


def test_nb_indistinguishable_saturated_classes_fall_back_to_finite_terms():
    # symmetric classes: |x - mu0| == |x - mu1| in float64 at 1e250, so the honest answer is the prior-only 0.5
    model = GaussianNaiveBayes().fit(np.array([[-2.0], [-1.0], [1.0], [2.0]]), [0, 0, 1, 1])
    np.testing.assert_array_equal(model.predict_proba([[1e250]]), [[0.5, 0.5]])


@pytest.mark.parametrize("make", [GaussianNaiveBayes, lambda: KNN(3), lambda: LogisticRegression(max_iter=5)])
def test_training_data_whose_variance_overflows_is_rejected_not_nan(make):
    X, y = _data()
    X[0, 0], X[1, 0] = BIG, -BIG
    with pytest.raises(ValueError, match="overflow"):
        make().fit(X, y)


# ---- 3. logistic divergence is an explicit ValueError, never a RuntimeWarning -----------------

def test_logistic_divergence_raises_value_error_without_runtime_warning():
    X, _ = _data(300, 4)
    y = (np.arange(300) % 5 == 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="diverged"):
            LogisticRegression(l2=1e6, standardize=False).fit(X, y)

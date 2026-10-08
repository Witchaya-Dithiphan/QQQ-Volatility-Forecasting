"""M6 XGBoost audit contract tests: hand-computed fixtures, validation, strict-JSON state, Q75 smoke."""
import inspect
import json

import numpy as np
import pytest

from src.modeling.ensemble.xgboost import XGBoostClassifier

X4 = np.array([[1.0], [2.0], [3.0], [4.0]])
G4 = np.array([0.5, 0.5, 0.5, -0.5])  # p=0.5, y=[0,0,0,1]
H4 = np.full(4, 0.25)


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


# ---- frozen constructor surface -------------------------------------------------

def test_constructor_matches_frozen_tuple_fields():
    names = list(inspect.signature(XGBoostClassifier.__init__).parameters)[1:]
    assert names == ["n_estimators", "max_depth", "learning_rate", "reg_lambda", "gamma", "min_child_weight"]
    m = XGBoostClassifier()
    assert (m.n_estimators, m.max_depth, m.learning_rate, m.reg_lambda, m.gamma) == (50, 2, 0.1, 1.0, 0.0)


# ---- hand-computed one-step fixtures --------------------------------------------

def test_first_step_gradients_and_leaf_weight_hand_computed():
    # y=[0,0,1,1], p=.5 -> g=[.5,.5,-.5,-.5], h=.25; split 2.5: GL=1,HL=.5,GR=-1,HR=.5, lam=1
    # w_L=-1/1.5=-2/3, w_R=+2/3; gain=.5*(1/1.5+1/1.5-0)=2/3
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    m = XGBoostClassifier(n_estimators=1, max_depth=1, learning_rate=0.1, reg_lambda=1.0, gamma=0.0, min_child_weight=0.5).fit(X, y)
    root = m.trees_[0]
    assert (root.feature, root.threshold) == (0, 2.5)
    assert root.left.leaf_weight == pytest.approx(-2 / 3)
    assert root.right.leaf_weight == pytest.approx(2 / 3)
    np.testing.assert_allclose(m.predict_proba(X)[:, 1], sigmoid(0.1 * np.array([-2, -2, 2, 2]) / 3), rtol=1e-12)
    assert m.loss_history_[0] == pytest.approx(np.mean(np.log1p(np.exp(np.array([-2, -2, 2, 2]) / 30 * np.array([1, 1, -1, -1])))))


def test_gain_lambda_every_denominator_and_leaf_weights():
    # split 3.5: GL=1.5,HL=.75,GR=-.5,HR=.25,G=1,H=1,lam=1
    # gain=.5*(2.25/1.75+.25/1.25-1/2)=0.4928571428...; w_L=-1.5/1.75, w_R=.5/1.25
    m = XGBoostClassifier(max_depth=1, reg_lambda=1.0, gamma=0.0, min_child_weight=0.25)
    root = m._build_tree(X4, G4, H4)
    assert (root.feature, root.threshold) == (0, 3.5)
    assert root.left.leaf_weight == pytest.approx(-1.5 / 1.75)
    assert root.right.leaf_weight == pytest.approx(0.5 / 1.25)


def test_gamma_subtracted_once_boundary():
    kw = dict(max_depth=1, reg_lambda=1.0, min_child_weight=0.25)
    gain = 0.5 * (2.25 / 1.75 + 0.25 / 1.25 - 0.5)
    assert not XGBoostClassifier(gamma=gain - 1e-4, **kw)._build_tree(X4, G4, H4).is_leaf()
    assert XGBoostClassifier(gamma=gain + 1e-4, **kw)._build_tree(X4, G4, H4).is_leaf()
    # a double-subtracted gamma would already have rejected gain/2
    assert not XGBoostClassifier(gamma=gain * 0.6, **kw)._build_tree(X4, G4, H4).is_leaf()


def test_lambda_shrinks_leaf_and_changes_gain():
    # lambda=3: w_R=.5/3.25, gain=.5*(2.25/3.75+.25/3.25-1/4)
    gain = 0.5 * (2.25 / 3.75 + 0.25 / 3.25 - 0.25)
    m = XGBoostClassifier(max_depth=1, reg_lambda=3.0, gamma=gain - 1e-6, min_child_weight=0.25)
    root = m._build_tree(X4, G4, H4)
    assert root.right.leaf_weight == pytest.approx(0.5 / 3.25)
    assert XGBoostClassifier(max_depth=1, reg_lambda=3.0, gamma=gain + 1e-6, min_child_weight=0.25)._build_tree(X4, G4, H4).is_leaf()


def test_min_child_weight_inclusive_boundary():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    g, h = np.array([0.5, 0.5, -0.5, -0.5]), np.full(4, 0.25)
    assert not XGBoostClassifier(max_depth=1, min_child_weight=0.5)._build_tree(X, g, h).is_leaf()  # HL==HR==.5 allowed
    assert XGBoostClassifier(max_depth=1, min_child_weight=0.5001)._build_tree(X, g, h).is_leaf()
    assert XGBoostClassifier(max_depth=1, min_child_weight=0.5001)._build_tree(X, g, h).leaf_weight == pytest.approx(0.0)


def test_max_depth_exact():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (np.sin(3 * X[:, 0]) + X[:, 1] > 0).astype(int)
    for depth in (1, 2, 3):
        m = XGBoostClassifier(n_estimators=3, max_depth=depth, min_child_weight=0.1).fit(X, y)
        assert max(m._tree_depth(t) for t in m.trees_) == depth


# ---- exact split thresholds -----------------------------------------------------

def _thresholds(node, out):
    if not node.is_leaf():
        out.append((node.feature, node.threshold))
        _thresholds(node.left, out)
        _thresholds(node.right, out)
    return out


def test_thresholds_are_midpoints_not_training_values():
    rng = np.random.default_rng(1)
    X = np.round(rng.normal(size=(120, 3)), 1)  # many duplicates
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    m = XGBoostClassifier(n_estimators=4, max_depth=3, min_child_weight=0.1).fit(X, y)
    for tree in m.trees_:  # every root sees all rows, so its threshold sits between two global adjacent values
        f, t = tree.feature, tree.threshold
        u = np.unique(X[:, f])
        i = np.searchsorted(u, t)
        assert t not in u and 0 < i < len(u) and u[i - 1] < t < u[i]
        assert t == pytest.approx((u[i - 1] + u[i]) / 2)


def test_midpoint_exact_two_values_with_duplicates():
    X = np.array([[1.0], [1.0], [2.0], [2.0]])
    m = XGBoostClassifier(max_depth=1, min_child_weight=0.5)
    root = m._build_tree(X, np.array([0.5, 0.5, -0.5, -0.5]), np.full(4, 0.25))
    assert root.threshold == 1.5


def test_adjacent_floats_still_partition_and_extremes_finite():
    a = 1.0
    b = float(np.nextafter(1.0, 2.0))
    X = np.array([[a], [a], [b], [b]])
    root = XGBoostClassifier(max_depth=1, min_child_weight=0.5)._build_tree(X, np.array([0.5, 0.5, -0.5, -0.5]), np.full(4, 0.25))
    assert not root.is_leaf() and a <= root.threshold < b
    Xe = np.array([[1e308], [1e308], [1.7e308], [1.7e308]])
    root = XGBoostClassifier(max_depth=1, min_child_weight=0.5)._build_tree(Xe, np.array([0.5, 0.5, -0.5, -0.5]), np.full(4, 0.25))
    assert np.isfinite(root.threshold) and 1e308 <= root.threshold < 1.7e308


def test_constant_feature_never_splits():
    X = np.ones((6, 1))
    root = XGBoostClassifier(max_depth=2, min_child_weight=0.0)._build_tree(X, np.array([.5, .5, .5, -.5, -.5, -.5]), np.full(6, .25))
    assert root.is_leaf()


def test_tie_prefers_lowest_feature_and_deterministic():
    col = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.column_stack([col, col, col])
    root = XGBoostClassifier(max_depth=1, min_child_weight=0.5)._build_tree(X, np.array([.5, .5, -.5, -.5]), np.full(4, .25))
    assert root.feature == 0


# ---- behavioral reference (independent brute force) -----------------------------

def _ref_tree(X, g, h, depth, md, lam, gam, mcw):
    G, H = g.sum(), h.sum()
    leaf = ("leaf", -G / (H + lam))
    if depth >= md:
        return leaf
    best = (0.0, None)
    for f in range(X.shape[1]):
        u = np.unique(X[:, f])
        for lo, hi in zip(u[:-1], u[1:]):
            t = (lo + hi) / 2
            L = X[:, f] <= t
            GL, HL, GR, HR = g[L].sum(), h[L].sum(), g[~L].sum(), h[~L].sum()
            if HL < mcw or HR < mcw:
                continue
            gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam)) - gam
            if gain > best[0]:
                best = (gain, (f, t, L))
    if best[1] is None:
        return leaf
    f, t, L = best[1]
    return ("split", f, t, _ref_tree(X[L], g[L], h[L], depth + 1, md, lam, gam, mcw), _ref_tree(X[~L], g[~L], h[~L], depth + 1, md, lam, gam, mcw))


def _ref_pred(node, x):
    while node[0] == "split":
        node = node[3] if x[node[1]] <= node[2] else node[4]
    return node[1]


@pytest.mark.parametrize("params", [(6, 2, 0.1, 1.0, 0.0), (5, 3, 0.3, 10.0, 0.1), (4, 1, 0.03, 1.0, 0.1)])
def test_matches_bruteforce_reference(params):
    n, md, lr, lam, gam = params
    rng = np.random.default_rng(7)
    X = np.round(rng.normal(size=(80, 4)), 2)
    y = ((X[:, 0] * X[:, 1] + X[:, 2]) > 0.2).astype(int)
    m = XGBoostClassifier(n, md, lr, lam, gam, min_child_weight=1.0).fit(X, y)
    s = np.zeros(len(X))
    for _ in range(n):
        p = sigmoid(s)
        tree = _ref_tree(X, p - y, p * (1 - p), 0, md, lam, gam, 1.0)
        s += lr * np.array([_ref_pred(tree, x) for x in X])
    np.testing.assert_allclose(m.predict_proba(X)[:, 1], sigmoid(s), rtol=1e-9, atol=1e-12)


def test_matches_xgboost_library_if_installed():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    y = ((X[:, 0] + X[:, 1] ** 2) > 0.8).astype(int)
    ref = xgb.XGBClassifier(n_estimators=20, max_depth=2, learning_rate=0.1, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,
                            tree_method="exact", base_score=0.5, n_jobs=1).fit(X, y)
    ours = XGBoostClassifier(20, 2, 0.1, 1.0, 0.0).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X)[:, 1], ref.predict_proba(X)[:, 1], atol=1e-3)


# ---- objective / stability / no early stopping ----------------------------------

def test_no_early_stopping_all_estimators_built():
    X = np.random.default_rng(0).normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(int)
    m = XGBoostClassifier(n_estimators=7, max_depth=2, learning_rate=1e-9).fit(X, y)
    assert len(m.trees_) == 7 and len(m.loss_history_) == 7


def test_extreme_values_stable():
    X = np.array([[-1e300], [-1e300], [1e300], [1e300]] * 5)
    y = np.array([0, 0, 1, 1] * 5)
    m = XGBoostClassifier(n_estimators=60, max_depth=1, learning_rate=1.0, min_child_weight=0.0).fit(X, y)
    p = m.predict_proba(X)
    assert np.isfinite(p).all() and np.isfinite(m.loss_history_).all()
    assert np.array_equal(m.predict(X), y)


def test_predict_threshold_ge_half():
    m = XGBoostClassifier.from_dict({**_state_template(), "trees": [{"leaf": 0.0}]})
    assert m.predict_proba(np.zeros((1, 2)))[0, 1] == 0.5
    assert m.predict(np.zeros((1, 2)))[0] == 1


# ---- validation -----------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"n_estimators": 0}, {"n_estimators": True}, {"n_estimators": 1.5}, {"max_depth": 0}, {"max_depth": 2.0},
    {"learning_rate": 0}, {"learning_rate": float("nan")}, {"learning_rate": float("inf")},
    {"reg_lambda": -1}, {"reg_lambda": float("nan")}, {"gamma": -0.1}, {"gamma": float("inf")},
    {"min_child_weight": -1}, {"min_child_weight": float("nan")}, {"learning_rate": "0.1"},
])
def test_bad_hyperparameters_rejected(kw):
    with pytest.raises(ValueError):
        XGBoostClassifier(**kw)


@pytest.mark.parametrize("X,y", [
    (np.ones(4), np.array([0, 1, 0, 1])),
    (np.ones((0, 2)), np.array([])),
    (np.ones((4, 0)), np.array([0, 1, 0, 1])),
    (np.array([[np.nan], [1], [2], [3]]), np.array([0, 1, 0, 1])),
    (np.array([[np.inf], [1], [2], [3]]), np.array([0, 1, 0, 1])),
    (np.ones((4, 1)), np.array([0, 1, 2, 1])),
    (np.ones((4, 1)), np.array([-1, 1, 0, 1])),
    (np.ones((4, 1)), np.array([0, 0.5, 1, 1])),
    (np.ones((4, 1)), np.array([0, np.nan, 1, 1])),
    (np.ones((4, 1)), np.array([0, 1, 0])),
    (np.ones((4, 1)), np.ones((4, 2))),
])
def test_bad_fit_inputs_rejected(X, y):
    with pytest.raises(ValueError):
        XGBoostClassifier(n_estimators=1).fit(X, y)


def test_predict_input_validation():
    m = XGBoostClassifier(n_estimators=2).fit(np.arange(8.0).reshape(4, 2), np.array([0, 0, 1, 1]))
    for bad in (np.ones(2), np.ones((2, 3)), np.array([[np.nan, 1.0]]), np.array([[np.inf, 1.0]]), np.ones((0, 2))):
        with pytest.raises(ValueError):
            m.predict_proba(bad)
    with pytest.raises(ValueError, match="not fitted"):
        XGBoostClassifier().predict(np.ones((1, 2)))


def test_float_and_bool_binary_labels_accepted():
    X = np.arange(8.0).reshape(4, 2)
    a = XGBoostClassifier(n_estimators=2, min_child_weight=0.5).fit(X, np.array([0, 0, 1, 1]))
    for y in (np.array([0.0, 0.0, 1.0, 1.0]), np.array([False, False, True, True])):
        assert XGBoostClassifier(n_estimators=2, min_child_weight=0.5).fit(X, y).to_dict() == a.to_dict()


# ---- state / persistence --------------------------------------------------------

def _state_template():
    return {"n_estimators": 1, "max_depth": 1, "learning_rate": 0.1, "reg_lambda": 1.0, "gamma": 0.0,
            "min_child_weight": 1.0, "n_features": 2, "classes": [0, 1], "trees": [], "loss_history": [0.6931471805599453]}


def _data():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(90, 3))
    return X, ((X[:, 0] + X[:, 1] * X[:, 2]) > 0.3).astype(int)


def test_strict_json_state_complete_and_exact_reload():
    X, y = _data()
    m = XGBoostClassifier(6, 3, 0.2, 2.0, 0.05, min_child_weight=0.5).fit(X, y)
    state = m.to_dict()
    for key in ("n_estimators", "max_depth", "learning_rate", "reg_lambda", "gamma", "min_child_weight", "n_features", "classes", "trees", "loss_history"):
        assert key in state
    assert not {"tol", "random_state", "max_trees"} & set(state)
    text = json.dumps(state, allow_nan=False)
    loaded = XGBoostClassifier.from_dict(json.loads(text))
    assert np.array_equal(m.predict_proba(X), loaded.predict_proba(X))
    assert np.array_equal(m.predict(X), loaded.predict(X))
    assert loaded.to_dict() == state
    assert (loaded.n_estimators, loaded.max_depth, loaded.learning_rate, loaded.reg_lambda, loaded.gamma, loaded.min_child_weight) == (6, 3, 0.2, 2.0, 0.05, 0.5)
    assert list(loaded.classes_) == [0, 1] and loaded.n_features_ == 3


def test_from_dict_rejects_malformed_and_unfitted_to_dict_raises():
    with pytest.raises(ValueError, match="not fitted"):
        XGBoostClassifier().to_dict()
    X, y = _data()
    good = XGBoostClassifier(2, 2).fit(X, y).to_dict()
    for broken in ({k: v for k, v in good.items() if k != "gamma"}, {**good, "trees": []}, {**good, "n_features": 0}, {**good, "learning_rate": -1}):
        with pytest.raises((ValueError, KeyError)):
            XGBoostClassifier.from_dict(broken)


def test_fit_twice_resets_and_is_deterministic():
    X, y = _data()
    m = XGBoostClassifier(4, 2)
    first = m.fit(X, y).to_dict()
    m.fit(X[:, :2] * 3, 1 - y)  # different shape and labels
    assert m.n_features_ == 2 and len(m.trees_) == 4 and len(m.loss_history_) == 4
    assert m.fit(X, y).to_dict() == first
    assert XGBoostClassifier(4, 2).fit(X, y).to_dict() == first


def test_fit_does_not_mutate_inputs():
    X, y = _data()
    X0, y0 = X.copy(), y.copy()
    XGBoostClassifier(2, 2).fit(X, y)
    assert np.array_equal(X, X0) and np.array_equal(y, y0)


# ---- Q75 real-data smoke, M2 metrics/threshold reuse ----------------------------

def test_q75_real_data_smoke_uses_m2_metrics_and_never_touches_test(monkeypatch):
    from src.modeling import datasets
    from src.modeling.metrics import classification_metrics, select_threshold

    def boom(*a, **k):
        raise AssertionError("Test split must not be accessed")
    monkeypatch.setattr(datasets, "load_test", boom)
    data = datasets.load_train_validation("with_spike")
    ytr, yva = data.train.y_classification, data.validation.y_classification
    assert set(np.unique(ytr)) <= {0, 1} and 0.05 < ytr.mean() < 0.5  # Q75 label, not a median split
    m = XGBoostClassifier(50, 2, 0.1, 1.0, 0.0).fit(data.train.X, ytr)
    proba = m.predict_proba(data.validation.X)[:, 1]
    pred = m.predict(data.validation.X)
    assert np.array_equal(pred, (proba >= 0.5).astype(int))
    metrics = classification_metrics(yva, proba, threshold=0.5)
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    assert tp + tn == (pred == yva).sum() and sum(map(sum, metrics["confusion_matrix"])) == len(yva)
    assert select_threshold(yva, proba)["selection_split"] == "validation"

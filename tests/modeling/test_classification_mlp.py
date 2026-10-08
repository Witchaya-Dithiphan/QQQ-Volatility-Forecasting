"""Behavioral tests for the scratch ReLU/sigmoid MLP classifier and its chronological epoch-selection adapter (M6)."""
import inspect
import json
import math

import numpy as np
import pytest

from src.modeling.classification.mlp import (EpochMonitor, MLPClassifier, batch_slices, forward, init_parameters,
                                             loss_and_gradients, train_epoch)
from src.modeling.configuration import load_config
from src.modeling.datasets import load_train_validation
from src.modeling.metrics import classification_metrics, loss, select_threshold


def blobs(n=200, d=4, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    X = rng.normal(size=(n, d)) + (2 * y[:, None] - 1) * signal * np.array([1.0, 0.5, 0.0, 0.0][:d])
    return X * np.array([5.0, 1.0, 0.1, 30.0][:d]) + 3.0, y


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# ---- forward / loss / gradients -------------------------------------------
def test_forward_hand_example_relu_then_sigmoid():
    W = [np.array([[1.0], [-1.0]]), np.array([[2.0]])]
    b = [np.array([0.5]), np.array([-1.0])]
    acts, logits = forward(W, b, np.array([[2.0, 1.0], [0.0, 3.0]]))
    np.testing.assert_allclose(acts[1], [[1.5], [0.0]])  # ReLU(1.5), ReLU(-2.5)
    np.testing.assert_allclose(logits, [2.0, -1.0])
    value, _, _ = loss_and_gradients(W, b, np.array([[2.0, 1.0], [0.0, 3.0]]), np.array([1, 0]), l2=0.0)
    expected = (-math.log(sigmoid(2.0)) - math.log(1 - sigmoid(-1.0))) / 2
    assert value == pytest.approx(expected)


def test_l2_penalty_applies_to_weights_not_biases():
    W = [np.array([[1.0, 2.0]]), np.array([[3.0], [4.0]])]
    b = [np.array([5.0, 6.0]), np.array([7.0])]
    X, y = np.array([[0.0]]), np.array([1])
    base, _, _ = loss_and_gradients(W, b, X, y, l2=0.0)
    pen, gW, gb = loss_and_gradients(W, b, X, y, l2=0.1)
    assert pen - base == pytest.approx(0.5 * 0.1 * (1 + 4 + 9 + 16))


@pytest.mark.parametrize("hidden", [[3], [4, 3]])
@pytest.mark.parametrize("l2", [0.0, 0.05])
def test_backprop_matches_finite_differences(hidden, l2):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(12, 3)), rng.integers(0, 2, 12)
    W, b = init_parameters([3, *hidden, 1], seed=3)
    b = [v + rng.normal(size=v.shape) * 0.1 for v in b]  # exercise biases off zero
    _, gW, gb = loss_and_gradients(W, b, X, y, l2)
    eps = 1e-6
    for params, grads in ((W, gW), (b, gb)):
        for layer, grad in zip(params, grads):
            for idx in np.ndindex(layer.shape):
                keep = layer[idx]
                layer[idx] = keep + eps; hi = loss_and_gradients(W, b, X, y, l2)[0]
                layer[idx] = keep - eps; lo = loss_and_gradients(W, b, X, y, l2)[0]
                layer[idx] = keep
                assert grad[idx] == pytest.approx((hi - lo) / (2 * eps), rel=1e-4, abs=1e-7)


def test_loss_finite_for_extreme_logits_and_probabilities_in_unit_interval():
    W, b = [np.array([[1e3]])], [np.array([0.0])]
    X, y = np.array([[-1e3], [1e3], [0.0]]), np.array([1, 0, 1])
    value, gW, gb = loss_and_gradients(W, b, X, y, 0.0)
    assert np.isfinite(value) and np.isfinite(gW[0]).all() and np.isfinite(gb[0]).all()


def test_init_is_seeded_and_he_scaled_with_zero_biases():
    a, ab = init_parameters([8, 16, 1], seed=42)
    c, _ = init_parameters([8, 16, 1], seed=42)
    d, _ = init_parameters([8, 16, 1], seed=7)
    assert all(np.array_equal(p, q) for p, q in zip(a, c)) and not np.array_equal(a[0], d[0])
    assert all((v == 0).all() for v in ab)
    assert a[0].shape == (8, 16) and a[1].shape == (16, 1)
    big = init_parameters([400, 1000], seed=0)[0][0]
    assert big.std() == pytest.approx(math.sqrt(2 / 400), rel=0.02)


# ---- chronological minibatches ---------------------------------------------
def test_batch_slices_are_contiguous_in_order_with_short_last_batch():
    assert batch_slices(70, 32) == [(0, 32), (32, 64), (64, 70)]
    assert batch_slices(64, 32) == [(0, 32), (32, 64)]
    assert batch_slices(5, 32) == [(0, 5)]


def test_train_epoch_equals_manual_sequential_sgd_without_shuffle():
    X, y = blobs(70, 3, seed=2)
    W, b = init_parameters([3, 4, 1], seed=5)
    W2, b2 = [w.copy() for w in W], [v.copy() for v in b]
    train_epoch(W, b, X, y, lr=0.01, l2=1e-3, batch_size=32)
    for lo, hi in [(0, 32), (32, 64), (64, 70)]:
        _, gW, gb = loss_and_gradients(W2, b2, X[lo:hi], y[lo:hi], 1e-3)
        for p, g in zip(W2 + b2, gW + gb):
            p -= 0.01 * g
    for p, q in zip(W + b, W2 + b2):
        np.testing.assert_array_equal(p, q)


def test_one_sgd_step_matches_sklearn_mlpclassifier():
    nn = pytest.importorskip("sklearn.neural_network")
    X, y = blobs(32, 3, seed=4)
    Z = (X - X.mean(0)) / X.std(0)
    W, b = init_parameters([3, 5, 1], seed=1)
    ref = nn.MLPClassifier(hidden_layer_sizes=(5,), solver="sgd", momentum=0.0, nesterovs_momentum=False, learning_rate_init=0.05,
                           alpha=1e-3 * 32, batch_size=32, shuffle=False, random_state=42, tol=0.0)
    ref.partial_fit(Z, y, classes=[0, 1])  # allocates sklearn state; its own weights are overwritten
    ref.coefs_, ref.intercepts_ = [w.copy() for w in W], [v.reshape(-1).copy() for v in b]
    ref.partial_fit(Z, y)
    train_epoch(W, b, Z, y, lr=0.05, l2=1e-3, batch_size=32)
    for ours, theirs in zip(W, ref.coefs_):
        np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-12)
    for ours, theirs in zip(b, ref.intercepts_):
        np.testing.assert_allclose(ours.reshape(-1), theirs, rtol=1e-9, atol=1e-12)


def test_trained_behavior_agrees_with_sklearn_reference():
    nn = pytest.importorskip("sklearn.neural_network")
    X, y = blobs(300, 4, seed=5)
    ours = MLPClassifier([16], learning_rate=0.05).fit_epochs(X, y, 200)
    Z = ours.scaler_.transform(X)
    ref = nn.MLPClassifier(hidden_layer_sizes=(16,), solver="sgd", momentum=0.0, nesterovs_momentum=False, learning_rate_init=0.05,
                           batch_size=32, shuffle=False, max_iter=200, tol=0.0, n_iter_no_change=1000, random_state=42, alpha=0.0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn ConvergenceWarning at max_iter is expected
        ref.fit(Z, y)
    assert np.mean(ours.predict(X) == y) >= 0.9 and np.mean(ref.predict(Z) == y) >= 0.9
    assert np.mean(ours.predict(X) == ref.predict(Z)) >= 0.9


def test_training_reduces_loss():
    X, y = blobs(150, 4, seed=6)
    short = MLPClassifier([8], learning_rate=0.05).fit_epochs(X, y, 1)
    long = MLPClassifier([8], learning_rate=0.05).fit_epochs(X, y, 100)
    ll = lambda m: loss("binary_cross_entropy", y, probabilities=m.predict_proba(X)[:, 1])["value"]
    assert ll(long) < ll(short)


# ---- early-stopping monitor -------------------------------------------------
def test_monitor_requires_min_delta_improvement_and_keeps_earliest_best():
    m = EpochMonitor(patience=2, min_delta=1e-6)
    assert [m.update(v) for v in (1.0, 0.9, 0.9 - 5e-7, 0.9 - 9e-7)] == [False, False, False, True]
    assert m.best_epoch == 2 and m.epoch == 4 and m.best_loss == 0.9  # sub-min_delta gains never move the best epoch


def test_monitor_counter_resets_on_real_improvement_and_ties_keep_earlier():
    m = EpochMonitor(patience=2, min_delta=1e-6)
    stops = [m.update(v) for v in (1.0, 1.0, 0.5, 0.5, 0.4)]
    assert stops == [False, False, False, False, False] and m.best_epoch == 5
    assert [m.update(0.4), m.update(0.4)] == [False, True] and m.best_epoch == 5


def test_first_epoch_is_always_an_improvement():
    m = EpochMonitor(patience=1, min_delta=1e-6)
    assert m.update(5.0) is False and m.best_epoch == 1


# ---- chronological adapter ---------------------------------------------------
def adapter_model(**kw):
    kw = {"hidden_layers": [4], "learning_rate": 0.05, "max_epochs": 40, "patience": 5, **kw}
    return MLPClassifier(**kw)


def test_internal_tail_and_original_day_purge_on_contiguous_positions():
    X, y = blobs(200, 4, seed=7)
    m = adapter_model().fit(X, y)
    s = m.selection_
    assert (s["n_tail"], s["tail_start"], s["n_subtrain"]) == (30, 170, 165)  # rows 165..169 purged, tail = 170..199


def test_purge_counts_original_trading_days_not_rows():
    X, y = blobs(200, 4, seed=7)
    positions = np.arange(200) * 2  # every other original day was removed (Non-Spike style)
    s = adapter_model().fit(X, y, original_positions=positions).selection_
    assert s["tail_start"] == 170 and s["n_subtrain"] == 168  # positions < 340 - 5 -> rows 0..167
    assert positions[s["n_subtrain"] - 1] < positions[s["tail_start"]] - 5 <= positions[s["n_subtrain"]]


def test_scaler_for_selection_is_subtrain_and_refit_scaler_is_full_train():
    X, y = blobs(200, 4, seed=8)
    m = adapter_model().fit(X, y)
    n = m.selection_["n_subtrain"]
    np.testing.assert_allclose(m.selection_["subtrain_scaler_mean"], X[:n].mean(axis=0))
    np.testing.assert_allclose(m.scaler_.mean_, X.mean(axis=0))
    assert not np.allclose(m.scaler_.mean_, X[:n].mean(axis=0))


def test_selection_curve_is_tail_bce_of_subtrain_model_at_each_epoch():
    X, y = blobs(200, 4, seed=9)
    m = adapter_model().fit(X, y)
    s, best = m.selection_, m.selection_["best_epoch"]
    probe = adapter_model().fit_epochs(X[:s["n_subtrain"]], y[:s["n_subtrain"]], best)  # same seed, subtrain scaler
    p = probe.predict_proba(X[s["tail_start"]:])[:, 1]
    expected = loss("binary_cross_entropy", y[s["tail_start"]:], probabilities=p)["value"]
    assert s["tail_bce"][best - 1] == pytest.approx(expected, rel=1e-12)
    assert len(s["tail_bce"]) == s["stopped_epoch"]
    assert s["best_epoch"] == 1 + int(np.argmin(s["tail_bce"][:best]))  # earliest best


def test_early_stopping_status_and_patience_distance():
    X, y = blobs(200, 4, seed=10, signal=0.0)  # pure noise -> tail loss stops improving quickly
    s = adapter_model(max_epochs=500, patience=4).fit(X, y).selection_
    assert s["status"] == "early_stopped" and s["stopped_epoch"] == s["best_epoch"] + 4 < 500


def test_max_epochs_status_when_never_stopped():
    X, y = blobs(200, 4, seed=10)
    s = adapter_model(max_epochs=2, patience=50).fit(X, y).selection_
    assert s["status"] == "max_epochs" and s["stopped_epoch"] == 2


def test_refit_reinitializes_and_trains_exactly_selected_epochs_without_early_stopping():
    X, y = blobs(200, 4, seed=11)
    m = adapter_model().fit(X, y)
    best = m.selection_["best_epoch"]
    assert m.epochs_trained_ == best
    fresh = adapter_model().fit_epochs(X, y, best)
    for p, q in zip(m.weights_ + m.biases_, fresh.weights_ + fresh.biases_):
        np.testing.assert_array_equal(p, q)  # same seed-42 init, full-Train scaler, exact epoch count


def test_external_validation_cannot_choose_epochs():
    assert list(inspect.signature(MLPClassifier.fit).parameters) == ["self", "X", "y", "original_positions"]


def test_adapter_is_deterministic():
    X, y = blobs(200, 4, seed=12)
    assert adapter_model().fit(X, y).to_json() == adapter_model().fit(X, y).to_json()


# ---- validation of inputs ----------------------------------------------------
@pytest.mark.parametrize("kwargs", [dict(hidden_layers=[]), dict(hidden_layers=[0]), dict(hidden_layers=[4.5]), dict(learning_rate=0),
                                    dict(l2=-1), dict(batch_size=0), dict(max_epochs=0), dict(patience=0), dict(min_delta=-1),
                                    dict(tail_fraction=0), dict(tail_fraction=1), dict(purge=-1), dict(seed=1.5)])
def test_bad_hyperparameters_rejected(kwargs):
    with pytest.raises(ValueError):
        MLPClassifier(**kwargs)


def test_bad_inputs_rejected():
    X, y = blobs(100, 3, seed=13)
    m = adapter_model()
    for bad_X, bad_y in ((X[:, 0], y), (X, y[:-1]), (X, np.where(y == 1, 2, 0)), (np.where(np.eye(100, 3) == 1, np.nan, X), y)):
        with pytest.raises(ValueError):
            m.fit(bad_X, bad_y)
    with pytest.raises(ValueError):
        m.fit(X, y, original_positions=np.arange(99))
    with pytest.raises(ValueError):
        m.fit(X, y, original_positions=np.r_[np.arange(99), 5])  # not strictly increasing
    with pytest.raises(ValueError):
        m.fit(X[:6], y[:6])  # too short for tail + purge
    with pytest.raises(ValueError):
        m.fit_epochs(X, y, 0)


def test_predict_checks_fit_state_and_dimensions():
    X, y = blobs(100, 3, seed=14)
    with pytest.raises(ValueError):
        adapter_model().predict_proba(X)
    m = adapter_model().fit_epochs(X, y, 2)
    with pytest.raises(ValueError):
        m.predict_proba(X[:, :2])
    with pytest.raises(ValueError):
        adapter_model().to_json()


# ---- probabilities / thresholds / persistence -------------------------------
def test_predict_proba_is_genuine_and_threshold_boundary_is_inclusive():
    X, y = blobs(100, 3, seed=15)
    m = adapter_model().fit_epochs(X, y, 5)
    p = m.predict_proba(X)
    assert p.shape == (100, 2) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    t = float(p[3, 1])
    assert m.predict(X, threshold=t)[3] == 1  # score == threshold -> positive
    assert m.predict(X, threshold=np.nextafter(t, 1.0))[3] == 0


def test_strict_json_exact_roundtrip_of_complete_state():
    X, y = blobs(200, 4, seed=16)
    m = adapter_model(hidden_layers=[5, 3], l2=1e-4).fit(X, y)
    text = m.to_json()
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    clone = MLPClassifier.from_json(text)
    assert clone.to_json() == text
    np.testing.assert_array_equal(clone.predict_proba(X), m.predict_proba(X))
    assert clone.selection_ == m.selection_ and clone.epochs_trained_ == m.epochs_trained_


def test_from_json_rejects_nonfinite_constants_and_wrong_kind():
    X, y = blobs(100, 3, seed=17)
    text = adapter_model().fit_epochs(X, y, 2).to_json()
    with pytest.raises(ValueError):
        MLPClassifier.from_json(text.replace('"seed": 42', '"seed": NaN', 1))
    with pytest.raises(ValueError):
        MLPClassifier.from_json(text.replace('"mlp_classifier"', '"other"', 1))


def test_compat_alias_is_the_scratch_class():
    from src.modeling import advanced
    assert advanced.MLP is MLPClassifier


def test_frozen_defaults_match_config():
    cfg = load_config()
    m, neural = MLPClassifier(), cfg["neural"]
    assert (m.batch_size, m.max_epochs, m.patience, m.seed) == (cfg["models"]["mlp"]["parameters"]["batch"], 1000, 50, 42)
    assert (m.min_delta, m.tail_fraction, m.purge) == (neural["min_delta"], neural["internal_tail_fraction"], neural["purge_original_trading_days"])
    assert cfg["models"]["mlp"]["parameters"]["shuffle"] is False


# ---- real-data smoke (Q75 labels, Train-only fit, Validation scoring) -------
def test_real_data_smoke_with_and_without_spike_positions():
    spike = load_train_validation("with_spike")
    clean = load_train_validation("non_spike")
    for data in (spike, clean):
        positions = np.searchsorted(spike.train.dates, data.train.dates)  # original timeline index
        m = MLPClassifier([16], learning_rate=0.01, max_epochs=15, patience=5).fit(data.train.X, data.train.y_classification, positions)
        s = m.selection_
        assert positions[s["n_subtrain"] - 1] < positions[s["tail_start"]] - 5
        p = m.predict_proba(data.validation.X)[:, 1]
        yv = data.validation.y_classification
        assert p.shape == yv.shape and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
        np.testing.assert_allclose(m.scaler_.mean_, data.train.X.mean(axis=0))  # Validation never fits preprocessing
        assert np.isfinite(loss("binary_cross_entropy", yv, probabilities=p)["value"])
        assert classification_metrics(yv, p)["roc_auc"]["value"] is not None
        assert select_threshold(yv, p)["selection_split"] == "validation"

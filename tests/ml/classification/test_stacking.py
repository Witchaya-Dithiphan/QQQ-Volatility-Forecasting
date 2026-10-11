"""Stacking: the fold and purge rules worked by hand, leakage probes, then a fold-by-fold rebuild.

sklearn's StackingClassifier has no purge, so it cannot be an exact reference. The
exact check here rebuilds the same folds out of each base model's own sklearn
reference and compares the meta features and the final probabilities.
"""

import numpy as np
import pytest

from src.ml.classification.stacking import BASES, META_FEATURES, Stacking, fold_plan
from src.ml.core.data import load
from src.ml.core.persist import load_npz, save_npz
from src.ml.core.preprocess import Standardizer
from src.ml.registry import build, model_names


def _problem(seed=0, rows=400):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(rows, 4))
    y = (X[:, 0] + 0.6 * X[:, 1] + 0.8 * rng.normal(size=rows) > 0).astype(int)
    return X, y


def test_fold_plan_on_a_hundred_day_timeline():
    # The first 40 days are history only; days 40..99 split into four blocks of 15.
    # With a 5-day purge, block [40, 55) may be trained on days 0..34 and no later.
    plan = fold_plan(np.arange(100), 100, folds=4, initial_fraction=0.4, purge=5)
    assert [(start, stop) for _, _, start, stop in plan] == [(40, 55), (55, 70), (70, 85), (85, 100)]
    assert [int(train.sum()) for train, _, _, _ in plan] == [35, 50, 65, 80]
    assert [int(block.sum()) for _, block, _, _ in plan] == [15, 15, 15, 15]
    for train, block, _, _ in plan:
        assert not (train & block).any()


def test_purge_counts_trading_days_not_rows():
    # Days 30..37 are missing, as if a spike filter had removed them. Block [40, 55)
    # must still stop its history at day 34, which leaves rows for days 0..29 only.
    # Counting 5 rows back from the block would wrongly drop days 25..29 as well.
    positions = np.concatenate([np.arange(30), np.arange(38, 100)])
    train, block, start, _ = fold_plan(positions, 100, folds=4, initial_fraction=0.4, purge=5)[0]
    assert start == 40
    assert positions[train].tolist() == list(range(30))
    assert positions[block].tolist() == list(range(40, 55))


def test_no_fold_trains_within_the_purge_window():
    X, y = _problem()
    model = Stacking().fit(X, y)
    assert len(model.folds_) == 4
    for fold in model.folds_:
        assert fold["last_train_position"] < fold["start"] - model.purge


def test_meta_learner_only_sees_out_of_sample_rows():
    X, y = _problem()
    model = Stacking().fit(X, y)
    first = int(len(X) * 0.4)
    assert model.oos_features_.shape == (len(X), 3)
    assert META_FEATURES == ("logistic_probability", "tree_probability", "knn_probability")
    # The first 40% never receives a meta feature, so it cannot reach the meta-learner.
    assert np.isnan(model.oos_features_[:first]).all()
    assert not np.isnan(model.oos_features_[first:]).any()
    assert model.meta_rows_.tolist() == [False] * first + [True] * (len(X) - first)
    assert model.meta_.n_samples_ == len(X) - first


@pytest.mark.parametrize("fold", [0, 1, 2, 3])
def test_a_blocks_meta_features_ignore_everything_from_its_purge_window_on(fold):
    """Leakage probe: rewrite the present and future, and the block's features must not move."""
    X, y = _problem(seed=1)
    base = Stacking().fit(X, y)
    start, stop = base.folds_[fold]["start"], base.folds_[fold]["stop"]
    block = slice(start, stop)

    relabelled = y.copy()
    relabelled[start - base.purge:] = 1 - relabelled[start - base.purge:]
    assert np.array_equal(Stacking().fit(X, relabelled).oos_features_[block], base.oos_features_[block])

    rewritten = X.copy()
    rewritten[start - base.purge:start] = 99.0
    rewritten[stop:] = -99.0
    assert np.array_equal(Stacking().fit(rewritten, y).oos_features_[block], base.oos_features_[block])


def test_prediction_is_the_meta_learner_over_bases_refitted_on_all_rows():
    X, y = _problem(seed=2)
    model = Stacking().fit(X[:300], y[:300])
    features = []
    for _, model_class, params, standardize in BASES:
        scaler = Standardizer().fit(X[:300]) if standardize else None
        train = scaler.transform(X[:300]) if scaler else X[:300]
        held = scaler.transform(X[300:]) if scaler else X[300:]
        features.append(model_class(**params).fit(train, y[:300]).predict_proba(held))
    features = np.column_stack(features)
    assert np.array_equal(model.base_probabilities(X[300:]), features)
    assert np.array_equal(model.predict_proba(X[300:]), model.meta_.predict_proba(features))
    assert np.array_equal(model.predict(X[300:]), (model.predict_proba(X[300:]) > 0.5).astype(int))


def test_single_class_history_fails_the_fold():
    X, y = _problem()
    y[:int(len(X) * 0.4)] = 0
    with pytest.raises(ValueError, match="single-class"):
        Stacking().fit(X, y)


@pytest.mark.parametrize("positions", [
    np.arange(400)[::-1],                     # not chronological
    np.r_[0, 0, np.arange(2, 400)],           # a repeated day
    np.arange(400) + 5,                       # beyond the timeline when n_positions=400
])
def test_invalid_positions_fail(positions):
    X, y = _problem()
    with pytest.raises(ValueError):
        Stacking().fit(X, y, positions=positions, n_positions=400)


def test_invalid_hyperparameters_and_data_fail():
    X, y = _problem()
    for kwargs in ({"folds": 0}, {"initial_fraction": 1.0}, {"purge": -1}):
        with pytest.raises(ValueError):
            Stacking(**kwargs).fit(X, y)
    with pytest.raises(ValueError):
        Stacking().fit(X, np.full(len(X), 2))


def test_readonly_input_is_preserved():
    X, y = _problem()
    X.setflags(write=False)
    Stacking().fit(X, y)
    assert not X.flags.writeable


def _rebuild_with_references(scratch, X, y, positions, n_positions):
    """The same protocol assembled from sklearn learners: (oos features, meta, final bases)."""
    from sklearn.linear_model import LogisticRegression

    def fit_bases(rows_X, rows_y):
        fitted = []
        for _, model_class, params, standardize in BASES:
            scaler = Standardizer().fit(rows_X) if standardize else None
            train = scaler.transform(rows_X) if scaler else rows_X
            probe = model_class(**params).fit(train, rows_y)  # carries n_samples_ for the reference's C
            fitted.append((scaler, probe.reference().fit(train, rows_y)))
        return fitted

    def probabilities(fitted, rows_X):
        return np.column_stack([
            model.predict_proba(scaler.transform(rows_X) if scaler else rows_X)[:, 1]
            for scaler, model in fitted
        ])

    oos = np.full((len(X), 3), np.nan)
    for train, block, _, _ in fold_plan(positions, n_positions, scratch.folds,
                                        scratch.initial_fraction, scratch.purge):
        oos[block] = probabilities(fit_bases(X[train], y[train]), X[block])
    rows = ~np.isnan(oos).any(axis=1)
    meta = LogisticRegression(C=np.inf, tol=1e-12, max_iter=200000).fit(oos[rows], y[rows])
    final = fit_bases(X, y)
    return oos, lambda rows_X: meta.predict_proba(probabilities(final, rows_X))[:, 1]


def _assert_matches_rebuild(scratch, X, y, positions, n_positions, held_out):
    oos, predict = _rebuild_with_references(scratch, X, y, positions, n_positions)
    rows = scratch.meta_rows_
    assert np.array_equal(np.isnan(oos), np.isnan(scratch.oos_features_))
    # The tree and k-NN bases are exact algorithms, so their columns must match exactly.
    assert np.allclose(scratch.oos_features_[rows, 1:], oos[rows, 1:], rtol=1e-9, atol=1e-12)
    # The logistic base is iterative; run to convergence it agrees to solver tolerance.
    assert np.allclose(scratch.oos_features_[rows, 0], oos[rows, 0], atol=1e-5)
    for part in (X, held_out):
        ours, theirs = scratch.predict_proba(part), predict(part)
        assert np.allclose(ours, theirs, atol=1e-4)
        assert np.array_equal(ours > 0.5, theirs > 0.5)


def test_logistic_base_is_run_to_convergence():
    # At the configured 5,000 steps the base stops with a gradient near 1e-3 on the real
    # folds; the stack must not inherit that, or its meta features are solver noise.
    X, y = _problem(seed=7)
    model = Stacking().fit(X, y)
    scaler, logistic = model.bases_[0]
    Z = scaler.transform(X)
    gradient = Z.T @ (logistic.predict_proba(Z) - y) / len(y)
    assert np.abs(gradient).max() < 1e-8


def test_meta_learner_reaches_the_reference_optimum_on_identical_features():
    """With the base gap removed, the meta step itself must agree with sklearn tightly."""
    from sklearn.linear_model import LogisticRegression

    X, y = _problem(seed=0, rows=500)
    scratch = Stacking().fit(X[:400], y[:400])
    rows = scratch.meta_rows_
    reference = LogisticRegression(C=np.inf, tol=1e-12, max_iter=200000).fit(
        scratch.oos_features_[rows], y[:400][rows])
    assert np.allclose(scratch.meta_.coef_, reference.coef_[0], rtol=1e-4)
    assert np.allclose(scratch.meta_.predict_proba(scratch.oos_features_[rows]),
                       reference.predict_proba(scratch.oos_features_[rows])[:, 1], atol=1e-6)


@pytest.mark.parametrize("seed", [0, 3])
def test_synthetic_matches_fold_by_fold_reference_rebuild(seed):
    X, y = _problem(seed=seed, rows=500)
    scratch = Stacking().fit(X[:400], y[:400])
    _assert_matches_rebuild(scratch, X[:400], y[:400], np.arange(400), 400, X[400:])


@pytest.mark.parametrize("variant", ["with_spike", "non_spike"])
def test_real_data_matches_fold_by_fold_reference_rebuild(variant):
    data = load(variant)
    timeline = load("with_spike").train.dates
    positions = np.searchsorted(timeline, data.train.dates)
    assert np.array_equal(timeline[positions], data.train.dates)
    scratch = Stacking().fit(data.train.X, data.train.y_classification,
                             positions=positions, n_positions=len(timeline))
    for fold in scratch.folds_:
        assert fold["last_train_position"] < fold["start"] - scratch.purge
    _assert_matches_rebuild(scratch, data.train.X, data.train.y_classification, positions,
                            len(timeline), data.validation.X)


def test_non_spike_folds_differ_from_a_row_count_split():
    """The whole point of positions: on non_spike, rows and trading days are not the same thing."""
    data = load("non_spike")
    timeline = load("with_spike").train.dates
    positions = np.searchsorted(timeline, data.train.dates)
    by_day = fold_plan(positions, len(timeline), 4, 0.4, 5)
    by_row = fold_plan(np.arange(len(positions)), len(positions), 4, 0.4, 5)
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(by_day, by_row))


def test_behavioural_reference_fits_the_same_three_learners():
    X, y = _problem(seed=4)
    reference = Stacking().fit(X, y).reference().fit(X, y)
    assert [name for name, _ in reference.estimators] == ["logistic", "tree", "knn"]
    assert reference.predict_proba(X).shape == (len(X), 2)


def test_save_load_preserves_scores(tmp_path):
    X, y = _problem(seed=5)
    original = Stacking().fit(X[:300], y[:300])
    path = tmp_path / "stacking.npz"
    save_npz(path, original.state_dict(), {"params": original.get_params()})
    arrays, metadata = load_npz(path)
    restored = Stacking().load_state(arrays, metadata)
    assert restored.get_params() == original.get_params()
    assert restored.folds_ == original.folds_
    assert np.array_equal(restored.predict_proba(X[300:]), original.predict_proba(X[300:]))
    assert np.array_equal(restored.predict(X[300:]), original.predict(X[300:]))


def test_registry_exposes_stacking():
    assert "stacking" in model_names()
    assert isinstance(build("stacking"), Stacking)

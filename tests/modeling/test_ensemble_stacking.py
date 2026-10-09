"""M6 contract: original-timeline OOS stacking, Q75 labels, scratch state."""
import ast
import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling.ensemble import stacking
from src.modeling.ensemble.stacking import StackingClassifier
from src.modeling.classification.logistic import LogisticRegression
from src.modeling.classification.decision_tree import DecisionTreeClassifier
from src.modeling.classification.knn import KNN
from src.modeling.preprocessing import Standardizer
from src.modeling.configuration import load_config
from src.modeling.datasets import load_train_validation
from src.modeling.metrics import classification_metrics, select_threshold, loss
from src.modeling.persistence import save_npz, load_npz, verify_reload


def fixture(gapped=False):
    original = np.arange('2020-01-01', '2020-04-11', dtype='datetime64[D]')
    positions = np.arange(len(original))
    if gapped:
        positions = positions[~np.isin(positions, [9, 11, 38, 39, 41, 42, 44, 46, 52, 70])]
    rng = np.random.default_rng(7)
    X = rng.normal(size=(len(positions), 3))
    X[:, 0] = positions  # identifiable rows, deliberately drifting mean
    y = (positions % 3 == 0).astype(int)
    return X, y, original[positions], original, positions


def model():
    return StackingClassifier(logistic_max_iter=30)


def fitted(gapped=False):
    X, y, dates, original, positions = fixture(gapped)
    m = model().fit(X, y, dates=dates, original_dates=original)
    return m, X, y, dates, original, positions


@pytest.mark.parametrize('gapped', [False, True])
def test_actual_purge_array_split_and_train_only_fit(monkeypatch, gapped):
    X, y, dates, original, positions = fixture(gapped)
    calls, scaling = [], []
    lr_fit, tree_fit, knn_fit, scale_fit = (LogisticRegression.fit,
        DecisionTreeClassifier.fit, KNN.fit, Standardizer.fit)
    def lr(self, data, labels):
        calls.append(('logistic', data.copy(), labels.copy()))
        return lr_fit(self, data, labels)
    def tree(self, data, labels):
        calls.append(('tree', data.copy(), labels.copy()))
        return tree_fit(self, data, labels)
    def knn(self, data, labels):
        calls.append(('knn', data.copy(), labels.copy()))
        return knn_fit(self, data, labels)
    def scale(self, data):
        scaling.append(data.copy())
        return scale_fit(self, data)
    monkeypatch.setattr(LogisticRegression, 'fit', lr)
    monkeypatch.setattr(DecisionTreeClassifier, 'fit', tree)
    monkeypatch.setattr(KNN, 'fit', knn)
    monkeypatch.setattr(Standardizer, 'fit', scale)
    m = model().fit(X, y, dates=dates, original_dates=original, original_positions=positions)
    eligible = np.flatnonzero(positions >= int(.4 * len(original)))
    blocks = np.array_split(eligible, 4)
    assert len(m.fold_records_) == 4
    for i, (block, record) in enumerate(zip(blocks, m.fold_records_)):
        allowed = np.flatnonzero(positions < positions[block[0]] - 5)
        assert record['validation_positions'] == positions[block].tolist()
        assert record['train_positions'] == positions[allowed].tolist()
        assert not set(positions[allowed]) & set(positions[block])
        assert positions[allowed[-1]] < positions[block[0]] - 5
        assert np.all(np.diff(dates[block]) > np.timedelta64(0, 'D'))
        np.testing.assert_array_equal(scaling[2*i], X[allowed])
        np.testing.assert_array_equal(scaling[2*i+1], X[allowed])
        np.testing.assert_allclose(calls[3*i][1], scale_fit(Standardizer(), X[allowed]).transform(X[allowed]))
        np.testing.assert_array_equal(calls[3*i+1][1], X[allowed])
        np.testing.assert_array_equal(calls[3*i+2][1], calls[3*i][1])
        for call in calls[3*i:3*i+3]:
            np.testing.assert_array_equal(call[2], y[allowed])
    # Four OOS triplets, one meta logistic, then three full-Train refits.
    assert [c[0] for c in calls] == ['logistic', 'tree', 'knn'] * 4 + ['logistic'] + ['logistic', 'tree', 'knn']
    np.testing.assert_array_equal(calls[12][1], m.oos_meta_features_)
    np.testing.assert_array_equal(calls[12][2], y[eligible])
    np.testing.assert_array_equal(m.oos_positions_, positions[eligible])
    np.testing.assert_array_equal(scaling[-1], X)
    np.testing.assert_array_equal(calls[-2][1], X)
    np.testing.assert_array_equal(scaling[-2], X)
    np.testing.assert_array_equal(calls[-1][1], m.preprocessors_[2].transform(X))
    np.testing.assert_array_equal(m.base_learners_[2].X_train_, calls[-1][1])
    assert m.preprocessors_[0] is not m.preprocessors_[2]
    for learner in (m.base_learners_[0], m.base_learners_[2], m.meta_learner_):
        assert learner.standardize is False and learner.scaler_ is None
    np.testing.assert_array_equal(m.base_learners_[2].y_train_, y)
    assert len(scaling) == 10
    assert m.status_ == 'completed'


def test_meta_features_are_actual_class_one_oos_probabilities(monkeypatch):
    X, y, dates, original, positions = fixture()
    predicted = []
    for cls in (LogisticRegression, DecisionTreeClassifier, KNN):
        method = cls.predict_proba
        def spy(self, data, method=method):
            p = method(self, data)
            predicted.append(p[:, 1].copy())
            return p
        monkeypatch.setattr(cls, 'predict_proba', spy)
    m = model().fit(X, y, dates=dates, original_dates=original)
    expected = np.vstack([np.column_stack(predicted[i:i+3]) for i in range(0, 12, 3)])
    np.testing.assert_array_equal(m.oos_meta_features_, expected)


def test_frozen_defaults_and_bounded_test_override():
    m = StackingClassifier()
    cfg = load_config()
    assert m.protocol_ == cfg['stacking']
    assert m.config_['logistic']['learning_rate'] == .05
    assert m.config_['logistic']['max_iter'] == 5000
    assert m.config_['decision_tree']['max_depth'] == cfg['models']['decision_tree']['grid']['max_depth'][0]
    assert m.config_['decision_tree']['min_samples_leaf'] == 5
    assert m.config_['knn']['n_neighbors'] == 3
    assert m.config_['preprocessing'] == ['standardize', 'none', 'standardize']
    assert m.config_['preprocessing_policy'] == 'base_specific'
    assert m.config_['meta_preprocessing'] == 'none'
    assert m.config_['meta_class_weight'] is None
    for n in (0, 5001, True, 1.5):
        with pytest.raises(ValueError):
            StackingClassifier(logistic_max_iter=n)


def test_probability_threshold_determinism_and_tree_pure_leaves():
    a, X, *_ = fitted()
    b, *_ = fitted()
    p = a.predict_proba(X)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_array_equal(p.sum(axis=1), np.ones(len(X)))
    np.testing.assert_array_equal(p, b.predict_proba(X))
    np.testing.assert_array_equal(a.predict(X), (p[:, 1] >= .5).astype(int))
    assert isinstance(a.base_learners_[0], LogisticRegression)
    assert isinstance(a.base_learners_[1], DecisionTreeClassifier)
    assert isinstance(a.base_learners_[2], KNN)
    # M5 pure-positive leaves need counts aligned with global classes.
    tree = a.base_learners_[1]
    toy = np.repeat(np.arange(20.)[:, None], 3, axis=1)
    tree.fit(toy, (toy[:, 0] >= 10).astype(int))
    np.testing.assert_array_equal(tree.predict_proba(toy)[:, 1], (toy[:, 0] >= 10).astype(int))
    a.meta_learner_.coefficients_[:] = 0
    a.meta_learner_.intercept_ = 0
    assert (a.predict(X) == 1).all()


@pytest.mark.parametrize('gapped', [False, True])
def test_complete_strict_json_npz_roundtrip(tmp_path, gapped):
    m, X, *_ = fitted(gapped)
    state = json.loads(json.dumps(m.to_dict(), allow_nan=False))
    assert state['protocol'] == load_config()['stacking']
    assert state['bases'] == [base.to_dict() for base in m.base_learners_]
    assert state['meta'] == m.meta_learner_.to_dict()
    assert len(state['preprocessors']) == 3
    path = tmp_path / 'model.npz'
    save_npz(path, {'oos_positions': m.oos_positions_}, state)
    arrays, state = load_npz(path)
    restored = StackingClassifier.from_dict(state)
    np.testing.assert_array_equal(m.predict_proba(X), restored.predict_proba(X))
    np.testing.assert_array_equal(m.predict(X), restored.predict(X))
    np.testing.assert_array_equal(arrays['oos_positions'], restored.oos_positions_)
    assert restored.to_dict() == m.to_dict()
    assert verify_reload(m.predict(X), restored.predict(X), labels=True)['passed']


@pytest.mark.parametrize('change', ['nan', 'labels', 'label_shape', 'date_order', 'date_duplicate', 'missing_date', 'position_duplicate', 'position_mismatch', 'short', 'one_class', 'one_class_fold', 'one_class_validation'])
def test_invalid_training_explicit_failure(change):
    X, y, dates, original, positions = fixture()
    kwargs = dict(dates=dates, original_dates=original, original_positions=positions)
    if change == 'nan': X[0, 0] = np.nan
    if change == 'labels': y = y.astype(float); y[0] = .5
    if change == 'label_shape': y = y[:, None]
    if change == 'date_order': kwargs['dates'] = dates[::-1]
    if change == 'date_duplicate': kwargs['dates'] = dates.copy(); kwargs['dates'][1] = dates[0]
    if change == 'missing_date': kwargs['dates'] = dates + np.timedelta64(1000, 'D')
    if change == 'position_duplicate': kwargs['original_positions'] = np.zeros(len(X), dtype=int)
    if change == 'position_mismatch': kwargs['original_positions'] = positions + 1
    if change == 'short': X, y = X[:6], y[:6]; kwargs = dict(dates=dates[:6], original_dates=original[:6])
    if change == 'one_class': y[:] = 0
    if change == 'one_class_fold': y[:40] = 0
    if change == 'one_class_validation': y[40:56] = 0
    m = model()
    with pytest.raises(ValueError): m.fit(X, y, **kwargs)
    assert m.status_ == 'failed' and m.error_['message']
    with pytest.raises(ValueError, match='not fitted'): m.predict_proba(X)


@pytest.mark.parametrize('change', ['dimensions', 'nonfinite', 'protocol', 'classes', 'oos', 'base_count'])
def test_corrupt_persistence_rejected(change):
    m, *_ = fitted()
    state = copy.deepcopy(m.to_dict())
    if change == 'dimensions': state['n_features'] += 1
    if change == 'nonfinite': state['meta']['intercept'] = float('nan')
    if change == 'protocol': state['protocol']['purge_original_trading_days'] = 0
    if change == 'classes': state['classes'] = [0]
    if change == 'oos': state['oos_positions'][1] = state['oos_positions'][0]
    if change == 'base_count': state['bases'].pop()
    with pytest.raises(ValueError): StackingClassifier.from_dict(state)


def test_prediction_input_and_fitted_state_validation():
    m = model()
    for method in (m.predict, m.predict_proba, m.to_dict):
        with pytest.raises(ValueError, match='not fitted'):
            method(np.ones((3, 3))) if method != m.to_dict else method()
    m, *_ = fitted()
    for X in (np.ones((2, 4)), np.array([[np.inf]*3]), np.ones(3), np.empty((0, 3))):
        with pytest.raises(ValueError): m.predict_proba(X)


def test_no_sklearn_or_test_access_in_production():
    source = inspect.getsource(stacking)
    imports = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any('sklearn' in ast.unparse(n) for n in imports)
    assert 'load_test' not in source


@pytest.mark.parametrize('variant', ['with_spike', 'non_spike'])
def test_real_q75_variants_validation_isolation(monkeypatch, variant):
    import src.modeling.datasets as datasets
    def forbidden(*a, **k): raise AssertionError('Test access')
    monkeypatch.setattr(datasets, 'load_test', forbidden)
    read_bytes = Path.read_bytes
    def guard(path):
        assert 'test' not in path.name.lower(), 'Test input accessed'
        return read_bytes(path)
    monkeypatch.setattr(Path, 'read_bytes', guard)
    data = load_train_validation(variant)
    original = load_train_validation('with_spike').train
    m = model().fit_dataset(data.train, original)
    eligible = np.searchsorted(original.dates, data.train.dates) >= int(.4 * len(original.X))
    np.testing.assert_array_equal(m.oos_labels_, data.train.y_classification[eligible])
    assert data.train.y_classification.mean() < .4  # accepted Q75, not median
    before = m.to_dict()
    restored = StackingClassifier.from_dict(json.loads(json.dumps(before, allow_nan=False)))
    assert restored.to_dict() == before
    np.testing.assert_array_equal(m.predict_proba(data.validation.X), restored.predict_proba(data.validation.X))
    np.testing.assert_array_equal(m.predict(data.validation.X), restored.predict(data.validation.X))
    score = m.predict_proba(data.validation.X)[:, 1]
    assert classification_metrics(data.validation.y_classification, score)['accuracy']['value'] is not None
    assert loss('log_loss', data.validation.y_classification, probabilities=score)['value'] is not None
    assert select_threshold(data.validation.y_classification, score)['selection_split'] == 'validation'
    assert m.to_dict() == before
    # Replacing external Validation cannot affect the Train-only adapter.
    from dataclasses import replace
    changed = replace(data, validation=replace(data.validation, X=data.validation.X * 100, y_classification=1-data.validation.y_classification))
    b = model().fit_dataset(changed.train, original)
    np.testing.assert_array_equal(score, b.predict_proba(data.validation.X)[:, 1])
    assert m.oos_positions_[-1] < len(original.X)
    assert max(max(r['validation_positions']) for r in m.fold_records_) < len(original.X)


def test_rng_unchanged_and_failed_refit_cannot_predict():
    np.random.seed(91)
    before = np.random.get_state()
    m, X, y, dates, original, _ = fitted()
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    with pytest.raises(ValueError):
        m.fit(X, np.zeros(len(y)), dates=dates, original_dates=original)
    with pytest.raises(ValueError, match='not fitted'): m.predict(X)


@pytest.mark.parametrize('change', ['original_duplicate', 'original_order', 'missing_dates', 'nat', 'date_length', 'float_positions', 'empty_matrix', 'one_dimensional'])
def test_date_matrix_contract(change):
    X, y, dates, original, positions = fixture()
    kwargs = dict(dates=dates, original_dates=original, original_positions=positions)
    if change == 'original_duplicate': original[1] = original[0]
    if change == 'original_order': kwargs['original_dates'] = original[::-1]
    if change == 'missing_dates': kwargs['dates'] = None
    if change == 'nat': dates[0] = np.datetime64('NaT', 'D')
    if change == 'date_length': kwargs['dates'] = dates[:-1]
    if change == 'float_positions': kwargs['original_positions'] = positions.astype(float)
    if change == 'empty_matrix': X = np.empty((0, 3))
    if change == 'one_dimensional': X = X[:, 0]
    with pytest.raises(ValueError): model().fit(X, y, **kwargs)


def test_persisted_oos_labels_must_match_full_train():
    m, *_ = fitted()
    state = m.to_dict()
    state['oos_labels'][0] = 1-state['oos_labels'][0]
    with pytest.raises(ValueError, match='OOS labels'):
        StackingClassifier.from_dict(state)


def test_frozen_tree_minimum_leaf_size():
    m, X, y, dates, original, _ = fitted()
    y[:] = 0
    y[[0, 1, 2, 8, 15, 30, 45, 60, 75, 90]] = 1
    # Test the reused base directly; every descendant leaf must have >=5 rows.
    tree = m.base_learners_[1].fit(X, y)
    def check(node):
        if node.is_leaf(): assert node.value.sum() >= 5
        else: check(node.left); check(node.right)
    check(tree.tree_)


def test_default_budget_runs_and_predicts():
    X, y, dates, original, _ = fixture()
    m = StackingClassifier().fit(X, y, dates=dates, original_dates=original)
    assert m.meta_learner_.max_iter == 5000
    assert m.base_learners_[0].max_iter == 5000
    assert np.isfinite(m.predict_proba(X)).all()


@pytest.mark.parametrize('variant', ['with_spike', 'non_spike'])
def test_m2_variant_adapter_ignores_external_validation(monkeypatch, variant):
    from types import SimpleNamespace
    X, y, dates, original, _ = fixture(variant == 'non_spike')
    train = SimpleNamespace(X=X, y_classification=y, dates=dates)
    calls = []
    def loader(name):
        calls.append(name)
        if name == variant:
            return SimpleNamespace(train=train, validation=object())
        return SimpleNamespace(train=SimpleNamespace(dates=original), validation=object())
    monkeypatch.setattr(stacking, 'load_train_validation', loader)
    m = model().fit_variant(variant)
    assert calls == ([variant] if variant == 'with_spike' else [variant, 'with_spike'])
    np.testing.assert_array_equal(m.base_learners_[2].y_train_, y)


@pytest.mark.parametrize('change', ['base_scaling', 'meta_scaling', 'knn_scaling', 'knn_weights', 'knn_dimensions', 'logistic_dimensions', 'logistic_budget', 'logistic_history', 'scaler_scale', 'scaler_missing', 'tree_dimensions', 'tree_counts'])
def test_complete_model_state_corruption_rejected(change):
    m, *_ = fitted()
    state = m.to_dict()
    if change == 'base_scaling': state['bases'][0]['standardize'] = True
    if change == 'meta_scaling': state['meta']['standardize'] = True
    if change == 'knn_scaling': state['bases'][2]['standardize'] = True
    if change == 'knn_weights': state['bases'][2]['weights'] = 'distance'
    if change == 'knn_dimensions': state['bases'][2]['X_train'][0].pop()
    if change == 'logistic_dimensions': state['bases'][0]['coefficients'].pop()
    if change == 'logistic_budget': state['meta']['max_iter'] += 1
    if change == 'logistic_history': state['meta']['n_iter'] += 1
    if change == 'scaler_scale': state['preprocessors'][2]['scale'][0] *= 2
    if change == 'scaler_missing': state['preprocessors'][2] = None
    if change == 'tree_dimensions': state['bases'][1]['n_features'] += 1
    if change == 'tree_counts': state['bases'][1]['tree'] = {'value': [-1, 3]}
    with pytest.raises(ValueError):
        StackingClassifier.from_dict(state)

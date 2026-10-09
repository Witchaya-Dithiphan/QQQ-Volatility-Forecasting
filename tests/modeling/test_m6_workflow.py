"""M6 workflow contracts: Train/Validation only, frozen grids, immutable verified runs."""
import dataclasses
import json
import numpy as np
import pytest

from src.modeling import m6_training as training
from src.modeling.configuration import load_config
from src.modeling.datasets import DatasetSplit, TrainValidation
from src.modeling.persistence import read_manifest, resume_check

CONFIG = load_config()
HASHES = dict(config='c', inputs='i', code='k', dependency_lock='d', runtime='r')
MODELS = ('random_forest', 'gradient_boosting', 'adaboost', 'svm', 'mlp', 'stacking', 'xgboost')
TINY = {
    'random_forest': [dict(n_estimators=3, max_depth=2, pca_components=None), dict(n_estimators=3, max_depth=2, pca_components=2)],
    'gradient_boosting': [dict(n_estimators=3, max_depth=1, learning_rate=.1)],
    'adaboost': [dict(n_estimators=3, learning_rate=.5)],
    'svm': [dict(C=.1, pca_components=None), dict(C=1, pca_components=2)],
    'mlp': [dict(hidden_layers=[4], learning_rate=.01, l2=0)],
    'stacking': [{}],
    'xgboost': [dict(n_estimators=3, max_depth=1, learning_rate=.1, reg_lambda=1, gamma=0)],
}
OPTIONS = {'svm': dict(max_iter=20), 'mlp': dict(max_epochs=4, patience=2), 'stacking': dict(logistic_max_iter=15)}


def synthetic(variant='with_spike'):
    rng = np.random.default_rng(22)
    original = np.arange('2020-01-01', '2020-06-09', dtype='datetime64[D]')
    positions = np.arange(len(original))
    if variant == 'non_spike':
        positions = np.delete(positions, np.arange(50, 60))
    def split(dates):
        X = rng.normal(size=(len(dates), 8))
        y = (np.arange(len(dates)) % 4 == 0).astype(int)
        X[:, 0] += 2 * y
        return DatasetSplit(dates, X, np.arange(len(dates), dtype=float), y, '0' * 64, 'synthetic')
    return TrainValidation(variant, split(original[positions]), split(np.arange('2021-01-01', '2021-03-01', dtype='datetime64[D]'))), original


def read(path, filename):
    return json.loads((path / filename).read_text())


def test_registrations_and_frozen_candidate_scope():
    assert training.M6_MODELS == MODELS
    counts = dict(random_forest=7, gradient_boosting=8, adaboost=6, svm=12, mlp=12, stacking=1, xgboost=8)
    for name, count in counts.items():
        assert len(training.candidates_for(name, CONFIG)) == count
    svm = training.candidates_for('svm', CONFIG)
    assert {(p['C'], p['pca_components']) for p in svm} == {(c, k) for c in [.1, 1, 10] for k in [None, 2, 4, 6]}
    forest = training.candidates_for('random_forest', CONFIG)
    assert all(p['use_selected_no_pca_hyperparameters'] for p in forest[4:])


@pytest.mark.parametrize('variant', ['with_spike', 'non_spike'])
@pytest.mark.parametrize('name', MODELS)
def test_scratch_artifacts_and_exact_reload(tmp_path, name, variant):
    inputs, original = synthetic(variant)
    path = training.train_m6_model(name, variant, tmp_path, 'tiny', inputs=inputs, original_dates=original,
                                  hashes=HASHES, candidate_overrides=TINY[name], fit_options=OPTIONS.get(name), reference=False)
    manifest = read_manifest(path)
    assert manifest['status'] == 'completed' and manifest['role'] == 'finalized_scratch'
    assert not manifest['test_accessed'] and manifest['input_identity']['train']['rows'] == len(inputs.train.X)
    assert resume_check(manifest, HASHES, path)['can_skip']
    results = read(path, 'search_results.json')
    assert len(results['candidates']) == len(TINY[name])
    assert all(c['metrics']['loss']['name'] == CONFIG['losses'][name] for c in results['candidates'])
    metrics = read(path, 'validation_metrics.json')
    score = training.score_of(name, training.load_m6_model(path), inputs.validation.X)
    expected = training.select_threshold(inputs.validation.y_classification, score, score_kind=metrics['score_kind'])
    assert expected['threshold'] == metrics['threshold']
    predictions = np.genfromtxt(path / 'validation_predictions.csv', delimiter=',', names=True, dtype=None, encoding='utf8')
    np.testing.assert_allclose(predictions['raw_score'], score, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(predictions['final_prediction'], score >= metrics['threshold'])
    assert read(path, 'load_verification.json')['passed']
    assert training.train_m6_model(name, variant, tmp_path, 'tiny', hashes=HASHES, resume=True, reference=False) == path
    assert training.m6_status(tmp_path, variant, name, 'tiny', hashes=HASHES)['can_skip']


@pytest.mark.parametrize('name', MODELS)
def test_reference_artifacts(tmp_path, name):
    inputs, original = synthetic()
    path = training.train_m6_model(name, 'with_spike', tmp_path, 'reference', inputs=inputs, original_dates=original,
                                  hashes=HASHES, candidate_overrides=TINY[name], fit_options=OPTIONS.get(name))
    reference = path.parent.parent / 'reference' / 'reference'
    manifest = read_manifest(reference)
    if manifest['status'] == 'skipped':
        assert manifest['reason'] and name == 'xgboost'
        return
    assert manifest['role'] == 'finalized_reference' and manifest['status'] == 'completed'
    assert resume_check(manifest, HASHES, reference)['can_skip']
    assert (reference / 'model.joblib').is_file()
    assert len(read(reference, 'search_results.json')['candidates']) == len(TINY[name])


def test_original_positions_and_q75_labels(monkeypatch):
    inputs, original = synthetic('non_spike')
    expected = np.searchsorted(original, inputs.train.dates)
    for name in ['mlp', 'stacking']:
        model, protocol = training.fit_candidate(name, TINY[name][0], inputs.train, original, CONFIG, OPTIONS[name])
        if name == 'mlp':
            assert protocol['purge_boundary_position'] == int(expected[protocol['tail_start']]) - 5
            assert protocol['n_subtrain'] == np.searchsorted(expected, protocol['purge_boundary_position'])
        else:
            np.testing.assert_array_equal(model.original_positions_, expected)
            np.testing.assert_array_equal(model.base_learners_[2].y_train_, inputs.train.y_classification)


def test_failure_evidence_and_candidate_error(tmp_path):
    inputs, original = synthetic()
    bad = dataclasses.replace(inputs.train, X=np.zeros_like(inputs.train.X), y_classification=np.arange(len(inputs.train.X)) % 2)
    inputs = dataclasses.replace(inputs, train=bad)
    with pytest.raises(ValueError):
        training.train_m6_model('adaboost', 'with_spike', tmp_path, 'failed', inputs=inputs, original_dates=original,
                                hashes=HASHES, candidate_overrides=TINY['adaboost'], reference=False)
    path = tmp_path / 'with_spike/classification/adaboost/scratch/failed'
    assert read_manifest(path)['status'] == 'failed'
    assert read(path, 'search_results.json')['candidates'][0]['error']
    assert (path / 'traceback.txt').is_file()


def test_no_test_loader_access(monkeypatch, tmp_path):
    from src.modeling import datasets
    monkeypatch.setattr(datasets, 'load_test', lambda *a, **k: pytest.fail('Test accessed'))
    inputs, original = synthetic()
    calls = []
    monkeypatch.setattr(training, 'load_train_validation', lambda variant: calls.append(variant) or inputs)
    training.train_m6_model('xgboost', 'with_spike', tmp_path, 'loader', hashes=HASHES,
                            candidate_overrides=TINY['xgboost'], reference=False)
    assert calls == ['with_spike']


def test_npz_contains_compact_numeric_state_and_rejects_tampering(tmp_path):
    state = {'weights': [[1., 2.], [3., 4.]], 'ragged': [[1.], [2., 3.]], 'flag': True}
    arrays = training._numeric_arrays(state)
    assert arrays['state_7_weights'].shape == (2, 2)
    inputs, original = synthetic()
    path = training.train_m6_model('svm', 'with_spike', tmp_path, 'tamper', inputs=inputs, original_dates=original,
                                  hashes=HASHES, candidate_overrides=TINY['svm'], fit_options=OPTIONS['svm'], reference=False)
    model = read(path, 'model.json')
    model['model']['coef'][0] += 1
    (path / 'model.json').write_text(json.dumps(model))
    with pytest.raises(ValueError, match='mismatch'):
        training.load_m6_model(path)
    assert not training.m6_status(tmp_path, 'with_spike', 'svm', 'tamper', hashes=HASHES)['can_skip']

def test_stacking_workflow_uses_only_configured_base_preprocessing():
    inputs, original = synthetic()
    model, _ = training.fit_candidate('stacking', {}, inputs.train, original, CONFIG, OPTIONS['stacking'])
    assert model.base_learners_[0].scaler_ is None  # the outer Train-only scaler already transformed logistic inputs
    assert model.base_learners_[2].scaler_ is None  # frozen stacking kNN uses raw features
    np.testing.assert_array_equal(model.base_learners_[2].X_train_, inputs.train.X)

def test_adaboost_loss_uses_native_votes_not_probability(tmp_path):
    inputs, original = synthetic()
    path = training.train_m6_model('adaboost', 'with_spike', tmp_path, 'loss', inputs=inputs, original_dates=original,
                                  hashes=HASHES, candidate_overrides=TINY['adaboost'], reference=False)
    model = training.load_m6_model(path)
    votes = model.decision_function(inputs.validation.X)
    expected = np.mean(np.exp(-(2 * inputs.validation.y_classification - 1) * votes))
    assert read(path, 'validation_metrics.json')['metrics']['loss']['value'] == pytest.approx(expected)


def test_invalid_stacking_fold_is_persisted(tmp_path):
    inputs, original = synthetic()
    inputs = dataclasses.replace(inputs, train=dataclasses.replace(inputs.train, y_classification=np.zeros(len(inputs.train.X), dtype=int)))
    with pytest.raises(ValueError, match='All M6'):
        training.train_m6_model('stacking', 'with_spike', tmp_path, 'badfold', inputs=inputs, original_dates=original,
                                hashes=HASHES, fit_options=OPTIONS['stacking'], reference=False)
    path = tmp_path / 'with_spike/classification/stacking/scratch/badfold'
    assert read_manifest(path)['stage'] == 'candidate_search'
    assert 'both classes' in read(path, 'search_results.json')['candidates'][0]['error']['message']


def test_production_grids_are_executed_and_forest_pca_reuses_no_pca_winner(monkeypatch):
    inputs, _ = synthetic()
    class Model:
        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), .7), np.full(len(X), .3)])
        def decision_function(self, X):
            return np.full(len(X), .3)
    for name in MODELS:
        calls = []
        def fit(params):
            calls.append(dict(params))
            return Model(), None
        _, _, search = training._search(name, training.candidates_for(name, CONFIG), CONFIG, inputs, fit, None)
        assert len(calls) == len(training.candidates_for(name, CONFIG))
        assert all(r['error'] is None for r in search['candidates'])
        if name == 'random_forest':
            selected = search['candidates'][0]['params']  # equal metrics: earliest candidate wins
            assert all({k: p[k] for k in ('max_depth', 'n_estimators')} ==
                       {k: selected[k] for k in ('max_depth', 'n_estimators')} for p in calls[4:])


def test_search_selection_ap_then_auc_not_accuracy(monkeypatch):
    from types import SimpleNamespace
    inputs, _ = synthetic()
    quality = [( .7, .99, .99), (.8, .6, .4), (.8, .9, .3)]
    def metrics(name, config, y, score, threshold, **kwargs):
        ap, auc, accuracy = quality[int(score[0])]
        result = training.classification_metrics(y, np.full(len(y), .5))
        result['loss'] = {'name': CONFIG['losses'][name], 'value': 0}
        result.update(average_precision={'value': ap}, roc_auc={'value': auc}, accuracy={'value': accuracy})
        return result
    monkeypatch.setattr(training, 'validation_metrics', metrics)
    models = [SimpleNamespace(predict_proba=lambda X, index=i: np.column_stack([np.zeros(len(X)), np.full(len(X), index)])) for i in range(3)]
    _, _, search = training._search('xgboost', [{'index': i} for i in range(3)], CONFIG, inputs,
                                    lambda p: (models[p['index']], None), None)
    assert search['selected']['id'] == 'c02'


@pytest.mark.parametrize('kind,score', [('probability', [.1, .2, .3, .4]), ('decision', [-2, -1, 0, 1])])
def test_threshold_f1_tie_prefers_recall_then_lower_threshold(monkeypatch, tmp_path, kind, score):
    # F1 = 2/3 at the first and third thresholds; first has recall=1, third=.5.
    name = 'svm' if kind == 'decision' else 'random_forest'
    inputs, _ = synthetic()
    validation = dataclasses.replace(inputs.validation, dates=inputs.validation.dates[:4], X=inputs.validation.X[:4],
                                     y_classification=np.array([1, 0, 1, 0]), y_regression=np.zeros(4))
    inputs = dataclasses.replace(inputs, validation=validation)
    path = tmp_path / kind
    path.mkdir()
    manifest = {'load_verification': None}
    training._persist_validation(path, manifest, name, CONFIG, inputs, None, np.array(score), np.array(score), {}, {})
    assert read(path, 'validation_metrics.json')['threshold'] == score[0]
    # Same F1 and recall: M2 chooses the lower threshold.
    from src.modeling import metrics
    real = metrics.classification_metrics
    def tied(y, score, threshold=.5):
        value = real(y, score, threshold=threshold)
        value['f1']['value'] = value['recall']['value'] = .5
        return value
    monkeypatch.setattr(metrics, 'classification_metrics', tied)
    assert training.select_threshold([0, 1, 0, 1], score, score_kind=kind)['threshold'] == score[0]


def test_validation_does_not_control_neural_epochs_or_stack_meta():
    from src.modeling.m6_reference import fit_reference
    inputs, original = synthetic('non_spike')
    changed = dataclasses.replace(inputs, validation=dataclasses.replace(inputs.validation, X=inputs.validation.X * 100,
                                                                       y_classification=1-inputs.validation.y_classification))
    for fit in (training.fit_candidate, fit_reference):
        for name in ('mlp', 'stacking'):
            first, protocol = fit(name, TINY[name][0], inputs.train, original, CONFIG, OPTIONS[name])
            second, again = fit(name, TINY[name][0], changed.train, original, CONFIG, OPTIONS[name])
            assert protocol == again
            np.testing.assert_array_equal(first.predict_proba(inputs.validation.X), second.predict_proba(inputs.validation.X))


@pytest.mark.parametrize('name', ['random_forest', 'svm', 'mlp', 'stacking', 'xgboost'])
@pytest.mark.parametrize('variant', ['with_spike', 'non_spike'])
def test_reduced_accepted_data_smoke(tmp_path, monkeypatch, name, variant):
    from src.modeling import datasets
    from src.modeling.contracts import TEST_PATH
    from pathlib import Path
    original_read = Path.read_bytes
    def no_test(path):
        assert path != TEST_PATH, 'Test data access prohibited'
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', no_test)
    monkeypatch.setattr(datasets, 'load_test', lambda **k: pytest.fail('Test accessed'))
    inputs = datasets.load_train_validation(variant)
    original = datasets.load_train_validation('with_spike').train.dates
    # Keep full Train timeline for purge/fold correctness; bound trees/epochs only.
    path = training.train_m6_model(name, variant, tmp_path, 'accepted', inputs=inputs, original_dates=original,
                                  hashes=HASHES, candidate_overrides=TINY[name], fit_options=OPTIONS.get(name), reference=True)
    assert read_manifest(path)['load_verification']['passed']
    assert inputs.train.y_classification.mean() < .4
    ref = path.parent.parent / 'reference/accepted'
    assert read_manifest(ref)['status'] == 'completed'


def test_reference_stack_meta_preprocessing_matches_scratch():
    from src.modeling.m6_reference import fit_reference
    inputs, original = synthetic()
    model, _ = fit_reference('stacking', {}, inputs.train, original, CONFIG, OPTIONS['stacking'])
    state = model.preprocessing_state(8)
    assert state['meta_preprocessor']['kind'] == 'standardize'
    assert len(state['meta_preprocessor']['scaler']['mean']) == 3


def test_optional_reference_dependency_is_honestly_skipped(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__
    def without_xgboost(name, *args, **kwargs):
        if name == 'xgboost':
            raise ImportError('xgboost unavailable for test')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_xgboost)
    inputs, original = synthetic()
    path = training.train_m6_model('xgboost', 'with_spike', tmp_path, 'optional', inputs=inputs, original_dates=original,
                                  hashes=HASHES, candidate_overrides=TINY['xgboost'])
    assert read_manifest(path)['status'] == 'completed'
    ref = read_manifest(path.parent.parent / 'reference/optional')
    assert ref['status'] == 'skipped' and 'unavailable' in ref['reason'] and not ref['required']
    assert not (path.parent.parent / 'reference/optional/model.joblib').exists()


@pytest.mark.parametrize('change', ['variant', 'bad_y', 'nonfinite', 'dates', 'overlap', 'dimensions'])
def test_invalid_inputs_are_failed_with_evidence(tmp_path, change):
    inputs, original = synthetic()
    if change == 'variant': inputs = dataclasses.replace(inputs, variant='non_spike')
    if change == 'bad_y': inputs = dataclasses.replace(inputs, train=dataclasses.replace(inputs.train, y_classification=np.full(len(inputs.train.X), 2)))
    if change == 'nonfinite': inputs = dataclasses.replace(inputs, train=dataclasses.replace(inputs.train, X=np.full_like(inputs.train.X, np.nan)))
    if change == 'dates': inputs = dataclasses.replace(inputs, train=dataclasses.replace(inputs.train, dates=inputs.train.dates[::-1]))
    if change == 'overlap': inputs = dataclasses.replace(inputs, validation=dataclasses.replace(inputs.validation, dates=inputs.train.dates[:len(inputs.validation.X)]))
    if change == 'dimensions': inputs = dataclasses.replace(inputs, train=dataclasses.replace(inputs.train, X=inputs.train.X[:, :3]))
    with pytest.raises(ValueError):
        training.train_m6_model('xgboost', 'with_spike', tmp_path, change, inputs=inputs, original_dates=original, hashes=HASHES, reference=False)
    manifest = read_manifest(tmp_path / f'with_spike/classification/xgboost/scratch/{change}')
    assert manifest['status'] == 'failed' and manifest['error']['message'] and not manifest['test_accessed']


def test_bad_config_and_unknown_model_are_rejected_before_fit(tmp_path):
    bad = dict(CONFIG, unexpected=True)
    with pytest.raises(ValueError, match='unknown'):
        training.train_m6_model('xgboost', 'with_spike', tmp_path, 'badconfig', config=bad, hashes=HASHES)
    with pytest.raises(ValueError, match='Invalid'):
        training.train_m6_model('missing', 'with_spike', tmp_path, 'unknown', hashes=HASHES)
    assert not list(tmp_path.iterdir())


def test_batch_and_resume_do_not_refit_or_read_data(monkeypatch, tmp_path):
    inputs, original = synthetic()
    paths = training.train_m6_batch(tmp_path, 'batch', names=('xgboost',), variants=('with_spike',), inputs=inputs,
                                   original_dates=original, hashes=HASHES, candidate_overrides=TINY['xgboost'], reference=False)
    assert len(paths) == 1
    monkeypatch.setattr(training, 'load_train_validation', lambda *a: pytest.fail('resume read data'))
    monkeypatch.setattr(training, 'fit_candidate', lambda *a: pytest.fail('resume refit'))
    assert training.train_m6_model('xgboost', 'with_spike', tmp_path, 'batch', hashes=HASHES, resume=True, reference=False) == paths[0]
    changed = dict(HASHES, code='changed')
    with pytest.raises(ValueError, match='incompatible'):
        training.train_m6_model('xgboost', 'with_spike', tmp_path, 'batch', hashes=changed, resume=True, reference=False)

@pytest.mark.parametrize('name,constructor', [('random_forest', 'RandomForestClassifier'), ('gradient_boosting', 'GradientBoostingClassifier'),
    ('adaboost', 'AdaBoostClassifier'), ('svm', 'LinearSVC'), ('mlp', 'MLPClassifier'), ('stacking', 'ReferenceStacking'), ('xgboost', 'XGBClassifier')])
def test_reference_constructor_protocol_is_reviewable(tmp_path, name, constructor):
    inputs, original = synthetic()
    path = training.train_m6_model(name, 'with_spike', tmp_path, 'constructor', inputs=inputs, original_dates=original,
                                  hashes=HASHES, candidate_overrides=TINY[name], fit_options=OPTIONS.get(name))
    ref = path.parent.parent / 'reference/constructor'
    manifest = read_manifest(ref)
    assert manifest['selected_hyperparameters']['reference_constructor'] == constructor
    assert manifest['selected_hyperparameters']['reference_parameters']
    assert read(ref, 'validation_metrics.json')['metrics']['loss']['value'] is not None
    if name == 'adaboost':
        assert manifest['selected_hyperparameters']['reference_protocol']['installed_estimator_keyword'] in ('estimator', 'base_estimator')

def test_reduced_runs_are_not_required_delivery(tmp_path):
    inputs, original = synthetic()
    path = training.train_m6_model('xgboost', 'with_spike', tmp_path, 'smokeonly', inputs=inputs, original_dates=original,
                                  hashes=HASHES, candidate_overrides=TINY['xgboost'])
    assert not read_manifest(path)['required']
    assert not read_manifest(path.parent.parent / 'reference/smokeonly')['required']

def test_scratch_stacking_preprocessor_includes_meta_scaler():
    inputs, original = synthetic()
    model, _ = training.fit_candidate('stacking', {}, inputs.train, original, CONFIG, OPTIONS['stacking'])
    state = training._preprocessor_state('stacking', model, 8)
    assert state['meta_preprocessor']['scaler']['mean'] == model.meta_learner_.scaler_.mean_.tolist()

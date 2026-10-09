"""Executable M6 Train/Validation searches and immutable M2 artifact lifecycle.

Production defaults execute the full frozen configuration. candidate_overrides and
fit_options are explicit reduced-test hooks, recorded in search results. No Test API
is accepted. sklearn/xgboost dependencies live exclusively in m6_reference.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
import numpy as np

from .artifacts import write_json, write_predictions
from .classification import LinearSVM, MLPClassifier
from .classification import _state
from .classification_training import _identity, _compact
from .configuration import load_config, validate_config
from .datasets import load_train_validation
from .ensemble import RandomForest, GradientBoosting, AdaBoost, StackingClassifier, XGBoostClassifier
from .ensemble.adaboost import AdaBoostFailure
from .expected_runs import _candidates
from .metrics import classification_metrics, loss, select_candidate, select_threshold
from .persistence import (capture_artifacts, create_run_directory, load_npz, make_manifest,
                          read_manifest, resume_check, save_npz, transition, verify_reload, _git_state)
from .preprocessing import PCA, Standardizer

M6_MODELS = ('random_forest', 'gradient_boosting', 'adaboost', 'svm', 'mlp', 'stacking', 'xgboost')
_CLASSES = dict(zip(M6_MODELS, (RandomForest, GradientBoosting, AdaBoost, LinearSVM, MLPClassifier,
                               StackingClassifier, XGBoostClassifier)))


def candidates_for(name, config=None):
    if name not in M6_MODELS:
        raise ValueError(f'Unsupported M6 classifier: {name}')
    config = load_config() if config is None else config
    candidates = _candidates(config['models'][name])
    counts = config['preprocessing']['pca']['components']
    if name == 'svm':
        return [{**p, 'pca_components': k} for p in candidates for k in [None, *counts]]
    if name == 'random_forest':
        return [{**p, 'pca_components': None} for p in candidates] + [
            dict(pca_components=k, use_selected_no_pca_hyperparameters=True) for k in counts]
    return candidates


def original_positions(dates, original_dates):
    dates, original = np.asarray(dates).astype('datetime64[ns]'), np.asarray(original_dates).astype('datetime64[ns]')
    if (dates.ndim != 1 or original.ndim != 1 or not len(dates) or not len(original)
            or np.isnat(dates).any() or np.isnat(original).any()
            or (np.diff(dates) <= np.timedelta64(0, 'ns')).any()
            or (np.diff(original) <= np.timedelta64(0, 'ns')).any()):
        raise ValueError('Dates must be unique and chronological')
    positions = np.searchsorted(original, dates)
    if (positions >= len(original)).any() or not np.array_equal(original[positions], dates):
        raise ValueError('Train dates require an exact original timeline join')
    return positions


def _pca(count, config):
    p = config['preprocessing']['pca']
    return PCA(count, rank_floor=p['rank_eigenvalue_floor'], degenerate_rtol=p['degenerate_relative_tolerance'],
               negative_tolerance=p['negative_eigenvalue_tolerance'])


class ForestPCA:
    """Train-fitted RF PCA adapter; forest implementation is unchanged."""
    def __init__(self, forest, scaler, pca):
        self.forest, self.scaler_, self.pca_ = forest, scaler, pca

    def predict_proba(self, X):
        return self.forest.predict_proba(self.pca_.transform(self.scaler_.transform(X)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)

    def to_dict(self):
        return {'forest': self.forest.to_dict(), 'scaler': _state.scaler_state(self.scaler_), 'pca': _state.pca_state(self.pca_)}

    @classmethod
    def from_dict(cls, state):
        return cls(RandomForest.from_dict(state['forest']), _state.scaler_from_state(state['scaler']), _state.pca_from_state(state['pca']))


def fit_candidate(name, params, train, original_dates, config, fit_options=None):
    options = dict(fit_options or {})
    allowed = {'svm': {'max_iter', 'patience'}, 'mlp': {'max_epochs', 'patience'}, 'stacking': {'logistic_max_iter'}}
    if set(options) - allowed.get(name, set()):
        raise ValueError(f'Unsupported fit options for {name}')
    fixed, neural, seed = config['models'][name]['parameters'], config['neural'], config['experiment']['seed']
    y = train.y_classification
    if name == 'random_forest':
        model = RandomForest(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                             min_samples_leaf=fixed['min_samples_leaf'], max_features=fixed['max_features'], random_state=seed)
        if params.get('pca_components') is not None:
            scaler = Standardizer(config['preprocessing']['standardizer']['variance_floor']).fit(train.X)
            pca = _pca(params['pca_components'], config).fit(scaler.transform(train.X))
            model.fit(pca.transform(scaler.transform(train.X)), y)
            return ForestPCA(model, scaler, pca), None
    elif name == 'gradient_boosting':
        model = GradientBoosting(**params, random_state=seed)
    elif name == 'adaboost':
        model = AdaBoost(**params, alpha_factor=config['adaboost']['alpha_factor'], epsilon=config['adaboost']['epsilon'], random_state=seed)
    elif name == 'xgboost':
        model = XGBoostClassifier(**params)
    elif name == 'svm':
        model = LinearSVM(**params, **options)
    elif name == 'mlp':
        if neural['refit_seed'] != seed or neural['refit_early_stopping']:
            raise ValueError('MLP adapter requires the frozen same-seed fixed-epoch refit protocol')
        model = MLPClassifier(**params, batch_size=fixed['batch'],
                              max_epochs=options.get('max_epochs', fixed['max_epochs']),
                              patience=options.get('patience', neural['patience']), min_delta=neural['min_delta'],
                              seed=seed, tail_fraction=neural['internal_tail_fraction'], purge=neural['purge_original_trading_days'])
        model.fit(train.X, y, original_positions=original_positions(train.dates, original_dates))
        return model, {**model.selection_, 'refit_epochs': model.epochs_trained_, 'refit_rows': len(y), 'refit_seed': seed}
    elif name == 'stacking':
        model = StackingClassifier(random_state=seed, **options)
        if model.protocol_ != config['stacking']:
            raise ValueError('Stacking adapter requires the frozen configured protocol')
        model.fit(train.X, y, dates=train.dates, original_dates=original_dates,
                  original_positions=original_positions(train.dates, original_dates))
        return model, {'folds': model.fold_records_, 'config': model.config_, 'protocol': model.protocol_,
                       'oos_positions': model.oos_positions_.tolist(), 'meta_fit': 'train_oos_only'}
    else:
        raise ValueError(f'Unsupported M6 classifier: {name}')
    model.fit(train.X, y)
    if getattr(model, 'status_', None) == 'failed':
        raise ValueError(f'{name} candidate failed')
    return model, None


def score_of(name, model, X):
    return model.decision_function(X) if name == 'svm' else model.predict_proba(X)[:, 1]


def validation_metrics(name, config, y, score, threshold, *, loss_scores=None):
    metrics = classification_metrics(y, score, threshold=threshold)
    loss_name = config['losses'][name]
    kwargs = {'scores': loss_scores if loss_scores is not None else score} if name in ('svm', 'adaboost') else {'probabilities': score}
    metrics['loss'] = {'name': loss_name, **loss(loss_name, y, epsilon=config['selection']['probability_epsilon'], **kwargs)}
    return metrics


def _numeric_arrays(state):
    """Mirror every numeric state leaf in pickle-free NPZ; JSON retains structure."""
    arrays = {}
    def walk(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, path + '_' + str(len(key)) + '_' + key)
        elif isinstance(value, (list, tuple)):
            try:
                array = np.asarray(value)
            except ValueError:  # unequal neural layer shapes
                array = None
            if array is not None and array.size and array.dtype.kind in 'biuf':
                arrays[path] = array
                return
            for index, child in enumerate(value):
                walk(child, path + '_' + str(index))
        elif isinstance(value, (bool, int, float, np.number)):
            arrays[path] = np.asarray(value)
    walk(state, 'state')
    return arrays or {'state_empty': np.array([0])}


def _preprocessor_state(name, model, n_features):
    if name == 'stacking':
        return {'kind': 'base_specific', 'preprocessors': model.to_dict()['preprocessors'], 'feature_count': n_features,
                'meta_preprocessor': {'kind': 'none', 'scaler': None}}
    scaler, pca = getattr(model, 'scaler_', None), getattr(model, 'pca_', None)
    return {'kind': 'standardize' if scaler else 'none', 'feature_count': n_features,
            'scaler': _state.scaler_state(scaler) if scaler else None, 'pca': _state.pca_state(pca)}


def load_m6_model(path):
    arrays, state = load_npz(Path(path) / 'model.npz')
    expected = _numeric_arrays(state)
    if set(arrays) != set(expected) or not all(np.array_equal(arrays[k], expected[k]) for k in arrays):
        raise ValueError('Model NPZ/JSON state mismatch')
    cls = {'forest_pca': ForestPCA, 'stacking_workflow': StackingClassifier}.get(state['adapter'], _CLASSES[state['name']])
    return cls.from_dict(state['model'])


def _search(name, candidates, config, inputs, fit, fit_options):
    records, selected_model, selected_score = [], None, None
    selected_id = None
    default = config['selection']['defaults']['decision' if name == 'svm' else 'probability']
    for index, candidate in enumerate(candidates):
        params = dict(candidate)
        record = dict(id=f'c{index:02d}', params=params, requested_params=dict(candidate), metrics=None, protocol=None, error=None)
        try:
            if params.pop('use_selected_no_pca_hyperparameters', False):
                no_pca = [r for r in records if r['error'] is None and r['params'].get('pca_components') is None]
                winner = select_candidate(no_pca, task='classification')
                params = {**winner['params'], **params}
                record['params'] = params
            model, record['protocol'] = fit(params)
            score = np.asarray(score_of(name, model, inputs.validation.X), dtype=float)
            votes = model.decision_function(inputs.validation.X) if name == 'adaboost' else None
            record['metrics'] = _compact(validation_metrics(name, config, inputs.validation.y_classification, score, default, loss_scores=votes))
            # Keep only the winner, rather than retaining every forest/network.
            winner = select_candidate([r for r in [*records, record] if r['error'] is None], task='classification')
            if winner['id'] == record['id']:
                selected_model, selected_score, selected_id = model, score, record['id']
        except (ValueError, AdaBoostFailure) as error:
            record['error'] = {'exception_type': type(error).__name__, 'message': str(error)}
        records.append(record)
    result = {'selection_rule': config['selection']['classification'], 'threshold_rule': config['selection']['threshold'],
              'score_kind': 'decision' if name == 'svm' else 'probability', 'fit_options': fit_options or {},
              'selected': next((r for r in records if r['id'] == selected_id), None), 'candidates': records}
    return selected_model, selected_score, result


def _validate_inputs(inputs, variant, config):
    if inputs.variant != variant:
        raise ValueError('Inputs do not match requested variant')
    for split in (inputs.train, inputs.validation):
        X, y = np.asarray(split.X), np.asarray(split.y_classification)
        if (X.ndim != 2 or not all(X.shape) or X.shape[1] != len(config['data']['feature_columns'])
                or not np.isfinite(X).all() or y.shape != (len(X),) or not np.isin(y, [0, 1]).all()
                or len(split.dates) != len(X)):
            raise ValueError('Invalid aligned features/Q75 classification labels')
        original_positions(split.dates, split.dates)
    if inputs.train.dates[-1] >= inputs.validation.dates[0]:
        raise ValueError('Validation must follow Train chronologically')


def m6_status(out_root, variant, name, run_id, *, hashes=None, implementation='scratch'):
    if name not in M6_MODELS or variant not in ('with_spike', 'non_spike') or implementation not in ('scratch', 'reference'):
        raise ValueError('Invalid M6 run identity')
    # Validate components without creating directories.
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Unsafe run ID')
    path = Path(out_root) / variant / 'classification' / name / implementation / run_id
    if not path.exists():
        return {'can_skip': False, 'status': 'missing', 'mismatches': {}}
    from .persistence import classify_run_directory
    cl = classify_run_directory(path)
    if cl['status'] != 'completed':
        return {'can_skip': False, 'status': 'legacy_incompatible', 'mismatches': {'classification': cl}}
    manifest = read_manifest(path)
    if hashes is None:
        from .runner import current_hashes
        hashes = current_hashes(variant)
    return resume_check(manifest, hashes, path)


def train_m6_model(name, variant, out_root, run_id, *, config=None, inputs=None, original_dates=None,
                   hashes=None, candidate_overrides=None, fit_options=None, reference=True, resume=False):
    if name not in M6_MODELS or variant not in ('with_spike', 'non_spike'):
        raise ValueError('Invalid M6 model/variant')
    if not resume:
        from .workflow import preflight
        preflight(out_root, variant, "classification", name, ("scratch", "reference") if reference else ("scratch",), run_id)
    config = load_config() if config is None else config
    validate_config(config)
    if hashes is None:
        from .runner import current_hashes
        hashes = current_hashes(variant)
    if resume:
        status = m6_status(out_root, variant, name, run_id, hashes=hashes)
        if status['can_skip']:
            if reference and not m6_status(out_root, variant, name, run_id, hashes=hashes, implementation='reference')['can_skip']:
                raise ValueError('Scratch is complete but reference is missing/incompatible; use a new run ID')
            return Path(out_root) / variant / 'classification' / name / 'scratch' / run_id
        if status['status'] != 'missing':
            raise ValueError(f'Cannot resume incompatible run: {status["mismatches"]}')
    inputs = load_train_validation(variant) if inputs is None else inputs
    if original_dates is None:
        original_dates = inputs.train.dates if variant == 'with_spike' else load_train_validation('with_spike').train.dates
    candidates = candidates_for(name, config) if candidate_overrides is None else list(candidate_overrides)
    path = create_run_directory(Path(out_root), variant, 'classification', name, 'scratch', run_id)
    manifest = make_manifest(run_id=run_id, variant=variant, task='classification', model=name, implementation='scratch',
                             lifecycle='supervised_tuned', hashes=hashes, role='finalized_scratch',
                             required=candidate_overrides is None and not fit_options)
    manifest['config_id'] = config['config_id']
    manifest['git_dirty'], manifest['git_revision'] = _git_state()
    transition(manifest, 'running', stage='input_validation')
    try:
        _validate_inputs(inputs, variant, config)
        original_positions(inputs.train.dates, original_dates)
        if not candidates:
            raise ValueError('Empty candidate scope')
        manifest['input_identity'] = {'train': _identity(inputs.train), 'validation': _identity(inputs.validation)}
        manifest['stage'] = 'candidate_search'
        write_json(path / 'config.json', config)
        model, score, search = _search(name, candidates, config, inputs,
                                      lambda p: fit_candidate(name, p, inputs.train, original_dates, config, fit_options), fit_options)
        search['candidate_scope'] = 'frozen_config' if candidate_overrides is None else 'explicit_override'
        write_json(path / 'search_results.json', search)
        if model is None:
            raise ValueError('All M6 candidates failed; see search_results.json')
        manifest['stage'] = 'persisting'
        adapter = 'forest_pca' if isinstance(model, ForestPCA) else None
        state = {'name': name, 'adapter': adapter, 'model': model.to_dict()}
        save_npz(path / 'model.npz', _numeric_arrays(state), state)
        preprocessing = _preprocessor_state(name, model, inputs.train.X.shape[1])
        save_npz(path / 'preprocessor.npz', _numeric_arrays(preprocessing), preprocessing)
        receipt_model = load_m6_model(path)
        after = score_of(name, receipt_model, inputs.validation.X)
        stored_arrays, stored_preprocessing = load_npz(path / 'preprocessor.npz')
        expected = _preprocessor_state(name, receipt_model, inputs.train.X.shape[1])
        receipt = {'preprocessor': {'passed': stored_preprocessing == expected and
                   set(stored_arrays) == set(_numeric_arrays(expected)) and
                   all(np.array_equal(v, _numeric_arrays(expected)[k]) for k, v in stored_arrays.items())}}
        _persist_validation(path, manifest, name, config, inputs, model, score, after, search['selected']['params'], receipt)
        if reference:
            from .m6_reference import train_m6_reference
            reference_path = train_m6_reference(name, variant, out_root, run_id, config=config, inputs=inputs,
                                                original_dates=original_dates, hashes=hashes, candidates=candidates,
                                                fit_options=fit_options, scratch_search=search)
            write_json(path / 'reference_result.json', {'path': reference_path.relative_to(Path(out_root)).as_posix(),
                                                       'status': read_manifest(reference_path)['status']})
        capture_artifacts(manifest, path)
        transition(manifest, 'completed', run_dir=path, stage='validation_complete')
    except Exception as error:
        _record_failure(path, manifest, error)
        raise
    write_json(path / 'manifest.json', manifest)
    return path


def _persist_validation(path, manifest, name, config, inputs, model, score, after, params, receipt):
    kind = 'decision' if name == 'svm' else 'probability'
    default = config['selection']['defaults'][kind]
    threshold = select_threshold(inputs.validation.y_classification, score, score_kind=kind)
    if threshold['threshold'] is None:
        raise ValueError(f'Threshold selection failed: {threshold["reason"]}')
    chosen = float(threshold['threshold'])
    votes = model.decision_function(inputs.validation.X) if name == 'adaboost' else None
    if votes is not None:
        save_npz(path / 'validation_loss_outputs.npz', {'scores': votes, 'target': inputs.validation.y_classification},
                 {'loss': config['losses'][name], 'score_kind': 'native_decision_votes', 'split': 'validation'})
    receipt.update(scores=verify_reload(score, after, rtol=config['persistence']['reload_rtol'], atol=config['persistence']['reload_atol']),
                   labels=verify_reload(score >= chosen, after >= chosen, labels=True))
    receipt['passed'] = all(part['passed'] for part in receipt.values())
    write_json(path / 'load_verification.json', receipt)
    manifest['load_verification'] = receipt
    manifest['selected_hyperparameters'] = params
    manifest['output_policy'] = {'threshold': chosen, 'score_kind': kind, 'default_threshold': default,
                                 'selection_split': threshold['selection_split'], 'rule': config['selection']['threshold']}
    write_json(path / 'validation_metrics.json', {**manifest['output_policy'],
               'metrics': validation_metrics(name, config, inputs.validation.y_classification, score, chosen, loss_scores=votes),
               'baseline': validation_metrics(name, config, inputs.validation.y_classification, score, default, loss_scores=votes)})
    write_predictions(path / 'validation_predictions.csv', dates=inputs.validation.dates, target=inputs.validation.y_classification,
                      raw=score, task='classification', threshold=chosen)


def _record_failure(path, manifest, error):
    (path / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf8')
    transition(manifest, 'failed', stage=manifest['stage'], error={
        'exception_type': type(error).__name__, 'message': str(error), 'traceback_path': 'traceback.txt'})
    write_json(path / 'manifest.json', manifest)


def train_m6_batch(out_root, run_id, *, variants=('with_spike', 'non_spike'), names=M6_MODELS, **kwargs):
    return [train_m6_model(name, variant, out_root, run_id, **kwargs) for variant in variants for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'status'))
    parser.add_argument('--model', required=True, choices=M6_MODELS)
    parser.add_argument('--variant', default='with_spike', choices=('with_spike', 'non_spike'))
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-reference', action='store_true')
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(m6_status(args.out_root, args.variant, args.model, args.run_id), allow_nan=False))
    else:
        print(train_m6_model(args.model, args.variant, args.out_root, args.run_id, resume=args.resume, reference=not args.no_reference))


if __name__ == '__main__':
    main()

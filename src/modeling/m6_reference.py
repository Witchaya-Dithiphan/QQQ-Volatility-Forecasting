"""Dedicated M6 sklearn/XGBoost behavioral references; no dependency enters scratch models.

Reference stacking uses the identical purged expanding Original Train folds, fits
meta only on Train OOS probabilities and refits bases on full variant Train. MLP
selects its own epoch budget on the internal purged tail with partial_fit, then
freshly refits; external Validation is used only for candidate/threshold selection.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import numpy as np

from .artifacts import file_sha256, write_json
from .classification import _state
from .classification_training import _identity
from .persistence import (capture_artifacts, create_run_directory, load_npz, load_reference,
                          make_manifest, save_npz, save_reference, transition, _git_state)
from .preprocessing import Standardizer
from .m6_training import (_pca, _numeric_arrays, _search, _persist_validation, _record_failure,
                          original_positions, score_of)


class ReferenceModel:
    """Persist estimator plus Train-fitted M2 preprocessing as a joblib object."""
    def __init__(self, estimator, scaler=None, pca=None):
        self.estimator, self.scaler, self.pca = estimator, scaler, pca

    def transform(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.pca.transform(X) if self.pca is not None else X

    def predict_proba(self, X):
        return self.estimator.predict_proba(self.transform(X))

    def decision_function(self, X):
        return self.estimator.decision_function(self.transform(X))

    def predict(self, X):
        return self.estimator.predict(self.transform(X))

    def preprocessing_state(self, n_features):
        return {'kind': 'standardize' if self.scaler is not None else 'none', 'feature_count': n_features,
                'scaler': _state.scaler_state(self.scaler) if self.scaler is not None else None,
                'pca': _state.pca_state(self.pca)}


def _logistic(config, seed, budget):
    from sklearn.linear_model import SGDClassifier
    params = dict(config['references']['logistic']['parameters'])
    params['random_state'] = seed
    return SGDClassifier(**params, penalty=None, max_iter=budget, tol=None)


class ReferenceStacking:
    def __init__(self, config, seed, budget):
        self.config, self.seed, self.budget = config, seed, budget

    def _fit_bases(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        config = self.config
        scaler = Standardizer(config['preprocessing']['standardizer']['variance_floor']).fit(X)
        logistic = _logistic(config, self.seed, self.budget).fit(scaler.transform(X), y)
        tree = DecisionTreeClassifier(max_depth=config['models']['decision_tree']['grid']['max_depth'][0],
                                      min_samples_leaf=config['models']['decision_tree']['grid']['min_samples_leaf'][0],
                                      random_state=self.seed).fit(X, y)
        knn = KNeighborsClassifier(n_neighbors=config['models']['knn']['grid']['k'][0], weights='uniform').fit(X, y)
        return [ReferenceModel(logistic, scaler), ReferenceModel(tree), ReferenceModel(knn)]

    @staticmethod
    def _features(X, bases):
        return np.column_stack([base.predict_proba(X)[:, 1] for base in bases])

    def fit(self, train, original_dates):
        positions = original_positions(train.dates, original_dates)
        protocol = self.config['stacking']
        eligible = np.flatnonzero(positions >= int(protocol['initial_original_fraction'] * len(original_dates)))
        blocks = np.array_split(eligible, protocol['folds'])
        features, labels, records = [], [], []
        for fold, block in enumerate(blocks):
            if not len(block):
                raise ValueError(f'Fold {fold}: empty OOS block')
            allowed = np.flatnonzero(positions < positions[block[0]] - protocol['purge_original_trading_days'])
            if (len(allowed) < self.config['models']['knn']['grid']['k'][0]
                    or len(np.unique(train.y_classification[allowed])) != 2
                    or len(np.unique(train.y_classification[block])) != 2):
                raise ValueError(f'Fold {fold}: invalid purged Train/OOS classes or size')
            bases = self._fit_bases(train.X[allowed], train.y_classification[allowed])
            features.append(self._features(train.X[block], bases))
            labels.append(train.y_classification[block])
            records.append({'fold': fold, 'train_positions': positions[allowed].tolist(),
                            'validation_positions': positions[block].tolist()})
        oos = np.vstack(features)
        meta_scaler = Standardizer(self.config['preprocessing']['standardizer']['variance_floor']).fit(oos)
        meta = _logistic(self.config, self.seed, self.budget).fit(meta_scaler.transform(oos), np.concatenate(labels))
        self.meta = ReferenceModel(meta, meta_scaler)
        self.bases = self._fit_bases(train.X, train.y_classification)
        self.protocol = {'folds': records, 'meta_fit': 'train_oos_only', 'oos_positions': positions[np.concatenate(blocks)].tolist()}
        return self

    def predict_proba(self, X):
        return self.meta.predict_proba(self._features(X, self.bases))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)

    def preprocessing_state(self, n_features):
        return {'kind': 'base_specific', 'feature_count': n_features,
                'preprocessors': [b.preprocessing_state(n_features) for b in self.bases],
                'meta_preprocessor': self.meta.preprocessing_state(len(self.bases))}


def _mlp(params, train, original_dates, config, options):
    from sklearn.neural_network import MLPClassifier
    from .classification.mlp import EpochMonitor
    fixed, neural = config['models']['mlp']['parameters'], config['neural']
    positions = original_positions(train.dates, original_dates)
    tail = len(positions) - int(np.ceil(neural['internal_tail_fraction'] * len(positions)))
    n_sub = int(np.searchsorted(positions, positions[tail] - neural['purge_original_trading_days']))
    if n_sub < 2 or tail >= len(positions):
        raise ValueError('Too few rows for neural subtrain/purge/tail')
    def estimator(n, seed):
        kwargs = {**config['references']['mlp']['parameters'], 'random_state': seed,
                  'hidden_layer_sizes': tuple(params['hidden_layers']), 'learning_rate_init': params['learning_rate'],
                  'alpha': params['l2'], 'batch_size': min(fixed['batch'], n)}
        return MLPClassifier(**kwargs)
    scaler = Standardizer(config['preprocessing']['standardizer']['variance_floor']).fit(train.X[:n_sub])
    Z, Ztail = scaler.transform(train.X[:n_sub]), scaler.transform(train.X[tail:])
    model = estimator(n_sub, config['experiment']['seed'])
    monitor = EpochMonitor(options.get('patience', neural['patience']), neural['min_delta'])
    curve = []
    for _ in range(options.get('max_epochs', fixed['max_epochs'])):
        model.partial_fit(Z, train.y_classification[:n_sub], classes=np.array([0, 1]))
        p = np.clip(model.predict_proba(Ztail)[:, 1], config['selection']['probability_epsilon'], 1 - config['selection']['probability_epsilon'])
        y = train.y_classification[tail:]
        curve.append(float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))))
        if monitor.update(curve[-1]):
            break
    refit_scaler = Standardizer(config['preprocessing']['standardizer']['variance_floor']).fit(train.X)
    model = estimator(len(train.X), neural['refit_seed'])
    Z = refit_scaler.transform(train.X)
    for _ in range(monitor.best_epoch):
        model.partial_fit(Z, train.y_classification, classes=np.array([0, 1]))
    return ReferenceModel(model, refit_scaler), {'best_epoch': monitor.best_epoch, 'stopped_epoch': monitor.epoch,
           'n_subtrain': n_sub, 'n_tail': len(positions) - tail, 'tail_bce': curve,
           'refit_epochs': monitor.best_epoch, 'scaler_fit': 'subtrain_then_fresh_full_train',
           'scope_note': 'sklearn SGD includes momentum and sample-scaled alpha; behavioral comparison only.'}


def fit_reference(name, params, train, original_dates, config, fit_options=None):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.exceptions import ConvergenceWarning
    options = fit_options or {}
    seed, fixed = config['experiment']['seed'], config['models'][name]['parameters']
    if name == 'mlp':
        return _mlp(params, train, original_dates, config, options)
    if name == 'stacking':
        model = ReferenceStacking(config, seed, options.get('logistic_max_iter', config['models']['logistic']['parameters']['max_iter']))
        model.fit(train, original_dates)
        return model, model.protocol
    scaler, pca, protocol = None, None, {}
    X = train.X
    if config['models'][name]['preprocess'] == 'standardize' or params.get('pca_components') is not None:
        scaler = Standardizer(config['preprocessing']['standardizer']['variance_floor']).fit(X)
        X = scaler.transform(X)
    if params.get('pca_components') is not None:
        pca = _pca(params['pca_components'], config).fit(X)
        X = pca.transform(X)
    kwargs = {k: v for k, v in params.items() if k != 'pca_components'}
    if name == 'random_forest':
        estimator = RandomForestClassifier(**kwargs, min_samples_leaf=fixed['min_samples_leaf'], max_features=fixed['max_features'],
                                           bootstrap=fixed['bootstrap'], random_state=seed, n_jobs=1)
        protocol['scope_note'] = 'sklearn averages leaf probabilities; scratch aggregates hard tree votes.'
    elif name == 'gradient_boosting':
        estimator = GradientBoostingClassifier(**kwargs, loss='log_loss', random_state=seed)
    elif name == 'adaboost':
        ref = config['references']['adaboost']['parameters']
        stump = DecisionTreeClassifier(max_depth=ref['max_depth'], criterion=ref['criterion'], random_state=seed)
        syntax = inspect.signature(AdaBoostClassifier).parameters
        key = 'estimator' if 'estimator' in syntax else 'base_estimator'
        kwargs[key] = stump
        if 'algorithm' in syntax and syntax['algorithm'].default != 'deprecated':
            kwargs['algorithm'] = 'SAMME'
        estimator = AdaBoostClassifier(**kwargs, random_state=seed)
        protocol.update(installed_estimator_keyword=key, algorithm=kwargs.get('algorithm', 'installed_discrete_default'),
                        scope_note='Discrete sklearn SAMME weighted Gini stump reference; scratch minimizes weighted error.')
    elif name == 'svm':
        estimator = LinearSVC(**kwargs, **config['references']['svm']['parameters'], max_iter=100000, tol=1e-6)
    elif name == 'xgboost':
        from xgboost import XGBClassifier
        estimator = XGBClassifier(**kwargs, objective='binary:logistic', tree_method='exact', n_jobs=1,
                                  random_state=seed, eval_metric='logloss', base_score=.5)
    else:
        raise ValueError(f'Unknown reference: {name}')
    try:
        estimator.fit(X, train.y_classification)
    except ConvergenceWarning as error:
        # Under warnings-as-errors this is an invalid reference candidate, not a silent fit.
        raise ValueError(f'Reference convergence failure: {error}') from error
    protocol['constructor'] = type(estimator).__name__
    return ReferenceModel(estimator, scaler, pca), protocol


def _constructor_parameters(estimator):
    """JSON description of installed APIs, including nested AdaBoost estimators."""
    def describe(value):
        if hasattr(value, 'get_params'):
            return {'constructor': type(value).__name__, 'parameters': _constructor_parameters(value)}
        if isinstance(value, np.generic):
            return describe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return {'value': None, 'reason': f'Installed constructor sentinel: {value}'}
        if isinstance(value, (list, tuple)):
            return [describe(v) for v in value]
        if isinstance(value, dict):
            return {k: describe(v) for k, v in value.items()}
        return value
    return {key: describe(value) for key, value in estimator.get_params(deep=False).items()}


def train_m6_reference(name, variant, out_root, run_id, *, config, inputs, original_dates, hashes, candidates,
                       fit_options=None, scratch_search=None):
    path = create_run_directory(Path(out_root), variant, 'classification', name, 'reference', run_id)
    manifest = make_manifest(run_id=run_id, variant=variant, task='classification', model=name, implementation='reference',
                             lifecycle='supervised_tuned', hashes=hashes, role='finalized_reference',
                             required=bool(scratch_search and scratch_search['candidate_scope'] == 'frozen_config' and not fit_options))
    manifest['config_id'] = config['config_id']
    manifest['git_dirty'], manifest['git_revision'] = _git_state()
    manifest['input_identity'] = {'train': _identity(inputs.train), 'validation': _identity(inputs.validation)}
    transition(manifest, 'running', stage='reference_search')
    try:
        # Missing optional dependencies are recorded honestly. No installation occurs.
        import sklearn
        import joblib
        if name == 'xgboost':
            import xgboost
        model, score, search = _search(name, candidates, config, inputs,
                                      lambda p: fit_reference(name, p, inputs.train, original_dates, config, fit_options), fit_options)
        search['candidate_scope'] = scratch_search['candidate_scope'] if scratch_search else 'provided'
        search['scope_note'] = 'Independent behavioral reference search; no internal state equality is claimed.'
        write_json(path / 'config.json', config)
        write_json(path / 'search_results.json', search)
        if model is None:
            raise ValueError('All reference candidates failed; see search_results.json')
        manifest['stage'] = 'persisting'
        save_reference(path / 'model.joblib', model)
        state = model.preprocessing_state(inputs.train.X.shape[1])
        arrays = _numeric_arrays(state)
        save_npz(path / 'preprocessor.npz', arrays, state)
        restored = load_reference(path / 'model.joblib', expected_sha256=file_sha256(path / 'model.joblib'))
        after = score_of(name, restored, inputs.validation.X)
        stored_arrays, stored = load_npz(path / 'preprocessor.npz')
        receipt = {'preprocessor': {'passed': stored == restored.preprocessing_state(inputs.train.X.shape[1]) and
                   set(arrays) == set(stored_arrays) and all(np.array_equal(arrays[k], stored_arrays[k]) for k in arrays)}}
        _persist_validation(path, manifest, name, config, inputs, model, score, after, search['selected']['params'], receipt)
        if name == 'stacking':
            constructor = type(model).__name__
            parameters = {'bases': [_constructor_parameters(b.estimator) for b in model.bases],
                          'meta': _constructor_parameters(model.meta.estimator), 'seed': model.seed,
                          'protocol': config['stacking']}
        else:
            constructor, parameters = type(model.estimator).__name__, _constructor_parameters(model.estimator)
        manifest['selected_hyperparameters'].update(reference_constructor=constructor, reference_parameters=parameters,
                                                    reference_protocol=search['selected']['protocol'])
        capture_artifacts(manifest, path)
        transition(manifest, 'completed', run_dir=path, stage='validation_complete')
    except ImportError as error:
        transition(manifest, 'skipped', reason=f'Optional reference dependency unavailable: {error}', stage='dependency_check')
    except Exception as error:
        _record_failure(path, manifest, error)
        raise
    write_json(path / 'manifest.json', manifest)
    return path

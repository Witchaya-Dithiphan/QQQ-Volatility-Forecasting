"""NumPy-only M6 stacking on purged Original Train positions.

Default bases select the first frozen M5 grid values: unweighted logistic
(l2=0, learning_rate=.05, max_iter=5000), Gini tree (depth=2, leaf>=5),
uniform kNN (k=3). Stacking owns separate Train-only standardizers for
logistic and kNN; their internal scaling is disabled. Tree and meta features
are unscaled. Meta logistic is unweighted with the same iteration budget
and the current M5 gradient-tolerance stopping rule.

logistic_max_iter is the sole bounded test injection (integer 1..5000),
and applies to both logistic learners. Dates are mandatory: filtered row
numbers can never stand in for the original trading timeline.
"""
from __future__ import annotations

import copy
import numpy as np

from ..artifacts import canonical_bytes
from ..configuration import load_config
from ..datasets import load_train_validation
from ..preprocessing import Standardizer, _matrix
from ..classification.logistic import LogisticRegression
from ..classification.decision_tree import DecisionTreeClassifier, DecisionTreeNode
from ..classification.knn import KNN


class _StackingTree(DecisionTreeClassifier):
    """Reuse M5 splitting, enforce frozen leaf size and align leaf counts.

    M5 stores only the counts of classes present in a leaf. Recount using
    global binary class order so pure-positive leaves retain P(1)=1.
    """
    def __init__(self, *, min_samples_leaf, **kwargs):
        super().__init__(**kwargs)
        self.min_samples_leaf = min_samples_leaf

    def _split_gain(self, parent, left, right):
        if min(len(left), len(right)) < self.min_samples_leaf:
            return 0.0
        return super()._split_gain(parent, left, right)

    def fit(self, X, y):
        super().fit(X, y)
        def recount(node, data, labels):
            if node.is_leaf():
                node.value = np.bincount(labels, minlength=2)
            else:
                mask = data[:, node.feature] <= node.threshold
                recount(node.left, data[mask], labels[mask])
                recount(node.right, data[~mask], labels[~mask])
        recount(self.tree_, np.asarray(X), np.asarray(y, dtype=int))
        return self


def _dates(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw) or raw.dtype.kind not in 'MU':
        raise ValueError(f'{name}: expected nonempty dates')
    try:
        dates = raw.astype('datetime64[ns]')
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'{name}: invalid dates') from exc
    if np.isnat(dates).any() or (dates[1:] <= dates[:-1]).any():
        raise ValueError(f'{name}: dates must be unique and chronological')
    return dates


def _labels(values, n, *, both=True):
    y = np.asarray(values)
    if y.shape != (n,) or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected aligned binary labels')
    if both and len(np.unique(y)) != 2:
        raise ValueError('Expected both classes')
    return y.astype(np.int64)


def _join(dates, original, supplied=None):
    positions = np.searchsorted(original, dates)
    if (positions >= len(original)).any() or not np.array_equal(original[positions], dates):
        raise ValueError('Dates require an exact one-to-one Original Train join')
    if supplied is not None:
        given = np.asarray(supplied)
        if given.dtype.kind not in 'iu' or given.shape != positions.shape or not np.array_equal(given, positions):
            raise ValueError('Original positions do not match exact date join')
    return positions


def _load_tree(state, n_features, depth, max_depth):
    node = DecisionTreeNode()
    if set(state) == {'value'}:
        counts = np.asarray(state['value'])
        if counts.shape != (2,) or counts.dtype.kind not in 'iu' or (counts < 0).any() or counts.sum() <= 0:
            raise ValueError('Invalid tree counts')
        node.value = counts
    else:
        if set(state) != {'feature', 'threshold', 'left', 'right'} or depth >= max_depth:
            raise ValueError('Invalid tree structure')
        feature = state['feature']
        if type(feature) is not int or not 0 <= feature < n_features or not np.isfinite(state['threshold']):
            raise ValueError('Invalid tree split')
        node.feature, node.threshold = feature, float(state['threshold'])
        node.left = _load_tree(state['left'], n_features, depth + 1, max_depth)
        node.right = _load_tree(state['right'], n_features, depth + 1, max_depth)
    return node


class StackingClassifier:
    def __init__(self, random_state=42, *, logistic_max_iter=None):
        frozen = load_config()
        budget = frozen['models']['logistic']['parameters']['max_iter']
        iterations = budget if logistic_max_iter is None else logistic_max_iter
        if type(iterations) is not int or not 1 <= iterations <= budget:
            raise ValueError('logistic_max_iter must be an integer in 1..5000')
        if type(random_state) is not int or not 0 <= random_state < 2**32:
            raise ValueError('Invalid random seed')
        self.random_state = random_state
        self.protocol_ = copy.deepcopy(frozen['stacking'])
        self.config_ = {
            'config_id': frozen['config_id'], 'config_sha256': frozen['config_sha256'],
            'logistic': {'learning_rate': frozen['models']['logistic']['parameters']['learning_rate'],
                         'max_iter': iterations, 'random_state': random_state},
            'decision_tree': {'max_depth': frozen['models']['decision_tree']['grid']['max_depth'][0],
                              'min_samples_leaf': frozen['models']['decision_tree']['grid']['min_samples_leaf'][0],
                              'random_state': random_state},
            'knn': {'n_neighbors': frozen['models']['knn']['grid']['k'][0]},
            'preprocessing': ['standardize', 'none', 'standardize'],
            'preprocessing_policy': 'base_specific', 'meta_preprocessing': 'none',
            'variance_floor': frozen['preprocessing']['standardizer']['variance_floor'],
            'meta_class_weight': None, 'logistic_l2': 0,
            'logistic_stopping': 'gradient_tolerance_M5', 'knn_weights': 'uniform',
        }
        self._clear()

    def _clear(self):
        self.base_learners_ = self.meta_learner_ = self.classes_ = None
        self.preprocessors_ = None
        self.fold_records_ = []
        self.status_, self.error_ = 'pending', None

    def _new_bases(self):
        return [LogisticRegression(**self.config_['logistic'], standardize=False),
                _StackingTree(**self.config_['decision_tree']), KNN(**self.config_['knn'], standardize=False)]

    def _fit_bases(self, X, y):
        scaler = Standardizer(self.config_['variance_floor']).fit(X)
        knn_scaler = Standardizer(self.config_['variance_floor']).fit(X)
        bases = self._new_bases()
        bases[0].fit(scaler.transform(X), y)
        bases[1].fit(X, y)
        bases[2].fit(knn_scaler.transform(X), y)
        return bases, [scaler, None, knn_scaler]

    @staticmethod
    def _features(X, bases, preprocessors):
        columns = []
        for base, scaler in zip(bases, preprocessors):
            p = np.asarray(base.predict_proba(scaler.transform(X) if scaler else X), dtype=np.float64)
            if p.shape != (len(X), 2) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.allclose(p.sum(axis=1), 1):
                raise ValueError('Invalid base probabilities')
            columns.append(p[:, 1])
        return np.column_stack(columns)

    def _blocks(self, positions, original_n):
        start = int(self.protocol_['initial_original_fraction'] * original_n)
        eligible = np.flatnonzero(positions >= start)
        blocks = np.array_split(eligible, self.protocol_['folds'])
        if any(not len(block) for block in blocks):
            raise ValueError('Not enough OOS rows for four folds')
        return blocks

    def fit(self, X, y, *, dates=None, original_dates=None, original_positions=None):
        self._clear()
        self.status_ = 'running'
        rng_state = np.random.get_state()  # M5 logistic seeds the global RNG.
        try:
            X = _matrix(X)
            y = _labels(y, len(X))
            dates = _dates(dates, 'Train')
            original = _dates(original_dates, 'Original Train')
            if len(dates) != len(X):
                raise ValueError('Dates and features must be aligned')
            positions = _join(dates, original, original_positions)
            blocks = self._blocks(positions, len(original))
            features, labels = [], []
            for fold, block in enumerate(blocks):
                allowed = np.flatnonzero(positions < positions[block[0]] - self.protocol_['purge_original_trading_days'])
                if len(allowed) < self.config_['knn']['n_neighbors']:
                    raise ValueError(f'Fold {fold}: insufficient purged training rows')
                _labels(y[allowed], len(allowed))
                _labels(y[block], len(block))
                bases, scalers = self._fit_bases(X[allowed], y[allowed])
                features.append(self._features(X[block], bases, scalers))
                labels.append(y[block])
                self.fold_records_.append({'fold': fold,
                    'train_positions': positions[allowed].tolist(),
                    'validation_positions': positions[block].tolist()})
            self.oos_meta_features_ = np.vstack(features)
            self.oos_labels_ = np.concatenate(labels)
            self.oos_positions_ = positions[np.concatenate(blocks)].copy()
            self.meta_learner_ = LogisticRegression(**self.config_['logistic'], standardize=False).fit(self.oos_meta_features_, self.oos_labels_)
            self.base_learners_, self.preprocessors_ = self._fit_bases(X, y)
            self.classes_ = np.array([0, 1], dtype=np.int64)
            self.n_features_ = X.shape[1]
            self.original_positions_ = positions.copy()
            self.train_dates_, self.original_dates_ = dates.copy(), original.copy()
            self.status_ = 'completed'
            return self
        except Exception as exc:
            self.base_learners_ = self.meta_learner_ = self.preprocessors_ = self.classes_ = None
            self.status_ = 'failed'
            self.error_ = {'exception_type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            np.random.set_state(rng_state)

    def fit_dataset(self, train, original_train):
        """Fit accepted DatasetSplit Q75 labels; no Validation argument exists."""
        return self.fit(train.X, train.y_classification, dates=train.dates,
                        original_dates=original_train.dates)

    def fit_variant(self, variant='with_spike'):
        """Use M2 accepted inputs; external Validation is never passed to fit."""
        data = load_train_validation(variant)
        original = data.train if variant == 'with_spike' else load_train_validation('with_spike').train
        return self.fit_dataset(data.train, original)

    def _fitted(self):
        if self.status_ != 'completed' or self.meta_learner_ is None or self.base_learners_ is None:
            raise ValueError('Model not fitted')

    def predict_proba(self, X):
        self._fitted()
        X = _matrix(X)
        if X.shape[1] != self.n_features_:
            raise ValueError('Feature dimension mismatch')
        features = self._features(X, self.base_learners_, self.preprocessors_)
        p = self.meta_learner_.predict_proba(features)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('Invalid meta probabilities')
        return p

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(np.int64)

    def to_dict(self):
        self._fitted()
        def scaler_state(scaler):
            if scaler is None:
                return None
            return {'mean': scaler.mean_.tolist(), 'variance': scaler.variance_.tolist(),
                    'scale': scaler.scale_.tolist(), 'constant': scaler.constant_.tolist(),
                    'warnings': list(scaler.warnings_)}
        state = {'type': 'StackingClassifier', 'version': 1,
            'random_state': self.random_state, 'config': copy.deepcopy(self.config_),
            'protocol': copy.deepcopy(self.protocol_), 'classes': self.classes_.tolist(),
            'n_features': self.n_features_, 'status': self.status_,
            'bases': [self.base_learners_[0].to_dict(), self.base_learners_[1].to_dict(), self.base_learners_[2].to_dict()],
            'meta': self.meta_learner_.to_dict(),
            'preprocessors': [scaler_state(scaler) for scaler in self.preprocessors_],
            'oos_positions': self.oos_positions_.tolist(),
            'oos_meta_features': self.oos_meta_features_.tolist(), 'oos_labels': self.oos_labels_.tolist(),
            'original_positions': self.original_positions_.tolist(),
            'train_dates': self.train_dates_.astype(str).tolist(),
            'original_dates': self.original_dates_.astype(str).tolist(),
            'fold_records': copy.deepcopy(self.fold_records_)}
        canonical_bytes(state)
        return state

    @classmethod
    def from_dict(cls, state):
        try:
            canonical_bytes(state)
            expected = {'type', 'version', 'random_state', 'config', 'protocol', 'classes',
                'n_features', 'status', 'bases', 'meta', 'preprocessors', 'oos_positions',
                'oos_meta_features', 'oos_labels', 'original_positions', 'train_dates',
                'original_dates', 'fold_records'}
            if set(state) != expected or state['type'] != 'StackingClassifier' or state['version'] != 1 or state['status'] != 'completed':
                raise ValueError('Invalid stacking state schema')
            m = cls(state['random_state'], logistic_max_iter=state['config']['logistic']['max_iter'])
            if state['config'] != m.config_ or state['protocol'] != m.protocol_ or state['classes'] != [0, 1]:
                raise ValueError('Stacking config/protocol/classes mismatch')
            n = state['n_features']
            if type(n) is not int or n < 1 or len(state['bases']) != 3 or len(state['preprocessors']) != 3:
                raise ValueError('Invalid dimensions or base count')
            m.train_dates_ = _dates(state['train_dates'], 'Train')
            m.original_dates_ = _dates(state['original_dates'], 'Original Train')
            m.original_positions_ = _join(m.train_dates_, m.original_dates_, state['original_positions'])
            blocks = m._blocks(m.original_positions_, len(m.original_dates_))
            oos = m.original_positions_[np.concatenate(blocks)]
            if not np.array_equal(oos, state['oos_positions']):
                raise ValueError('Invalid OOS positions')
            records = []
            for fold, block in enumerate(blocks):
                allowed = m.original_positions_[m.original_positions_ < m.original_positions_[block[0]] - m.protocol_['purge_original_trading_days']]
                if len(allowed) < m.config_['knn']['n_neighbors']:
                    raise ValueError('Invalid persisted fold')
                records.append({'fold': fold, 'train_positions': allowed.tolist(), 'validation_positions': m.original_positions_[block].tolist()})
            if records != state['fold_records']:
                raise ValueError('Invalid persisted purge record')
            m.fold_records_, m.oos_positions_ = records, oos
            m.oos_meta_features_ = _matrix(state['oos_meta_features'])
            if m.oos_meta_features_.shape != (len(oos), 3) or ((m.oos_meta_features_ < 0) | (m.oos_meta_features_ > 1)).any():
                raise ValueError('Invalid persisted meta features')
            m.oos_labels_ = _labels(state['oos_labels'], len(oos))
            for block in np.array_split(m.oos_labels_, 4):
                _labels(block, len(block))
            def logistic(data, dimension):
                template = LogisticRegression(**m.config_['logistic'], standardize=False)
                parameters = ('learning_rate', 'max_iter', 'tol', 'l2', 'class_weight', 'standardize', 'random_state')
                expected_keys = set(parameters) | {'coefficients', 'intercept', 'scaler', 'loss_history', 'n_iter', 'converged'}
                if set(data) != expected_keys or any(data[key] != getattr(template, key) for key in parameters) or data['standardize'] is not False or data['scaler'] is not None:
                    raise ValueError('Invalid logistic state/config')
                coefficients = np.asarray(data['coefficients'], dtype=float)
                history = np.asarray(data['loss_history'], dtype=float)
                if coefficients.shape != (dimension,) or not np.isfinite(coefficients).all() or not np.isscalar(data['intercept']) or not np.isfinite(data['intercept']):
                    raise ValueError('Invalid logistic dimensions/state')
                if type(data['n_iter']) is not int or not 1 <= data['n_iter'] <= template.max_iter or history.shape != (data['n_iter'],) or not np.isfinite(history).all() or type(data['converged']) is not bool:
                    raise ValueError('Invalid logistic convergence state')
                return LogisticRegression.from_dict(data)
            bases = m._new_bases()
            bases[0] = logistic(state['bases'][0], n)
            tree = state['bases'][1]
            if set(tree) != {'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_features', 'classes', 'tree'} or any(tree[key] != getattr(bases[1], key) for key in ('max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state')) or tree['n_features'] != n or tree['classes'] != [0, 1]:
                raise ValueError('Invalid tree state/config')
            _load_tree(tree['tree'], n, 0, bases[1].max_depth)
            bases[1] = DecisionTreeClassifier.from_dict(tree)
            knn = state['bases'][2]
            if set(knn) != {'n_neighbors', 'weights', 'standardize', 'scaler', 'X_train', 'y_train'} or knn['n_neighbors'] != m.config_['knn']['n_neighbors'] or knn['weights'] != 'uniform' or knn['standardize'] is not False or knn['scaler'] is not None:
                raise ValueError('Invalid kNN state/config')
            data = _matrix(knn['X_train'])
            if data.shape != (len(m.train_dates_), n):
                raise ValueError('Invalid kNN dimensions')
            labels = _labels(knn['y_train'], len(data))
            if not np.array_equal(labels[np.concatenate(blocks)], m.oos_labels_):
                raise ValueError('OOS labels do not match full Train labels')
            for record in records:
                rows = np.searchsorted(m.original_positions_, record['train_positions'])
                _labels(labels[rows], len(rows))
            bases[2] = KNN.from_dict(knn)
            if not np.isfinite(bases[2].X_train_).all():
                raise ValueError('Invalid kNN numeric range')
            scalers = []
            if state['preprocessors'][1] is not None:
                raise ValueError('Invalid tree preprocessing')
            for scaler_state in (state['preprocessors'][0], state['preprocessors'][2]):
                if not isinstance(scaler_state, dict) or set(scaler_state) != {'mean', 'variance', 'scale', 'constant', 'warnings'}:
                    raise ValueError('Invalid preprocessing schema')
                scaler = Standardizer(m.config_['variance_floor'])
                for name in ('mean', 'variance', 'scale'):
                    a = np.asarray(scaler_state[name], dtype=float)
                    if a.shape != (n,) or not np.isfinite(a).all():
                        raise ValueError('Invalid scaler dimensions/state')
                    setattr(scaler, name + '_', a)
                constant = np.asarray(scaler_state['constant'])
                if constant.shape != (n,) or constant.dtype.kind != 'b' or (scaler.variance_ < 0).any() or (scaler.scale_ <= 0).any():
                    raise ValueError('Invalid scaler statistics')
                scaler.constant_ = constant
                if not np.array_equal(constant, scaler.variance_ <= scaler.variance_floor) or not np.array_equal(scaler.scale_, np.where(constant, 1., np.sqrt(scaler.variance_))):
                    raise ValueError('Inconsistent scaler statistics')
                scaler.warnings_ = [f'Constant feature {i}: variance <= {scaler.variance_floor}' for i in np.flatnonzero(constant)]
                if scaler.warnings_ != scaler_state['warnings']:
                    raise ValueError('Invalid scaler warnings')
                scalers.append(scaler)
            m.base_learners_, m.preprocessors_ = bases, [scalers[0], None, scalers[1]]
            m.meta_learner_ = logistic(state['meta'], 3)
            m.classes_, m.n_features_, m.status_ = np.array([0, 1]), n, 'completed'
            return m
        except (KeyError, TypeError, IndexError, OverflowError) as exc:
            raise ValueError('Invalid stacking state') from exc

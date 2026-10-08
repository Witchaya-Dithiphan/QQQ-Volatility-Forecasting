"""
AdaBoost classifier (thin wrapper over sklearn.ensemble.AdaBoostClassifier).
"""

from sklearn.ensemble import AdaBoostClassifier

from src.modeling.ensemble.sklearn_wrapper import SklearnEnsemble


class AdaBoost(SklearnEnsemble):
    _cls = AdaBoostClassifier
    _defaults = {"n_estimators": 50, "learning_rate": 1.0, "random_state": 42}

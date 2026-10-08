"""
Random Forest classifier (thin wrapper over sklearn.ensemble.RandomForestClassifier).
"""

from sklearn.ensemble import RandomForestClassifier

from src.modeling.ensemble.sklearn_wrapper import SklearnEnsemble


class RandomForest(SklearnEnsemble):
    _cls = RandomForestClassifier
    _defaults = {"n_estimators": 100, "max_depth": None, "random_state": 42}

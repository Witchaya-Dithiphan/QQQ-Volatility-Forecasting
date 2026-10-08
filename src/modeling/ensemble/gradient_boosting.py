"""
Gradient Boosting classifier (thin wrapper over sklearn.ensemble.GradientBoostingClassifier).
"""

from sklearn.ensemble import GradientBoostingClassifier

from src.modeling.ensemble.sklearn_wrapper import SklearnEnsemble


class GradientBoosting(SklearnEnsemble):
    _cls = GradientBoostingClassifier
    _defaults = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}

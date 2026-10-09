"""Ensemble methods."""

from src.modeling.ensemble.adaboost import AdaBoost
from src.modeling.ensemble.gradient_boosting import GradientBoosting
from src.modeling.ensemble.random_forest import RandomForest
from src.modeling.ensemble.stacking import StackingClassifier

__all__ = ["AdaBoost", "GradientBoosting", "RandomForest", "StackingClassifier"]

from .xgboost import XGBoostClassifier
__all__ += ['XGBoostClassifier']

"""Model name -> class.

Both developers edit this file, so keep it mechanical: one import, one dict entry,
sorted by name. That keeps merge conflicts to a single obvious line.
"""
from __future__ import annotations

from .classification.adaboost import AdaBoost
from .classification.decision_tree import DecisionTree
from .classification.gaussian_nb import GaussianNB
from .classification.gradient_boosting import GradientBoosting
from .classification.knn import KNN
from .classification.logistic import Logistic
from .classification.perceptron import Perceptron
from .classification.random_forest import RandomForest
from .classification.slp import SLP
from .clustering.agglomerative import Agglomerative
from .clustering.kmeans import KMeans
from .core.base import BaseModel
from .regression.elastic_net import ElasticNet
from .regression.multiple_linear import MultipleLinear
from .regression.polynomial import Polynomial
from .regression.simple_linear import SimpleLinear

MODELS: dict[str, type[BaseModel]] = {
    "adaboost": AdaBoost,
    "agglomerative": Agglomerative,
    "decision_tree": DecisionTree,
    "elastic_net": ElasticNet,
    "gaussian_nb": GaussianNB,
    "gradient_boosting": GradientBoosting,
    "kmeans": KMeans,
    "knn": KNN,
    "logistic": Logistic,
    "perceptron": Perceptron,
    "random_forest": RandomForest,
    "slp": SLP,
    "multiple_linear": MultipleLinear,
    "polynomial": Polynomial,
    "simple_linear": SimpleLinear,
}


def model_names() -> list[str]:
    return sorted(MODELS)


def build(name: str, **hyperparams) -> BaseModel:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; available: {model_names()}")
    return MODELS[name](**hyperparams)


def models_for_task(task: str) -> list[str]:
    return sorted(name for name, factory in MODELS.items() if factory.task == task)

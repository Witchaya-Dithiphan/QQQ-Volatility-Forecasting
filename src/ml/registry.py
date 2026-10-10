"""Model name -> class.

Both developers edit this file, so keep it mechanical: one import, one dict entry,
sorted by name. That keeps merge conflicts to a single obvious line.
"""
from __future__ import annotations

from .core.base import BaseModel
from .regression.multiple_linear import MultipleLinear

MODELS: dict[str, type[BaseModel]] = {
    "multiple_linear": MultipleLinear,
}


def model_names() -> list[str]:
    return sorted(MODELS)


def build(name: str, **hyperparams) -> BaseModel:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; available: {model_names()}")
    return MODELS[name](**hyperparams)


def models_for_task(task: str) -> list[str]:
    return sorted(name for name, factory in MODELS.items() if factory.task == task)

import pytest

from src.ml.core.base import BaseModel
from src.ml.registry import MODELS, build, model_names, models_for_task


def test_every_registered_entry_is_a_base_model_subclass():
    for name, factory in MODELS.items():
        assert issubclass(factory, BaseModel), name


def test_registered_name_matches_the_class_attribute():
    for name, factory in MODELS.items():
        assert factory.name == name


def test_every_registered_model_declares_a_known_task():
    for name, factory in MODELS.items():
        assert factory.task in ("regression", "classification", "clustering"), name


def test_build_returns_a_fresh_instance():
    name = next(iter(MODELS))
    assert build(name) is not build(name)


def test_every_model_constructs_with_no_arguments():
    """The trainer rebuilds models with model_class() before load_state()."""
    for name in model_names():
        build(name)


def test_unknown_model_name_lists_what_is_available():
    with pytest.raises(KeyError, match="available"):
        build("does_not_exist")


def test_model_names_are_sorted_for_stable_merge_conflicts():
    assert model_names() == sorted(model_names())


def test_models_for_task_filters():
    assert "multiple_linear" in models_for_task("regression")
    assert "multiple_linear" not in models_for_task("classification")

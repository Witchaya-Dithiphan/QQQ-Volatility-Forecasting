"""BaseModel is the contract both developers code against, so it has to fail loudly."""
import numpy as np
import pytest

from src.ml.core.base import BaseModel


class Dummy(BaseModel):
    name = "dummy"
    task = "regression"

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def get_params(self):
        return {"scale": self.scale}

    def fit(self, X, y=None):
        self.mean_ = float(np.asarray(y).mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_ * self.scale)

    def state_dict(self):
        return {"mean": np.array([self.mean_])}

    def load_state(self, arrays, metadata):
        self.mean_ = float(arrays["mean"][0])
        self.scale = metadata["params"]["scale"]
        return self


def test_subclass_can_fit_and_predict():
    model = Dummy().fit(np.zeros((3, 2)), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(model.predict(np.zeros((3, 2))), 4.0)


def test_state_round_trip_restores_predictions():
    X, y = np.zeros((3, 2)), np.array([1.0, 2.0, 3.0])
    original = Dummy(scale=3.0).fit(X, y)
    restored = Dummy().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.allclose(original.predict(X), restored.predict(X))


def test_instantiating_the_interface_directly_is_rejected():
    with pytest.raises(TypeError):
        BaseModel()


def test_incomplete_subclass_fails_at_construction_not_at_runtime():
    """A missing method must surface immediately, not after a long training run."""

    class Incomplete(BaseModel):
        name = "incomplete"
        task = "regression"

        def get_params(self):
            return {}

    with pytest.raises(TypeError):
        Incomplete()


def test_classification_helpers_raise_when_not_implemented():
    with pytest.raises(NotImplementedError):
        Dummy().decision_function(np.zeros((1, 2)))
    with pytest.raises(NotImplementedError):
        Dummy().predict_proba(np.zeros((1, 2)))


def test_reference_reports_absence_by_name():
    with pytest.raises(NotImplementedError, match="Dummy"):
        Dummy().reference()


def test_every_hyperparameter_has_a_default():
    """trainer rebuilds models with model_class() before load_state(), so this must work."""
    assert Dummy().get_params() == {"scale": 2.0}

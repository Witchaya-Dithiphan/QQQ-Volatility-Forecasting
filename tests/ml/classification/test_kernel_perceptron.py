import numpy as np
import pytest

from src.ml.classification.kernel_perceptron import KernelPerceptron
from src.ml.core.persist import load_npz, save_npz


def test_rbf_kernel_perceptron_converges_on_xor():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    model = KernelPerceptron(gamma=2.0, max_epochs=100).fit(X, y)
    assert model.converged_
    assert model.history_["updates"][-1] == 0
    assert model.n_epochs_ < 100
    assert np.array_equal(model.predict(X), y)


def test_one_mistake_updates_kernel_weight_and_bias():
    model = KernelPerceptron(gamma=1.0, max_epochs=1).fit([[0.0]], [1])
    assert np.array_equal(model.alpha_, [1.0])
    assert model.intercept_ == 1.0
    assert model.decision_function([[0.0], [1.0]]) == pytest.approx([2.0, 1.0 + np.exp(-1.0)])


def test_saved_model_keeps_scores_and_convergence_status(tmp_path):
    X = np.array([[-1.0], [1.0]])
    model = KernelPerceptron().fit(X, np.array([0, 1]))
    path = tmp_path / "kernel_perceptron.npz"
    save_npz(path, model.state_dict(), {"params": model.get_params()})
    arrays, metadata = load_npz(path)
    restored = KernelPerceptron().load_state(arrays, metadata)
    assert restored.converged_ == model.converged_
    assert restored.n_epochs_ == model.n_epochs_
    assert np.allclose(restored.decision_function(X), model.decision_function(X))


def test_rejects_invalid_kernel_and_labels():
    with pytest.raises(ValueError, match="gamma"):
        KernelPerceptron(gamma=0)
    with pytest.raises(ValueError, match="binary"):
        KernelPerceptron().fit([[0.0], [1.0]], [0, 2])

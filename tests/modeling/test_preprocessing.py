"""Population scaling and deterministic PCA numerical edge cases."""
import numpy as np
import pytest
from src.modeling.preprocessing import Standardizer, PCA


def test_train_only_population_scaling_constants_and_float64():
    X = np.array([[1, 2, 1], [3, 2, 1 + 1e-7]], dtype=np.float64)
    before = X.copy()
    fitted = Standardizer().fit(X)
    np.testing.assert_array_equal(fitted.mean_, X.mean(axis=0))
    np.testing.assert_array_equal(fitted.variance_, X.var(axis=0, ddof=0))
    np.testing.assert_array_equal(fitted.scale_, [1., 1., 1.])
    assert len(fitted.warnings_) == 2
    transformed = fitted.transform([[5, 99, 80]])
    np.testing.assert_array_equal(transformed, [[3., 0., 0.]])
    assert transformed.dtype == np.float64
    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("X", [[], [[float("nan")]], [[float("inf")]], [1, 2]])
def test_scaler_invalid_input(X):
    with pytest.raises(ValueError): Standardizer().fit(X)


def test_unfitted_and_feature_mismatch():
    with pytest.raises(ValueError): Standardizer().transform([[1, 2]])
    fitted = Standardizer().fit([[1, 2], [2, 3]])
    with pytest.raises(ValueError): fitted.transform([[1]])
    with pytest.raises(ValueError): PCA(1).transform([[1, 2]])


def test_pca_covariance_sign_and_orthogonality():
    X = Standardizer().fit_transform(np.random.default_rng(42).normal(size=(30, 4)))
    before = X.copy()
    fitted = PCA(3).fit(X)
    expected = np.linalg.eigvalsh(X.T @ X / (len(X) - 1))[::-1]
    np.testing.assert_allclose(fitted.eigenvalues_, expected)
    np.testing.assert_allclose(fitted.components_ @ fitted.components_.T, np.eye(3), atol=1e-12)
    for vector in fitted.components_: assert vector[np.argmax(np.abs(vector))] > 0
    np.testing.assert_array_equal(fitted.transform(X), X @ fitted.components_.T)
    np.testing.assert_array_equal(PCA(3).fit(X).components_, fitted.components_)
    np.testing.assert_array_equal(X, before)


def test_degenerate_eigenspace_independent_of_eigh_basis(monkeypatch):
    X = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    expected = PCA(2).fit(X).components_
    original = np.linalg.eigh
    def rotated(cov):
        vals, vectors = original(cov)
        rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
        return vals, vectors @ rotation
    monkeypatch.setattr(np.linalg, "eigh", rotated)
    actual = PCA(2).fit(X).components_
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual, np.eye(2), atol=1e-14)


@pytest.mark.parametrize("components,X", [(0, [[1, 2], [2, 3]]), (True, [[1, 2], [2, 3]]), (3, [[1, 2], [2, 3]]), (2, [[1, 1], [-1, -1]]), (1, [[0, 0], [0, 0]]), (1, [[1, 2]])])
def test_pca_rank_component_and_sample_guards(components, X):
    with pytest.raises(ValueError): PCA(components).fit(X)


def test_small_negative_eigenvalue_clamped_and_large_negative_rejected(monkeypatch):
    monkeypatch.setattr(np.linalg, "eigh", lambda cov: (np.array([-1e-14, 2.]), np.eye(2)))
    assert PCA(1).fit([[1, 0], [-1, 0]]).eigenvalues_[-1] == 0
    monkeypatch.setattr(np.linalg, "eigh", lambda cov: (np.array([-0.1, 2.]), np.eye(2)))
    with pytest.raises(ValueError, match="negative"): PCA(1).fit([[1, 0], [-1, 0]])


@pytest.mark.parametrize("perturbation", [np.spacing(1.0), 5e-11, -5e-11])
def test_pca_near_tie_sign_uses_lowest_index(monkeypatch, perturbation):
    vector = np.array([1., -(1. + perturbation)])
    vector /= np.linalg.norm(vector)
    other = np.array([-vector[1], vector[0]])
    monkeypatch.setattr(np.linalg, "eigh", lambda cov: (np.array([1., 2.]), np.column_stack([other, vector])))
    fitted = PCA(1).fit([[1., -1.], [-1., 1.]])
    assert fitted.components_[0, 0] > 0
    np.testing.assert_allclose(fitted.components_[0], [1 / np.sqrt(2), -1 / np.sqrt(2)], atol=1e-10)


def test_pca_requires_centered_train_input_without_recentering():
    with pytest.raises(ValueError, match="centered"):
        PCA(1).fit([[1., 2.], [2., 4.], [3., 6.]])
    standardized = Standardizer().fit_transform(np.random.default_rng(7).normal(size=(40, 3)))
    noisy = standardized + 1e-12
    before = noisy.copy()
    fitted = PCA(2).fit(noisy)
    np.testing.assert_allclose(fitted.eigenvalues_, np.linalg.eigvalsh(noisy.T @ noisy / 39)[::-1])
    np.testing.assert_array_equal(noisy, before)

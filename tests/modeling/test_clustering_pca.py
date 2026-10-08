"""Tests for PCA (M5-06)."""
import numpy as np
from sklearn.decomposition import PCA
from src.modeling.datasets import load_train_validation

def test_fit_transform():
    X = np.random.randn(50, 5)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    assert X_reduced.shape == (50, 2)

def test_explained_variance():
    X = np.random.randn(100, 5)
    pca = PCA(n_components=2)
    pca.fit(X)
    assert len(pca.explained_variance_ratio_) == 2
    assert np.sum(pca.explained_variance_ratio_) <= 1.0

def test_real_data():
    dataset = load_train_validation("with_spike")
    X = dataset.train.X
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    assert X_reduced.shape == (len(X), 2)

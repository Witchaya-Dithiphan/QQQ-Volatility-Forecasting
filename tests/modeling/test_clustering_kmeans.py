"""Tests for k-Means (M5-05)."""
import numpy as np
from sklearn.cluster import KMeans
from src.modeling.datasets import load_train_validation

def test_fit():
    X = np.random.randn(50, 3)
    m = KMeans(n_clusters=3, random_state=42)
    m.fit(X)
    assert m.cluster_centers_.shape == (3, 3)

def test_predict():
    X = np.random.randn(50, 3)
    m = KMeans(n_clusters=3, random_state=42)
    m.fit(X)
    labels = m.predict(X)
    assert labels.shape == (50,) and np.all(np.isin(labels, [0, 1, 2]))

def test_real_data():
    dataset = load_train_validation("with_spike")
    X = dataset.train.X
    m = KMeans(n_clusters=3, random_state=42)
    m.fit(X)
    labels = m.predict(X)
    assert labels.shape == (len(X),)

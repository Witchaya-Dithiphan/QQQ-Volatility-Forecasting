"""Tests for k-Nearest Neighbors (M5-04)."""
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from src.modeling.datasets import load_train_validation

def test_fit_predict():
    X, y = np.random.randn(50, 3), (np.random.randn(50) > 0).astype(int)
    m = KNeighborsClassifier(n_neighbors=5)
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (50,)

def test_proba():
    X, y = np.random.randn(50, 3), (np.random.randn(50) > 0).astype(int)
    m = KNeighborsClassifier(n_neighbors=5)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_real_data():
    dataset = load_train_validation("with_spike")
    train_X, train_y_reg = dataset.train.X, dataset.train.y_regression
    train_y = (train_y_reg > np.median(train_y_reg)).astype(int)
    m = KNeighborsClassifier(n_neighbors=5)
    m.fit(train_X, train_y)
    acc = np.mean(m.predict(train_X) == train_y)
    assert acc > 0.5

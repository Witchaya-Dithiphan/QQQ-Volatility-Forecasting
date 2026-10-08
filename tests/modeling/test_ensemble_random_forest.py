"""Tests for Random Forest (M5-09)."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.modeling.datasets import load_train_validation

def test_fit():
    X, y = np.random.randn(50, 3), (np.random.randn(50) > 0).astype(int)
    m = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    m.fit(X, y)
    assert m.n_estimators == 10

def test_predict():
    X, y = np.random.randn(50, 3), (np.random.randn(50) > 0).astype(int)
    m = RandomForestClassifier(n_estimators=10, random_state=42)
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (50,) and np.all(np.isin(pred, [0, 1]))

def test_proba():
    X, y = np.random.randn(50, 3), (np.random.randn(50) > 0).astype(int)
    m = RandomForestClassifier(n_estimators=10, random_state=42)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_real_data():
    dataset = load_train_validation("with_spike")
    train_X, train_y_reg = dataset.train.X, dataset.train.y_regression
    train_y = (train_y_reg > np.median(train_y_reg)).astype(int)
    m = RandomForestClassifier(n_estimators=10, random_state=42)
    m.fit(train_X, train_y)
    acc = np.mean(m.predict(train_X) == train_y)
    assert acc > 0.5

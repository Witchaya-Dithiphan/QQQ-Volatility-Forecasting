"""Tests for k-Nearest Neighbors (M5-04, reference/sklearn wrapper)."""
import json
from pathlib import Path

import numpy as np

from src.modeling.classification.knn import KNN
from src.modeling.datasets import load_train_validation


def test_fit_predict():
    """Fit synthetic, check predict shape."""
    X, y = np.random.randn(50, 3), (np.random.randn(50) > 0).astype(int)
    m = KNN(n_neighbors=5).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (50,)
    assert np.all(np.isin(pred, [0, 1]))


def test_proba():
    """Check predict_proba sums to 1."""
    X, y = np.random.randn(50, 3), (np.random.randn(50) > 0).astype(int)
    m = KNN(n_neighbors=5).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_real_data():
    """Load Train/Val, fit, predict, save artifacts."""
    dataset = load_train_validation("with_spike")
    train_X, train_y_reg = dataset.train.X, dataset.train.y_regression
    val_X, val_y_reg = dataset.validation.X, dataset.validation.y_regression
    
    median = np.median(train_y_reg)
    train_y = (train_y_reg > median).astype(int)
    val_y = (val_y_reg > median).astype(int)
    
    m = KNN(n_neighbors=5).fit(train_X, train_y)
    
    train_pred = m.predict(train_X)
    val_pred = m.predict(val_X)
    
    train_acc = np.mean(train_pred == train_y)
    val_acc = np.mean(val_pred == val_y)
    
    print(f"  Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
    assert not np.any(np.isnan(train_pred)) and not np.any(np.isnan(val_pred))
    
    # Save artifacts
    out_dir = Path("outputs/modeling/with_spike/classification/knn/scratch/M5_04")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "train_predictions.npy", train_pred)
    np.save(out_dir / "val_predictions.npy", val_pred)
    
    metadata = {
        "model": "KNN",
        "n_neighbors": 5,
        "metric": "euclidean",
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "train_samples": len(train_X),
        "val_samples": len(val_X),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

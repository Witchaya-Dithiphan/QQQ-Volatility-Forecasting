"""Shared input/state validation and scaler persistence for the M5 scratch classifiers."""
import numpy as np

from ..preprocessing import Standardizer

NOT_FITTED = "Model not fitted"


def check_X(X, n_features=None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if not np.isfinite(X).all():
        raise ValueError("X must be finite")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"Feature mismatch: expected {n_features}, got {X.shape[1]}")
    return X


def check_Xy(X, y):
    X = check_X(X)
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) != len(X):
        raise ValueError("X and y must have same number of samples")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only binary labels {0, 1}")
    if y.min() == y.max():
        raise ValueError("y must contain both classes 0 and 1")
    return X, y.astype(np.int64)


def check_param(name, value, *, integer=False, allow_zero=False):
    ok = isinstance(value, (int, np.integer)) and not isinstance(value, bool) if integer else isinstance(value, (int, float, np.number)) and np.isfinite(value)
    if not ok or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} {'integer' if integer else 'number'}, got {value!r}")


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def apply_scaler(scaler, X):
    """Standardize X; inputs whose standardization leaves float64 range are rejected instead of becoming inf/NaN."""
    with np.errstate(all="ignore"):
        out = scaler.transform(X)
    if not np.isfinite(out).all():
        raise ValueError("Standardization overflow: values exceed the float64-safe range")
    return out


def fit_scaler(X, standardize):
    """Fit on the training matrix only; returns (scaler or None, transformed X)."""
    if not standardize:
        return None, X
    with np.errstate(all="ignore"):
        scaler = Standardizer().fit(X)
    if not (np.isfinite(scaler.mean_).all() and np.isfinite(scaler.variance_).all()):
        raise ValueError("Standardization overflow: Train mean/variance exceed the float64-safe range")
    return scaler, apply_scaler(scaler, X)


def scaler_to_dict(scaler):
    if scaler is None:
        return None
    return {"variance_floor": scaler.variance_floor, "mean": scaler.mean_.tolist(), "variance": scaler.variance_.tolist(),
            "constant": scaler.constant_.tolist(), "scale": scaler.scale_.tolist()}


def scaler_from_dict(d):
    if d is None:
        return None
    scaler = Standardizer(d["variance_floor"])
    scaler.mean_, scaler.variance_ = np.array(d["mean"]), np.array(d["variance"])
    scaler.constant_, scaler.scale_ = np.array(d["constant"], dtype=bool), np.array(d["scale"])
    scaler.warnings_ = []
    return scaler

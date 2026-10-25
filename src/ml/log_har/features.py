"""HAR regressors in log space, built from the eight accepted features (no new data).

The HAR model (Corsi, 2009) explains future volatility with daily, weekly and monthly
realised volatility; here those are |return_1d|, historical_volatility_5d and
historical_volatility_20d. The X variant adds the intraday range and a leverage term,
because negative returns raise later volatility (Corsi & Renò, 2012).
"""
from __future__ import annotations

import numpy as np

from ..core.contracts import FEATURE_COLUMNS

ANNUALIZATION = 252
# One Train day has a return of exactly zero; 1% a year keeps its log finite and is
# far below every realised value the HAR terms otherwise take.
DAILY_FLOOR = 0.01
NAMES = {"3": ("d", "w", "m"), "X": ("d", "w", "m", "r", "lev")}
_COL = {name: i for i, name in enumerate(FEATURE_COLUMNS)}


def har_features(X, kind: str = "X") -> np.ndarray:
    if kind not in NAMES:
        raise ValueError(f"Unknown HAR feature set: {kind!r}")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_COLUMNS) or not np.isfinite(X).all():
        raise ValueError("Expected a finite matrix of the eight accepted features")
    logged = ["historical_volatility_5d", "historical_volatility_20d"] + (["intraday_range"] if kind == "X" else [])
    if (X[:, [_COL[name] for name in logged]] <= 0).any():
        # A zero range or volatility (a halted or flat day) has no log; fail instead of fitting on -inf.
        raise ValueError("HAR features need positive volatility and range inputs")
    daily = np.log(np.maximum(np.abs(X[:, _COL["return_1d"]]) * np.sqrt(ANNUALIZATION), DAILY_FLOOR))
    weekly = np.log(X[:, _COL["historical_volatility_5d"]])
    monthly = np.log(X[:, _COL["historical_volatility_20d"]])
    if kind == "3":
        return np.column_stack([daily, weekly, monthly])
    rng = np.log(X[:, _COL["intraday_range"]])
    leverage = np.minimum(X[:, _COL["return_5d"]], 0.0)
    return np.column_stack([daily, weekly, monthly, rng, leverage])


def log_target(y) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Volatility targets must be positive and finite")
    return np.log(y)


def back_transform(log_prediction, residual_variance: float) -> np.ndarray:
    """Mean of a log-normal: exp(mu + s^2 / 2), so predictions are not biased low."""
    return np.exp(np.asarray(log_prediction, dtype=np.float64) + residual_variance / 2)

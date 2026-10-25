"""Log-HAR fitted on QQQ with ridge shrinkage towards coefficients pooled from other ETFs.

    min_{a, b} ||z - a - F b||^2 + lam ||b - prior||^2,    lam = kappa * n

On centred data the solution is b = (Fc'Fc + lam I)^{-1} (Fc'zc + lam prior), with the
intercept recovered from the means and never penalised. kappa = 0 is plain OLS on QQQ;
kappa = inf keeps the prior and estimates only QQQ's own level.
"""
from __future__ import annotations

import numpy as np

from ..core.base import BaseModel
from .features import NAMES, back_transform, har_features, log_target


def _residual_variance(residual: np.ndarray) -> float:
    # ddof=1 for every arm, so the back-transform is computed the same way whatever kappa is.
    return float(np.sum(residual ** 2) / (len(residual) - 1))


class LogHAR(BaseModel):
    name = "log_har"
    task = "regression"

    def __init__(self, features: str = "X", kappa: float = 0.0, prior=None):
        if features not in NAMES:
            raise ValueError(f"Unknown HAR feature set: {features!r}")
        if not (kappa == np.inf or (np.isfinite(kappa) and kappa >= 0)):
            raise ValueError("kappa must be >= 0 or inf")
        self.features = features
        self.kappa = float(kappa)
        self.prior = None if prior is None else np.asarray(prior, dtype=np.float64)

    def get_params(self) -> dict:
        return {"features": self.features, "kappa": None if self.kappa == np.inf else self.kappa,
                "kappa_is_inf": self.kappa == np.inf,
                "prior": None if self.prior is None else self.prior.tolist()}

    def fit(self, X, y=None) -> "LogHAR":
        F, z = har_features(X, self.features), log_target(y)
        if self.kappa > 0 and (self.prior is None or self.prior.shape != (F.shape[1],)):
            raise ValueError("kappa > 0 needs a prior with one coefficient per HAR feature")
        f_mean, z_mean = F.mean(axis=0), z.mean()
        Fc, zc = F - f_mean, z - z_mean
        if self.kappa == 0:
            self.coef_, *_ = np.linalg.lstsq(Fc, zc, rcond=None)
        elif self.kappa == np.inf:
            self.coef_ = self.prior.copy()
        else:
            lam = self.kappa * len(F)
            self.coef_ = np.linalg.solve(Fc.T @ Fc + lam * np.eye(F.shape[1]), Fc.T @ zc + lam * self.prior)
        self.intercept_ = float(z_mean - f_mean @ self.coef_)
        residual = z - (F @ self.coef_ + self.intercept_)
        self.residual_variance_ = _residual_variance(residual)
        self.history_ = {"iteration": [0], "loss": [float(np.mean(residual ** 2))]}
        return self

    def predict_log(self, X) -> np.ndarray:
        return har_features(X, self.features) @ self.coef_ + self.intercept_

    def predict(self, X) -> np.ndarray:
        return back_transform(self.predict_log(X), self.residual_variance_)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"coef": np.asarray(self.coef_), "intercept": np.array([self.intercept_]),
                "residual_variance": np.array([self.residual_variance_])}

    def load_state(self, arrays: dict, metadata: dict) -> "LogHAR":
        params = metadata["params"]
        self.features = params["features"]
        self.kappa = np.inf if params["kappa_is_inf"] else float(params["kappa"])
        self.prior = None if params["prior"] is None else np.asarray(params["prior"], dtype=np.float64)
        self.coef_ = np.asarray(arrays["coef"], dtype=np.float64)
        self.intercept_ = float(arrays["intercept"][0])
        self.residual_variance_ = float(arrays["residual_variance"][0])
        return self

    def reference(self):
        return _SklearnLogHAR(self.features, self.kappa, self.prior)


class _SklearnLogHAR:
    """The same estimator assembled from scikit-learn, for parity checks only.

    kappa = 0 is LinearRegression on the HAR features. kappa > 0 is Ridge(alpha=lam)
    on the shifted target z - F prior, whose coefficients plus the prior equal the
    transfer-ridge solution (substitute d = b - prior in the objective).
    """

    def __init__(self, features, kappa, prior):
        self.features, self.kappa, self.prior = features, kappa, prior

    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression, Ridge

        F, z = har_features(X, self.features), log_target(y)
        if self.kappa == 0:
            estimator = LinearRegression().fit(F, z)
            self.coef_, self.intercept_ = estimator.coef_, float(estimator.intercept_)
        elif self.kappa == np.inf:
            # Infinite shrinkage keeps the prior; only the level is fitted, which is an
            # intercept-only regression on the shifted target.
            estimator = LinearRegression().fit(np.zeros((len(F), 1)), z - F @ self.prior)
            self.coef_, self.intercept_ = self.prior.copy(), float(estimator.intercept_)
        else:
            estimator = Ridge(alpha=self.kappa * len(F)).fit(F, z - F @ self.prior)
            self.coef_, self.intercept_ = estimator.coef_ + self.prior, float(estimator.intercept_)
        self.residual_variance_ = _residual_variance(z - (F @ self.coef_ + self.intercept_))
        return self

    def predict(self, X):
        return back_transform(har_features(X, self.features) @ self.coef_ + self.intercept_, self.residual_variance_)


def pooled_prior(frames, features: str = "X") -> tuple[np.ndarray, list[np.ndarray]]:
    """Within estimator over several assets: demean each asset, stack, OLS without intercept.

    Removing each asset's own means drops its volatility level, which differs across
    markets, and keeps the shared dynamics (how strongly today's volatility persists).
    Also returns each asset's own slope for checking that those dynamics are similar.
    """
    stacked_F, stacked_z, per_asset = [], [], []
    for X, y in frames:
        F, z = har_features(X, features), log_target(y)
        Fc, zc = F - F.mean(axis=0), z - z.mean()
        stacked_F.append(Fc)
        stacked_z.append(zc)
        own, *_ = np.linalg.lstsq(Fc, zc, rcond=None)
        per_asset.append(own)
    beta, *_ = np.linalg.lstsq(np.vstack(stacked_F), np.concatenate(stacked_z), rcond=None)
    return beta, per_asset

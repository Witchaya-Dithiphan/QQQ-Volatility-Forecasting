"""Strict-JSON helpers and M2 preprocessing (de)serialization shared by scratch classifiers."""
import json

import numpy as np

from ..preprocessing import PCA, Standardizer


def dumps(state: dict) -> str:
    return json.dumps(state, sort_keys=True, allow_nan=False, separators=(",", ": "))


def loads(text: str) -> dict:
    def reject(token):
        raise ValueError(f"Non-finite JSON constant: {token}")
    return json.loads(text, parse_constant=reject)


def scaler_state(scaler: Standardizer) -> dict:
    return {"variance_floor": scaler.variance_floor, "mean": scaler.mean_.tolist(), "variance": scaler.variance_.tolist(),
            "scale": scaler.scale_.tolist(), "constant": scaler.constant_.tolist(), "warnings": list(scaler.warnings_)}


def scaler_from_state(state: dict) -> Standardizer:
    scaler = Standardizer(state["variance_floor"])
    scaler.mean_, scaler.variance_ = np.array(state["mean"]), np.array(state["variance"])
    scaler.scale_, scaler.constant_ = np.array(state["scale"]), np.array(state["constant"], dtype=bool)
    scaler.warnings_ = list(state["warnings"])
    return scaler


def pca_state(pca: PCA | None) -> dict | None:
    if pca is None:
        return None
    return {"n_components": pca.n_components, "rank": pca.rank_, "n_features": pca.n_features_, "components": pca.components_.tolist(),
            "eigenvalues": pca.eigenvalues_.tolist(), "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}


def pca_from_state(state: dict | None) -> PCA | None:
    if state is None:
        return None
    pca = PCA(state["n_components"])
    pca.rank_, pca.n_features_ = state["rank"], state["n_features"]
    pca.components_, pca.eigenvalues_ = np.array(state["components"]), np.array(state["eigenvalues"])
    pca.explained_variance_ratio_ = np.array(state["explained_variance_ratio"])
    return pca

# src/ml/core/artifacts.py
"""Read a finished training run back from disk: its numbers, and the model itself."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..registry import MODELS, model_names
from .persist import load_npz

# The files that define what a run would predict; hashing them freezes the selection.
HASHED_FILES = ("config.json", "model.npz", "model.json", "preprocessor.npz", "preprocessor.json")


@dataclass(frozen=True)
class Run:
    name: str
    task: str
    variant: str
    directory: Path
    config: dict
    metrics: dict
    dates: np.ndarray
    target: np.ndarray
    score: np.ndarray
    prediction: np.ndarray
    history: dict | None


def run_directory(root: Path, variant: str, name: str) -> Path:
    return Path(root) / variant / MODELS[name].task / name


def read_run(root: Path, variant: str, name: str) -> Run:
    directory = run_directory(root, variant, name)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "validation_metrics.json").read_text(encoding="utf-8"))
    with (directory / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    history_path = directory / "history.json"
    return Run(
        name=name,
        task=config["task"],
        variant=variant,
        directory=directory,
        config=config,
        metrics=metrics,
        dates=np.array([row["Date"] for row in rows], dtype="datetime64[D]"),
        target=np.array([float(row["target"]) for row in rows]),
        score=np.array([float(row["raw_score"]) for row in rows]),
        prediction=np.array([float(row["final_prediction"]) for row in rows]),
        history=json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else None,
    )


def read_runs(root: Path, variant: str, *, task: str | None = None, require_all: bool = True) -> list[Run]:
    names = [name for name in model_names() if task is None or MODELS[name].task == task]
    missing = [
        name for name in names
        if not (run_directory(root, variant, name) / "validation_metrics.json").is_file()
    ]
    if missing and require_all:
        raise FileNotFoundError(f"{variant}: no finished run for {missing}")
    return [read_run(root, variant, name) for name in names if name not in missing]


def preprocess(X, arrays: dict, metadata: dict) -> np.ndarray:
    """The transform the trainer applied before this model saw the data."""
    X = np.array(X, dtype=np.float64, copy=True)
    if metadata["standardized"]:
        X = (X - arrays["mean"]) / arrays["scale"]
        X[:, arrays["constant"].astype(bool)] = 0.0
    if metadata["pca_components"]:
        X = X @ arrays["components"].T
    return X


def load_model(directory: Path):
    """Return (model rebuilt from model.npz, function mapping raw features to its inputs)."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    arrays, metadata = load_npz(directory / "model.npz")
    model = MODELS[config["model"]]().load_state(arrays, metadata)
    scaler_arrays, scaler_metadata = load_npz(directory / "preprocessor.npz")
    return model, lambda X: preprocess(X, scaler_arrays, scaler_metadata)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_digest(path: Path) -> str:
    """SHA-256 of an artifact that is the same on every checkout.

    Git stores the JSON artifacts with LF, while the Windows machine that wrote them
    used CRLF. Hashing raw bytes would make a lock built here fail to verify on a fresh
    clone, so line endings of JSON are normalised first. Binary files are hashed as is.
    """
    data = Path(path).read_bytes()
    if Path(path).suffix == ".json":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()

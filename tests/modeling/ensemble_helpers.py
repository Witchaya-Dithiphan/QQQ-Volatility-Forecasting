"""Shared helpers for M6 scratch ensemble tests (sklearn is a test-only reference)."""
import ast
import json
from pathlib import Path

import numpy as np

ENSEMBLE_DIR = Path("src/modeling/ensemble")


def synthetic(n=200, d=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    signal = X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n)
    return X, (signal > 0).astype(int)


def imported_modules(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def strict_roundtrip(state):
    """Strict JSON: NaN/Infinity rejected on dump, parsed back on load."""
    return json.loads(json.dumps(state, allow_nan=False))


def real_q75():
    from src.modeling.datasets import load_train_validation
    data = load_train_validation("with_spike")
    return data.train.X, data.train.y_classification, data.validation.X, data.validation.y_classification

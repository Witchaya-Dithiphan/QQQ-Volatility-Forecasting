"""Strict scratch-vs-reference comparison. Policy lives in AGENTS.md section 6.

The failure message names the worst index and both values because the point of this
gate is diagnosing the mismatch; "arrays are not almost equal" sends you back to the
debugger with nothing to go on.
"""
from __future__ import annotations

import numpy as np

RTOL = 1e-5
ATOL = 1e-8


class ParityError(AssertionError):
    """Raised when a scratch model disagrees with its reference implementation."""


def _report(label: str, scratch: np.ndarray, reference: np.ndarray) -> str:
    difference = np.abs(scratch - reference)
    worst = int(np.argmax(difference))
    mismatching = int((difference > ATOL).sum())
    return (
        f"{label}: max abs diff {difference.max():.3e} at index {worst} "
        f"(scratch={scratch[worst]!r}, reference={reference[worst]!r}; "
        f"{mismatching}/{len(difference)} elements differ by more than {ATOL:g})"
    )


def align_clusters(scratch_labels, reference_labels) -> np.ndarray:
    """Rename reference clusters to the scratch naming by majority overlap.

    Cluster ids carry no meaning, so two implementations can produce the same
    partition under different names.
    """
    scratch_labels = np.asarray(scratch_labels)
    reference_labels = np.asarray(reference_labels)
    mapping = {}
    for cluster in np.unique(reference_labels):
        members = reference_labels == cluster
        values, counts = np.unique(scratch_labels[members], return_counts=True)
        mapping[cluster] = values[int(np.argmax(counts))]
    return np.array([mapping[c] for c in reference_labels])


def _compare_scores(scratch, reference, X, label: str) -> None:
    for method in ("decision_function", "predict_proba"):
        try:
            mine = np.asarray(getattr(scratch, method)(X), dtype=np.float64)
        except (NotImplementedError, AttributeError):
            continue
        try:
            theirs = np.asarray(getattr(reference, method)(X), dtype=np.float64)
        except (NotImplementedError, AttributeError):
            continue
        # sklearn's predict_proba is (n, 2); scratch models return (n,).
        if theirs.ndim == 2 and theirs.shape[1] == 2:
            theirs = theirs[:, 1]
        if mine.ndim == 2 and mine.shape[1] == 2:
            mine = mine[:, 1]
        if mine.shape != theirs.shape:
            raise ParityError(f"{label}.{method}: shape {mine.shape} != {theirs.shape}")
        if not np.allclose(mine, theirs, rtol=RTOL, atol=ATOL):
            raise ParityError(_report(f"{label}.{method}", mine, theirs))


def assert_parity(scratch, reference, X, y=None, *, task: str, label: str = "predict") -> dict:
    """Raise ParityError unless the scratch model matches its reference.

    `svm` is the one model allowed a different criterion (objective-level rather than
    weight-level) because hinge loss is non-smooth; see AGENTS.md section 6.
    """
    scratch_prediction = np.asarray(scratch.predict(X), dtype=np.float64)
    reference_prediction = np.asarray(reference.predict(X), dtype=np.float64)

    if scratch_prediction.shape != reference_prediction.shape:
        raise ParityError(
            f"{label}: shape {scratch_prediction.shape} != {reference_prediction.shape}"
        )

    if task == "regression":
        if not np.allclose(scratch_prediction, reference_prediction, rtol=RTOL, atol=ATOL):
            raise ParityError(_report(label, scratch_prediction, reference_prediction))

    elif task == "classification":
        if not np.array_equal(scratch_prediction, reference_prediction):
            disagreements = int((scratch_prediction != reference_prediction).sum())
            raise ParityError(
                f"{label}: {disagreements}/{len(scratch_prediction)} labels differ"
            )
        _compare_scores(scratch, reference, X, label)

    elif task == "clustering":
        aligned = align_clusters(scratch_prediction, reference_prediction)
        if not np.array_equal(scratch_prediction, aligned):
            disagreements = int((scratch_prediction != aligned).sum())
            raise ParityError(
                f"{label}: cluster partition differs after aligning ids "
                f"({disagreements}/{len(aligned)} points in a different group)"
            )

    else:
        raise ValueError(f"Unknown task: {task!r}; expected one of regression/classification/clustering")

    return {
        "task": task,
        "n_samples": int(len(scratch_prediction)),
        "passed": True,
        "rtol": RTOL,
        "atol": ATOL,
    }

# tests/ml/test_finalize.py
import csv
import json
import shutil

import numpy as np
import pytest

from src.ml.core.artifacts import read_run, run_directory
from src.ml.core.contracts import VARIANTS
from src.ml.core.data import load
from src.ml.core.trainer import OUTPUT_ROOT
from src.ml.finalize import (
    LOCK_NAME,
    RECEIPT_NAME,
    build_lock,
    check_receipt,
    evaluate_holdout,
    segment_metrics,
    verify_lock,
    write_lock,
)


@pytest.fixture()
def root(tmp_path):
    """The trained runs as they were before finalize: the real one-shot outputs are left out."""
    for variant in VARIANTS:
        shutil.copytree(OUTPUT_ROOT / variant, tmp_path / variant,
                        ignore=shutil.ignore_patterns("test_metrics.json", "test_predictions.csv"))
    return tmp_path


@pytest.fixture(scope="module")
def stand_in():
    """Validation dressed up as the holdout, with the first 54 rows flagged as spike-affected."""
    validation = load("with_spike").validation
    segments = np.array(["spike_affected"] * 54 + ["non_spike"] * 317)
    return validation, segments


def test_lock_covers_all_38_runs_and_names_a_headline_per_task(root):
    lock = build_lock(root)
    assert lock["run_count"] == 38 and len(lock["runs"]) == 38
    assert set(lock["headline"]) == {f"{v}/{t}" for v in VARIANTS for t in ("regression", "classification")}
    assert lock["headline"]["with_spike/regression"] == "elastic_net"


def test_lock_refuses_a_run_whose_reload_check_failed(root):
    path = run_directory(root, "with_spike", "knn") / "load_verification.json"
    path.write_text(json.dumps({"passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="reload verification"):
        build_lock(root)


def test_verify_lock_needs_a_lock_file(root):
    with pytest.raises(FileNotFoundError, match="dry-run"):
        verify_lock(root)


def test_changed_artifact_blocks_finalize_before_scoring(root, stand_in):
    holdout, segments = stand_in
    write_lock(root)
    path = run_directory(root, "non_spike", "logistic") / "model.npz"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="non_spike/logistic"):
        evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=10)
    assert not list(root.rglob("test_metrics.json"))


def test_holdout_evaluation_writes_every_output_once(root, stand_in):
    holdout, segments = stand_in
    write_lock(root)
    result = evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=20,
                              command="pytest")
    assert len(list(root.rglob("test_metrics.json"))) == 38
    assert len(list(root.rglob("test_predictions.csv"))) == 38
    for variant in VARIANTS:
        assert (root / "final" / variant / "leaderboard_classification.csv").is_file()
        assert (root / "final" / variant / "leaderboard_regression.md").is_file()
    assert (root / "final" / "segmented_metrics.csv").is_file()
    receipt = json.loads((root / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["holdout_sha256"] == holdout.sha256 and receipt["run_count"] == 38
    assert receipt["segment_rows"] == {"spike_affected": 54, "non_spike": 317}
    assert receipt["previous_receipts"] == [] and receipt == result["receipt"]

    with pytest.raises(FileExistsError, match="--rerun-reason"):
        evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=20)

    evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=20,
                     rerun_reason="fixture rerun")
    again = json.loads((root / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert again["rerun_reason"] == "fixture rerun" and len(again["previous_receipts"]) == 1


def test_scoring_validation_through_the_finalize_path_matches_the_stored_metrics(root, stand_in):
    holdout, segments = stand_in
    write_lock(root)
    evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=10)
    for name, key in (("knn", "average_precision"), ("xgboost", "roc_auc")):
        directory = run_directory(root, "with_spike", name)
        scored = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8"))
        stored = read_run(root, "with_spike", name).metrics
        assert scored[key]["value"] == pytest.approx(stored[key]["value"], rel=1e-12)
        assert scored["segments"]["spike_affected"]["rows"] == 54
    with (run_directory(root, "with_spike", "knn") / "test_predictions.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 371 and rows[0]["diagnostic_segment"] == "spike_affected"


def test_misaligned_segments_are_rejected(root, stand_in):
    holdout, segments = stand_in
    write_lock(root)
    with pytest.raises(ValueError, match="do not line up"):
        evaluate_holdout(holdout, holdout.dates[::-1], segments, artifact_root=root, replicates=10)
    with pytest.raises(ValueError, match="do not line up"):
        evaluate_holdout(holdout, holdout.dates, np.array(["calm"] * 371), artifact_root=root, replicates=10)


def test_one_class_segment_is_reported_with_a_reason():
    target = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.2, 0.7])
    segments = np.array(["spike_affected"] * 3 + ["non_spike"] * 3)
    result = segment_metrics("classification", target, score, (score > 0.5).astype(float), segments)
    assert result["spike_affected"]["rows"] == 3 and result["spike_affected"]["positives"] == 0
    assert result["spike_affected"]["average_precision"] == {"value": None, "reason": "one-class subset"}
    assert result["non_spike"]["average_precision"]["value"] == 1.0
    assert "roc_curve" not in result["non_spike"]


def test_receipt_check_passes_when_nothing_was_finalized(root):
    assert check_receipt(root, None) == []


def test_holdout_leaderboard_says_whether_each_model_beats_the_best_baseline(root, stand_in):
    holdout, segments = stand_in
    write_lock(root)
    evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=20, command="pytest")
    header = (root / "final" / "with_spike" / "leaderboard_classification.md").read_text(
        encoding="utf-8").splitlines()[2]
    assert "beats_best_baseline" in header and "tied_with_leader" in header


def test_a_crash_while_scoring_leaves_no_holdout_output_behind(root, stand_in, monkeypatch):
    # The held-out read cannot be repeated, so a failure part-way must not leave some
    # models scored on disk with no receipt to say the split was opened.
    holdout, segments = stand_in
    write_lock(root)

    def explode(*args, **kwargs):
        raise RuntimeError("late failure")

    monkeypatch.setattr("src.ml.finalize.leaderboard", explode)
    with pytest.raises(RuntimeError, match="late failure"):
        evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=20, command="pytest")
    assert list(root.rglob("test_metrics.json")) == []
    assert list(root.rglob("test_predictions.csv")) == []
    assert not (root / RECEIPT_NAME).exists() and not (root / "final").exists()


def test_the_lock_cannot_be_rewritten_after_the_holdout_was_evaluated(root):
    write_lock(root)
    (root / RECEIPT_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already evaluated"):
        write_lock(root)


def test_an_unserialisable_metric_is_caught_before_anything_is_written(root, stand_in, monkeypatch):
    # write_json refuses NaN. That refusal must happen while scoring, not half-way
    # through writing 38 folders.
    holdout, segments = stand_in
    write_lock(root)
    calls = []

    def late_nan(*args, **kwargs):
        calls.append(1)
        # Only a late model misbehaves, so earlier folders would already be on disk
        # if writing were interleaved with scoring.
        rows = float("nan") if len(calls) == 20 else 1
        return {"spike_affected": {"rows": rows}, "non_spike": {"rows": 1}}

    monkeypatch.setattr("src.ml.finalize.segment_metrics", late_nan)
    with pytest.raises(ValueError):
        evaluate_holdout(holdout, holdout.dates, segments, artifact_root=root, replicates=20, command="pytest")
    assert list(root.rglob("test_metrics.json")) == []
    assert list(root.rglob("test_predictions.csv")) == []
    assert not (root / RECEIPT_NAME).exists() and not (root / "final").exists()


def test_the_lock_does_not_depend_on_the_line_endings_of_a_checkout(root):
    # Git stores the JSON artifacts with LF, but the machine that trained them wrote
    # CRLF. A lock built from raw bytes would not verify on a fresh clone.
    before = build_lock(root)["runs"]
    for path in root.rglob("*.json"):
        data = path.read_bytes()
        flipped = data.replace(b"\r\n", b"\n") if b"\r\n" in data else data.replace(b"\n", b"\r\n")
        path.write_bytes(flipped)
    assert build_lock(root)["runs"] == before


def test_a_changed_binary_artifact_still_changes_the_lock(root):
    before = build_lock(root)["runs"]
    target = root / "with_spike" / "classification" / "knn" / "model.npz"
    target.write_bytes(target.read_bytes() + b"\r\n")
    assert build_lock(root)["runs"]["with_spike/knn"]["model.npz"] != before["with_spike/knn"]["model.npz"]

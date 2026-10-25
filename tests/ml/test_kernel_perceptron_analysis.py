import json

import numpy as np
import pandas as pd
import pytest

from config import PROJECT_ROOT
from src.ml.kernel_perceptron_analysis import ROOT, aligned


def test_prediction_alignment_checks_dates_and_targets():
    reference = pd.DataFrame({"Date": ["2025-01-01", "2025-01-02"], "target": [0, 1]})
    aligned(reference, reference.copy())
    changed = reference.copy()
    changed.loc[1, "target"] = 0
    with pytest.raises(ValueError, match="dates or targets"):
        aligned(reference, changed)


def test_supplemental_analysis_preserves_locked_outputs_and_matches_saved_scores():
    report = ROOT / "report"
    receipt = json.loads((ROOT / "test_receipt.json").read_text(encoding="utf-8"))
    parity = json.loads((report / "reference_parity.json").read_text(encoding="utf-8"))
    assert all(row["passed"] and row["prediction_agreement"] == 1 for row in parity)
    training = json.loads((report / "training_reference.json").read_text(encoding="utf-8"))
    assert all(row["converged"] and row["train_accuracy"] == 1 for row in training)
    assert {row["variant"]: row["same_update_history"] for row in training} == {
        "with_spike": False, "non_spike": True,
    }
    for variant in ("with_spike", "non_spike"):
        folds = pd.read_csv(report / "walk_forward.csv").query("variant == @variant")
        assert len(folds) == 5 and folds.iloc[-1]["fold"] == "pooled"
        assert folds.iloc[:4]["converged"].all()
        segments = pd.read_csv(report / "segmented_metrics.csv").query("variant == @variant")
        assert int(segments["rows"].sum()) == receipt["test_rows"]
        assert int(segments["positives"].sum()) == 73
        for split in ("validation", "test"):
            directory = ROOT / variant / "classification" / "kernel_perceptron"
            metrics = json.loads((directory / f"{split}_metrics.json").read_text(encoding="utf-8"))
            board = pd.read_csv(report / f"leaderboard_{split}_{variant}.csv")
            kernel = board.set_index("model").loc["kernel_perceptron"]
            assert kernel["study"] == "post-submission"
            assert np.isclose(kernel["average_precision"], metrics["average_precision"]["value"])
            source = (PROJECT_ROOT / "outputs" / "modeling" /
                      ("report" if split == "validation" else "final") / variant /
                      "leaderboard_classification.csv")
            locked = pd.read_csv(source).set_index("model")
            for model in locked.index:
                assert np.isclose(board.set_index("model").loc[model, "average_precision"],
                                  locked.loc[model, "average_precision"])
            paired = json.loads((report / f"bootstrap_{split}_{variant}.json").read_text(encoding="utf-8"))
            score = paired["intervals"]["kernel_perceptron"]["average_precision"]
            assert np.isclose(score["point"], kernel["average_precision"])
            assert score["low"] <= score["point"] <= score["high"]

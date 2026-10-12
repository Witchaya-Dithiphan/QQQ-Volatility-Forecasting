import numpy as np
import pytest

from src.ml.core.calibration import PlattCalibrator, precision_threshold, probability_metrics


def test_platt_calibration_improves_miscalibrated_log_loss_without_changing_order():
    probability = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 0.90, 0.99])
    target = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    before = probability_metrics(target, probability, threshold=0.5)["log_loss"]["value"]
    calibrator = PlattCalibrator().fit(probability, target)
    calibrated = calibrator.predict_proba(probability)
    after = probability_metrics(target, calibrated, threshold=0.5)["log_loss"]["value"]

    assert after < before
    assert calibrator.slope_ > 0
    np.testing.assert_array_equal(np.argsort(calibrated), np.argsort(probability))
    assert np.all((calibrated > 0) & (calibrated < 1))


def test_platt_state_is_json_serializable():
    import json

    calibrator = PlattCalibrator().fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    json.dumps(calibrator.state_dict(), allow_nan=False)


def test_precision_threshold_respects_recall_constraint_and_tie_policy():
    target = np.array([1, 0, 1, 0, 1, 0])
    score = np.array([0.95, 0.90, 0.80, 0.70, 0.60, 0.10])
    selected = precision_threshold(target, score, min_recall=2 / 3)

    assert selected["threshold"] == pytest.approx(0.8)
    assert selected["metrics"]["precision"]["value"] == pytest.approx(2 / 3)
    assert selected["metrics"]["recall"]["value"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("bad", [-0.1, 0.0, 1.1, np.nan])
def test_precision_threshold_rejects_invalid_minimum_recall(bad):
    with pytest.raises(ValueError):
        precision_threshold([0, 1], [0.1, 0.9], min_recall=bad)

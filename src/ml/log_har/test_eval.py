"""Score the Validation-locked log-HAR arms once on Test and on the post-snapshot holdout.

    python -m src.ml.log_har.test_eval --allow-test

Test was opened for the 19-model study on 2026-10-11, so these Test numbers are
post-hoc. The holdout (QQQ days after the accepted snapshot) is continuous every day,
which makes it more useful here than for the classifiers. Nothing is refit, and
everything that does not need Test is loaded and checked before the receipt is
written, so only a failure while reading Test itself can use up the one read.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

from ..core.artifacts import load_model
from ..core.contracts import FEATURE_COLUMNS, REGRESSION_TARGET, VARIANTS
from ..core.data import load, load_segments, load_test
from ..core.report import baselines
from ..transfer_learning.sources import MANIFEST_PATH, qqq_extended
from .experiment import ARMS, CLASSROOM, ROOT, digest, evaluate, scores
from .model import LogHAR


def verify_decision() -> dict:
    decision = json.loads((ROOT / "decision.json").read_text(encoding="utf-8"))
    hashes = decision["hashes"]
    current = {p.name: digest(p) for p in sorted((PROJECT_ROOT / "src" / "ml" / "log_har").glob("*.py"))}
    if current != hashes["code"]:
        raise ValueError("log-HAR code changed after the Validation decision was locked")
    if digest(MANIFEST_PATH) != hashes["source_manifest"] or digest(ROOT / "report" / "prior.json") != hashes["prior"]:
        raise ValueError("Source manifest or ETF prior changed after the decision")
    for kind, file_name in (("models", "model.npz"), ("configs", "config.json")):
        for key, value in hashes[kind].items():
            if digest(ROOT / key / file_name) != value:
                raise ValueError(f"{file_name} changed after the decision: {key}")
    if load("with_spike").validation.sha256 != hashes["validation_split"]:
        raise ValueError("The Validation split differs from the one the decision was locked on")
    return decision


def load_arm(variant: str, arm: str) -> LogHAR:
    directory = ROOT / variant / arm
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    return LogHAR().load_state(dict(np.load(directory / "model.npz")), {"params": config["params"]})


def load_classroom(variant: str, name: str):
    directory = PROJECT_ROOT / "outputs" / "modeling" / variant / "regression" / name
    model, transform = load_model(directory)
    saved = pd.read_csv(directory / "test_predictions.csv", float_precision="round_trip")["final_prediction"]
    return model, transform, saved.to_numpy(dtype=np.float64)


def classroom_predictions(loaded, held, hold_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The locked classroom model's saved Test predictions (checked by reloading it) and its holdout forecast."""
    model, transform, saved = loaded
    again = np.maximum(model.predict(transform(held.X)), 0.0)  # the trainer clips forecasts at zero
    if not np.allclose(saved, again, rtol=1e-10, atol=1e-12):
        raise ValueError("A reloaded classroom model does not reproduce its finalized Test predictions")
    return saved, np.maximum(model.predict(transform(hold_X)), 0.0)


def segment_rows(variant: str, y: np.ndarray, segments: np.ndarray, compared: dict) -> list[dict]:
    rows = []
    for segment in np.unique(segments):
        mask = segments == segment
        for name, prediction in compared.items():
            rows.append({"variant": variant, "segment": segment, "rows": int(mask.sum()), "model": name,
                         **scores(y[mask], prediction[mask])})
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_test:
        raise PermissionError("Test requires --allow-test")
    out_root = ROOT / "test"
    receipt = ROOT / "test_receipt.json"
    if receipt.exists():
        raise FileExistsError(f"log-HAR Test was already evaluated: {receipt}")
    decision = verify_decision()

    # Everything below until the receipt needs no Test data.
    extended = qqq_extended()
    accepted_end = pd.Timestamp("2026-08-21")  # last accepted Test date; checked against Test below
    after = extended.loc[(pd.to_datetime(extended["Date"]) > accepted_end) & extended[REGRESSION_TARGET].notna()]
    if after.empty:
        raise ValueError("No holdout rows after the accepted snapshot")
    hold_dates = pd.to_datetime(after["Date"]).to_numpy()
    hold_X = after[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    hold_y = after[REGRESSION_TARGET].to_numpy(dtype=np.float64)
    hold_split = SimpleNamespace(X=hold_X)
    arms = {variant: {arm: load_arm(variant, arm) for arm in ARMS} for variant in VARIANTS}
    classroom = {variant: {name: load_classroom(variant, name) for name in CLASSROOM} for variant in VARIANTS}
    datasets = {variant: load(variant) for variant in VARIANTS}

    # Lock before the read: a crash after this point must not leave Test re-readable.
    receipt.write_text(json.dumps({"status": "opened", "post_hoc": True,
                                   "opened_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}),
                       encoding="utf-8", newline="\n")
    held = load_test(allow_test=True)
    if pd.Timestamp(held.dates[-1]) != accepted_end:
        raise ValueError("The holdout boundary does not match the last Test date")
    segment_dates, segments = load_segments("test", allow_test=True)
    if not np.array_equal(segment_dates, held.dates):
        raise ValueError("Test segments are not aligned with the Test split")

    pending = {}
    for variant in VARIANTS:
        ds = datasets[variant]
        test_compared, hold_compared = {}, {}
        for arm, model in arms[variant].items():
            test_compared[arm], hold_compared[arm] = model.predict(held.X), model.predict(hold_X)
        for name, loaded in classroom[variant].items():
            test_compared[name], hold_compared[name] = classroom_predictions(loaded, held, hold_X)
        for name, (_, prediction) in baselines("regression", ds.train, held, ds.threshold).items():
            test_compared[name] = prediction
        for name, (_, prediction) in baselines("regression", ds.train, hold_split, ds.threshold).items():
            hold_compared[name] = prediction
        test_result = evaluate(held.y_regression, test_compared)
        test_result.update(variant=variant, split="test", post_hoc=True)
        hold_result = {"variant": variant, "split": "holdout", "rows": int(len(hold_y)),
                       "dates": [str(hold_dates[0])[:10], str(hold_dates[-1])[:10]],
                       "note": f"Blind but short ({len(hold_y)} days); no interval is reported.",
                       "models": {name: scores(hold_y, prediction) for name, prediction in hold_compared.items()}}
        pending[variant] = (test_result, hold_result, test_compared, hold_compared,
                            segment_rows(variant, held.y_regression, segments, test_compared))

    for variant, (test_result, hold_result, test_compared, hold_compared, rows) in pending.items():
        out = out_root / variant
        out.mkdir(parents=True, exist_ok=True)
        (out / "test_metrics.json").write_text(json.dumps(test_result, indent=1), encoding="utf-8", newline="\n")
        (out / "holdout_metrics.json").write_text(json.dumps(hold_result, indent=1), encoding="utf-8", newline="\n")
        pd.DataFrame({"Date": [str(d)[:10] for d in held.dates], "segment": segments, "target": held.y_regression,
                      **test_compared}).to_csv(out / "test_predictions.csv", index=False, lineterminator="\n")
        pd.DataFrame({"Date": [str(d)[:10] for d in hold_dates], "target": hold_y, **hold_compared}).to_csv(
            out / "holdout_predictions.csv", index=False, lineterminator="\n")
        pd.DataFrame(rows).to_csv(out / "segment_metrics.csv", index=False, lineterminator="\n")
    receipt.write_text(json.dumps({
        "status": "completed", "post_hoc": True,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": "Post-submission log-HAR study; arms and kappa locked on Validation (decision.json) before this run. "
                  "Test had already been opened for the 19-model study.",
        "locked_model": decision["locked_model"], "decision_sha256": digest(ROOT / "decision.json"),
        "test_sha256": held.sha256, "test_rows": int(len(held.y_regression)), "holdout_rows": int(len(hold_y)),
    }, indent=1), encoding="utf-8", newline="\n")
    for variant, (test_result, hold_result, *_rest) in pending.items():
        print(f"== {variant}: Test {test_result['rows']} rows (post-hoc), holdout {hold_result['rows']} rows")
        for row in test_result["leaderboard"]:
            hold = hold_result["models"][row["model"]]
            print(f"  {row['model']:22s} Test RMSE {row['rmse']:.5f} R2 {row['r2']:+.3f} QLIKE {row['qlike']} "
                  f"| holdout RMSE {hold['rmse']:.5f} QLIKE {hold['qlike']}")


if __name__ == "__main__":
    main()

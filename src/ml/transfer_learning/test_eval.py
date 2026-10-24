"""Score the Validation-locked transfer models once on Test and on the post-snapshot holdout.

    python -m src.ml.transfer_learning.test_eval --allow-test

Test was already opened for the 19-model study (2026-10-11), so these Test numbers are
post-hoc. The holdout (QQQ days after the accepted snapshot, labelled with the same
Train Q75) did not exist when anything here was designed; it is the only blind check.
Nothing is refit: every model is reloaded from the checkpoint hashed in decision.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

from ..core.artifacts import load_model
from ..core.bootstrap import block_indices, difference, interval, replicate
from ..core.contracts import FEATURE_COLUMNS, VARIANTS
from ..core.data import load, load_test
from ..core.metrics import classification_metrics
from ..core.preprocess import Standardizer
from ..core.report import PERSISTENCE_FEATURE, classification_scores
from .experiment import CONFIG, FEATURES_PATH, RUNS, build, mlp_dir, sha256, write_predictions
from .sources import MANIFEST_PATH, ROOT, qqq_extended
from .windows import make_windows, timeline

OUT = ROOT / "test"


def verify_decision() -> dict:
    decision = json.loads((ROOT / "decision.json").read_text(encoding="utf-8"))
    hashes = decision["hashes"]
    current = {p.name: sha256(p) for p in sorted((PROJECT_ROOT / "src" / "ml" / "transfer_learning").glob("*.py"))}
    if current != hashes["code"]:
        raise ValueError("Transfer-learning code changed after the Validation decision was locked")
    if sha256(MANIFEST_PATH) != hashes["source_manifest"]:
        raise ValueError("Source manifest changed after the decision")
    for key, digest in hashes["models"].items():
        if sha256(RUNS / key / "model.npz") != digest:
            raise ValueError(f"Checkpoint changed after the decision: {key}")
    return decision


def scores(y, score, label) -> dict:
    metrics = classification_metrics(y, score, predicted_labels=label)
    out = classification_scores(y, score, label)
    out["confusion_matrix"] = metrics["confusion_matrix"]
    return out


def evaluate(name: str, y: np.ndarray, compared: dict, bootstrap: bool) -> dict:
    result = {"rows": int(len(y)), "positives": int(y.sum()), "models": {}}
    reps = {}
    indices = block_indices(len(y)) if bootstrap else None
    for model, (score, label) in compared.items():
        point = scores(y, score, label)
        entry = {"point": point}
        if bootstrap:
            reps[model] = replicate(classification_scores, (y, score, label), indices)["average_precision"]
            entry["ap_ci"] = interval(point["average_precision"], reps[model])
        result["models"][model] = entry
    if bootstrap:
        primary = CONFIG["primary_arm"]
        ap = {m: result["models"][m]["point"]["average_precision"] for m in compared}
        result["paired_ap_differences"] = {
            f"{primary} - {other}": difference(ap[other], reps[other], ap[primary], reps[primary])
            for other in compared if other != primary}
    return result


def score_variant(variant: str, decision: dict, held, test_dates, test_features, hold) -> dict:
    """Seed-averaged arms, the variant's own MLP and persistence on Test and on the holdout."""
    ds = load(variant)
    scaler = Standardizer().fit(ds.train.X)  # each variant's arms were trained on its own Train fit
    kept, _, test_X = make_windows(test_dates, scaler.transform(test_features), held.dates, CONFIG["lookback"])
    if len(kept) != len(held.dates):
        raise ValueError("Every Test day must keep a full window")
    _, _, hold_X = make_windows(hold["timeline_dates"], scaler.transform(hold["timeline_features"]),
                                hold["dates"], CONFIG["lookback"])

    test_compared, hold_compared = {}, {}
    for arm in decision["also_evaluated"]:
        t_scores, h_scores = [], []
        for seed in CONFIG["seeds"]:
            run = RUNS / variant / arm / f"seed_{seed}"
            backbone = json.loads((run / "config.json").read_text(encoding="utf-8"))["backbone"]
            model = build(backbone, seed).load_state(dict(np.load(run / "model.npz")))
            t_scores.append(model.predict_proba(test_X))
            h_scores.append(model.predict_proba(hold_X))
        for target, values in ((test_compared, t_scores), (hold_compared, h_scores)):
            score = np.mean(values, axis=0)
            target[arm] = (score, (score > 0.5).astype(float))

    mlp, transform = load_model(mlp_dir(variant))
    saved = pd.read_csv(mlp_dir(variant) / "test_predictions.csv", float_precision="round_trip")
    if not np.allclose(saved["raw_score"].to_numpy(), mlp.predict_proba(transform(held.X)), rtol=1e-10, atol=0):
        raise ValueError(f"Reloaded {variant} MLP does not reproduce its finalized Test scores")
    test_compared["mlp"] = (saved["raw_score"].to_numpy(), saved["final_prediction"].to_numpy(dtype=float))
    hold_mlp = mlp.predict_proba(transform(hold["raw"]))
    hold_compared["mlp"] = (hold_mlp, (hold_mlp > 0.5).astype(float))
    column = FEATURE_COLUMNS.index(PERSISTENCE_FEATURE)
    for target, X in ((test_compared, held.X), (hold_compared, hold["raw"])):
        realised = X[:, column].copy()
        target["baseline_persistence"] = (realised, (realised > ds.threshold).astype(float))

    y_test = held.y_classification.astype(np.float64)
    test_result = evaluate("test", y_test, test_compared, bootstrap=True)
    hold_result = evaluate("holdout", hold["y"], hold_compared, bootstrap=False)
    hold_result["note"] = ("Blind but tiny: AP with this few positive days is descriptive only; "
                           "see holdout_predictions.csv day by day.")
    hold_result["dates"] = [str(hold["dates"][0])[:10], str(hold["dates"][-1])[:10]]
    return {"test": (test_result, test_compared), "holdout": (hold_result, hold_compared)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_test:
        raise PermissionError("Test requires --allow-test")
    receipt = ROOT / "test_receipt.json"
    if receipt.exists():
        raise FileExistsError(f"Transfer-learning Test was already evaluated: {receipt}")
    decision = verify_decision()
    # Lock before the read: a crash after this point must not leave Test re-readable.
    receipt.write_text(json.dumps({"status": "opened", "post_hoc": True,
                                   "opened_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}),
                       encoding="utf-8")

    held = load_test(allow_test=True)
    test_dates, test_features = timeline(pd.read_csv(FEATURES_PATH, float_precision="round_trip"))
    rows = np.searchsorted(test_dates, held.dates)
    if not np.allclose(test_features[rows], held.X, rtol=0, atol=1e-12):
        raise ValueError("Interim features disagree with the accepted Test split")

    extended = qqq_extended()
    ext_dates, ext_features = timeline(extended)
    after = extended.loc[(pd.to_datetime(extended["Date"]) > pd.Timestamp(held.dates[-1]))
                         & extended["target_volatility_5d"].notna()]
    threshold = load("with_spike").threshold  # one Train Q75 for both variants
    hold = {"timeline_dates": ext_dates, "timeline_features": ext_features,
            "dates": pd.to_datetime(after["Date"]).to_numpy(),
            "y": (after["target_volatility_5d"].to_numpy() > threshold).astype(np.float64),
            "raw": after[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)}

    results = {variant: score_variant(variant, decision, held, test_dates, test_features, hold) for variant in VARIANTS}

    y_test = held.y_classification.astype(np.float64)
    for variant, result in results.items():
        out = OUT / variant
        out.mkdir(parents=True, exist_ok=True)
        test_result, test_compared = result["test"]
        hold_result, hold_compared = result["holdout"]
        for name, (score, label) in test_compared.items():
            write_predictions(out / f"{name}_test_predictions.csv", held.dates, y_test, score, label)
        pd.DataFrame({"Date": [str(d)[:10] for d in hold["dates"]], "target": hold["y"].astype(int),
                      **{name: score for name, (score, _) in hold_compared.items()}}
                     ).to_csv(out / "holdout_predictions.csv", index=False)
        (out / "test_metrics.json").write_text(json.dumps(test_result, indent=1), encoding="utf-8")
        (out / "holdout_metrics.json").write_text(json.dumps(hold_result, indent=1), encoding="utf-8")
    receipt.write_text(json.dumps({
        "status": "completed",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "post_hoc": True,
        "reason": "Post-submission transfer-learning study; model, seeds and label rule locked on Validation "
                  "(decision.json) before this run. Test had already been opened for the 19-model study.",
        "locked_model": decision["locked_model"], "variants": list(VARIANTS),
        "decision_sha256": sha256(ROOT / "decision.json"),
        "test_sha256": held.sha256, "test_rows": int(len(y_test)),
        "holdout_rows": int(len(hold["y"])), "holdout_positives": int(hold["y"].sum()),
    }, indent=1), encoding="utf-8")
    for variant, result in results.items():
        for split in ("test", "holdout"):
            res = result[split][0]
            print(variant, split, res["rows"], "rows", res["positives"], "positives")
            for model, entry in res["models"].items():
                p = entry["point"]
                print(f"  {model:22s} AP {p['average_precision']} AUC {p['roc_auc']} CM {p['confusion_matrix']}")


if __name__ == "__main__":
    main()

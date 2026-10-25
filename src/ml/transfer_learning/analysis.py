"""Post-hoc paired comparisons of the GRU arms with the baselines, read from saved predictions.

    python -m src.ml.transfer_learning.analysis

Written after test_receipt.json existed: it opens no data split, refits nothing and only
pairs the Validation/Test CSVs already on disk, so it cannot change any locked number.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..core.bootstrap import block_indices, difference, replicate
from ..core.contracts import VARIANTS
from ..core.data import load
from scipy.stats import spearmanr

from ..core.metrics import classification_metrics
from ..core.report import baselines, classification_scores
from .experiment import CONFIG, RUNS, build, mlp_dir, qqq_data
from .sources import ROOT
from .train import tail_split

PAIRS = [("mlp", "G-scratch"), ("mlp", "G-full"), ("baseline_persistence", "G-scratch"),
         ("G-scratch", "G-full"), ("G-scratch", "G-probe"), ("T-scratch", "T-full")]


def validation_scores(variant: str) -> dict:
    out = {}
    for arm in {a for pair in PAIRS for a in pair} - {"mlp", "baseline_persistence"}:
        frames = [pd.read_csv(RUNS / variant / arm / f"seed_{s}" / "validation_predictions.csv") for s in (0, 1, 2)]
        score = np.mean([f["raw_score"] for f in frames], axis=0)
        out[arm] = (frames[0]["target"].to_numpy(), score, (score > 0.5).astype(float))
    mlp = pd.read_csv(mlp_dir(variant) / "validation_predictions.csv")
    out["mlp"] = (mlp["target"].to_numpy(), mlp["raw_score"].to_numpy(), mlp["final_prediction"].to_numpy(dtype=float))
    return out


def test_scores(variant: str) -> dict:
    out = {}
    for path in (ROOT / "test" / variant).glob("*_test_predictions.csv"):
        frame = pd.read_csv(path)
        out[path.name.replace("_test_predictions.csv", "")] = (
            frame["target"].to_numpy(), frame["raw_score"].to_numpy(), frame["final_prediction"].to_numpy(dtype=float))
    return out


def paired(scores: dict) -> dict:
    y = next(iter(scores.values()))[0]
    if any(not np.array_equal(entry[0], y) for entry in scores.values()):
        raise ValueError("Predictions are not paired on the same days")
    indices = block_indices(len(y))
    ap, reps = {}, {}
    for name, (target, score, label) in scores.items():
        ap[name] = classification_scores(target, score, label)["average_precision"]
        reps[name] = replicate(classification_scores, (target, score, label), indices)["average_precision"]
    return {f"{b} - {a}": difference(ap[a], reps[a], ap[b], reps[b]) for a, b in PAIRS if a in ap and b in ap}


def generalization(variant: str) -> list[dict]:
    """ROC-AUC on the fitted rows, the early-stopping tail and Validation, plus how each model's
    Validation score tracks the latest day's features (Spearman). Train/Validation only."""
    data = qqq_data(CONFIG["lookback"], variant)
    train, val = data["train"], data["validation"]
    sub, tail = tail_split(train["positions"])
    columns = {"historical_volatility_5d": 2, "historical_volatility_20d": 3, "rsi_14": 6}
    rows = []
    for arm in ("T-scratch", "T-full", "G-scratch", "G-full"):
        for seed in CONFIG["seeds"]:
            run = RUNS / variant / arm / f"seed_{seed}"
            backbone = json.loads((run / "config.json").read_text(encoding="utf-8"))["backbone"]
            model = build(backbone, seed).load_state(dict(np.load(run / "model.npz")))
            fitted, scored = model.predict_proba(train["X"]), model.predict_proba(val["X"])
            auc = lambda y, s: classification_metrics(y, s)["roc_auc"]["value"]
            row = {"variant": variant, "arm": arm, "seed": seed, "auc_fitted_rows": auc(train["y"][sub], fitted[sub]),
                   "auc_train_tail": auc(train["y"][tail], fitted[tail]), "auc_validation": auc(val["y"], scored)}
            for name, column in columns.items():
                row[f"spearman_{name}"] = float(spearmanr(scored, val["X"][:, -1, column]).statistic)
            rows.append(row)
    return rows


def main() -> None:
    diagnostics = [row for variant in VARIANTS for row in generalization(variant)]
    (ROOT / "report").mkdir(exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(ROOT / "report" / "generalization_diagnostics.csv", index=False)
    result = {}
    for variant in VARIANTS:
        validation = validation_scores(variant)
        ds = load(variant)
        score, label = baselines("classification", ds.train, ds.validation, ds.threshold)["baseline_persistence"]
        validation["baseline_persistence"] = (ds.validation.y_classification, score, label)
        result[variant] = {"validation": paired(validation), "test": paired(test_scores(variant))}
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report" / "gru_vs_baselines.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for variant, splits in result.items():
        for split in ("validation", "test"):
            for name, d in splits[split].items():
                print(f"{variant:10s} {split:10s} {name:28s} {d['point']:+.4f} [{d['low']:+.3f}, {d['high']:+.3f}]")


if __name__ == "__main__":
    main()

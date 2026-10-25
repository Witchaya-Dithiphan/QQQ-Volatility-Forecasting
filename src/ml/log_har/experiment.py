"""Fit every log-HAR arm on QQQ Train, choose kappa on Validation, lock the decision.

    python -m src.ml.log_har.experiment validate
    python -m src.ml.log_har.experiment summarize

Never opens the held-out split. The prior comes from the ten ETFs only (cut at the last
QQQ Train date), so QQQ enters the transfer arm once, through its own likelihood.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

from ..core.bootstrap import block_indices, difference, interval, replicate
from ..core.contracts import FEATURE_COLUMNS, REGRESSION_TARGET, VARIANTS
from ..core.data import load
from ..core.report import baselines, regression_scores
from ..transfer_learning.sources import MANIFEST_PATH, TICKERS, load_source, text_sha256
from .features import NAMES
from .model import LogHAR, pooled_prior

ROOT = PROJECT_ROOT / "outputs" / "experiments" / "log_har"
KAPPAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, np.inf)
# arm -> (feature set, kappa or "selected", uses the ETF prior)
ARMS = {"H3-qqq": ("3", 0.0, False), "HX-qqq": ("X", 0.0, False),
        "HX-pool": ("X", np.inf, True), "HX-prior": ("X", "selected", True)}
PRIMARY = "HX-prior"
CLASSROOM = ("elastic_net", "multiple_linear", "polynomial", "simple_linear")


def qlike(y, prediction) -> float | None:
    """Patton (2011) loss on variances: mean(r - log r - 1) with r = (y / prediction)^2."""
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if (prediction <= 0).any():
        return None
    ratio = (y / prediction) ** 2
    return float(np.mean(ratio - np.log(ratio) - 1))


def scores(y, prediction) -> dict:
    return {**regression_scores(y, prediction), "qlike": qlike(y, prediction)}


def select_kappa(path: list[dict]) -> float:
    """Lowest Validation RMSE; ties keep the earlier (less shrunk) kappa."""
    best = min(row["validation_rmse"] for row in path)
    return next(row["kappa"] for row in path if row["validation_rmse"] == best)


def etf_frames() -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    frames, rows = [], {}
    for ticker in TICKERS:
        frame = load_source(ticker)  # raises if a label reaches past the QQQ Train cutoff
        frames.append((frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64),
                       frame[REGRESSION_TARGET].to_numpy(dtype=np.float64)))
        rows[ticker] = int(len(frame))
    return frames, rows


def kappa_text(kappa: float) -> str:
    return "inf" if kappa == np.inf else repr(float(kappa))


def write_predictions(path: Path, dates, y, prediction) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["Date", "target", "raw_score", "final_prediction"])
        for d, t, p in zip(dates, y, prediction):
            writer.writerow([str(d)[:10], repr(float(t)), repr(float(p)), repr(float(p))])


def save_run(directory: Path, model: LogHAR, ds, extra: dict) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "model.npz", **model.state_dict())
    prediction = model.predict(ds.validation.X)
    reloaded = LogHAR().load_state(dict(np.load(directory / "model.npz")), {"params": model.get_params()})
    again = reloaded.predict(ds.validation.X)
    (directory / "load_verification.json").write_text(json.dumps(
        {"passed": bool(np.array_equal(prediction, again)), "max_abs_diff": float(np.abs(prediction - again).max())}),
        encoding="utf-8", newline="\n")
    write_predictions(directory / "validation_predictions.csv", ds.validation.dates, ds.validation.y_regression, prediction)
    (directory / "config.json").write_text(json.dumps(
        {"model": "log_har", "params": model.get_params(), "variant": ds.variant,
         "coef": dict(zip(NAMES[model.features], model.coef_.tolist())), "intercept": model.intercept_,
         "residual_variance": model.residual_variance_, "train_rows": int(len(ds.train.X)), **extra}, indent=1),
        encoding="utf-8", newline="\n")
    return prediction


def validate() -> None:
    frames, rows = etf_frames()
    beta, per_asset = pooled_prior(frames, "X")
    (ROOT / "report").mkdir(parents=True, exist_ok=True)
    (ROOT / "report" / "prior.json").write_text(json.dumps(
        {"features": list(NAMES["X"]), "beta_pool": beta.tolist(), "rows": rows,
         "per_asset": {t: b.tolist() for t, b in zip(TICKERS, per_asset)}}, indent=1), encoding="utf-8", newline="\n")
    path_rows = []
    for variant in VARIANTS:
        ds = load(variant)
        path = []
        for kappa in KAPPAS:
            model = LogHAR("X", kappa, beta).fit(ds.train.X, ds.train.y_regression)
            rmse = regression_scores(ds.validation.y_regression, model.predict(ds.validation.X))["rmse"]
            path.append({"variant": variant, "kappa": kappa, "validation_rmse": rmse,
                         **{f"coef_{n}": c for n, c in zip(NAMES["X"], model.coef_)}})
        chosen = select_kappa(path)
        path_rows += [{**row, "kappa": kappa_text(row["kappa"]), "selected": row["kappa"] == chosen} for row in path]
        for arm, (features, kappa, uses_prior) in ARMS.items():
            kappa = chosen if kappa == "selected" else kappa
            model = LogHAR(features, kappa, beta if uses_prior else None).fit(ds.train.X, ds.train.y_regression)
            prediction = save_run(ROOT / variant / arm, model, ds, {"arm": arm, "kappa_grid": [kappa_text(k) for k in KAPPAS]
                                                                    if arm == PRIMARY else None})
            print(f"{variant} {arm:9s} kappa={kappa_text(kappa):5s} "
                  f"Val RMSE {regression_scores(ds.validation.y_regression, prediction)['rmse']:.5f}")
    pd.DataFrame(path_rows).to_csv(ROOT / "report" / "lambda_path.csv", index=False, lineterminator="\n")


def read_prediction(path: Path, dates) -> np.ndarray:
    frame = pd.read_csv(path, float_precision="round_trip")
    if not np.array_equal(pd.to_datetime(frame["Date"]).to_numpy(), dates):
        raise ValueError(f"{path} is not aligned with the evaluation dates")
    return frame["final_prediction"].to_numpy(dtype=np.float64)


def compared_on_validation(variant: str) -> tuple[np.ndarray, dict]:
    ds = load(variant)
    dates, y = ds.validation.dates, ds.validation.y_regression
    compared = {arm: read_prediction(ROOT / variant / arm / "validation_predictions.csv", dates) for arm in ARMS}
    for name in CLASSROOM:
        compared[name] = read_prediction(PROJECT_ROOT / "outputs" / "modeling" / variant / "regression" / name
                                         / "validation_predictions.csv", dates)
    for name, (_, prediction) in baselines("regression", ds.train, ds.validation, ds.threshold).items():
        compared[name] = prediction
    return y, compared


PAIRS = [("HX-qqq", PRIMARY), ("H3-qqq", PRIMARY), ("HX-pool", PRIMARY), ("elastic_net", PRIMARY),
         ("multiple_linear", PRIMARY), ("polynomial", PRIMARY), ("baseline_persistence", PRIMARY),
         ("H3-qqq", "HX-qqq"), ("elastic_net", "H3-qqq"),
         # Neither side was chosen on Validation, so this pair is free of the kappa-selection bias.
         ("HX-qqq", "HX-pool")]


def evaluate(y: np.ndarray, compared: dict) -> dict:
    """Point scores, block-bootstrap intervals and paired RMSE differences (b - a; negative = b better)."""
    indices = block_indices(len(y))
    points, reps = {}, {}
    for name, prediction in compared.items():
        points[name] = scores(y, prediction)
        reps[name] = replicate(scores, (y, prediction), indices)
    board = []
    for name in compared:
        board.append({"model": name, **points[name],
                      "rmse_ci": interval(points[name]["rmse"], reps[name]["rmse"]),
                      "qlike_ci": interval(points[name]["qlike"], reps[name]["qlike"]) if points[name]["qlike"] is not None else None})
    board.sort(key=lambda row: row["rmse"])
    paired = {}
    for a, b in PAIRS:
        if a in points and b in points:
            paired[f"{b} - {a}"] = {metric: difference(points[a][metric], reps[a][metric], points[b][metric], reps[b][metric])
                                    for metric in ("rmse", "qlike") if points[a][metric] is not None and points[b][metric] is not None}
    return {"rows": int(len(y)), "leaderboard": board, "paired_differences": paired}


def digest(path: Path) -> str:
    path = Path(path)
    if path.suffix in {".py", ".json", ".csv", ".md"}:
        return text_sha256(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize() -> None:
    prior = json.loads((ROOT / "report" / "prior.json").read_text(encoding="utf-8"))
    path = pd.read_csv(ROOT / "report" / "lambda_path.csv")
    decision = {"locked_model": f"{PRIMARY}: transfer ridge towards the ETF prior, kappa chosen on Validation RMSE",
                "kappa_grid": [kappa_text(k) for k in KAPPAS], "variants": {}, "hashes": {"models": {}}}
    coefficient_rows = [{"source": "beta_pool (10 ETFs)", "variant": "both", **dict(zip(NAMES["X"], prior["beta_pool"]))}]
    coefficient_rows += [{"source": f"ETF {t}", "variant": "both", **dict(zip(NAMES["X"], b))} for t, b in prior["per_asset"].items()]
    for variant in VARIANTS:
        y, compared = compared_on_validation(variant)
        result = evaluate(y, compared)
        failed = [arm for arm in ARMS
                  if not json.loads((ROOT / variant / arm / "load_verification.json").read_text(encoding="utf-8"))["passed"]]
        if failed:
            raise ValueError(f"Checkpoints that do not reproduce Validation: {failed}")
        selected = path[(path["variant"] == variant) & path["selected"]]["kappa"].iloc[0]
        paired = result["paired_differences"]
        to_qqq, to_en = paired[f"{PRIMARY} - HX-qqq"]["rmse"], paired[f"{PRIMARY} - elastic_net"]["rmse"]
        criteria = {"selected_kappa": str(selected),
                    "transfer_helps": bool(to_qqq["high"] is not None and to_qqq["high"] < 0),
                    "beats_elastic_net": bool(to_en["high"] is not None and to_en["high"] < 0)}
        unbiased = paired["HX-pool - HX-qqq"]["rmse"]
        criteria["pool_beats_qqq_only_unselected"] = bool(unbiased["high"] is not None and unbiased["high"] < 0)
        result.update(variant=variant, split="validation", criteria=criteria,
                      note="kappa was chosen on these rows and the grid contains kappa=0 (HX-qqq itself), so "
                           "HX-prior - HX-qqq is tilted towards HX-prior by selection; HX-pool - HX-qqq compares "
                           "two arms that were both fixed in advance and carries no such bias.")
        (ROOT / variant).mkdir(parents=True, exist_ok=True)
        (ROOT / variant / "validation_summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
        decision["variants"][variant] = {"criteria": criteria,
                                         "validation_rmse": {row["model"]: row["rmse"] for row in result["leaderboard"]}}
        for arm in ARMS:
            config = json.loads((ROOT / variant / arm / "config.json").read_text(encoding="utf-8"))
            coefficient_rows.append({"source": arm, "variant": variant, **config["coef"]})
            decision["hashes"]["models"][f"{variant}/{arm}"] = digest(ROOT / variant / arm / "model.npz")
            decision["hashes"]["configs"] = {**decision["hashes"].get("configs", {}),
                                             f"{variant}/{arm}": digest(ROOT / variant / arm / "config.json")}
        print(f"== {variant}: Validation {result['rows']} rows, selected kappa {selected}")
        for row in result["leaderboard"]:
            ci = row["rmse_ci"]
            print(f"  {row['model']:22s} RMSE {row['rmse']:.5f} [{ci['low']:.4f}, {ci['high']:.4f}] "
                  f"MAE {row['mae']:.4f} R2 {row['r2']:+.3f} QLIKE {row['qlike'] if row['qlike'] is None else round(row['qlike'], 4)}")
        for name, metrics in paired.items():
            d = metrics["rmse"]
            print(f"  {name:32s} dRMSE {d['point']:+.5f} [{d['low']:+.4f}, {d['high']:+.4f}]")
        print(" ", criteria)
    pd.DataFrame(coefficient_rows).to_csv(ROOT / "report" / "coefficients.csv", index=False, lineterminator="\n")
    decision["hashes"].update(
        code={p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        source_manifest=digest(MANIFEST_PATH), prior=digest(ROOT / "report" / "prior.json"),
        validation_split=load("with_spike").validation.sha256)
    (ROOT / "decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "summarize"))
    args = parser.parse_args(argv)
    validate() if args.command == "validate" else summarize()


if __name__ == "__main__":
    main()

"""Train every arm on QQQ Train and score it on Validation; never opens the held-out split.

    python -m src.ml.transfer_learning.experiment train --backbone transformer --seed 0 --variant with_spike
    python -m src.ml.transfer_learning.experiment summarize

Arms share one seed's initial weights and fresh head, so a difference between them is
the pretraining, not the draw. All settings below were fixed before any Validation run.

Both dataset variants of the 19-model study are run. Self-supervised pretraining on the
other ETFs never sees QQQ, so it is done once per seed and shared; DAPT, the scaler and
fine-tuning use the variant's own Train rows. In non_spike the removed spike days lose
their *labels* only: their features stay in the timeline as context, because cutting
them out of a window would splice days that were not adjacent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

from ..core.bootstrap import block_indices, difference, interval, replicate
from ..core.contracts import VARIANTS
from ..core.data import load
from ..core.preprocess import Standardizer
from ..core.report import baselines, classification_scores
from .nn import Model
from .sources import MANIFEST_PATH, ROOT, TICKERS, load_source
from .train import finetune, pretrain
from .windows import contiguous_windows, make_windows, timeline

FEATURES_PATH = PROJECT_ROOT / "data" / "interim" / "qqq_features.csv"
RUNS = ROOT / "runs"
CONFIG = {
    "lookback": 60,
    "model": {"d_model": 32, "n_heads": 4, "n_layers": 2, "d_ff": 64, "dropout": 0.1},
    "batch": 128,
    "ssl": {"epochs": 30, "lr": 1e-3},
    "dapt": {"epochs": 20, "lr": 5e-4},
    "finetune": {"max_epochs": 100, "patience": 10, "head_lr": 1e-3,
                 "encoder_lr_pretrained": 3e-4, "encoder_lr_scratch": 1e-3},
    "weight_decay": 1e-4,
    "seeds": [0, 1, 2],
    "label_rule": "probability > 0.5",
    "primary_arm": "T-full",
}
ARMS = ("scratch", "ssl", "full", "probe")
PREFIX = {"transformer": "T", "gru": "G"}


def mlp_dir(variant: str) -> Path:
    return PROJECT_ROOT / "outputs" / "modeling" / variant / "classification" / "mlp"


def qqq_data(lookback: int, variant: str) -> dict:
    """Standardized QQQ windows for Train and Validation; context stops at the last Validation day."""
    ds = load(variant)
    frame = pd.read_csv(FEATURES_PATH, float_precision="round_trip")
    frame = frame.loc[pd.to_datetime(frame["Date"]) <= pd.Timestamp(ds.validation.dates[-1])]
    dates, features = timeline(frame)
    for split in (ds.train, ds.validation):
        rows = np.searchsorted(dates, split.dates)
        if not np.allclose(features[rows], split.X, rtol=0, atol=1e-12):
            raise ValueError(f"Interim features disagree with the accepted {split.name} split")
    scaler = Standardizer().fit(ds.train.X)  # the same Train-only fit the 19-model trainer uses
    scaled = scaler.transform(features)
    out = {"dataset": ds, "scaler": scaler}
    for split in (ds.train, ds.validation):
        kept, positions, X = make_windows(dates, scaled, split.dates, lookback)
        out[split.name] = {"dates": split.dates[kept], "X": X, "y": split.y_classification[kept].astype(np.float64),
                           "positions": positions}
    if len(out["validation"]["dates"]) != len(ds.validation.dates):
        raise ValueError("Every Validation day must keep a full window")
    return out


def source_windows(lookback: int) -> tuple[np.ndarray, dict]:
    """Each ticker standardized on its own pre-cutoff rows, then cut into windows."""
    chunks, counts = [], {}
    for ticker in TICKERS:
        _, features = timeline(load_source(ticker))
        windows = contiguous_windows(Standardizer().fit_transform(features), lookback)
        chunks.append(windows)
        counts[ticker] = len(windows)
    return np.concatenate(chunks), counts


def write_predictions(path: Path, dates, y, score, labels=None) -> None:
    """labels defaults to the arms' rule (probability > 0.5); baselines pass the labels they were scored with."""
    labels = (np.asarray(score) > 0.5) if labels is None else np.asarray(labels)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Date", "target", "raw_score", "final_prediction"])
        for d, t, s, label in zip(dates, y, score, labels):
            writer.writerow([str(d)[:10], int(t), repr(float(s)), int(label)])


def build(backbone: str, seed: int) -> Model:
    return Model(backbone, lookback=CONFIG["lookback"], seed=seed, **CONFIG["model"])


def shared_ssl(backbone: str, seed: int, monitor: np.ndarray) -> tuple[dict, dict]:
    """SSL state of this seed, trained once on the ETFs and reused by both variants."""
    directory = RUNS / f"{PREFIX[backbone]}-pretrain" / f"seed_{seed}"
    if (directory / "history.json").is_file():
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        return dict(np.load(directory / "ssl.npz")), history
    sources, counts = source_windows(CONFIG["lookback"])
    model = build(backbone, seed)
    ssl = pretrain(model, sources, epochs=CONFIG["ssl"]["epochs"], lr=CONFIG["ssl"]["lr"], batch=CONFIG["batch"],
                   seed=seed, monitor=monitor, weight_decay=CONFIG["weight_decay"])
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "ssl.npz", **model.state_dict())
    history = {"ssl": ssl, "source_windows": counts, "n_source_windows": int(len(sources))}
    (directory / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    return model.state_dict(), history


def train_seed(backbone: str, seed: int, variant: str) -> None:
    started = time.time()
    batch, ft = CONFIG["batch"], CONFIG["finetune"]
    data = qqq_data(CONFIG["lookback"], variant)
    train, val = data["train"], data["validation"]
    # The monitor slice only feeds the logged loss curve; it never steers training.
    ssl_state, _ = shared_ssl(backbone, seed, train["X"][:512])

    model = build(backbone, seed).load_state(ssl_state)
    dapt_history = pretrain(model, train["X"], epochs=CONFIG["dapt"]["epochs"], lr=CONFIG["dapt"]["lr"],
                            batch=batch, seed=100 + seed, monitor=train["X"][:512], weight_decay=CONFIG["weight_decay"])
    dapt_state = model.state_dict()
    dapt_dir = RUNS / variant / f"{PREFIX[backbone]}-dapt" / f"seed_{seed}"
    dapt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(dapt_dir / "dapt.npz", **dapt_state)
    (dapt_dir / "history.json").write_text(json.dumps(
        {"dapt": dapt_history, "n_qqq_windows": int(len(train["X"]))}, indent=1), encoding="utf-8")

    starts = {"scratch": None, "ssl": ssl_state, "full": dapt_state, "probe": dapt_state}
    for arm in ARMS:
        model = build(backbone, seed)
        if starts[arm] is not None:
            model.load_state(starts[arm])
        model.reset_head(seed)
        encoder_lr = {"scratch": ft["encoder_lr_scratch"], "probe": None}.get(arm, ft["encoder_lr_pretrained"])
        history = finetune(model, train["X"], train["y"], train["positions"], seed=seed, encoder_lr=encoder_lr,
                           head_lr=ft["head_lr"], max_epochs=ft["max_epochs"], patience=ft["patience"],
                           batch=batch, weight_decay=CONFIG["weight_decay"])
        out = RUNS / variant / f"{PREFIX[backbone]}-{arm}" / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "model.npz", **model.state_dict())
        score = model.predict_proba(val["X"])
        reloaded = build(backbone, seed).load_state(dict(np.load(out / "model.npz"))).predict_proba(val["X"])
        (out / "load_verification.json").write_text(json.dumps(
            {"passed": bool(np.array_equal(score, reloaded)), "max_abs_diff": float(np.abs(score - reloaded).max())}),
            encoding="utf-8")
        write_predictions(out / "validation_predictions.csv", val["dates"], val["y"], score)
        (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        (out / "config.json").write_text(json.dumps(
            {"arm": f"{PREFIX[backbone]}-{arm}", "variant": variant, "backbone": backbone, "seed": seed,
             "encoder_lr": encoder_lr,
             "start": None if starts[arm] is None else ("ssl" if arm == "ssl" else "ssl+dapt"),
             "model": model.config, "experiment": CONFIG,
             "scaler": {"mean": data["scaler"].mean_.tolist(), "scale": data["scaler"].scale_.tolist()}}, indent=1),
            encoding="utf-8")
        print(f"{variant} {backbone} seed {seed} {arm}: best epoch {history['best_epoch']} "
              f"tail AP {history['best_tail_ap']:.4f}", flush=True)
    print(f"done {variant} {backbone} seed {seed} in {time.time() - started:.0f}s", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"], float_precision="round_trip")


def arm_entries(variant: str, dates: np.ndarray, y: np.ndarray) -> dict:
    """{arm: {"seeds": {seed: score}, "ensemble": score}} for every arm that finished all seeds."""
    entries = {}
    for directory in sorted((RUNS / variant).glob("*-*")):
        if directory.name.endswith("-dapt"):
            continue
        seeds = {}
        for seed in CONFIG["seeds"]:
            path = directory / f"seed_{seed}" / "validation_predictions.csv"
            if path.is_file():
                frame = read_predictions(path)
                if not (np.array_equal(frame["Date"].to_numpy(), dates) and np.array_equal(frame["target"], y)):
                    raise ValueError(f"{path} is not aligned with Validation")
                seeds[seed] = frame["raw_score"].to_numpy()
        if len(seeds) == len(CONFIG["seeds"]):
            entries[directory.name] = {"seeds": seeds, "ensemble": np.mean(list(seeds.values()), axis=0)}
    return entries


def summarize_variant(variant: str) -> tuple[dict, dict, dict]:
    ds = load(variant)
    dates, y = ds.validation.dates, ds.validation.y_classification.astype(np.float64)
    arms = arm_entries(variant, dates, y)
    failed = [str(p) for p in (RUNS / variant).glob("*/seed_*/load_verification.json")
              if not json.loads(p.read_text(encoding="utf-8"))["passed"]]
    if failed:
        raise ValueError(f"Checkpoints that do not reproduce their Validation scores: {failed}")
    mlp = read_predictions(mlp_dir(variant) / "validation_predictions.csv")
    if not np.array_equal(mlp["Date"].to_numpy(), dates):
        raise ValueError("MLP predictions are not aligned with Validation")
    persistence_score, persistence_label = baselines("classification", ds.train, ds.validation, ds.threshold)["baseline_persistence"]
    compared = {name: (entry["ensemble"], (entry["ensemble"] > 0.5).astype(float)) for name, entry in arms.items()}
    compared["mlp"] = (mlp["raw_score"].to_numpy(), mlp["final_prediction"].to_numpy(dtype=float))
    compared["baseline_persistence"] = (persistence_score, persistence_label)

    indices = block_indices(len(y))
    points, reps = {}, {}
    for name, (score, label) in compared.items():
        points[name] = classification_scores(y, score, label)
        reps[name] = replicate(classification_scores, (y, score, label), indices)

    def seed_ap(arm, s):
        score = arms[arm]["seeds"][s]
        return classification_scores(y, score, (score > 0.5).astype(float))["average_precision"]

    table = []
    for name in compared:
        row = {"model": name, **{k: points[name][k] for k in ("average_precision", "roc_auc", "f1", "recall", "precision")},
               "ap_ci": interval(points[name]["average_precision"], reps[name]["average_precision"])}
        if name in arms:
            row["seed_ap"] = {str(s): seed_ap(name, s) for s in CONFIG["seeds"]}
        table.append(row)
    table.sort(key=lambda r: -r["average_precision"])

    def diff(a, b):
        return difference(points[a]["average_precision"], reps[a]["average_precision"],
                          points[b]["average_precision"], reps[b]["average_precision"])
    pairs = [("T-scratch", "T-full"), ("T-scratch", "T-ssl"), ("T-ssl", "T-full"), ("T-probe", "T-full"),
             ("mlp", "T-full"), ("baseline_persistence", "T-full"), ("mlp", "T-scratch"),
             ("G-scratch", "G-full"), ("G-full", "T-full"), ("G-scratch", "T-scratch")]
    differences = {f"{b} - {a}": diff(a, b) for a, b in pairs if a in compared and b in compared}

    primary = CONFIG["primary_arm"]
    seed_wins = sum(seed_ap(primary, s) > seed_ap("T-scratch", s) for s in CONFIG["seeds"])
    transfer_helps = bool(points[primary]["average_precision"] > points["T-scratch"]["average_precision"] and seed_wins >= 2)
    beats = all(points[primary]["average_precision"] - points[b]["average_precision"] >= 0.05
                and differences[f"{primary} - {b}"]["excludes_zero"] and differences[f"{primary} - {b}"]["low"] > 0
                for b in ("mlp", "baseline_persistence"))
    summary = {
        "variant": variant, "split": "validation", "rows": int(len(y)), "positives": int(y.sum()),
        "train_rows": int(len(ds.train.dates)),
        "note": "Validation is the selection split for this post-submission study; intervals are moving-block "
                "bootstrap (20 days, 2000 replicates) on paired days and are exploratory.",
        "leaderboard": table, "paired_ap_differences": differences,
        "criteria": {"transfer_helps": transfer_helps, "seed_wins_vs_scratch": int(seed_wins),
                     "clearly_beats_baselines": bool(beats)},
    }
    (ROOT / variant).mkdir(parents=True, exist_ok=True)
    (ROOT / variant / "validation_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"== {variant}: Validation {len(y)} rows, {int(y.sum())} positives")
    for row in table:
        print(f"{row['model']:22s} AP {row['average_precision']:.4f} "
              f"[{row['ap_ci']['low']:.3f}, {row['ap_ci']['high']:.3f}] AUC {row['roc_auc']:.4f} {row.get('seed_ap', '')}")
    for name, d in differences.items():
        print(f"{name:32s} {d['point']:+.4f} [{d['low']:+.3f}, {d['high']:+.3f}]")
    print(summary["criteria"])
    models = {f"{variant}/{name}/seed_{s}": sha256(RUNS / variant / name / f"seed_{s}" / "model.npz")
              for name in arms for s in CONFIG["seeds"]}
    return summary, models, {"arms": sorted(arms), "train_split": ds.train.sha256,
                             "validation_ap": {name: points[name]["average_precision"] for name in compared}}


def summarize() -> None:
    variants, models = {}, {}
    for variant in VARIANTS:
        summary, digests, facts = summarize_variant(variant)
        models.update(digests)
        variants[variant] = {**facts, "criteria": summary["criteria"]}
    if variants["with_spike"]["arms"] != variants["non_spike"]["arms"]:
        raise ValueError("Both variants must have the same finished arms before locking")
    code = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    decision = {
        "locked_model": f"{CONFIG['primary_arm']} mean probability over seeds {CONFIG['seeds']}, per variant",
        "also_evaluated": variants["with_spike"]["arms"], "baselines": ["mlp", "baseline_persistence"],
        "config": CONFIG, "variants": variants,
        "hashes": {"code": code, "source_manifest": sha256(MANIFEST_PATH), "models": models,
                   "validation_split": load("with_spike").validation.sha256},
    }
    (ROOT / "decision.json").write_text(json.dumps(decision, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--backbone", choices=sorted(PREFIX), default="transformer")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--variant", choices=VARIANTS, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args(argv)
    if args.command == "train":
        train_seed(args.backbone, args.seed, args.variant)
    else:
        summarize()


if __name__ == "__main__":
    main()

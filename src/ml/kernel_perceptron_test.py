"""Evaluate the already selected extra model on Test, separately from the locked 19-model study."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone

from config import PROJECT_ROOT

from .classification.kernel_perceptron import KernelPerceptron
from .core.artifacts import preprocess, sha256_file
from .core.data import load_test
from .core.persist import load_npz
from .core.trainer import evaluate, write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_test:
        raise PermissionError("Test requires --allow-test")

    root = PROJECT_ROOT / "outputs" / "experiments" / "kernel_perceptron"
    receipt = root / "test_receipt.json"
    if receipt.exists():
        raise FileExistsError(f"Kernel Perceptron Test was already evaluated: {receipt}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if {row["variant"] for row in summary} != {"with_spike", "non_spike"}:
        raise ValueError("Expected selected runs for both variants before opening Test")

    prepared = []
    for row in summary:
        directory = root / row["variant"] / "classification" / "kernel_perceptron"
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config["selected_candidate"] != row["selected"] or config["model"] != "kernel_perceptron":
            raise ValueError(f"Selected model does not match summary: {row['variant']}")
        arrays, metadata = load_npz(directory / "model.npz")
        model = KernelPerceptron().load_state(arrays, metadata)
        if not model.converged_:
            raise ValueError(f"Selected model has not converged: {row['variant']}")
        scaler, scaler_metadata = load_npz(directory / "preprocessor.npz")
        prepared.append((row, directory, model, scaler, scaler_metadata))

    holdout = load_test(allow_test=True)
    pending = []
    runs = []
    for row, directory, model, scaler, scaler_metadata in prepared:
        metrics, prediction, scores = evaluate(
            model, preprocess(holdout.X, scaler, scaler_metadata),
            holdout.y_regression, holdout.y_classification, "classification",
        )
        json.dumps(metrics, allow_nan=False)
        results = [
            [str(date)[:10], int(target), float(score), int(label)]
            for date, target, score, label in zip(
                holdout.dates, holdout.y_classification, scores, prediction
            )
        ]
        pending.append((directory, metrics, results))
        runs.append({
            "variant": row["variant"],
            "selected": row["selected"],
            "gamma": model.gamma,
            "model_sha256": sha256_file(directory / "model.npz"),
            "config_sha256": sha256_file(directory / "config.json"),
        })

    for directory, metrics, results in pending:
        write_json(directory / "test_metrics.json", metrics)
        with (directory / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Date", "target", "raw_score", "final_prediction"])
            writer.writerows(results)
    write_json(receipt, {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": "Post-submission exploratory Kernel Perceptron evaluation; hyperparameters selected on Validation",
        "test_sha256": holdout.sha256,
        "test_rows": len(holdout.X),
        "runs": runs,
    })


if __name__ == "__main__":
    main()

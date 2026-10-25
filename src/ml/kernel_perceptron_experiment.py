"""Run the extra Kernel Perceptron experiment without changing the locked model inventory."""
from __future__ import annotations

from config import PROJECT_ROOT

from .classification.kernel_perceptron import KernelPerceptron
from .core.data import load
from .core.preprocess import Standardizer
from .core.trainer import train_model, write_json


def main() -> None:
    root = PROJECT_ROOT / "outputs" / "experiments" / "kernel_perceptron"
    summary = []
    for variant in ("with_spike", "non_spike"):
        result = train_model(
            KernelPerceptron,
            grid={"gamma": [0.5, 0.75, 1.0, 1.5, 2.0], "max_epochs": [1000]},
            variant=variant,
            output_root=root,
        )
        candidates = [
            {
                "id": row["id"],
                "gamma": row["model"].gamma,
                "epochs": row["model"].n_epochs_,
                "converged": row["model"].converged_,
                "average_precision": row["metrics"]["average_precision"]["value"],
                "roc_auc": row["metrics"]["roc_auc"]["value"],
            }
            for row in result["search_results"]
        ]
        data = load(variant)
        train_X = Standardizer().fit_transform(data.train.X)
        train_correct = bool((result["model"].predict(train_X) == data.train.y_classification).all())
        if not all(row["converged"] for row in candidates) or not train_correct or not result["load_verification"]["passed"]:
            raise RuntimeError(f"Kernel Perceptron convergence or reload check failed: {variant}")
        summary.append({"variant": variant, "selected": result["selected"], "train_correct": train_correct, "candidates": candidates})
        print(f"{variant}: {result['selected']} converged at epoch {result['model'].n_epochs_}")
    write_json(root / "summary.json", summary)


if __name__ == "__main__":
    main()

"""Save notebook charts that the report shows as PNG files.

    python docs/report/tools/export_notebook_figures.py

The report shows exactly the charts the executed notebooks display, so the figures are
taken from the saved cell outputs instead of being redrawn by separate code. Re-run the
notebooks first if their sections change.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "figures"
# (notebook, cell id) -> one path under outputs/figures per image the cell draws, in output order.
# notebooks/ holds charts the main chapters use that only exist as notebook output;
# appendix/ holds the post-submission and post-presentation experiments.
FIGURES = {
    ("03_regression.ipynb", "0d09d523"): ["notebooks/03_elastic_net_coefficients.png"],
    ("03_regression.ipynb", "1537318e"): ["notebooks/03_regression_training_loss.png"],
    ("04_classification.ipynb", "72f95e97"): ["notebooks/04_adaboost_stumps.png"],
    ("05_results_comparison.ipynb", "2661895e"): ["notebooks/05_feature_importance_by_variant.png"],
    ("05_results_comparison.ipynb", "8cc8714b"): ["notebooks/05_validation_scores_over_time.png"],
    ("05_results_comparison.ipynb", "9d871a60"): ["notebooks/05_walk_forward_vs_validation.png"],
    ("04_classification.ipynb", "kernel-separability"): ["appendix/perceptron_separability.png"],
    ("04_classification.ipynb", "kernel-train-plots"): ["appendix/kernel_validation_with_spike.png", "appendix/kernel_validation_non_spike.png"],
    ("04_classification.ipynb", "kernel-analysis-train"): ["appendix/kernel_walk_forward_importance.png"],
    ("05_results_comparison.ipynb", "kernel-test-plots"): ["appendix/kernel_test_with_spike.png", "appendix/kernel_test_non_spike.png"],
    ("05_results_comparison.ipynb", "kernel-analysis-test"): ["appendix/kernel_test_segments.png"],
    ("04_classification.ipynb", "transfer-train-plots"): ["appendix/transfer_validation_with_spike.png", "appendix/transfer_validation_non_spike.png"],
    ("04_classification.ipynb", "transfer-analysis-train"): ["appendix/transfer_validation_paired.png"],
    ("05_results_comparison.ipynb", "transfer-test-plots"): ["appendix/transfer_test_with_spike.png", "appendix/transfer_test_non_spike.png"],
    ("05_results_comparison.ipynb", "transfer-analysis-test"): ["appendix/transfer_test_paired.png"],
    ("03_regression.ipynb", "logh-train-plots"): ["appendix/logh_coefficients.png", "appendix/logh_kappa_path.png", "appendix/logh_validation_forecasts.png"],
    ("03_regression.ipynb", "logh-analysis-train"): ["appendix/logh_validation_paired.png"],
    ("05_results_comparison.ipynb", "logh-test-plots"): ["appendix/logh_test_forecasts.png", "appendix/logh_test_segments.png", "appendix/logh_holdout.png"],
    ("05_results_comparison.ipynb", "logh-analysis-test"): ["appendix/logh_test_paired.png"],
}


def main() -> None:
    cache = {}
    for (notebook, cell_id), names in FIGURES.items():
        if notebook not in cache:
            cells = json.loads((ROOT / "notebooks" / notebook).read_text(encoding="utf-8"))["cells"]
            cache[notebook] = {cell.get("id"): cell for cell in cells}
        images = [output["data"]["image/png"] for output in cache[notebook][cell_id]["outputs"]
                  if "image/png" in output.get("data", {})]
        if len(images) != len(names):
            raise ValueError(f"{notebook}:{cell_id} drew {len(images)} images, expected {len(names)}")
        for name, image in zip(names, images):
            target = OUT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(image))
            print(f"{notebook}:{cell_id} -> {(OUT / name).relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

"""Save the rendered figures of the post-submission notebook sections as PNG files for the report.

    python docs/report/tools/export_notebook_figures.py

The appendices show exactly the charts the executed notebooks display, so the figures are
taken from the saved cell outputs instead of being redrawn by separate code. Re-run the
notebooks first if their sections change.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "figures" / "appendix"
# (notebook, cell id) -> one file name per image the cell draws, in output order
FIGURES = {
    ("04_classification.ipynb", "kernel-separability"): ["perceptron_separability.png"],
    ("04_classification.ipynb", "kernel-train-plots"): ["kernel_validation_with_spike.png", "kernel_validation_non_spike.png"],
    ("04_classification.ipynb", "kernel-analysis-train"): ["kernel_walk_forward_importance.png"],
    ("05_results_comparison.ipynb", "kernel-test-plots"): ["kernel_test_with_spike.png", "kernel_test_non_spike.png"],
    ("05_results_comparison.ipynb", "kernel-analysis-test"): ["kernel_test_segments.png"],
    ("04_classification.ipynb", "transfer-train-plots"): ["transfer_validation_with_spike.png", "transfer_validation_non_spike.png"],
    ("04_classification.ipynb", "transfer-analysis-train"): ["transfer_validation_paired.png"],
    ("05_results_comparison.ipynb", "transfer-test-plots"): ["transfer_test_with_spike.png", "transfer_test_non_spike.png"],
    ("05_results_comparison.ipynb", "transfer-analysis-test"): ["transfer_test_paired.png"],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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
            (OUT / name).write_bytes(base64.b64decode(image))
            print(f"{notebook}:{cell_id} -> {(OUT / name).relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

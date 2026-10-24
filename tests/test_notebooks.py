# tests/test_notebooks.py
import json

import pytest

from config import PROJECT_ROOT

MODEL_NOTEBOOKS = ("03_regression.ipynb", "04_classification.ipynb", "05_results_comparison.ipynb")
# 06 studies the spike rule on Train; it compares no model with a library, so it has no parity demo.
NOTEBOOKS = (*MODEL_NOTEBOOKS, "06_spike_analysis.ipynb")
FORBIDDEN = ("load_test(", "test_labeled", "test_flagged", "TEST_LABELED_DATA_PATH", "TEST_FLAGGED_PATH",
             "TEST_DATA_PATH", "train_model(", "src.ml.run train")
PARITY_TAG = "parity-demo"


def _cells(name):
    notebook = json.loads((PROJECT_ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return notebook["cells"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_explains_itself_in_markdown(name):
    cells = _cells(name)
    assert len(cells) >= 16
    assert cells[0]["cell_type"] == "markdown"
    assert sum(cell["cell_type"] == "markdown" for cell in cells) >= len(cells) // 3


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_never_trains_or_opens_the_test_split(name):
    code = [cell for cell in _cells(name) if cell["cell_type"] == "code"]
    for cell in code:
        source = "".join(cell["source"])
        for needle in FORBIDDEN:
            assert needle not in source, f"{name}: {needle}"
        if ".fit(" in source:
            assert PARITY_TAG in cell["metadata"].get("tags", []), f"{name}: fit outside the parity demo"


@pytest.mark.parametrize("name", MODEL_NOTEBOOKS)
def test_notebook_has_exactly_one_parity_demo(name):
    tagged = [cell for cell in _cells(name) if PARITY_TAG in cell["metadata"].get("tags", [])]
    assert len(tagged) == 1
    assert "assert_parity(" in "".join(tagged[0]["source"])


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_saved_executed(name):
    code = [cell for cell in _cells(name) if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code)
    assert not any(output.get("output_type") == "error" for cell in code for output in cell["outputs"])


@pytest.mark.parametrize("name,prefix", [("04_classification.ipynb", "kernel-train"),
                                         ("05_results_comparison.ipynb", "kernel-test")])
def test_kernel_supplement_has_ranking_and_rendered_graphs(name, prefix):
    cells = {cell.get("id"): cell for cell in _cells(name)}
    ranking = cells[f"{prefix}-board"]
    plots = cells[f"{prefix}-plots"]
    assert "kernel_perceptron" in "".join(ranking["source"])
    assert "AP rank" in "".join(ranking["source"])
    assert len([output for output in plots["outputs"]
                if "image/png" in output.get("data", {})]) == 2
    analysis = cells["kernel-analysis-train" if prefix == "kernel-train" else "kernel-analysis-test"]
    assert any("image/png" in output.get("data", {}) for output in analysis["outputs"])


@pytest.mark.parametrize("name,prefix", [("04_classification.ipynb", "transfer-train"),
                                         ("05_results_comparison.ipynb", "transfer-test")])
def test_transfer_supplement_has_ranking_and_rendered_graphs(name, prefix):
    cells = {cell.get("id"): cell for cell in _cells(name)}
    ranking = "".join(cells[f"{prefix}-board"]["source"])
    assert "T-full" in ranking and "AP rank" in ranking and "post-presentation" in ranking
    for cell_id in (f"{prefix}-plots", f"transfer-analysis-{prefix.split('-')[1]}"):
        outputs = cells[cell_id]["outputs"]
        assert any("image/png" in output.get("data", {}) for output in outputs), cell_id
    # Both variants are drawn, like every other model in the study.
    assert len([o for o in cells[f"{prefix}-plots"]["outputs"] if "image/png" in o.get("data", {})]) == 2

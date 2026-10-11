# tests/test_notebooks.py
import json

import pytest

from config import PROJECT_ROOT

NOTEBOOKS = ("03_regression.ipynb", "04_classification.ipynb", "05_results_comparison.ipynb")
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


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_exactly_one_parity_demo(name):
    tagged = [cell for cell in _cells(name) if PARITY_TAG in cell["metadata"].get("tags", [])]
    assert len(tagged) == 1
    assert "assert_parity(" in "".join(tagged[0]["source"])


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_saved_executed(name):
    code = [cell for cell in _cells(name) if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code)
    assert not any(output.get("output_type") == "error" for cell in code for output in cell["outputs"])

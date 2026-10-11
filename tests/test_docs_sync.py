# tests/test_docs_sync.py
import pytest

from config import PROJECT_ROOT
from src.ml.docs_sync import MARKER, sync, write_cards
from src.ml.registry import model_names

DOCUMENTS = [path for path in [PROJECT_ROOT / "README.md", *(PROJECT_ROOT / "docs").rglob("*.md")]
             if "archive" not in path.parts and "superpowers" not in path.parts
             and MARKER.search(path.read_text(encoding="utf-8"))]


def test_sync_replaces_only_the_generated_block(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ml.docs_sync.PROJECT_ROOT", tmp_path)
    (tmp_path / "table.md").write_text("| a |\n| --- |\n| 1 |\n", encoding="utf-8")
    document = tmp_path / "doc.md"
    document.write_text("before\n<!-- BEGIN GENERATED: table.md -->\nstale\n<!-- END GENERATED -->\nafter\n",
                        encoding="utf-8")
    assert sync(document) is True
    assert document.read_text(encoding="utf-8") == (
        "before\n<!-- BEGIN GENERATED: table.md -->\n| a |\n| --- |\n| 1 |\n<!-- END GENERATED -->\nafter\n")
    assert sync(document) is False


def test_a_block_pointing_at_a_missing_file_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ml.docs_sync.PROJECT_ROOT", tmp_path)
    document = tmp_path / "doc.md"
    document.write_text("<!-- BEGIN GENERATED: nope.md -->\n<!-- END GENERATED -->\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        sync(document)


def test_one_card_is_written_per_model(tmp_path):
    written = write_cards(report_root=tmp_path)
    assert sorted(path.stem for path in written) == model_names()
    assert "with_spike" in written[0].read_text(encoding="utf-8")


def test_at_least_the_readme_and_the_model_cards_carry_generated_blocks():
    names = {path.name for path in DOCUMENTS}
    assert {"README.md", "MODEL_CARDS.md"} <= names


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
def test_generated_blocks_match_their_source_files(document):
    for match in MARKER.finditer(document.read_text(encoding="utf-8")):
        source = (PROJECT_ROOT / match["path"]).read_text(encoding="utf-8")
        assert match["body"] == source, f"{document.name}: run `python -m src.ml.docs_sync`"

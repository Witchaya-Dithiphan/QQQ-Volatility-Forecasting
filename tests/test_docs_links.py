# tests/test_docs_links.py
import re

import pytest

from config import PROJECT_ROOT

LINK = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:|#)([^)#\s]+)(?:#[^)]*)?\)")
DOCUMENTS = sorted(
    path for path in [PROJECT_ROOT / "README.md", PROJECT_ROOT / "AGENTS.md", PROJECT_ROOT / "CLAUDE.md",
                      *(PROJECT_ROOT / "docs").rglob("*.md")]
    if "archive" not in path.parts and "superpowers" not in path.parts
)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
def test_relative_links_resolve(document):
    broken = [target for target in LINK.findall(document.read_text(encoding="utf-8"))
              if not (document.parent / target).exists()]
    assert broken == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
def test_documents_do_not_claim_report_or_finalize_are_missing(document):
    text = document.read_text(encoding="utf-8")
    for stale in ("ยังไม่ implement", "not implemented yet", "exit 1) — เป็นงาน Phase C"):
        assert stale not in text

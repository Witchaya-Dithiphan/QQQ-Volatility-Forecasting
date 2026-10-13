# tests/test_docs_links.py
import re
from pathlib import PurePosixPath

import pytest

from config import PROJECT_ROOT

LINK = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:|#)([^)#\s]+)(?:#[^)]*)?\)")
DOCUMENTS = sorted(
    path for path in [PROJECT_ROOT / "README.md", *(PROJECT_ROOT / "docs").rglob("*.md")]
    if "archive" not in path.parts
)
# Coding-assistant files are ignored by git: they can sit on a developer's disk, so a link
# to one resolves there and is dead for everyone who clones the repository.
LOCAL_ONLY = {"AGENTS.md", "CLAUDE.md", ".claude", ".codex", ".superpowers"}


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
def test_relative_links_resolve(document):
    broken = [target for target in LINK.findall(document.read_text(encoding="utf-8"))
              if not (document.parent / target).exists()]
    assert broken == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
def test_no_link_points_at_a_file_git_ignores(document):
    local = [target for target in LINK.findall(document.read_text(encoding="utf-8"))
             if LOCAL_ONLY.intersection(PurePosixPath(target).parts)]
    assert local == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
def test_documents_do_not_claim_report_or_finalize_are_missing(document):
    text = document.read_text(encoding="utf-8")
    for stale in ("ยังไม่ implement", "not implemented yet", "exit 1) — เป็นงาน Phase C"):
        assert stale not in text


EVIDENCE = re.compile(r"`((?:outputs|src|notebooks|docs|tests|deliverables|configs)/[^`*<>]+)`")


def test_every_traceability_evidence_path_exists():
    text = (PROJECT_ROOT / "docs" / "report" / "TRACEABILITY.md").read_text(encoding="utf-8")
    paths = EVIDENCE.findall(text)
    assert len(paths) >= 25
    assert [path for path in paths if not (PROJECT_ROOT / path).exists()] == []

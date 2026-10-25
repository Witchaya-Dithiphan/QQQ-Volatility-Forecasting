# tests/test_docs_links.py
import os
import re
import subprocess

import pytest

from config import PROJECT_ROOT

LINK = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:|#)([^)#\s]+)(?:#[^)]*)?\)")


def _shipped_files():
    """Paths git would put in a commit: tracked files plus new files that are not ignored."""
    try:
        listing = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=PROJECT_ROOT, capture_output=True, check=True,
        ).stdout.decode("utf-8")
    except (OSError, subprocess.CalledProcessError):
        # A copy without .git (such as a ZIP download) holds only shipped files, so every
        # project file on disk counts; virtual environments and caches are skipped.
        return {path.relative_to(PROJECT_ROOT).as_posix() for path in PROJECT_ROOT.rglob("*")
                if path.is_file() and not any(part.startswith(".") or part in {"tmp", "node_modules"}
                                              for part in path.relative_to(PROJECT_ROOT).parts[:-1])}
    return {path for path in listing.split("\0") if path and (PROJECT_ROOT / path).exists()}


SHIPPED = _shipped_files()
# Working notes and coding-assistant files are ignored by git. They can sit on a developer's
# disk, so a link to one resolves there and is dead for everyone who clones the repository.
DOCUMENTS = sorted(
    PROJECT_ROOT / path for path in SHIPPED
    if path == "README.md" or (path.startswith("docs/") and path.endswith(".md") and "/archive/" not in path)
)


def _targets(document):
    return LINK.findall(document.read_text(encoding="utf-8"))


def _in_repository(document, target):
    relative = os.path.relpath(os.path.normpath(document.parent / target), PROJECT_ROOT).replace(os.sep, "/")
    return relative in SHIPPED or any(path.startswith(relative.rstrip("/") + "/") for path in SHIPPED)


def test_every_shipped_document_is_checked():
    names = {path.relative_to(PROJECT_ROOT).as_posix() for path in DOCUMENTS}
    assert {"README.md", "docs/README.md", "docs/design/MODEL_CARDS.md"} <= names
    assert not any(name.startswith(("docs/plan/", "docs/superpowers/")) for name in names)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
def test_relative_links_resolve(document):
    broken = [target for target in _targets(document) if not (document.parent / target).exists()]
    assert broken == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix())
def test_no_link_points_at_a_file_git_ignores(document):
    local = [target for target in _targets(document)
             if (document.parent / target).exists() and not _in_repository(document, target)]
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

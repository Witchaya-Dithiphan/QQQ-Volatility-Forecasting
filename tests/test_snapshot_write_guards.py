"""The refresh writers must never replace the immutable snapshot or its manifest."""

from pathlib import Path

import pytest

import src.download_qqq_data as download_module
from src.download_qqq_data import (
    DataContractError,
    write_csv_atomically,
    write_json_atomically,
)


@pytest.fixture
def protected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point the protected paths at throwaway files so a missing guard harms nothing."""
    snapshot = tmp_path / "raw" / "qqq_daily.csv"
    manifest = tmp_path / "manifests" / "qqq_daily_snapshot.json"
    for path in (snapshot, manifest):
        path.parent.mkdir(parents=True)
        path.write_text("original", encoding="utf-8")
    monkeypatch.setattr(download_module, "RAW_DATA_PATH", snapshot)
    monkeypatch.setattr(download_module, "RAW_DATA_MANIFEST_PATH", manifest)
    return {"snapshot": snapshot, "manifest": manifest}


@pytest.mark.parametrize("target", ["snapshot", "manifest"])
def test_csv_writer_refuses_protected_paths_even_with_overwrite(
    protected: dict[str, Path], target: str
) -> None:
    with pytest.raises(DataContractError):
        write_csv_atomically([], protected[target], overwrite=True)
    assert protected[target].read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("target", ["snapshot", "manifest"])
def test_json_writer_refuses_protected_paths(protected: dict[str, Path], target: str) -> None:
    with pytest.raises(DataContractError):
        write_json_atomically({"status": "refreshed"}, protected[target])
    assert protected[target].read_text(encoding="utf-8") == "original"


def test_json_writer_still_writes_an_ordinary_report(
    protected: dict[str, Path], tmp_path: Path
) -> None:
    report = tmp_path / "reports" / "download_report.json"
    write_json_atomically({"status": "refreshed"}, report)
    assert '"status": "refreshed"' in report.read_text(encoding="utf-8")

"""Portable report references survive relocation of an artifact tree."""

import shutil
from pathlib import Path

from src.report_paths import report_relative_path


def test_report_paths_are_relative_and_survive_relocation(tmp_path: Path) -> None:
    source_tree = tmp_path / "first_location"
    report = source_tree / "outputs" / "reports" / "report.json"
    artifact = source_tree / "data" / "processed" / "train.csv"
    report.parent.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")
    artifact.write_text("Date\n", encoding="utf-8")

    reference = report_relative_path(artifact, report)
    assert reference == "../../data/processed/train.csv"
    assert not Path(reference).is_absolute()
    assert (report.parent / reference).resolve() == artifact.resolve()

    moved_tree = tmp_path / "second_location"
    shutil.copytree(source_tree, moved_tree)
    moved_report = moved_tree / "outputs" / "reports" / "report.json"
    moved_artifact = moved_tree / "data" / "processed" / "train.csv"
    assert (moved_report.parent / reference).resolve() == moved_artifact.resolve()


def test_report_path_reference_is_its_filename(tmp_path: Path) -> None:
    report = tmp_path / "outputs" / "reports" / "report.json"
    assert report_relative_path(report, report) == "report.json"

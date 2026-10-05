"""Portable references to artifacts recorded in JSON reports."""

import os
from pathlib import Path


def report_relative_path(artifact_path: str | Path, report_path: str | Path) -> str:
    """Return an artifact path relative to the directory holding its report.

    Both paths are resolved before calculating the reference, so callers may
    supply absolute or relative paths from the project root. Forward slashes
    keep the serialized JSON independent of the host path separator.

    Raises:
        ValueError: If Windows paths are on different drives and cannot be
            represented by a relative reference.
    """
    artifact = Path(artifact_path).resolve(strict=False)
    report_directory = Path(report_path).resolve(strict=False).parent
    try:
        return Path(os.path.relpath(artifact, report_directory)).as_posix()
    except ValueError as exc:
        raise ValueError(
            "Report and referenced artifact must be on the same drive: "
            f"{report_path}, {artifact_path}"
        ) from exc

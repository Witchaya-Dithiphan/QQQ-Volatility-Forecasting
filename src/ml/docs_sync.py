# src/ml/docs_sync.py
"""Copy generated result tables into the documents that quote them.

Numbers typed by hand drifted from the artifacts once already. A document now quotes a
table by naming its file between two markers, and a test fails when the two differ.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from config import PROJECT_ROOT

from .core.artifacts import read_run
from .core.contracts import VARIANTS
from .core.figures import write_markdown_table
from .core.trainer import OUTPUT_ROOT
from .registry import MODELS, model_names

MARKER = re.compile(
    r"<!-- BEGIN GENERATED: (?P<path>[^ ]+) -->\n(?P<body>.*?)<!-- END GENERATED -->", re.DOTALL
)
FIELDS = {"regression": ("rmse", "mae", "r2"),
          "classification": ("average_precision", "roc_auc", "f1", "recall", "precision", "specificity", "accuracy"),
          "clustering": ("silhouette", "inertia")}


def _values(metrics: dict, task: str) -> dict:
    source = metrics["clipped_metrics"] if task == "regression" else metrics
    return {field: source[field]["value"] for field in FIELDS[task]}


def write_cards(*, artifact_root=None, report_root=None) -> list[Path]:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    report_root = Path(report_root or artifact_root / "report")
    written = []
    for name in model_names():
        task, rows = MODELS[name].task, []
        for variant in VARIANTS:
            run = read_run(artifact_root, variant, name)
            params = json.dumps(run.config["params"], sort_keys=True)
            rows.append({"variant": variant, "split": "validation (selection score)", "params": params,
                         **_values(run.metrics, task)})
            holdout_path = run.directory / "test_metrics.json"
            if holdout_path.is_file():
                holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
                rows.append({"variant": variant, "split": "test (one-shot)", "params": params,
                             **_values(holdout, task)})
        path = report_root / "cards" / f"{name}.md"
        write_markdown_table(rows, path, columns=list(rows[0]), title=f"ผลจริงของ `{name}`")
        written.append(path)
    return written


def sync(document: Path) -> bool:
    document = Path(document)
    text = document.read_text(encoding="utf-8")

    def fill(match: re.Match) -> str:
        body = (PROJECT_ROOT / match["path"]).read_text(encoding="utf-8")
        return f"<!-- BEGIN GENERATED: {match['path']} -->\n{body}<!-- END GENERATED -->"

    updated = MARKER.sub(fill, text)
    if updated != text:
        document.write_text(updated, encoding="utf-8", newline="\n")
    return updated != text


def main() -> int:
    from .report_tables import write_tables

    write_cards()
    write_tables()
    documents = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").rglob("*.md"))]
    for document in documents:
        if "archive" in document.parts or "superpowers" in document.parts:
            continue
        if sync(document):
            print(f"updated {document.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

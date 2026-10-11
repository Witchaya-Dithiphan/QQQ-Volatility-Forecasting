# src/ml/core/comparison.py
"""with_spike vs non_spike, scored on the one Validation set both variants share.

The two variants differ in more than the spike rows themselves, so a difference here
is the effect of the whole removal policy and must be worded that way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import figures
from .artifacts import read_runs
from .bootstrap import REPLICATES, block_indices, difference
from .report import INTERVAL_FIELDS, PRIMARY, as_entry, score_entries
from .trainer import OUTPUT_ROOT

CONFOUND_NOTE = (
    "ผลต่างในตารางนี้คือผลของ **นโยบายตัดแถวทั้งชุด** ไม่ใช่ผลของ spike เพียงอย่างเดียว: "
    "การตัด 213 แถวทำให้ class prior ของ Train เปลี่ยนจาก 25.0% เป็น 18.4%, จำนวนแถวลดจาก 1,733 เป็น 1,520 "
    "และ regime ที่ผันผวนสูงหายไปจาก Train พร้อมกัน ทั้งสาม confound นี้แยกจากกันไม่ได้ด้วยการทดลองนี้ "
    "delta = non_spike − with_spike บน Validation 371 แถวชุดเดียวกัน ช่วง 95% จาก paired moving-block bootstrap "
    "(block 20 วัน) และเมื่อเทียบ 17 โมเดลพร้อมกัน ควรคาดว่าจะมีราว 1 แถวที่ช่วงไม่คร่อมศูนย์โดยบังเอิญ"
)


def paired_comparison(*, artifact_root=None, replicates: int = REPLICATES, require_all: bool = True) -> list[dict]:
    artifact_root = Path(artifact_root or OUTPUT_ROOT)
    rows = []
    for task in ("regression", "classification"):
        field, higher = PRIMARY[task]
        first = {run.name: run for run in read_runs(artifact_root, "with_spike", task=task, require_all=require_all)}
        second = {run.name: run for run in read_runs(artifact_root, "non_spike", task=task, require_all=require_all)}
        indices = None
        for name in sorted(set(first) & set(second)):
            a, b = first[name], second[name]
            if not (np.array_equal(a.dates, b.dates) and np.array_equal(a.target, b.target)):
                raise ValueError(f"{name}: the two variants were not scored on the same Validation rows")
            if indices is None:
                indices = block_indices(len(a.target), replicates=replicates)
            scored_a, scored_b = score_entries(task, [as_entry(a), as_entry(b)], indices)
            row = {"model": name, "task": task, "primary_metric": field}
            for metric in INTERVAL_FIELDS[task]:
                gap = difference(scored_a["point"][metric], scored_a["replicates"][metric],
                                 scored_b["point"][metric], scored_b["replicates"][metric])
                row[f"{metric}_with_spike"] = scored_a["point"][metric]
                row[f"{metric}_non_spike"] = scored_b["point"][metric]
                row[f"{metric}_delta"] = gap["point"]
                row[f"{metric}_delta_low"], row[f"{metric}_delta_high"] = gap["low"], gap["high"]
                row[f"{metric}_excludes_zero"] = gap["excludes_zero"]
            row["primary_excludes_zero"] = row[f"{field}_excludes_zero"]
            if row["primary_excludes_zero"] is not True:
                row["verdict"] = "inconclusive"
            else:
                row["verdict"] = "non_spike better" if (row[f"{field}_delta"] > 0) == higher else "with_spike better"
            rows.append(row)

    first = {run.name: run for run in read_runs(artifact_root, "with_spike", task="clustering", require_all=require_all)}
    second = {run.name: run for run in read_runs(artifact_root, "non_spike", task="clustering", require_all=require_all)}
    for name in sorted(set(first) & set(second)):
        a, b = first[name].metrics["silhouette"]["value"], second[name].metrics["silhouette"]["value"]
        rows.append({"model": name, "task": "clustering", "primary_metric": "silhouette",
                     "silhouette_with_spike": a, "silhouette_non_spike": b, "silhouette_delta": b - a,
                     "primary_excludes_zero": None, "verdict": "no interval"})
    return rows


def write_comparison(rows: list[dict], *, report_root: Path, figure_root: Path) -> list[Path]:
    report_root, figure_root = Path(report_root), Path(figure_root)
    csv_path, md_path = report_root / "comparison_a_vs_b.csv", report_root / "comparison_a_vs_b.md"
    figures.write_leaderboard(rows, csv_path, sort_by="model")
    display = []
    for row in rows:
        metric = row["primary_metric"]
        low, high = row.get(f"{metric}_delta_low"), row.get(f"{metric}_delta_high")
        display.append({
            "model": row["model"], "task": row["task"], "metric": metric,
            "with_spike": row[f"{metric}_with_spike"], "non_spike": row[f"{metric}_non_spike"],
            "delta": row[f"{metric}_delta"],
            "delta_95": "n/a" if low is None else f"[{low:+.4f}, {high:+.4f}]",
            "verdict": row["verdict"],
        })
    figures.write_markdown_table(display, md_path, columns=list(display[0]),
                                 title="Paired comparison: non_spike − with_spike (Validation)",
                                 notes=(CONFOUND_NOTE,))
    written = [csv_path, md_path]
    for task, (metric, _) in PRIMARY.items():
        bars = [(row["model"], row[f"{metric}_delta"], row[f"{metric}_delta_low"], row[f"{metric}_delta_high"])
                for row in rows if row["task"] == task]
        path = figure_root / f"comparison_delta_{task}.png"
        figures.plot_interval_bars(bars, path, title=f"non_spike − with_spike ({metric}, Validation)",
                                   xlabel=f"delta in {metric} with 95% paired block-bootstrap interval")
        written.append(path)
    return written

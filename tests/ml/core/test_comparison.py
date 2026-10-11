# tests/ml/core/test_comparison.py
import numpy as np
import pytest

from src.ml.core.artifacts import read_run
from src.ml.core.comparison import CONFOUND_NOTE, paired_comparison, write_comparison
from src.ml.core.trainer import OUTPUT_ROOT


@pytest.fixture(scope="module")
def rows():
    return paired_comparison(replicates=100)


def test_every_model_gets_exactly_one_row(rows):
    assert len(rows) == 19
    assert sum(row["task"] == "classification" for row in rows) == 13


def test_delta_is_non_spike_minus_with_spike(rows):
    row = next(row for row in rows if row["model"] == "knn")
    a = read_run(OUTPUT_ROOT, "with_spike", "knn").metrics["average_precision"]["value"]
    b = read_run(OUTPUT_ROOT, "non_spike", "knn").metrics["average_precision"]["value"]
    assert row["average_precision_with_spike"] == pytest.approx(a)
    assert row["average_precision_delta"] == pytest.approx(b - a)
    assert row["average_precision_delta_low"] <= row["average_precision_delta"] <= row["average_precision_delta_high"]


def test_verdict_is_inconclusive_whenever_the_interval_covers_zero(rows):
    for row in rows:
        if row["task"] == "clustering":
            assert row["verdict"] == "no interval"
        elif row["primary_excludes_zero"] is not True:
            assert row["verdict"] == "inconclusive"
        else:
            assert row["verdict"] in ("non_spike better", "with_spike better")


def test_lower_rmse_counts_as_better(rows):
    for row in rows:
        if row["task"] == "regression" and row["primary_excludes_zero"] is True:
            assert (row["rmse_delta"] < 0) == (row["verdict"] == "non_spike better")


def test_written_markdown_leads_with_the_confound_statement(rows, tmp_path):
    written = write_comparison(rows, report_root=tmp_path / "r", figure_root=tmp_path / "f")
    text = (tmp_path / "r" / "comparison_a_vs_b.md").read_text(encoding="utf-8")
    assert CONFOUND_NOTE in text
    for needle in ("25.0%", "18.4%", "1,733", "1,520"):
        assert needle in CONFOUND_NOTE
    assert all(path.stat().st_size > 0 for path in written)


def test_mismatched_validation_rows_are_rejected(monkeypatch):
    import src.ml.core.comparison as module

    original = module.read_runs

    def shifted(root, variant, **kwargs):
        runs = original(root, variant, **kwargs)
        if variant == "non_spike":
            runs = [type(run)(**{**run.__dict__, "target": np.roll(run.target, 1)}) for run in runs]
        return runs

    monkeypatch.setattr(module, "read_runs", shifted)
    with pytest.raises(ValueError, match="same Validation rows"):
        paired_comparison(replicates=10)

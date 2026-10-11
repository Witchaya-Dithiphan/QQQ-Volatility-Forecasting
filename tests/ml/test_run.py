import json

import pytest

from src.ml.core.trainer import expand_grid
from src.ml.registry import MODELS
from src.ml.run import _load_grid, main


def test_train_a_single_model_succeeds(tmp_path):
    assert main(["train", "--model", "multiple_linear", "--variant", "with_spike",
                 "--output-root", str(tmp_path)]) == 0
    assert (tmp_path / "with_spike" / "regression" / "multiple_linear" / "model.npz").is_file()


def test_train_all_covers_every_registered_model(tmp_path):
    from src.ml.registry import model_names

    assert main(["train", "--all", "--variant", "with_spike", "--output-root", str(tmp_path)]) == 0
    for name in model_names():
        assert list(tmp_path.rglob(f"{name}/model.npz"))


def test_train_reports_reload_status_on_stdout(tmp_path, capsys):
    main(["train", "--model", "multiple_linear", "--variant", "non_spike", "--output-root", str(tmp_path)])
    assert "reload=ok" in capsys.readouterr().out


def test_grid_file_is_applied(tmp_path):
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"multiple_linear": {"fit_intercept": [True, False]}}), encoding="utf-8")
    assert main(["train", "--model", "multiple_linear", "--variant", "with_spike",
                 "--grid", str(grid), "--output-root", str(tmp_path)]) == 0
    results = json.loads(
        (tmp_path / "with_spike" / "regression" / "multiple_linear" / "search_results.json").read_text(encoding="utf-8")
    )
    assert len(results) == 2


@pytest.mark.parametrize("name", sorted(MODELS))
def test_configured_grid_only_names_constructor_arguments(name):
    """Every candidate the config produces must be something the model can be built from."""
    import inspect

    accepted = set(inspect.signature(MODELS[name].__init__).parameters) - {"self"}
    candidates = expand_grid(_load_grid(None, name))
    assert candidates
    for candidate in candidates:
        assert set(candidate) <= accepted
        MODELS[name](**candidate)


def test_tuple_grid_from_config_becomes_named_candidates():
    candidates = expand_grid(_load_grid(None, "xgboost"))
    assert len(candidates) == 8
    assert candidates[0] == {"n_estimators": 50, "max_depth": 1, "learning_rate": 0.1,
                             "reg_lambda": 1, "gamma": 0}
    assert candidates[7] == {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.03,
                             "reg_lambda": 10, "gamma": 0.1}


def test_tuple_grid_file_is_applied(tmp_path):
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"xgboost": {
        "tuples": [[2, 1, 0.1, 1, 0], [3, 2, 0.1, 10, 0.1]],
        "tuple_fields": ["n_estimators", "max_depth", "learning_rate", "reg_lambda", "gamma"],
    }}), encoding="utf-8")
    assert main(["train", "--model", "xgboost", "--variant", "with_spike",
                 "--grid", str(grid), "--output-root", str(tmp_path)]) == 0
    results = json.loads(
        (tmp_path / "with_spike" / "classification" / "xgboost" / "search_results.json").read_text(encoding="utf-8")
    )
    assert [result["params"]["n_estimators"] for result in results] == [2, 3]


def test_unknown_model_name_fails_without_traceback(tmp_path, capsys):
    assert main(["train", "--model", "nope", "--variant", "with_spike", "--output-root", str(tmp_path)]) == 1
    assert "available" in capsys.readouterr().err


def test_unknown_variant_is_rejected_by_the_parser(tmp_path):
    with pytest.raises(SystemExit):
        main(["train", "--model", "multiple_linear", "--variant", "nope", "--output-root", str(tmp_path)])


def test_model_and_all_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        main(["train", "--model", "multiple_linear", "--all", "--variant", "with_spike"])


def test_report_fails_cleanly_when_runs_are_missing(tmp_path, capsys):
    assert main(["report", "--variant", "with_spike", "--output-root", str(tmp_path)]) == 1
    assert "no finished run" in capsys.readouterr().err


def test_report_writes_leaderboards_from_the_committed_artifacts(tmp_path, capsys):
    assert main(["report", "--variant", "non_spike", "--report-root", str(tmp_path / "r"),
                 "--figure-root", str(tmp_path / "f"), "--replicates", "20"]) == 0
    assert (tmp_path / "r" / "non_spike" / "leaderboard_classification.csv").is_file()
    assert "leaderboard_regression.md" in capsys.readouterr().out


def test_finalize_requires_the_explicit_allow_test_flag(capsys):
    assert main(["finalize"]) == 1
    assert "--allow-test" in capsys.readouterr().err


def test_compare_writes_the_paired_table(tmp_path, capsys):
    assert main(["compare", "--report-root", str(tmp_path / "r"), "--figure-root", str(tmp_path / "f"),
                 "--replicates", "20"]) == 0
    assert (tmp_path / "r" / "comparison_a_vs_b.csv").is_file()


def test_analyze_writes_importance_and_coefficients(tmp_path):
    assert main(["analyze", "--variant", "with_spike", "--report-root", str(tmp_path / "r"),
                 "--figure-root", str(tmp_path / "f"), "--repeats", "2", "--skip-walk-forward"]) == 0
    report = tmp_path / "r" / "with_spike"
    assert (report / "feature_importance.csv").is_file() and (report / "linear_coefficients.csv").is_file()
    assert (tmp_path / "f" / "with_spike" / "feature_importance.png").stat().st_size > 0
    assert not (report / "walk_forward.csv").exists()


def test_analyze_can_walk_forward_a_chosen_model(tmp_path):
    assert main(["analyze", "--variant", "with_spike", "--report-root", str(tmp_path / "r"),
                 "--figure-root", str(tmp_path / "f"), "--repeats", "2", "--replicates", "20",
                 "--models", "gaussian_nb"]) == 0
    lines = (tmp_path / "r" / "with_spike" / "walk_forward.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6


@pytest.fixture()
def copied_runs(tmp_path):
    import shutil

    from src.ml.core.contracts import VARIANTS
    from src.ml.core.trainer import OUTPUT_ROOT

    for variant in VARIANTS:
        # As the runs were before finalize: the real one-shot outputs are left out.
        shutil.copytree(OUTPUT_ROOT / variant, tmp_path / variant,
                        ignore=shutil.ignore_patterns("test_metrics.json", "test_predictions.csv"))
    return tmp_path


@pytest.fixture()
def forbid_holdout(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the held-out split was opened")

    monkeypatch.setattr("src.ml.core.data.load_test", refuse)
    monkeypatch.setattr("src.ml.core.data.load_segments", refuse)


def test_finalize_dry_run_writes_the_lock_without_opening_the_holdout(copied_runs, forbid_holdout, capsys):
    assert main(["finalize", "--dry-run", "--output-root", str(copied_runs)]) == 0
    assert (copied_runs / "selection_lock.json").is_file()
    assert "38 runs" in capsys.readouterr().out


def test_finalize_without_a_lock_stops_before_reading_the_holdout(copied_runs, forbid_holdout, capsys):
    assert main(["finalize", "--allow-test", "--output-root", str(copied_runs)]) == 1
    assert "dry-run" in capsys.readouterr().err


def test_finalize_refuses_a_second_run_before_reading_the_holdout(copied_runs, forbid_holdout, capsys):
    main(["finalize", "--dry-run", "--output-root", str(copied_runs)])
    (copied_runs / "finalize_receipt.json").write_text("{}", encoding="utf-8")
    assert main(["finalize", "--allow-test", "--output-root", str(copied_runs)]) == 1
    assert "--rerun-reason" in capsys.readouterr().err


def test_finalize_scores_whatever_the_loaders_return(copied_runs, monkeypatch, capsys):
    import numpy as np

    from src.ml.core.data import load

    validation = load("with_spike").validation
    segments = np.array(["spike_affected"] * 54 + ["non_spike"] * 317)
    monkeypatch.setattr("src.ml.core.data.load_test", lambda *, allow_test=False: validation)
    monkeypatch.setattr("src.ml.core.data.load_segments",
                        lambda split, *, allow_test=False: (validation.dates, segments))
    main(["finalize", "--dry-run", "--output-root", str(copied_runs)])
    assert main(["finalize", "--allow-test", "--output-root", str(copied_runs), "--replicates", "10"]) == 0
    assert (copied_runs / "finalize_receipt.json").is_file()
    assert "finalize_receipt.json" in capsys.readouterr().out


def test_finalize_dry_run_refuses_to_relock_after_the_holdout_was_evaluated(copied_runs, forbid_holdout, capsys):
    (copied_runs / "finalize_receipt.json").write_text("{}", encoding="utf-8")
    assert main(["finalize", "--dry-run", "--output-root", str(copied_runs)]) == 1
    assert "already evaluated" in capsys.readouterr().err

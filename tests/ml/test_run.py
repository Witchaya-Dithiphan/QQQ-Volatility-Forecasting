import json

import pytest

from src.ml.run import main


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


def test_unknown_model_name_fails_without_traceback(tmp_path, capsys):
    assert main(["train", "--model", "nope", "--variant", "with_spike", "--output-root", str(tmp_path)]) == 1
    assert "available" in capsys.readouterr().err


def test_unknown_variant_is_rejected_by_the_parser(tmp_path):
    with pytest.raises(SystemExit):
        main(["train", "--model", "multiple_linear", "--variant", "nope", "--output-root", str(tmp_path)])


def test_model_and_all_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        main(["train", "--model", "multiple_linear", "--all", "--variant", "with_spike"])


def test_report_is_honest_about_not_being_implemented(capsys):
    assert main(["report", "--variant", "with_spike"]) == 1
    assert "not implemented" in capsys.readouterr().err


def test_finalize_requires_the_explicit_allow_test_flag(capsys):
    assert main(["finalize"]) == 1
    assert "--allow-test" in capsys.readouterr().err

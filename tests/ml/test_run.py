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


def test_report_is_honest_about_not_being_implemented(capsys):
    assert main(["report", "--variant", "with_spike"]) == 1
    assert "not implemented" in capsys.readouterr().err


def test_finalize_requires_the_explicit_allow_test_flag(capsys):
    assert main(["finalize"]) == 1
    assert "--allow-test" in capsys.readouterr().err
